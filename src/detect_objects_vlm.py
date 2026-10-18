#!/usr/bin/env python3
"""
Extract objects from video frames using a vision LLM via Ollama.

Sends one mid-shot frame per shot (cut-aware, needs cuts.json) to the vision
model and asks for a structured list of notable objects. Falls back to fixed
5-second interval sampling when no cuts.json exists.

Writes objects.json (merged segments) and objects_from_vlm.json (run summary
+ raw per-frame detections).

Usage:
  python src/detect_objects_vlm.py <video_id>
  python src/detect_objects_vlm.py <video_id> --model qwen3-vl:8b
"""
import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
import prompts
import vlm

# Concurrent VLM requests — tunable via the PARALLEL_WORKERS env var. Default is 1 (serial) —
# measured 2026-07-15, concurrent requests to this model/backend don't
# parallelize (see config.py's PARALLEL_WORKERS comment for the numbers).
PARALLEL_WORKERS = config.PARALLEL_WORKERS

# Passed on every chat call: without it Ollama unloads the model after ~5 min
# idle, and a long non-VLM phase (4K decode, cut detection on a big file)
# silently pays a full model reload on the next call. 30m comfortably covers
# any per-file gap within a batch.
OLLAMA_KEEP_ALIVE = vlm.OLLAMA_KEEP_ALIVE  # back-compat name

# The prompts live in prompts.py (one module, so a prompt experiment swaps
# them in one place — prompts.override()). Module-level __getattr__ (PEP 562)
# keeps `from detect_objects_vlm import STILL_OBJECT_PROMPT` working for
# outside importers; inside this module use prompts.get(), since a bare
# global would not go through __getattr__.
def __getattr__(name):
    if name in prompts.PROMPT_NAMES:
        return prompts.get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Canonical person-noun set lives in tag_filter (whose clean_tags drops any
# multi-word tag containing one); this exact-match check in _call_vlm is the
# early, cheap first line of the same defense.
from tag_filter import PERSON_NOUNS as _PERSON_NOUNS  # noqa: E402

# Interior sampling for long shots (2026-07-10): one
# mid-shot frame is right for cut-heavy content but structurally blind on
# pans and slow moves — a 30 s skyline pan is ONE shot, so only whatever sat
# center-frame at its midpoint was ever seen (the real reason the NYC night
# clips under-tagged, not run variance). Shots longer than _LONG_SHOT_SEC get
# one sample point per ~_SHOT_SAMPLE_SEC instead. pHash dedup at extraction
# already dropped near-identical frames, so a STATIC long shot has few frames
# to offer anyway — only real content change generates extra VLM calls.
_LONG_SHOT_SEC = 8.0
_SHOT_SAMPLE_SEC = 5.0
# Multi-pass detection (2026-07-10, measured on the golden-set harness):
# each frame is sent DETECT_PASSES times and the noun lists unioned —
# temp-0.2 variance becomes recall. 2 passes captured the ENTIRE measurable
# gain (mean nice-to-have hit-rate 52%→64%, empty-result frames eliminated);
# 3 passes added nothing on top. Repeat passes on the same frame are cheap
# (harness: +28% wall time, not 2× — model hot, image context cached).
# Deliberately a fixed constant, not a config/UI option: a number measured
# to be good, nothing to tune.
# Real-pipeline A/B (a 23-shot clip, 23 frames, 4 parallel workers, warm model):
# 1-pass 71.1s / 56 classes; 2-pass 92.4s / 97 classes → 1.30× time,
# +44/−11 KEPT classes after clean_tags (license plate, headlights,
# side mirror, SUV, dashboard screen… — the small-object recall this
# pipeline exists for).
DETECT_PASSES = 2
# Sharpest-frame selection (2026-07-10): instead of
# blindly taking the frame nearest each sample point, score the few nearest
# candidates by variance of Laplacian (standard no-reference focus measure)
# and send the sharpest — removes the "VLM saw motion blur" failure class.
_SHARPNESS_CANDIDATES = 5


# Deterministic 'night' tag (2026-07-11): a narrow,
# ALWAYS-SAFE signature for unambiguous near-total-darkness content
# (starfields, dark cityscape pans, unlit bokeh) — NOT a general day/night
# classifier. Investigated and rejected: plain mean/median brightness, and
# darkness+color-saturation, both correctly flag true dark-night frames but
# ALSO fire on daytime-but-dim content (overcast skies, shadowed nature,
# dim indoor rooms, studio backdrops) — real golden-set measurement showed
# a lit night skyline (mean luma 88) and an overcast daytime shot (mean 76)
# overlap completely; brightness alone can't tell them apart. This
# signature deliberately stays narrow: calibrated against every golden-set
# frame (29 entries, ~100 frames), the true near-black night frames top out
# at median luma 16.7, while the next-darkest NON-night frame in the whole
# set (a daytime tennis court) sits at 25.1 — a clean, wide gap. The
# flat-frac guard exists because two frames in the set are solid-black
# END-CARD/credit screens (solid-black title and credit cards):
# median luma 0 but >90% pure-black pixels with no texture, vs. real dark
# scenes which stay under ~65% even at their darkest. A LIT night scene
# (a well-lit skyline, streetlights) is NOT dark overall and deliberately
# falls OUTSIDE this signature — that case is left to VLM prompt tuning
# (see docs/findings.md; the explicit VLM day/night instruction tried earlier was
# REJECTED for false-tagging golden-hour footage as night).
_NIGHT_MEDIAN_MAX = 18.0
_NIGHT_FLAT_FRAC_MAX = 90.0


def _is_dark_night_frame(path: str) -> bool:
    """True only for unambiguous near-total-darkness WITH real content
    (see the constants' docstring above for the calibration)."""
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    if float(np.median(img)) >= _NIGHT_MEDIAN_MAX:
        return False
    flat_frac = float((img < 5).mean()) * 100
    return flat_frac < _NIGHT_FLAT_FRAC_MAX


def apply_deterministic_tags(nouns: list[str], frame_path: str) -> list[str]:
    """Non-VLM, pixel-based tags appended after VLM detection for one
    frame. Currently just the narrow 'night' darkness signature above;
    a hook point for any future deterministic (rule-based, not model-
    judged) tag. Called once per frame — not per detection pass, since
    the result is deterministic and never varies by pass."""
    if "night" not in nouns and _is_dark_night_frame(frame_path):
        return nouns + ["night"]
    return nouns


def _sharpness(path: str) -> float:
    """Variance of Laplacian; higher = sharper. 0.0 for unreadable files."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _pick_sharpest(frames: list, target_ts: float, cache: dict) -> dict | None:
    """The sharpest of the (up to _SHARPNESS_CANDIDATES) frames nearest
    target_ts. cache: path → sharpness, shared per video so overlapping
    candidate sets never score a frame twice."""
    candidates = sorted(frames,
                        key=lambda f: abs(f["timestamp_sec"] - target_ts))
    candidates = candidates[:_SHARPNESS_CANDIDATES]
    best = None
    best_score = -1.0
    for f in candidates:
        score = cache.setdefault(f["path"], _sharpness(f["path"]))
        if score > best_score:
            best, best_score = f, score
    return best


def _end_ts(index: list, cuts_data: dict) -> float:
    """End of the last shot: the file's duration when cuts.json has it (the
    last kept frame is early when dedup dropped a static tail, which made a
    static final shot look ~0 s long and go untagged), never earlier than
    the last frame."""
    last = index[-1]["timestamp_sec"] if index else 0.0
    return max(last, float(cuts_data.get("duration_sec") or 0.0))


def _select_shot_frames(index: list, cuts_path: Path) -> list[dict]:
    """Return representative frames per shot using cuts.json boundaries:
    the sharpest frame near the middle of a normal shot; long shots are cut
    into ~_SHOT_SAMPLE_SEC slices with the sharpest frame picked PER SLICE.
    Slicing (not "nearest N frames per sample point") matters on sparse
    shots: extraction's pHash dedup keeps only visually-distinct frames, so
    a 20 s rally shot may hold 4 frames total — a global nearest-N window
    would let one sharpest frame win every sample point, silently dropping
    the other distinct views (caught on real footage 2026-07-10). A slice
    with no frames is a stretch where nothing changed — it costs nothing."""
    cuts_data = json.loads(cuts_path.read_text())
    # PySceneDetect always writes frame 0 as the first cut entry — strip it so
    # we don't get a 0.0→0.0 degenerate slot in boundaries.
    cut_times = [c["timestamp_sec"] for c in cuts_data["cuts"] if c.get("frame_num", 1) > 0]
    boundaries = [0.0] + cut_times + [_end_ts(index, cuts_data)]

    selected = []
    seen: set[int] = set()
    sharp_cache: dict[str, float] = {}
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        duration = e - s
        if duration < 0.3:
            continue
        shot = [f for f in index if s <= f["timestamp_sec"] <= e]
        if not shot:
            continue
        n_slices = (1 if duration <= _LONG_SHOT_SEC
                    else max(2, round(duration / _SHOT_SAMPLE_SEC)))
        slice_len = duration / n_slices
        for k in range(n_slices):
            lo = s + k * slice_len
            hi = s + (k + 1) * slice_len
            in_slice = [f for f in shot if lo <= f["timestamp_sec"] <= hi]
            if not in_slice:
                continue
            frame = _pick_sharpest(in_slice, (lo + hi) / 2, sharp_cache)
            if frame is not None and frame["frame_idx"] not in seen:
                seen.add(frame["frame_idx"])
                selected.append(frame)
    selected.sort(key=lambda f: f["timestamp_sec"])
    return selected


def _select_interval_frames(index: list, interval_sec: float = 5.0) -> list[dict]:
    """Return frames at a fixed interval when no cuts.json is available."""
    if not index:
        return []
    selected = [index[0]]
    last_ts = index[0]["timestamp_sec"]
    for f in index[1:]:
        if f["timestamp_sec"] - last_ts >= interval_sec:
            selected.append(f)
            last_ts = f["timestamp_sec"]
    return selected


class VLMCallFailed(RuntimeError):
    """The VLM backend call itself errored (connection refused, HTTP 5xx,
    timeout) — as opposed to the model responding with nothing usable,
    which is a legitimate empty result. The distinction matters because
    an empty result is recorded as "zero tags", which is correct for a
    genuinely empty frame but silent data corruption when the server simply
    died mid-batch."""


def _call_vlm(frame_entry: dict, model: str | None = None,
              prompt: str | None = None,
              raise_on_error: bool = False) -> list[str]:
    """Send one frame to the vision model; return list of object nouns.
    prompt overrides OBJECT_PROMPT (used by the stills path, which passes
    STILL_OBJECT_PROMPT). raise_on_error=True raises VLMCallFailed when
    the call itself errors instead of swallowing it into an empty list —
    the pipeline uses this to tell an Ollama outage apart from
    an empty frame; the default stays warn-and-continue for the golden
    harness / tier scripts, which want per-image resilience."""
    img_path = Path(frame_entry["path"])
    if not img_path.exists():
        return []

    model = model or config.VISION_MODEL
    # prompts.get(), not a bare global: module-level __getattr__ (PEP 562)
    # is only consulted for attribute access from OUTSIDE the module.
    prompt = prompt or prompts.get("OBJECT_PROMPT")

    is_qwen = "qwen" in model.lower()
    # qwen3-vl needs 2048 to fit thinking block + JSON answer; smaller models use less
    num_predict = 2048 if is_qwen else 512
    # temperature=0 collapses minicpm-v into refusing to detect vehicles on outdoor scenes.
    # temperature=0.5 caused too much variance (one entry dropped from 121→12 detections).
    # 0.2 balances consistency (close to old 0.1) while avoiding the vehicle-detection collapse at 0.
    temperature = 0 if is_qwen else 0.2

    try:
        text = vlm.chat(
            [{"role": "system", "content": prompts.get("SYSTEM_PROMPT")},
             {"role": "user", "content": prompt, "images": [vlm.image_b64(img_path)]}],
            model=model, temperature=temperature, num_predict=num_predict)
        return _parse_vlm_objects(text)
    except Exception as e:
        if raise_on_error:
            raise VLMCallFailed(str(e)) from e
        print(f"  WARNING: VLM call failed ({e})")
        return []


def _parse_vlm_objects(raw_text: str) -> list[str]:
    """Pure parsing: raw VLM response text → cleaned noun list. No network
    calls — split out of _call_vlm specifically so the dirty-output
    handling (the actual source of real bugs: <think> blocks, single-quote
    dicts, {"name": ...} wrappers, hedging parentheticals) is unit-testable
    without Ollama running. Any malformed/unexpected shape returns []
    rather than raising, so it's safe to call directly in tests."""
    text = config.strip_thinking(raw_text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1:
        return []
    parsed = None
    if end > start:
        fragment = text[start:end]
        try:
            parsed = json.loads(fragment)
        except json.JSONDecodeError:
            # minicpm-v sometimes uses single quotes — try ast as fallback
            import ast
            try:
                parsed = ast.literal_eval(fragment)
            except Exception:
                parsed = None
    if parsed is None:
        # Truncated answer (the token cap hit mid-list — MiniCPM-V 4.7 copying
        # a whole headline into an entry, 2026-10-15): keep every COMPLETE
        # quoted entry and close the list, instead of returning nothing.
        i = text.find("[", start)
        last = text.rfind('",')
        if i == -1 or last < i:
            return []
        try:
            parsed = json.loads('{"objects": ' + text[i:last + 1] + "]}")
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("objects", [])
    if not isinstance(raw, list):
        return []
    nouns = []
    for n in raw:
        # some models return {"name": "hat"} dicts instead of plain strings
        if isinstance(n, dict):
            n = n.get("name") or n.get("object") or n.get("label") or ""
        n = str(n).lower().strip().replace("_", " ")
        # drop hedging parentheticals the model sometimes adds, e.g.
        # "liquid (possibly coffee)" — the prompt asks for a bare noun
        n = re.sub(r"\s*\([^)]*\)", "", n).strip()
        if (
            n not in _PERSON_NOUNS
            and "example item" not in n  # defensive: format-placeholder leakage
            and 1 < len(n) <= 30
            and len(n.split()) <= 3
            and "<" not in n
        ):
            nouns.append(n)
    return nouns


def _merge_segments(
    frame_results: list,
    gap_sec: float = 10.0,
    model_name: str = "vlm",
    boundaries: list | None = None,
) -> list:
    """Build segments from per-frame detections.

    When shot boundaries are provided (cut-aware mode), two detections of the
    same noun only merge if they fall within the same shot. This prevents a
    30-second ad from collapsing 9 separate car appearances into one segment.
    Falls back to time-based gap_sec when boundaries is None.
    """
    short = model_name.split(":")[0]

    def _same_shot(ts1: float, ts2: float) -> bool:
        if boundaries is None:
            return ts2 - ts1 <= gap_sec
        for i in range(len(boundaries) - 1):
            is_last = i == len(boundaries) - 2
            lo, hi = boundaries[i], boundaries[i + 1]
            if lo <= ts1 < hi or (is_last and ts1 == hi):
                return lo <= ts2 < hi or (is_last and ts2 == hi)
        return False

    by_key: dict[tuple, list] = defaultdict(list)
    for frame in frame_results:
        ts = frame["timestamp_sec"]
        for noun in frame["nouns"]:
            by_key[(noun, f"{short}:{noun}")].append(ts)

    segments = []
    for (cls, detector), hits in by_key.items():
        hits.sort()
        seg_start = hits[0]; seg_end = hits[0]; count = 1
        for ts in hits[1:]:
            if _same_shot(seg_end, ts):
                seg_end = ts; count += 1
            else:
                segments.append({
                    "class": cls, "detector": detector,
                    "start_time": round(seg_start, 3), "end_time": round(seg_end, 3),
                    "best_time": round(seg_start, 3), "peak_confidence": 1.0,
                    "frame_count": count, "best_bbox": None,
                })
                seg_start = ts; seg_end = ts; count = 1
        segments.append({
            "class": cls, "detector": detector,
            "start_time": round(seg_start, 3), "end_time": round(seg_end, 3),
            "best_time": round(seg_start, 3), "peak_confidence": 1.0,
            "frame_count": count, "best_bbox": None,
        })

    segments.sort(key=lambda s: (-s["frame_count"], s["class"]))
    return segments


def detect_objects_vlm(video_id: str, model: str | None = None,
                       object_prompt: str | None = None) -> list:
    frames_dir   = config.DATA_DIR / video_id / "frames"
    objects_path = config.DATA_DIR / video_id / "objects.json"
    cuts_path    = config.DATA_DIR / video_id / "cuts.json"
    sentinel     = config.DATA_DIR / video_id / "objects_from_vlm.json"

    active_model = model or config.VISION_MODEL

    if not frames_dir.exists():
        print(f"ERROR: {frames_dir} not found — run extract_frames first.")
        sys.exit(1)

    config.check_vlm(active_model)

    index = json.loads((frames_dir / "index.json").read_text())

    shot_boundaries = None
    cuts_data = json.loads(cuts_path.read_text()) if cuts_path.exists() else None
    has_cuts = cuts_data and len(cuts_data.get("cuts", [])) > 0

    if has_cuts:
        frames_to_process = _select_shot_frames(index, cuts_path)
        # Strip frame-0 entries (PySceneDetect convention) to avoid 0.0→0.0 degenerate slot
        cut_times = [c["timestamp_sec"] for c in cuts_data["cuts"] if c.get("frame_num", 1) > 0]
        shot_boundaries = [0.0] + cut_times + [_end_ts(index, cuts_data)]
        print(f"Objects ({active_model}) for {video_id}: {len(frames_to_process)} shots (cut-aware)")
    else:
        frames_to_process = _select_interval_frames(index, interval_sec=5.0)
        print(f"Objects ({active_model}) for {video_id}: {len(frames_to_process)} frames (5s interval — no cuts detected)")

    t0 = time.time()
    total = len(frames_to_process)
    lock = threading.Lock()
    frame_results = []
    completed = 0

    def _process_frame(frame):
        ts = frame["timestamp_sec"]
        nouns: list[str] = []
        responded = False
        for _ in range(DETECT_PASSES):
            try:
                found = _call_vlm(frame, model=active_model,
                                  prompt=object_prompt, raise_on_error=True)
            except VLMCallFailed as e:
                print(f"  WARNING: VLM call failed ({e})")
                continue
            responded = True
            for n in found:
                if n not in nouns:  # dedup within frame (keeps segment counts honest)
                    nouns.append(n)
        if not responded:
            # Every pass failed at the transport level — the model never
            # saw this frame. Returns None (≠ empty list) so the caller
            # can count real failures; deterministic pixel tags are
            # skipped too, so a file can't be recorded as tagged off a
            # lone 'night' tag the VLM had no part in.
            return ts, None
        return ts, apply_deterministic_tags(nouns, frame["path"])

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {}
        for i, frame in enumerate(frames_to_process):
            futures[pool.submit(_process_frame, frame)] = frame
            if i < PARALLEL_WORKERS:
                time.sleep(0.5)  # stagger first submissions so Ollama loads cleanly
        failed = 0
        for future in as_completed(futures):
            ts, nouns = future.result()
            with lock:
                completed += 1
                if nouns is None:
                    failed += 1
                    print(f"  [{completed}/{total}] t={ts:.1f}s … FAILED "
                          f"(no VLM response)", flush=True)
                elif nouns:
                    frame_results.append({"timestamp_sec": ts, "nouns": nouns})
                    print(f"  [{completed}/{total}] t={ts:.1f}s … {', '.join(nouns)}", flush=True)
                else:
                    print(f"  [{completed}/{total}] t={ts:.1f}s … —", flush=True)

    # A mostly-failed run means the VLM server died mid-batch, not that the frames
    # were empty — raising here (instead of returning a thin/empty result)
    # is what keeps tagger.py from writing an empty tag list that reads as
    # "nothing in this video". A few isolated failures are
    # tolerated as transient (llama-server 500s happen) — the frame count
    # below makes them visible.
    if total and failed > total / 2:
        raise VLMCallFailed(
            f"{failed}/{total} frames got no VLM response — the {config.VLM_BACKEND} "
            f"backend looks unavailable; no partial result "
            "written")
    if failed:
        print(f"  WARNING: {failed}/{total} frames got no VLM response — "
              f"tags for this file may be incomplete", flush=True)

    frame_results.sort(key=lambda r: r["timestamp_sec"])
    elapsed = time.time() - t0
    new_segments = _merge_segments(frame_results, gap_sec=10.0, model_name=active_model, boundaries=shot_boundaries)

    objects_path.write_text(json.dumps(new_segments, indent=2))

    unique = set(s["class"] for s in new_segments)
    sentinel.write_text(json.dumps({
        "video_id":       video_id,
        "model":          active_model,
        "elapsed_sec":    round(elapsed, 1),
        "noun_count":     len(new_segments),
        "unique_classes": len(unique),
        "frame_count":    len(frame_results),
        "total_frames":   len(frames_to_process),
        "classes":        sorted(unique),
        "raw_frames":     frame_results,
    }, indent=2))

    mins, secs = divmod(int(elapsed), 60)
    print(f"\nDone → {sentinel}  ({mins}m {secs}s)")
    print(f"  {len(new_segments)} segments, {len(unique)} unique classes from {len(frame_results)}/{len(frames_to_process)} frames")
    if unique:
        print(f"  Classes: {', '.join(sorted(unique))}")
    return new_segments


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--model", default=None, help="Override vision model")
    args = ap.parse_args()
    detect_objects_vlm(args.video_id, model=args.model)
