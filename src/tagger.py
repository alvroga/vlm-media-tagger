#!/usr/bin/env python3
"""
Tag and describe video and still-image files with a local VLM, and write
the results as JSON. The end-to-end entry point; pipeline.py is the
detection core it wraps.

Per file:
  video  probe → shot cuts → frames → VLM tags (2 passes per shot) →
         tag cleanup → video summary → [profanity filter] → [translation]
         RAW camera formats PyAV can't decode (BRAW, R3D, ARRIRAW, X-OCN,
         ProRes RAW) get a 720p proxy via DaVinci Resolve Studio first,
         when it is installed (resolve_proxy.py).
  image  decode (PIL, then macOS `sips` for HEIC/PSD/camera RAW) → one
         1280px frame → VLM tags with the still prompt → description

Usage:
  python src/tagger.py clip.mp4                       # JSON to stdout
  python src/tagger.py footage/ -r -o results.json    # a folder, recursively
  python src/tagger.py photo.heic --lang es --no-description

Progress goes to stderr, so stdout carries only the JSON.
"""
import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import opus_translate
import pipeline
import profanity
import resolve_proxy
import translation_models
from media_probe import UndecodableMediaError
from tag_filter import clean_tag, clean_tags

TOOL_VERSION = "0.1.0"

# Known-RAW extensions: PyAV/ffmpeg can't decode these at all. .mxf is NOT
# here: it is also a normal delivery container; the decode probe sorts out
# ARRIRAW/X-OCN essence at pipeline time.
RAW_EXTS = {".braw", ".r3d", ".ari", ".arx", ".dng"}
VIDEO_EXTS = {
    ".mp4", ".mov", ".mxf", ".avi", ".mkv", ".m4v", ".mts", ".m2ts",
    ".mpg", ".mpeg", ".webm", ".wmv",
} | RAW_EXTS
# .dng stays a VIDEO extension: CinemaDNG sequences are folders of .dng
# frames, and the stills path would tag every frame of a clip separately.
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".psd", ".heic", ".heif",
    ".bmp", ".webp", ".gif",
    ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2",
}

STILL_MAX_DIMENSION = 1280  # same measured sweet spot as extract_frames

RESOLVE_PROXY_FILENAME = "resolve_proxy_720p.mp4"
# 720p, not resolve_proxy's own 1080p default: 360p loses detected classes,
# 1080p adds nothing over 720p but render and decode time.
RESOLVE_PROXY_WIDTH = 1280
RESOLVE_PROXY_HEIGHT = 720

# Connected lazily, once per PROCESS — Resolve startup is ~24s, so it must
# not happen per file. {"resolve": obj, "ours_to_quit": bool} once set.
_resolve_state: dict | None = None
RESOLVE_ALLOWED = True  # --no-resolve sets this False


class ResolveUnavailableError(RuntimeError):
    """DaVinci Resolve Studio isn't installed, scriptable, or allowed — its
    own status, so a RAW file it can't reach isn't reported as broken."""


# -- DaVinci Resolve RAW fallback ---------------------------------------------

def _get_resolve():
    """Connect to (or launch headless) Resolve on first use, then reuse the
    same connection for the rest of the run."""
    global _resolve_state
    if not RESOLVE_ALLOWED:
        raise RuntimeError("DaVinci Resolve Studio is disabled (--no-resolve)")
    if _resolve_state is None:
        resolve, ours_to_quit = resolve_proxy.ensure_resolve_running()
        _resolve_state = {"resolve": resolve, "ours_to_quit": ours_to_quit}
    return _resolve_state["resolve"]


def shutdown_resolve() -> None:
    """Quit Resolve if this run launched or claimed it. Call once, at the
    end of the run (main() does, in a finally block). Never raises."""
    global _resolve_state
    if _resolve_state is None:
        return
    state, _resolve_state = _resolve_state, None
    if state["ours_to_quit"]:
        quit_started = time.time()
        try:
            state["resolve"].Quit()
            print(f"  Resolve Quit() took {time.time() - quit_started:.1f}s")
        except Exception as e:
            print(f"  WARNING: Resolve Quit() failed after "
                  f"{time.time() - quit_started:.1f}s "
                  f"({e.__class__.__name__}: {e}) — it may still be running")
        resolve_proxy.release_ownership()


def _process_raw_via_resolve(media: Path, video_id: str, decode_error: Exception,
                             model: str | None = None) -> list | None:
    """Transcode a 720p proxy of `media` via Resolve into its working dir,
    then run the normal pipeline on the proxy. Returns segments, or None if
    the transcode failed; raises ResolveUnavailableError if Resolve can't
    be used at all."""
    out_dir = config.DATA_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = out_dir / RESOLVE_PROXY_FILENAME

    if proxy_path.exists():
        print(f"    reusing cached Resolve proxy: {proxy_path}")
    else:
        try:
            resolve = _get_resolve()
        except RuntimeError as e:
            print(f"    SKIP: DaVinci Resolve Studio unavailable ({e})")
            raise ResolveUnavailableError(str(e)) from e
        print(f"    not decodable by PyAV/ffmpeg ({decode_error}) — transcoding a "
              f"{RESOLVE_PROXY_HEIGHT}p proxy via DaVinci Resolve Studio…")
        result = resolve_proxy._render_clip(
            resolve, str(media), str(proxy_path),
            width=RESOLVE_PROXY_WIDTH, height=RESOLVE_PROXY_HEIGHT)
        if result is None:
            print("    SKIP: DaVinci Resolve Studio transcode failed")
            return None

    print("    running detection pipeline on the proxy…")
    return pipeline.run(str(proxy_path), model=model, video_id=video_id)


# -- still images -------------------------------------------------------------

def _decode_image(path: Path, work_dir: Path):
    """PIL first; macOS `sips` for what PIL can't open (HEIC, PSD, camera
    RAW). Returns an RGB PIL image, or None."""
    from PIL import Image
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        pil_error = e
    if shutil.which("sips"):
        converted = work_dir / "decoded.jpg"
        proc = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(path), "--out", str(converted)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and converted.exists():
            try:
                return Image.open(converted).convert("RGB")
            except Exception:
                pass
    print(f"    not decodable as an image ({pil_error})")
    return None


def _prepare_still_frames(path: Path, video_id: str) -> bool:
    """Write the image as a one-frame frames/ dir (+index.json) so
    detect_objects_vlm and summarize run on it unchanged: with no cuts.json
    they fall back to interval sampling, which selects exactly that frame."""
    work_dir = config.DATA_DIR / video_id
    frames_dir = work_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)  # never mix with stale frames
    frames_dir.mkdir(parents=True, exist_ok=True)

    img = _decode_image(path, work_dir)
    if img is None:
        return False
    w, h = img.size
    if max(w, h) > STILL_MAX_DIMENSION:
        scale = STILL_MAX_DIMENSION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    frame_path = frames_dir / "frame_000000.jpg"
    img.save(frame_path, quality=85)
    (frames_dir / "index.json").write_text(json.dumps([{
        "frame_idx": 0, "source_frame": 0, "timestamp_sec": 0.0,
        "path": str(frame_path),
    }], indent=2))
    return True


# -- post-processing ----------------------------------------------------------

def _duration_sec(path: Path) -> float | None:
    """Container duration (the frame index can't tell: pHash dedup may
    leave a single frame for a static shot)."""
    import av
    try:
        with av.open(str(path)) as c:
            return round(c.duration / av.time_base, 2) if c.duration else None
    except Exception:
        return None


def _clean_segments(segments: list) -> list:
    """Segments with tag_filter applied to their class (dropped when the
    class is), in the JSON's own shape. Several raw classes can clean to the
    same tag ("red car", "car"); they stay separate segments."""
    out = []
    for s in segments:
        cls = clean_tag(s["class"])
        if cls:
            out.append({"tag": cls, "start_sec": s["start_time"],
                        "end_sec": s["end_time"], "frame_count": s["frame_count"]})
    out.sort(key=lambda s: (s["start_sec"], s["tag"]))
    return out


def _finish_tags(segments: list, lang: str) -> tuple[list, list]:
    """(final tags, English tags). Order matters: clean (English word lists)
    → profanity filter → translate, so nothing profane reaches the
    translator and opus_translate's own hallucination check compares
    against the filtered English."""
    tags = clean_tags(sorted(set(s["class"] for s in segments)))
    if config.FILTER_PROFANITY:
        tags = profanity.filter_tags(tags, "en")
    return (opus_translate.translate_tags(tags, lang) if lang != "en" else tags), tags


def _finish_text(text: str, lang: str) -> tuple[str, str]:
    """(final description, English description), same ordering as tags."""
    if text and config.FILTER_PROFANITY:
        text = profanity.redact(text, "en")
    if text and lang != "en":
        return opus_translate.translate_text(text, lang), text
    return text, text


# -- per file -----------------------------------------------------------------

def process_file(path: Path, *, model: str | None, lang: str, describe: bool,
                 force: bool, keep_workdir: bool) -> dict:
    """Process one file and return its JSON record. Never raises for a
    per-file problem; the record's status says what happened."""
    from detect_objects_vlm import STILL_OBJECT_PROMPT, detect_objects_vlm
    from summarize import summarize_image, summarize_video

    t0 = time.time()
    ext = path.suffix.lower()
    kind = "image" if ext in IMAGE_EXTS else "video"
    record = {"path": str(path), "type": kind, "status": None}
    video_id = config.video_id_from_path(str(path), unique=True)

    def done(status: str, **fields) -> dict:
        record.update(status=status, **fields)
        record["elapsed_sec"] = round(time.time() - t0, 1)
        if not keep_workdir:
            work = config.DATA_DIR / video_id
            if work.is_dir() and not work.is_symlink():
                shutil.rmtree(work, ignore_errors=True)
        return record

    if kind == "image":
        if not _prepare_still_frames(path, video_id):
            return done("image-undecodable")
        segments = detect_objects_vlm(video_id, model=model,
                                      object_prompt=STILL_OBJECT_PROMPT)
        if not segments:
            # One retry: on busy designs the model sometimes merges
            # everything into one >3-word entry the parser drops, giving
            # zero tags; a second call at temperature 0.2 usually lands.
            print("    no tags detected — retrying once")
            segments = detect_objects_vlm(video_id, model=model,
                                          object_prompt=STILL_OBJECT_PROMPT)
        tags, tags_en = _finish_tags(segments, lang)
        desc, desc_en = _finish_text(summarize_image(video_id, model) if describe else "", lang)
        return done("ok", source="native", tags=tags, description=desc,
                    **({"tags_en": tags_en, "description_en": desc_en} if lang != "en" else {}))

    # R3D ships as numbered chunks; only _001.R3D is processed — Resolve
    # plays the rest through as one clip. Checked before any decode so it
    # never reads as a failure.
    if not resolve_proxy.filter_r3d_chunks([str(path)]):
        return done("redundant-chunk",
                    error="not the first chunk of its R3D clip (covered by _001.R3D)")

    source = "native"
    try:
        segments = pipeline.run(str(path), model=model, force=force, video_id=video_id)
    except UndecodableMediaError as e:
        try:
            segments = _process_raw_via_resolve(path, video_id, e, model=model)
        except ResolveUnavailableError as resolve_err:
            return done("resolve-unavailable", error=f"{e}; {resolve_err}")
        if segments is None:
            return done("undecodable", error=str(e))
        source = "resolve-proxy"
    except SystemExit as e:
        return done("pipeline-failed", error=str(e))

    work = config.DATA_DIR / video_id
    cuts = json.loads((work / "cuts.json").read_text()) if (work / "cuts.json").exists() else {}
    decoded = work / RESOLVE_PROXY_FILENAME if source == "resolve-proxy" else path
    tags, tags_en = _finish_tags(segments, lang)
    desc, desc_en = _finish_text(summarize_video(video_id, model) if describe else "", lang)
    return done(
        "ok", source=source,
        duration_sec=_duration_sec(decoded),
        # scenedetect reports 0 scenes for a clip with no cuts: one shot
        shot_count=max(1, cuts.get("scene_count") or 0),
        tags=tags, description=desc, segments=_clean_segments(segments),
        **({"tags_en": tags_en, "description_en": desc_en} if lang != "en" else {}))


def collect_inputs(inputs: list[str], recursive: bool) -> list[Path]:
    files = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files += sorted(f for f in it if f.is_file()
                            and f.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS
                            and not f.name.startswith("._"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"WARNING: not found, skipped: {p}", file=sys.stderr)
    return files


def main() -> int:
    global RESOLVE_ALLOWED
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="video/image files or folders")
    ap.add_argument("-o", "--output", default="-", help="JSON output path (default: stdout)")
    ap.add_argument("-r", "--recursive", action="store_true", help="descend into folders")
    ap.add_argument("--model", default=None,
                    help=f"vision model; with VLM_BACKEND=ollama an Ollama model name (default {config.VISION_MODEL})")
    ap.add_argument("--lang", default=config.OUTPUT_LANGUAGE,
                    help="output language: en, or one in translation_models/manifest.json")
    ap.add_argument("--no-description", action="store_true", help="tags only (faster)")
    ap.add_argument("--filter-profanity", action="store_true",
                    help="drop/redact profanity present in the media itself")
    ap.add_argument("--no-resolve", action="store_true",
                    help="never launch DaVinci Resolve for RAW files")
    ap.add_argument("--force", action="store_true", help="redo cached cuts/frames")
    ap.add_argument("--keep-workdir", action="store_true",
                    help=f"keep per-file frames/cuts under {config.DATA_DIR}")
    a = ap.parse_args()

    if a.filter_profanity:
        config.FILTER_PROFANITY = True
    RESOLVE_ALLOWED = not a.no_resolve
    model = a.model or config.VISION_MODEL
    if a.lang != "en":
        if not translation_models.is_downloadable(a.lang):
            ap.error(f"no translation models for {a.lang!r} in the manifest")
        if translation_models.missing_components(a.lang):
            print(f"Downloading {a.lang} translation models "
                  f"({translation_models.download_size(a.lang) / 1e6:.0f} MB)…", file=sys.stderr)
            ok, msg = translation_models.download_language(a.lang)
            if not ok:
                print(f"ERROR: translation model download failed: {msg}", file=sys.stderr)
                return 1

    files = collect_inputs(a.inputs, a.recursive)
    if not files:
        ap.error("no video or image files found")
    config.check_vlm(model)

    out = {"tool": "vlm-media-tagger", "version": TOOL_VERSION, "model": model,
           "language": a.lang, "generated_at": datetime.now(timezone.utc).isoformat(),
           "files": []}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            for i, f in enumerate(files, 1):
                print(f"[{i}/{len(files)}] {f}")
                rec = process_file(f, model=model, lang=a.lang, describe=not a.no_description,
                                   force=a.force, keep_workdir=a.keep_workdir)
                print(f"    → {rec['status']} ({rec['elapsed_sec']}s)")
                out["files"].append(rec)
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            shutdown_resolve()

    text = json.dumps(out, indent=2, ensure_ascii=False)
    if a.output == "-":
        print(text)
    else:
        Path(a.output).write_text(text + "\n")
        print(f"wrote {a.output}", file=sys.stderr)
    return 0 if all(r["status"] in ("ok", "redundant-chunk") for r in out["files"]) else 2


if __name__ == "__main__":
    sys.exit(main())
