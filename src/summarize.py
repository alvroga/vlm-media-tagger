#!/usr/bin/env python3
"""
Generate a natural-language description of a still image, or a summary of a
video, for the output JSON's "description" field.

Hierarchical approach (validated 2026-07-08 on a 12-min/163-shot 4K file):
caption a sample of the SAME per-shot frames detect_objects_vlm.py already
selects, then combine those captions into one final summary via a second,
text-only call. minicpm-v handles both steps fine — no separate model
needed. Captioning is capped at MAX_SUMMARY_FRAMES regardless of shot
count: captioning every shot of a long video would roughly double VLM cost
on top of object detection, for no measured gain in summary quality (8
frames sampled across 163 shots already gave a fully coherent summary).

Usage:
  python src/summarize.py <video_id>
  python src/summarize.py <video_id> --model qwen3-vl:8b
"""
import argparse
import json
import re

import config
import prompts
import vlm

MAX_SUMMARY_FRAMES = 10
# Still descriptions get more room: the 4.7 set asks for 4-8 sentences plus
# quoted credits, and text-dense posters came within a few tokens of 512.
STILL_DESCRIPTION_TOKENS = 1024

# Prompts live in prompts.py — see detect_objects_vlm.py's note. Module
# __getattr__ (PEP 562) keeps outside importers working; inside this module
# use prompts.get() directly, since a bare global would not go through it.
def __getattr__(name):
    if name in prompts.PROMPT_NAMES:
        return prompts.get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



def _video_duration_sec(video_id: str) -> float:
    """Approximate duration from the last extracted frame's timestamp.
    Only used for length-bucketing the summary (order-of-magnitude, not
    timing-precise) — pHash dedup can drop a static tail and shave a few
    seconds off, which doesn't matter for the buckets below."""
    frames_dir = config.DATA_DIR / video_id / "frames"
    try:
        index = json.loads((frames_dir / "index.json").read_text())
        return index[-1]["timestamp_sec"] if index else 0.0
    except Exception:
        return 0.0


def _shot_count(video_id: str) -> int:
    """Real shot count from cuts.json, PySceneDetect's frame-0 marker
    entry stripped — same interpretation detect_objects_vlm._select_shot_
    frames uses for shot boundaries. 1 for a single continuous take (no
    real cuts detected, or no cuts.json at all — the interval-sampling
    fallback path in _select_summary_frames below)."""
    cuts_path = config.DATA_DIR / video_id / "cuts.json"
    try:
        cuts_data = json.loads(cuts_path.read_text())
    except Exception:
        return 1
    cut_times = [c for c in cuts_data.get("cuts", []) if c.get("frame_num", 1) > 0]
    return len(cut_times) + 1


# Length-scaling added 2026-07-18 (hands-on review): summarize_video used a
# flat "2-3 sentences" for every video, so a 10s broll clip and a 20min
# video came back equally long. Revised same day:
# an initial duration-only version was wrong too — it forced a Stockholm
# skyline timelapse (~24s, but only 2 real shots — a slow pan/dissolve
# piece, documented in its own golden-entry notes as "a moody, low-
# object-count timelapse") into 2 sentences' worth of content it didn't
# have, and the model padded by listing every minor/repeated detail
# instead. The right proxy is content DENSITY — avg seconds per real shot
# (duration/cuts), not raw duration: "a 10s video with 5 cuts might be a
# lot to explain; a 10min video of a steady single shot not". A single continuous take (shot_count<=1) always gets a short
# target regardless of how long it runs.
def _target_summary_length(duration_sec: float, shot_count: int) -> str:
    if shot_count <= 1:
        return "1-2 concise sentences"
    avg_shot_sec = duration_sec / shot_count
    if avg_shot_sec <= 3:
        return "4-5 concise sentences, covering the range of distinct content shown"
    if avg_shot_sec <= 10:
        return "3-4 concise sentences"
    if avg_shot_sec <= 30:
        return "2 concise sentences"
    return "1-2 concise sentences"


def _select_summary_frames(video_id: str) -> list[dict]:
    """Same per-shot/interval frame selection detect_objects_vlm.py uses,
    subsampled down to MAX_SUMMARY_FRAMES if there are more shots than that."""
    from detect_objects_vlm import _select_interval_frames, _select_shot_frames

    frames_dir = config.DATA_DIR / video_id / "frames"
    cuts_path = config.DATA_DIR / video_id / "cuts.json"
    index = json.loads((frames_dir / "index.json").read_text())

    cuts_data = json.loads(cuts_path.read_text()) if cuts_path.exists() else None
    has_cuts = cuts_data and len(cuts_data.get("cuts", [])) > 0
    frames = (
        _select_shot_frames(index, cuts_path)
        if has_cuts
        else _select_interval_frames(index, interval_sec=5.0)
    )

    if len(frames) <= MAX_SUMMARY_FRAMES:
        return frames
    step = len(frames) / MAX_SUMMARY_FRAMES
    return [frames[int(i * step)] for i in range(MAX_SUMMARY_FRAMES)]


def _caption_frame(path: str, model: str, prompt: str | None = None,
                   num_predict: int = 512) -> str:
    # Resolved at call time, not as a default argument, so a prompt
    # override set after import still applies.
    prompt = prompt or prompts.get("CAPTION_PROMPT")
    # num_predict caps runaway generation: a missing cap once let a
    # description spiral into 40+ repeats of one sentence.
    text = vlm.chat([{"role": "user", "content": prompt, "images": [vlm.image_b64(path)]}],
                    model=model, temperature=0.2, num_predict=num_predict)
    return config.strip_thinking(text).strip()


def clean_description(text: str) -> str:
    """Defensive post-filter for description text: strip prompt-scaffolding
    echoes ("Sentence 1:" labels) seen during prompt tuning, stray think
    markers, and repeated sentences, and flatten line breaks — the written
    description should be one plain paragraph. Shared with the golden harness
    so its expectations always describe post-filter output (what would
    actually reach the output).

    The think-marker and repeat guards exist because of a real capture
    (2026-07-11): a thinking-enabled call produced the same sentence 40+
    times separated by unpaired </think> markers — strip_thinking only
    removes PAIRED blocks, so the debris sailed through. think=False and
    num_predict now prevent it at the source; this is defense in depth."""
    text = re.sub(r"(?im)^\s*sentence\s*\d\s*:\s*", "", text)
    text = re.sub(r"</?think>", " ", text)
    # drop "12. " enumeration debris from degenerate list-style repetition
    text = re.sub(r"(?<=[.!?])\s+\d+\.\s+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # de-duplicate identical sentences (order-preserving) — a legitimate
    # 2-4 sentence description never repeats one verbatim
    seen, kept = set(), []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        key = sentence.lower().strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(sentence)
    return " ".join(kept)


def summarize_image(video_id: str, model: str | None = None) -> str:
    """Still-image counterpart of summarize_video: one direct
    composition-focused description call on the single prepared frame (no
    caption→combine hierarchy — that prompt talks about 'a video' and would
    read oddly for a photo). Returns "" on failure."""
    active_model = model or config.VISION_MODEL
    frames_dir = config.DATA_DIR / video_id / "frames"
    try:
        index = json.loads((frames_dir / "index.json").read_text())
        if not index:
            return ""
        return clean_description(
            _caption_frame(index[0]["path"], active_model,
                           prompt=prompts.get("IMAGE_CAPTION_PROMPT"),
                           num_predict=STILL_DESCRIPTION_TOKENS))
    except Exception:
        return ""


def summarize_video(video_id: str, model: str | None = None) -> str:
    """Returns a summary (length scaled to shot density — see
    _target_summary_length), or "" if nothing could be captioned (e.g. no
    frames extracted yet, or every VLM call failed)."""
    active_model = model or config.VISION_MODEL
    frames = _select_summary_frames(video_id)
    length_instruction = _target_summary_length(
        _video_duration_sec(video_id), _shot_count(video_id))

    captions = []
    for frame in frames:
        try:
            caption = _caption_frame(frame["path"], active_model)
        except Exception:
            continue
        if caption:
            captions.append(caption)

    if not captions:
        return ""

    joined = "\n".join(f"- {c}" for c in captions)
    # The combine prompt and its tuning rationale live in prompts.py.
    prompt = prompts.get("VIDEO_SUMMARY_PROMPT").format(
        joined=joined, length_instruction=length_instruction)
    try:
        text = vlm.chat([{"role": "user", "content": prompt}],
                        model=active_model, temperature=0.2, num_predict=512)
    except Exception:
        return ""
    return clean_description(config.strip_thinking(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--model", default=None, help="Override vision model")
    args = ap.parse_args()
    print(summarize_video(args.video_id, model=args.model))
