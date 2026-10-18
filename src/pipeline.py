#!/usr/bin/env python3
"""
Run the full detection pipeline on one video:
  1. detect_cuts      → data/<video_id>/cuts.json
  2. extract_frames   → data/<video_id>/frames/
  3. detect_objects_vlm → data/<video_id>/objects.json

Usage:
  python src/pipeline.py /path/to/video.mp4
  python src/pipeline.py /path/to/video.mp4 --fps 2 --threshold 15 --model minicpm-v
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import av

import config
from detect_cuts import detect_cuts
from detect_objects_vlm import detect_objects_vlm
from extract_frames import extract_frames
from media_probe import UndecodableMediaError, probe_video


def _frames_cache_usable(frames_dir: Path, force: bool,
                         cuts_regenerated: bool) -> bool:
    """Can step 2's cached frames be reused as-is?

    Three ways they can't, beyond the obvious --force:

    - `cuts_regenerated`: step 1 just (re)ran, so whatever produced these
      frames didn't finish or wasn't this source file. The source-mismatch
      guard in step 1 covers the common case by flipping `force` on, but
      only when cuts.json still EXISTS to compare against — a run that
      died between writing frames and writing cuts.json (or a hand-deleted
      cuts.json) would otherwise silently reuse frames extracted from a
      different video, since a proxy and its original share one video_id.
    - index.json unparseable: a truncated write (crash/full disk) would
      otherwise raise JSONDecodeError deep inside detect_objects_vlm
      instead of just regenerating, which is the whole point of a cache.
    - index.json empty: no frames to detect against; redo rather than
      hand step 3 an empty list.
    """
    index_path = frames_dir / "index.json"
    if force or cuts_regenerated or not index_path.exists():
        return False
    try:
        return bool(json.loads(index_path.read_text()))
    except Exception:
        print("      cached frames/index.json is unreadable — re-extracting")
        return False


def _container_duration(video_path: Path) -> float | None:
    """The file's duration, for the last shot's end (the last extracted frame
    can be seconds early when dedup kept only the start of a static tail)."""
    try:
        with av.open(str(video_path)) as c:
            return round(c.duration / av.time_base, 3) if c.duration else None
    except Exception:
        return None


def run(video: str, sample_fps: float = 2.0, threshold: float = 15.0,
        model: str | None = None, force: bool = False,
        video_id: str | None = None) -> list:
    """Raises UndecodableMediaError for media this machine can't decode
    (RAW camera formats) — callers decide whether to fall back or skip.

    video_id: explicit override (tagger.py passes a path-hash-qualified id,
    see config.video_id_from_path, so same-named files in different folders
    don't collide on the same data/ dir). Falls back to the plain
    filename-stem derivation."""
    video_path = Path(video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    probe_video(video_path)  # fail fast on BRAW/R3D/ARRIRAW etc.
    config.check_vlm(model)

    if video_id is None:
        video_id = config.video_id_from_path(video)
    out_dir = config.DATA_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Shot cuts
    cuts_path = out_dir / "cuts.json"
    if cuts_path.exists() and not force:
        # Cache is only valid for the same source file — a proxy and its
        # original share a video_id, so a source switch must invalidate it.
        try:
            cached_source = json.loads(cuts_path.read_text()).get("source")
        except Exception:
            cached_source = None
        if cached_source != str(video_path):
            print(f"[1/3] cached data came from a different source "
                  f"({cached_source}) — redoing cuts + frames")
            force = True

    if cuts_path.exists() and not force:
        print("[1/3] cuts.json exists — skipping (use --force to redo)")
        cuts_regenerated = False
    else:
        cuts_regenerated = True
        print(f"[1/3] Detecting shot cuts in {video_path.name} …")
        try:
            cuts, fps = detect_cuts(str(video_path), threshold=threshold)
        except UndecodableMediaError:
            raise
        except Exception as e:
            # probe_video() above only checks that PyAV can decode ONE frame
            # (media_probe.py) — not a guarantee PySceneDetect's own backend
            # (OpenCV) can produce valid timestamps for the whole file. Found
            # 2026-07-24 on an Atomos Ninja V ProRes RAW .mov: PyAV decodes a
            # token frame fine, but scenedetect's OpenCV backend returns NaN
            # timestamps deep in its decode thread ("cannot convert float NaN
            # to integer") — a plain .mov extension, so it isn't in RAW_EXTS
            # and the upfront probe let it through. Apple ProRes RAW needs
            # DaVinci Resolve Studio (licensed native decode) same as
            # BRAW/R3D/ARRIRAW/X-OCN; treating any scene-detection failure as
            # undecodable routes it into the existing Resolve fallback
            # instead of crashing the batch on an opaque exception.
            raise UndecodableMediaError(
                f"{video_path.name}: shot-cut detection failed ({e})") from e
        cuts_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source":       str(video_path),
            "fps":          fps,
            "duration_sec": _container_duration(video_path),
            "threshold":    threshold,
            "scene_count":  len(cuts),
            "cuts":         cuts,
        }, indent=2, ensure_ascii=False))
        print(f"      {len(cuts)} scenes → {cuts_path}")

    # 2. Frames
    frames_dir = out_dir / "frames"
    if _frames_cache_usable(frames_dir, force, cuts_regenerated):
        print("[2/3] frames/index.json exists — skipping (use --force to redo)")
    else:
        print(f"[2/3] Extracting frames at {sample_fps} fps …")
        try:
            extract_frames(str(video_path), str(frames_dir), sample_fps=sample_fps)
        except UndecodableMediaError:
            raise
        except Exception as e:
            # Same reasoning as step 1's catch, one decode stage later:
            # probe_video() only proves PyAV can decode ONE frame, and
            # extract_frames decodes the WHOLE file — a stream that dies
            # partway through (variable-format MXF, a codec PyAV opens but
            # can't sustain) lands here. Routing it to UndecodableMediaError
            # sends it into tagger.py's Resolve fallback
            # like any other decode failure, instead of surfacing as a
            # generic "pipeline-failed" error. NOTE: step 3 below deliberately gets NO equivalent
            # wrapper — it's VLM inference, not decoding, so an Ollama
            # outage there must NOT be misfiled as undecodable media (it
            # already exits cleanly via check_vlm's SystemExit, which
            # tagger.py maps to "pipeline-failed").
            raise UndecodableMediaError(
                f"{video_path.name}: frame extraction failed ({e})") from e

    # 3. Objects via VLM
    print("[3/3] Detecting objects …")
    segments = detect_objects_vlm(video_id, model=model)

    tags = sorted(set(s["class"] for s in segments))
    print(f"\nTags for {video_path.name}: {', '.join(tags) if tags else '(none)'}")
    return segments


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to the video file")
    ap.add_argument("--fps",       type=float, default=2.0,  help="Frame sampling rate")
    ap.add_argument("--threshold", type=float, default=15.0, help="Cut detection sensitivity (lower = more cuts)")
    ap.add_argument("--model",     default=None, help="Override vision model")
    ap.add_argument("--force",     action="store_true", help="Redo cuts/frames even if cached")
    args = ap.parse_args()
    try:
        run(args.video, sample_fps=args.fps, threshold=args.threshold,
            model=args.model, force=args.force)
    except UndecodableMediaError as e:
        raise SystemExit(f"Cannot decode this file with PyAV/ffmpeg: {e}\n"
                         "RAW camera formats need the DaVinci Resolve Studio "
                         "proxy path (see tagger.py).") from e
