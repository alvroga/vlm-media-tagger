#!/usr/bin/env python3
"""
Extract frames from a video at a target sample rate using PyAV.
Usage: python src/extract_frames.py video.mp4 --fps 2 --output data/<video_id>/frames
"""
import argparse
import json
from pathlib import Path

import av
import imagehash

import config  # noqa: F401 — ensures .env is loaded

_DEDUP_THRESHOLD = 8  # Hamming distance; 0 = disabled


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_fps: float = 2.0,
    max_dimension: int = 1280,
    dedup_threshold: int = _DEDUP_THRESHOLD,
) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = []

    frame_idx = 0
    saved = 0
    skipped_dedup = 0
    last_hash = None

    # Explicit close (via `with`) rather than leaving it to refcounting: a
    # batch run walks thousands of files through here, and a decode failure
    # mid-file (now converted to UndecodableMediaError by pipeline.py) exits
    # this function by exception — the one path where relying on GC to
    # release the container's file handle is least predictable.
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # average_rate can be None on streams with no declared rate (some MXF/
        # transport-stream flavors) — sample every frame rather than crash.
        src_fps = float(stream.average_rate or sample_fps)
        every_n = max(1, int(src_fps / sample_fps))
        # Timestamps on the same clock as detect_cuts (0 = first frame): a
        # stream that starts at a non-zero pts (MPEG-TS/AVCHD, MXF or MOV
        # with an edit list / timecode start) would otherwise put every frame
        # in the wrong shot slot.
        start_pts = stream.start_time or 0

        print(f"Source: {src_fps:.2f} fps  →  sampling 1 of every {every_n} frames")

        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_idx % every_n == 0:
                    ts_sec = (max(0.0, float((frame.pts - start_pts) * stream.time_base))
                              if frame.pts is not None else frame_idx / src_fps)
                    img = frame.to_image()
                    w, h = img.size
                    if max(w, h) > max_dimension:
                        scale = max_dimension / max(w, h)
                        img = img.resize((int(w * scale), int(h * scale)))

                    if dedup_threshold > 0:
                        h = imagehash.dhash(img)
                        if last_hash is not None and (h - last_hash) < dedup_threshold:
                            skipped_dedup += 1
                            frame_idx += 1
                            continue
                        last_hash = h

                    fname = f"frame_{saved:06d}.jpg"
                    img.save(output / fname, quality=85)
                    index.append({
                        "frame_idx":     saved,
                        "source_frame":  frame_idx,
                        "timestamp_sec": round(ts_sec, 4),
                        "path":          str(output / fname),
                    })
                    saved += 1

                    if saved % 500 == 0:
                        print(f"  {saved} frames extracted  ({ts_sec/60:.1f} min into file)")

                frame_idx += 1

    if skipped_dedup:
        print(f"  pHash dedup skipped {skipped_dedup} near-duplicate frames")

    (output / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Done — {saved} frames → {output_dir}")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps",              type=float, default=2.0)
    ap.add_argument("--output",           required=True)
    ap.add_argument("--dedup-threshold",  type=int,   default=_DEDUP_THRESHOLD,
                    help="pHash Hamming distance threshold (0 = disabled)")
    args = ap.parse_args()
    extract_frames(args.video, args.output, args.fps, dedup_threshold=args.dedup_threshold)
