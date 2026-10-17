#!/usr/bin/env python3
"""
Detect shot cuts (hard cuts, fades, dissolves) using PySceneDetect.
Runs directly on the video file — no frames needed.
Usage:
  python src/detect_cuts.py /path/to/video.mp4
  python src/detect_cuts.py /path/to/video.mp4 --threshold 27
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

import config
from timecode_utils import seconds_to_tc


def detect_cuts(video_path: str, threshold: float = 15.0) -> tuple[list[dict], float]:
    video = open_video(video_path)
    fps = video.frame_rate

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    cuts = []
    for start_tc, _ in scene_list:
        frame_num = start_tc.frame_num
        ts = start_tc.seconds
        cuts.append({
            "frame_num":     frame_num,
            "timestamp_sec": round(ts, 4),
            "timecode":      seconds_to_tc(ts, float(fps), drop_frame=False),
        })

    return cuts, float(fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="ContentDetector sensitivity (lower = more cuts detected)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    print(f"Detecting shot cuts in {video_path.name} …", flush=True)
    cuts, fps = detect_cuts(str(video_path), threshold=args.threshold)
    print(f"Found {len(cuts)} scenes ({len(cuts) - 1} cuts)", flush=True)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source":       str(video_path),
        "fps":          fps,
        "threshold":    args.threshold,
        "scene_count":  len(cuts),
        "cuts":         cuts,
    }

    out_dir = config.DATA_DIR / config.video_id_from_path(args.video)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cuts.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
