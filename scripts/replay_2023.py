#!/usr/bin/env python3
"""Replay a logged tournament dataset offscreen and record it to a video.

Fully headless (EGL) — no X server or display needed. Use after running a
game with eval_2023.py, which logs a dataset.pkl under logs_2023/.

Example:
    python scripts/replay_2023.py logs_2023/eval-*/Game_0/1_tournament_balanced/dataset.pkl
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

# Must be set before mujoco is imported: makes the GL loader resolve through
# EGL, matching the headless context the viewer creates.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from air_hockey_challenge.utils.replay_dataset import replay_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to a logged dataset.pkl")
    parser.add_argument("--out", default=None, help="Output dir for the video (default: alongside the dataset)")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve() if args.out else dataset.parent

    viewer_params = {
        "headless": True,
        "camera_params": {
            "static": dict(distance=3.0, elevation=-45.0, azimuth=90.0, lookat=(0.0, 0.0, 0.0))
        },
        "default_camera_mode": "static",
        "hide_menu_on_startup": True,
    }
    replay_dataset(
        "tournament",
        dataset_path=str(dataset),
        record=True,
        viewer_params=viewer_params,
        record_dir=str(out),
    )

    # mushroom-rl records with the mp4v codec, which browser-based players
    # (e.g. VSCode) cannot decode — re-encode in place to H.264.
    videos = sorted(out.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if videos and shutil.which("ffmpeg"):
        video = videos[-1]
        tmp = video.with_suffix(".h264.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
             "-preset", "fast", str(tmp)],
            check=True,
        )
        tmp.replace(video)
        print(f"Video written to: {video}")
    elif videos:
        print(f"Video written to: {videos[-1]} (mp4v codec — ffmpeg not found, "
              "may not play in VSCode; use VLC or install ffmpeg)")


if __name__ == "__main__":
    main()
