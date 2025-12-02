from __future__ import annotations
from pathlib import Path
from typing import List
import json

import cv2
import numpy as np


def extract_frames(video_path: Path, out_dir: Path) -> List[str]:
    """
    Extract all frames from a video using OpenCV and save them as PNGs.

    Args:
        video_path: Path to a .mp4/.mov/etc file.
        out_dir   : Directory where frames will be written.

    Returns:
        List of file paths (as strings) to the saved frame images, in order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    frame_paths: List[str] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_file = out_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(frame_file), frame)
        frame_paths.append(str(frame_file))
        frame_idx += 1

    cap.release()
    return frame_paths


def build_manifest(
    video_path: Path,
    npy_path: Path,
    out_manifest: Path,
    out_frames_dir: Path,
) -> None:
    """
    Build a JSON manifest linking each video frame to its metadata.

    Assumes the .npy file is shape (N, 16) with:
        - cols 0:3  -> x,y,z position
        - cols 12:16 -> x_min, y_min, x_max, y_max (bbox in pixels)
    """
    frames = extract_frames(video_path, out_frames_dir)  # list of frame paths
    meta = np.load(npy_path)                             # (N, 16)

    if meta.shape[0] != len(frames):
        raise ValueError(
            f"Frame count mismatch: {len(frames)} frames vs {meta.shape[0]} rows in {npy_path}"
        )

    records = []
    for i, frame_path in enumerate(frames):
        x, y, z = meta[i, 0:3]
        x_min, y_min, x_max, y_max = meta[i, 12:16]

        records.append(
            dict(
                video=str(video_path),
                frame_idx=int(i),
                image_path=str(frame_path),
                bbox=[float(x_min), float(y_min), float(x_max), float(y_max)],
                pos=[float(x), float(y), float(z)],
            )
        )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w") as f:
        json.dump(records, f, indent=2)
