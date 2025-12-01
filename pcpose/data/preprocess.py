

from pathlib import Path
import cv2
import numpy as np
import json

def build_manifest(video_path: Path, npy_path: Path, out_manifest: Path, out_frames_dir: Path):
    frames = extract_frames(video_path, out_frames_dir)  # returns list of frame paths
    meta = np.load(npy_path)  # shape (N, 16)

    assert len(frames) == meta.shape[0]

    records = []
    for i, frame_path in enumerate(frames):
        x, y, z = meta[i, 0:3]
        # ... rot, vel if you care
        x_min, y_min, x_max, y_max = meta[i, 12:16]

        records.append(dict(
            video=str(video_path),
            frame_idx=int(i),
            image_path=str(frame_path),
            bbox=[float(x_min), float(y_min), float(x_max), float(y_max)],
            pos=[float(x), float(y), float(z)],
            # add vel, rot if needed
        ))

    with open(out_manifest, "w") as f:
        json.dump(records, f, indent=2)


def extract_frames(video_path: Path, out_dir: Path):
    """
    Extracts all frames from a video using cv2 and saves them as PNGs.

    Args:
        video_path: Path to the .mp4/.mov file.
        out_dir: Directory to save extracted frames.

    Returns:
        List of paths to extracted frame images, in order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    frame_paths = []
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
