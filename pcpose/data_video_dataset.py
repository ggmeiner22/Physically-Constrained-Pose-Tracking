from __future__ import annotations
import csv, os, random
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

def _read_manifest(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows

def _video_length(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

class VideoClipDataset(Dataset):
    """
    Samples fixed-length clips from videos listed in a CSV manifest.

    Manifest CSV header (example):
        path,scenario,anchor_x,anchor_y,cable_length,slide_y,mu_slide
        datasets/pendulum/clip1.mp4,hanging,128,0,120,,
        datasets/track/clip3.mp4,sliding,,,,128,0.5

    Each sample: dict with
        "frames": FloatTensor (T, C, H, W) normalized to ImageNet stats
        "scenario": str ("hanging"|"sliding"|"dropping")
        "params": dict with any provided physics overrides (floats)
    """
    def __init__(
        self,
        manifest_path: str,
        clip_len: int = 32,
        resize: Tuple[int, int] = (224, 224),
        device: str = "cpu",
        sampling: str = "random",  # or "sequential"
        seed: int = 42,
    ):
        super().__init__()
        self.rows = _read_manifest(manifest_path)
        self.clip_len = clip_len
        self.resize = resize
        self.device = device
        self.sampling = sampling
        self.rng = random.Random(seed)

        # Precompute simple index of (video_idx, max_start)
        self.index: List[Tuple[int, int]] = []
        self.lengths: List[int] = []
        for i, row in enumerate(self.rows):
            vpath = row["image_path"]
            T = _video_length(vpath)
            self.lengths.append(T)
            max_start = max(0, T - clip_len)
            # To keep it simple, index one entry per possible start (capped for huge videos)
            # You can thin this if videos are very long.
            span = min(max_start + 1, 10_000)  # safety cap
            for s in range(span if sampling == "sequential" else max(1, span)):
                self.index.append((i, s))

        # Shuffle index for random sampling
        if sampling == "random":
            self.rng.shuffle(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def _load_clip(self, vpath: str, start: int) -> np.ndarray:
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open {vpath}")
        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: List[np.ndarray] = []
        t = 0
        while t < self.clip_len:
            ok, frame = cap.read()
            if not ok:
                break
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            t += 1
        cap.release()
        if len(frames) < self.clip_len:
            # pad last frame
            while len(frames) < self.clip_len:
                frames.append(frames[-1])
        arr = np.stack(frames, axis=0)  # (T, H, W, C) uint8
        return arr

    def __getitem__(self, idx: int) -> Dict[str, object]:
        vid_idx, start = self.index[idx]
        row = self.rows[vid_idx]
        vpath = row["path"]
        T_total = self.lengths[vid_idx]
        if self.sampling == "random":
            # pick random start
            start = self.rng.randint(0, max(0, T_total - self.clip_len)) if T_total >= self.clip_len else 0

        clip_np = self._load_clip(vpath, start)                 # (T,H,W,C)
        x = torch.from_numpy(clip_np).float() / 255.0           # (T,H,W,C)
        x = x.permute(0, 3, 1, 2).contiguous()                  # (T,C,H,W)
        # Normalize for ResNet
        x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)

        # Scenario & params
        scenario = row.get("scenario", "hanging").strip()
        params: Dict[str, float] = {}
        def putf(key: str):
            val = row.get(key, "")
            if val is not None and str(val).strip() != "":
                try:
                    params[key] = float(val)
                except ValueError:
                    pass
        for k in ["anchor_x", "anchor_y", "cable_length", "slide_y", "mu_slide", "pendulum_damping"]:
            putf(k)

        return {
            "frames": x,                 # (T,C,H,W) float32
            "scenario": scenario,        # str
            "params": params,            # dict of floats
            "video_path": vpath,         # for logging
        }
