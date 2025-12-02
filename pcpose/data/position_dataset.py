# pcpose/data/position_dataset.py
from __future__ import annotations

import json
from typing import Tuple

import cv2
import torch
from torch.utils.data import Dataset

from pcpose.data.utils import to_tensor  # <- correct import


class PositionDataset(Dataset):
    """
    Dataset for training the position regressor.

    Each manifest record should have:
        - "image_path": path to RGB image
        - "bbox": [x_min, y_min, x_max, y_max] in pixels
        - "pos": [x, y, z] target position
    """
    def __init__(
        self,
        manifest_path: str,
        crop_size: int = 128,
        pad: float = 0.1,
        use_pred_boxes: bool = False,
        detector=None,
        transform=None,
    ) -> None:
        with open(manifest_path, "r") as f:
            self.records = json.load(f)
        self.crop_size = crop_size
        self.pad = pad
        self.use_pred_boxes = use_pred_boxes
        self.detector = detector  # optional, for future use
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img = cv2.imread(rec["image_path"])[..., ::-1]  # BGR -> RGB
        H, W, _ = img.shape

        x_min, y_min, x_max, y_max = rec["bbox"]

        # Padding around bbox
        w_box = x_max - x_min
        h_box = y_max - y_min
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        scale = 1.0 + self.pad
        half = 0.5 * scale * max(w_box, h_box)

        x0 = int(max(0, cx - half))
        x1 = int(min(W, cx + half))
        y0 = int(max(0, cy - half))
        y1 = int(min(H, cy + half))

        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = cv2.resize(img, (self.crop_size, self.crop_size))
        else:
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))

        if self.transform is not None:
            crop_t = self.transform(crop)
        else:
            crop_t = to_tensor(crop)  # (3, Hc, Wc)

        pos = torch.tensor(rec["pos"], dtype=torch.float32)
        bbox_norm = torch.tensor(
            [x_min / W, y_min / H, x_max / W, y_max / H],
            dtype=torch.float32,
        )

        return crop_t, bbox_norm, pos
