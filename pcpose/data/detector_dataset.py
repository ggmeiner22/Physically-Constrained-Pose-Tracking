from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import cv2
import torch
from torch.utils.data import Dataset


class DetectorDataset(Dataset):
    """
    Dataset for training the bounding box detector.

    JSON manifest format (list of dicts):
        - "image_path": path to an RGB image
        - "bbox": [x_min, y_min, x_max, y_max] in pixel coords
    """
    def __init__(self, manifest_path: str, transform=None, check_files: bool = True) -> None:
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.transform = transform

        with open(self.manifest_path, "r") as f:
            raw_records = json.load(f)

        if check_files:
            records = []
            for r in raw_records:
                p = Path(r["image_path"])
                if p.is_file():
                    records.append(r)
                else:
                    print(f"[DetectorDataset] WARNING: missing image {p}, skipping.")
            self.records = records
        else:
            self.records = raw_records

        print(f"[DetectorDataset] Loaded {len(self.records)} usable records from {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        img_path = rec["image_path"]
        bbox = rec["bbox"]

        img = cv2.imread(img_path)
        if img is None:
            # Should be rare now, but just in case
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # BGR -> RGB with positive strides
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img_t = self.transform(img)
        else:
            img_t = torch.from_numpy(img).float() / 255.0
            img_t = img_t.permute(2, 0, 1).contiguous()

        bbox_t = torch.tensor(bbox, dtype=torch.float32)
        return img_t, bbox_t
