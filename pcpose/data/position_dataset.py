import torch, json, cv2
from utils import to_tensor

class PositionDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, crop_size=128, pad=0.1, use_pred_boxes=False, detector=None, transform=None):
        with open(manifest_path) as f:
            self.records = json.load(f)
        self.crop_size = crop_size
        self.pad = pad
        self.use_pred_boxes = use_pred_boxes
        self.detector = detector
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["image_path"])[..., ::-1]

        H, W, _ = img.shape

        if self.use_pred_boxes and self.detector is not None:
            # run detector once per frame (you can optimize with caching later)
            img_tensor = to_tensor(img).unsqueeze(0)  # your existing preprocessing
            with torch.no_grad():
                bbox_pred = self.detector(img_tensor)[0].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox_pred
        else:
            x_min, y_min, x_max, y_max = rec["bbox"]

        # pad & crop
        w = x_max - x_min
        h = y_max - y_min
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        scale = 1 + self.pad

        half = 0.5 * scale * max(w, h)
        x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half))
        y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half))

        crop = img[y0:y1, x0:x1]

        if self.transform:
            crop = self.transform(crop)

        pos = torch.tensor(rec["pos"], dtype=torch.float32)

        # optionally also return bbox normalized + frame-level stuff as features
        bbox_norm = torch.tensor([
            x_min / W, y_min / H, x_max / W, y_max / H
        ], dtype=torch.float32)

        return crop, bbox_norm, pos
