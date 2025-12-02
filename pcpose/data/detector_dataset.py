import torch, json, cv2

class DetectorDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, transform=None):
        with open(manifest_path) as f:
            self.records = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["image_path"])[..., ::-1]  # BGR -> RGB
        if self.transform:
            img = self.transform(img)
        bbox = torch.tensor(rec["bbox"], dtype=torch.float32)
        return img, bbox
