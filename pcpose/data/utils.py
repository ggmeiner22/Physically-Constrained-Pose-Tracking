import torch
import numpy as np
import cv2

def to_tensor(img):
    """
    Convert a single HxWx3 uint8 image (RGB) into a torch tensor CxHxW normalized to [0,1].
    """
    if isinstance(img, np.ndarray):
        # HWC uint8 → CHW float32
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC → CHW
        return tensor
    else:
        raise TypeError("to_tensor expects a numpy array (H,W,3).")

def build_crops(frames, bboxes, H, W, pad=0.1, out_size=128):
    """
    Build cropped image tensors and normalized bbox features.

    Args:
        frames: (N, 3, H, W) float tensor OR np.ndarray
        bboxes: (N, 4) tensor/array in pixel coords [x_min, y_min, x_max, y_max]
        H, W: height and width of full frame
        pad: % padding around the bbox
        out_size: crop resize dimension (square)

    Returns:
        crops: (N, 3, out_size, out_size) tensor
        bbox_norm: (N, 4) tensor normalized to [0,1]
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()

    crops = []
    bbox_norm = []

    for i in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[i]

        w = x_max - x_min
        h = y_max - y_min
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        scale = 1 + pad
        half = 0.5 * scale * max(w, h)

        x0 = int(max(0, cx - half))
        x1 = int(min(W, cx + half))
        y0 = int(max(0, cy - half))
        y1 = int(min(H, cy + half))

        # extract crop
        frame_np = frames[i].permute(1, 2, 0).cpu().numpy() * 255.0
        frame_np = frame_np.astype(np.uint8)

        crop = frame_np[y0:y1, x0:x1]
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        crop = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0

        crops.append(crop)

        bbox_norm.append([
            x_min / W, y_min / H,
            x_max / W, y_max / H
        ])

    crops = torch.stack(crops, dim=0)
    bbox_norm = torch.tensor(bbox_norm, dtype=torch.float32)

    return crops, bbox_norm