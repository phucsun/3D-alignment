"""Visual-style embedding for an object's own image crop (CLIP), for the
optional visual-similarity term in `matching.weight_calibration`.

Requires a source image with the detection's region actually marked (a
green-highlighted segmentation overlay, as produced by this project's 2D
detection step) - most of this project's datasets have no such imagery at
all (the CloudCompare/LiDAR scenes are point-cloud-only, and several DA3
runs kept only the raw video frames without an overlay), so this module is
usable only where that overlay was actually saved; see `CONTRIBUTIONS.md`
for which datasets qualify.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def extract_green_box_crop(image_path: str, pad_frac: float = 0.03) -> Image.Image | None:
    """Crop the region highlighted by this project's green segmentation
    overlay (`multiview_segmentation`'s review exports) - the alpha-blended
    green mask is detected by simple color dominance (G channel well above
    both R and B), then the tight bounding box of its largest region is
    cropped with a small margin. Returns `None` if no green region is found
    (nothing to crop - e.g. a raw, unannotated frame).

    Note the returned crop still carries the green tint from the overlay
    (door texture/shading remains visible through it, but exact original
    color is not recoverable from this image alone) - a real fidelity
    limitation of using already-exported review images instead of the raw
    detector output, worth stating alongside any result computed from it.
    """
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mask = (g > r + 25) & (g > b + 25) & (g > 80)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = img.shape[0], img.shape[1]
    pad_x = int((x1 - x0) * pad_frac)
    pad_y = int((y1 - y0) * pad_frac)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    return Image.open(image_path).convert("RGB").crop((x0, y0, x1, y1))


class ClipEmbedder:
    """Thin wrapper around OpenAI CLIP's image encoder - loads the model
    once, reused across many crops (loading `ViT-B/32` per call would be
    needlessly slow)."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        import clip
        import torch

        self._torch = torch
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.device = device

    def embed(self, image: Image.Image) -> np.ndarray:
        """Unit-normalized CLIP embedding of one crop."""
        with self._torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().numpy()

    def embed_image_path(self, image_path: str, pad_frac: float = 0.03) -> np.ndarray | None:
        crop = extract_green_box_crop(image_path, pad_frac=pad_frac)
        return None if crop is None else self.embed(crop)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
