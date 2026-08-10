"""Grounding DINO + SAM2 door/window detection on a single 2D image.

Kept as a thin, lazily-imported wrapper (heavy ML deps: torch,
transformers, ultralytics - the `segmentation` extra in pyproject.toml)
so the rest of the detection pipeline (`multiview_source.py`,
`multiview_segmentation.py`) has no hard dependency on them and can be
imported/tested without a GPU or these packages installed.

This is the headless equivalent of `door_window_segmentation_in_2D.ipynb`'s
Grounded-SAM cells: same model choice and rationale (Grounding DINO for
open-vocabulary text-prompted detection, SAM2 for mask quality), just
packaged so `pipelines/multiview_pipeline.py` can call it without a
notebook - visual QA is a saved overlay `.jpg` per image (`save_overlay`)
instead of an inline `plt.imshow`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TEXT_PROMPT = "door. window."
DEFAULT_CLASS_COLORS = {"door": (60, 180, 75), "window": (255, 130, 0)}


def _canonical_label(text_label: str) -> str | None:
    label = text_label.lower()
    if "door" in label:
        return "door"
    if "window" in label:
        return "window"
    return None


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class DetectedInstance:
    label: str  # "door" or "window"
    conf: float
    box: list[float]  # [x1, y1, x2, y2] in pixels
    mask: np.ndarray  # (H, W) bool, same resolution as the input image


class DoorWindowSegmenter:
    """Loads Grounding DINO + SAM2 once; call `.segment(image_rgb)` per
    image. Construction requires the `segmentation` extra
    (`pip install -e ".[segmentation]"`)."""

    def __init__(
        self,
        text_prompt: str = DEFAULT_TEXT_PROMPT,
        box_threshold: float = 0.5,
        text_threshold: float = 0.25,
        nms_iou_thres: float = 0.7,
        cross_label_iou_thres: float = 0.7,
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_id: str = "sam2.1_b.pt",
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from ultralytics import SAM

        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_thres = nms_iou_thres
        self.cross_label_iou_thres = cross_label_iou_thres
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._torch = torch
        self._gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
        self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(self.device)
        self._sam = SAM(sam_model_id)

    def _detect_boxes(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cv2
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image_rgb)
        inputs = self._gdino_processor(images=pil_image, text=self.text_prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self._gdino_model(**inputs)
        result = self._gdino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        boxes, scores, labels = [], [], []
        for box, score, text_label in zip(result["boxes"], result["scores"], result["labels"]):
            label = _canonical_label(text_label)
            if label is not None:
                boxes.append(box.tolist())
                scores.append(float(score))
                labels.append(label)
        if not boxes:
            return np.zeros((0, 4)), np.array([]), np.array([])

        boxes, scores, labels = np.array(boxes), np.array(scores), np.array(labels)

        # 1) NMS within each label (several text synonyms can match the same object)
        keep = []
        for label in np.unique(labels):
            idxs = np.where(labels == label)[0]
            xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes[idxs].tolist()]
            nms_idx = cv2.dnn.NMSBoxes(xywh, scores[idxs].tolist(), score_threshold=0.0, nms_threshold=self.nms_iou_thres)
            keep.extend(idxs[np.array(nms_idx).flatten()])
        keep = np.array(keep)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # 2) merge overlapping boxes across the 2 labels, keep higher-confidence one
        order = np.argsort(scores)[::-1]
        final = []
        for i in order:
            if all(_iou(boxes[i], boxes[j]) < self.cross_label_iou_thres for j in final):
                final.append(i)
        final = np.array(final)
        return boxes[final], scores[final], labels[final]

    def segment(self, image_rgb: np.ndarray) -> list[DetectedInstance]:
        import cv2

        boxes, scores, labels = self._detect_boxes(image_rgb)
        if len(boxes) == 0:
            return []

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        sam_result = self._sam.predict(image_bgr, bboxes=boxes, verbose=False)[0]
        masks = sam_result.masks.data.cpu().numpy() if sam_result.masks is not None else []
        h, w = image_rgb.shape[:2]

        instances = []
        for box, label, conf, mask in zip(boxes, labels, scores, masks):
            mask_bool = mask.astype(bool)
            if mask_bool.shape[:2] != (h, w):
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            instances.append(DetectedInstance(label=str(label), conf=float(conf), box=box.tolist(), mask=mask_bool))
        return instances


def save_overlay(
    image_rgb: np.ndarray,
    instances: list[DetectedInstance],
    out_path: str,
    class_colors: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    """Save a visual-QA overlay (mask + box + label) to disk - the
    headless replacement for a notebook's inline `plt.imshow`."""
    import cv2

    class_colors = class_colors or DEFAULT_CLASS_COLORS
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for inst in instances:
        color = class_colors.get(inst.label, (0, 0, 255))
        overlay[inst.mask] = (0.5 * np.array(color) + 0.5 * overlay[inst.mask]).astype(np.uint8)
        x1, y1, x2, y2 = (int(v) for v in inst.box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{inst.label} {inst.conf:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, overlay)
