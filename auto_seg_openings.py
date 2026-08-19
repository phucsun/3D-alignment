"""Auto-detect door/window point clusters from a DA3 results.npz (GroundingDINO + SAM2),
backproject to 3D, merge across views, and save as OpeningClusters for gravity_align.

Usage:  python auto_seg_openings.py <results.npz> <out_clusters.pkl> [stride]
Outputs a pickle: list of {"category": "door"|"window", "points": (N,3) float64}.
CPU-friendly: `stride` subsamples frames (default 3). Weights download on first run.
"""
import os, sys, pickle
import numpy as np

NPZ = sys.argv[1]
OUT = sys.argv[2]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 3
DEPTH_RANGE = (0.05, 30.0)
MERGE_DISTANCE = 0.6
TEXT_PROMPT = "door. window."
BOX_THRESHOLD = float(os.environ.get("BOX_THR", "0.5"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THR", "0.25"))
NMS_IOU_THRES, CROSS_LABEL_IOU_THRES = 0.7, 0.7

Z = np.load(NPZ)
D, EX, IN, IM = (Z["depth"].astype(np.float32), Z["extrinsics"].astype(np.float64),
                 Z["intrinsics"].astype(np.float64), Z["image"])
views = list(range(0, len(D), STRIDE))
print(f"[load] {len(D)} views, running {len(views)} (stride {STRIDE})", flush=True)

import cv2, torch
from PIL import Image as PILImage
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[models] device={DEVICE}; loading GDINO + SAM2 ...", flush=True)
gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(DEVICE)
sam = SAM("sam2.1_b.pt")
print("[models] ready", flush=True)

def canon(t):
    t = t.lower(); return "door" if "door" in t else ("window" if "window" in t else None)
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

def detect(rgb):
    pil = PILImage.fromarray(rgb)
    inp = gp(images=pil, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = gm(**inp)
    try:
        res = gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD, target_sizes=[pil.size[::-1]])[0]
    except TypeError:
        res = gp.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD, target_sizes=[pil.size[::-1]])[0]
    boxes, scores, labels = [], [], []
    for b, s, tl in zip(res["boxes"], res["scores"], res["labels"]):
        lab = canon(tl if isinstance(tl, str) else str(tl))
        if lab: boxes.append(b.tolist()); scores.append(float(s)); labels.append(lab)
    if not boxes: return np.zeros((0, 4)), np.array([]), np.array([])
    boxes, scores, labels = np.array(boxes), np.array(scores), np.array(labels)
    keep = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        xywh = [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes[idx].tolist()]
        nms = cv2.dnn.NMSBoxes(xywh, scores[idx].tolist(), 0.0, NMS_IOU_THRES)
        keep.extend(idx[np.array(nms).flatten()])
    keep = np.array(keep); boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    order = np.argsort(scores)[::-1]; final = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < CROSS_LABEL_IOU_THRES for j in final): final.append(i)
    final = np.array(final)
    return boxes[final], scores[final], labels[final]

def backproject(mask, depth, K, R, t):
    valid = mask & (depth > DEPTH_RANGE[0]) & (depth < DEPTH_RANGE[1]); ys, xs = np.nonzero(valid)
    if len(xs) == 0: return np.zeros((0, 3))
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]; d = depth[ys, xs]
    return (np.stack([(xs-cx)/fx*d, (ys-cy)/fy*d, d], 1) - t) @ R

raw = []
for n, i in enumerate(views):
    rgb = IM[i]; boxes, scores, labels = detect(rgb)
    if len(boxes):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        s = sam.predict(bgr, bboxes=boxes, verbose=False)[0]
        masks = s.masks.data.cpu().numpy() if s.masks is not None else []
        dep = D[i]
        for box, lab, mask in zip(boxes, labels, masks):
            mb = mask.astype(bool)
            if mb.shape != dep.shape:
                mb = cv2.resize(mb.astype(np.uint8), (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            pts = backproject(mb, dep, IN[i], EX[i][:3,:3], EX[i][:3,3])
            if len(pts) >= 50: raw.append({"label": lab, "points": pts})
    if n % 5 == 0:
        print(f"[detect] {n+1}/{len(views)} views, {len(raw)} raw instances", flush=True)

# merge instances of same label whose centroids are close
clusters = []
for inst in raw:
    c0 = inst["points"].mean(0); m = None
    for c in clusters:
        if c["label"] == inst["label"] and np.linalg.norm(np.concatenate(c["pl"]).mean(0) - c0) < MERGE_DISTANCE:
            m = c; break
    clusters.append({"label": inst["label"], "pl": [inst["points"]]}) if m is None else m["pl"].append(inst["points"])
out = [{"category": c["label"], "points": np.concatenate(c["pl"]).astype(np.float64)} for c in clusters]
out = [o for o in out if len(o["points"]) >= 150]
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "wb") as f: pickle.dump(out, f)
print(f"[done] {len(out)} opening clusters -> {OUT}", flush=True)
for k, o in enumerate(out):
    print(f"   [{k}] {o['category']}  n={len(o['points'])}  center={np.round(o['points'].mean(0),2)}", flush=True)
