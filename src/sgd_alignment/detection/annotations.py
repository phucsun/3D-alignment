"""Manual ground-truth window/door annotations.

Automatic detection (geometric heuristics, PointNet++ 3D segmentation,
YOLO-World 2D detection on multi-view renders) all proved unreliable on
the available data (Section: see project discussion). This module loads
hand-authored annotations instead, so the downstream SGD/matching/
alignment work isn't blocked on solving detection first.

YAML format (one file per scene):

    openings:
      - category: door         # "door" or "window"
        center: [x, y, z]       # 3D center of the opening
        width: 0.9              # extent along the wall (meters)
        height: 2.1             # vertical extent (meters)
        normal: [nx, ny, nz]     # wall's outward normal at this opening
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sgd_alignment.common.types import Detection3D

WORLD_UP = np.array([0.0, 0.0, 1.0])


def load_annotations(path: str | Path, up: np.ndarray = WORLD_UP) -> list[Detection3D]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    detections = []
    for entry in raw.get("openings", []):
        normal = np.asarray(entry["normal"], dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        v_axis = up - np.dot(up, normal) * normal
        v_axis = v_axis / np.linalg.norm(v_axis)
        u_axis = np.cross(v_axis, normal)
        u_axis = u_axis / np.linalg.norm(u_axis)

        detections.append(
            Detection3D(
                category=entry["category"],
                center=np.asarray(entry["center"], dtype=np.float64),
                u_axis=u_axis,
                v_axis=v_axis,
                normal=normal,
                width=float(entry["width"]),
                height=float(entry["height"]),
            )
        )
    return detections
