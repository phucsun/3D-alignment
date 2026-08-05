"""Export a colored point cloud (.ply) for visual review of detection results.

Colors: exterior wall points get a distinct hue per wall; everything else
from the original scan stays neutral gray; each detected window/door gets
a small synthetic marker patch (a dense grid of points spanning its
bounding box, offset slightly in front of the wall) so the opening is
visible even though, by definition, the real scan has few/no points there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyElement, PlyData

from sgd_alignment.common.types import Detection3D, Plane, PointCloud

GRAY = (170, 170, 170)
WALL_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
]
WINDOW_COLOR = (255, 0, 0)   # red, matches the paper's Figure 10 convention
DOOR_COLOR = (0, 200, 0)     # green, matches the paper's Figure 10 convention


def _marker_patch(det: Detection3D, spacing: float = 0.02, offset: float = 0.03) -> np.ndarray:
    """A dense flat grid of points spanning the detection's box, pushed
    slightly out along the wall normal so it renders in front of the wall
    instead of z-fighting with it."""
    n_u = max(2, int(det.width / spacing))
    n_v = max(2, int(det.height / spacing))
    us = np.linspace(-det.width / 2, det.width / 2, n_u)
    vs = np.linspace(-det.height / 2, det.height / 2, n_v)
    uu, vv = np.meshgrid(us, vs)
    points = (
        det.center
        + uu[..., None] * det.u_axis
        + vv[..., None] * det.v_axis
        + offset * det.normal
    )
    return points.reshape(-1, 3)


def inverse_transform_detections(detections: list[Detection3D], R: np.ndarray) -> list[Detection3D]:
    """Map detections found on an upright-aligned cloud back to the
    original (un-rotated) frame, so they can be exported alongside the
    original point cloud instead of a re-oriented copy of it.

    `R` is the same rotation matrix returned by `plane_fitting.align_upright`
    (aligned = original @ R.T); the inverse for row-vectors is `@ R`.
    """
    transformed = []
    for det in detections:
        transformed.append(
            Detection3D(
                category=det.category,
                center=det.center @ R,
                u_axis=det.u_axis @ R,
                v_axis=det.v_axis @ R,
                normal=det.normal @ R,
                width=det.width,
                height=det.height,
                thickness=det.thickness,
            )
        )
    return transformed


def export_segmentation_ply(
    pc: PointCloud,
    walls: list[Plane],
    detections: list[Detection3D],
    out_path: str | Path,
) -> Path:
    n = len(pc)
    colors = np.tile(np.array(GRAY, dtype=np.uint8), (n, 1))

    for i, wall in enumerate(walls):
        colors[wall.inlier_indices] = WALL_PALETTE[i % len(WALL_PALETTE)]

    marker_points = []
    marker_colors = []
    for det in detections:
        patch = _marker_patch(det)
        color = WINDOW_COLOR if det.category == "window" else DOOR_COLOR
        marker_points.append(patch)
        marker_colors.append(np.tile(np.array(color, dtype=np.uint8), (len(patch), 1)))

    all_points = pc.points
    all_colors = colors
    if marker_points:
        all_points = np.vstack([all_points, *marker_points])
        all_colors = np.vstack([all_colors, *marker_colors])

    vertex = np.zeros(
        len(all_points),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = all_colors[:, 0], all_colors[:, 1], all_colors[:, 2]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(out_path))
    return out_path
