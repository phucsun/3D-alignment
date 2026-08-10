"""Load hand-segmented window/door point selections exported from CloudCompare.

Workflow this supports: in CloudCompare, select the points belonging to
one opening (its frame/surrounding wall patch) and create a new scalar
field from that selection (named e.g. `cua_ra_vao` for the door, `cua_so_1`
.. `cua_so_N` for windows). Exporting the whole cloud as .ply then yields
one `scalar_<name>` column per opening, valued 0 for points in that
selection and NaN everywhere else. This module turns each such column
directly into a `Detection3D` - no manual coordinate typing needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData

from sgd_alignment.common.types import Detection3D, PointCloud
from sgd_alignment.detection.opening_geometry import (
    nearest_wall_normal,
    orient_walls_outward,
    points_to_detection,
)
from sgd_alignment.detection.plane_fitting import estimate_up_vector_manhattan, extract_wall_planes

WORLD_UP = np.array([0.0, 0.0, 1.0])
DOOR_FIELD_PREFIX = "scalar_cua_ra_vao"
WINDOW_FIELD_PREFIX = "scalar_cua_so"


def _field_category(field_name: str) -> str | None:
    # CloudCompare exports sometimes have a stray "-" instead of "_" in the
    # door field name (e.g. "scalar_cua_ra-vao_2") depending on how the
    # scalar field was typed/renamed by hand - normalize before matching
    # the prefix so one typo doesn't silently drop a whole opening.
    normalized = field_name.replace("-", "_")
    if normalized.startswith(DOOR_FIELD_PREFIX):
        return "door"
    if normalized.startswith(WINDOW_FIELD_PREFIX):
        return "window"
    return None


def load_manual_segmentation(
    path: str | Path,
    is_outdoor: bool,
    up: np.ndarray | None = None,
) -> list[Detection3D]:
    """Read a CloudCompare-exported .ply with one `scalar_<name>` column
    per hand-segmented opening and return one Detection3D per opening.

    `is_outdoor` must say which side of the wall this scan was taken from
    - required to orient each wall's normal correctly, see
    `_orient_walls_outward`.

    Each opening's wall normal is taken from the nearest wall plane
    detected on the *whole* point cloud (via `plane_fitting`), not fit
    independently per opening - see `opening_geometry.points_to_detection`
    for why.

    Deliberately skips `filter_exterior_walls`: that step exists to tell
    real walls apart from furniture/shelving when searching *blindly*
    across a whole scene, and it systematically rejects walls that have a
    lot of window/door area (more see-through -> more points "beyond" it,
    the same signal the filter uses to reject furniture). Here we already
    know roughly where each opening is (from the hand-picked selection),
    so just taking the geometrically nearest wall plane is both simpler
    and more reliable - a manually selected window/door cluster is not
    going to be near a furniture surface by coincidence.
    """
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    pc = PointCloud(points=points)

    scene_up = up if up is not None else estimate_up_vector_manhattan(pc)
    walls = extract_wall_planes(pc, up=scene_up)
    oriented_normals = orient_walls_outward(walls, pc, is_outdoor)

    detections = []
    for field_name in vertex.dtype.names:
        category = _field_category(field_name)
        if category is None:
            continue
        mask = ~np.isnan(vertex[field_name])
        if not np.any(mask):
            continue
        selected = points[mask]
        wall_normal = nearest_wall_normal(selected.mean(axis=0), walls, oriented_normals)
        detections.append(points_to_detection(selected, category, scene_up, wall_normal))
    return detections
