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
from sgd_alignment.detection.plane_fitting import estimate_up_vector_manhattan, extract_wall_planes

WORLD_UP = np.array([0.0, 0.0, 1.0])
DOOR_FIELD_PREFIX = "scalar_cua_ra_vao"
WINDOW_FIELD_PREFIX = "scalar_cua_so"


def _field_category(field_name: str) -> str | None:
    if field_name.startswith(DOOR_FIELD_PREFIX):
        return "door"
    if field_name.startswith(WINDOW_FIELD_PREFIX):
        return "window"
    return None


def _nearest_wall_normal(centroid: np.ndarray, walls, pc: PointCloud) -> np.ndarray | None:
    if not walls:
        return None
    best = min(walls, key=lambda w: abs(w.signed_distance(centroid[None, :])[0]))
    return best.normal


def _mask_to_detection(
    points: np.ndarray,
    category: str,
    up: np.ndarray,
    wall_normal: np.ndarray | None,
) -> Detection3D:
    """Measure width/height of the selected points in a wall-aligned
    (u, v) frame, using the *actual detected wall's* normal (already
    reliably outward-oriented, see `load_manual_segmentation`) rather
    than re-deriving orientation from this small cluster alone: a PCA
    fit on just the selection has a sign-ambiguous normal, and using a
    "points away from the scene centroid" heuristic to fix that sign
    is NOT reliable here, since indoor and outdoor point clouds are
    different physical surfaces (interior vs. exterior faces of the same
    wall) with unrelated overall shapes/centroids - not simply a rotated
    copy of each other, so that heuristic can disagree between the two
    scenes for the very same physical wall.
    """
    centroid = points.mean(axis=0)

    if wall_normal is not None:
        normal = wall_normal
    else:
        # fallback if no wall plane was found nearby: PCA on the
        # selection itself (orientation may be inconsistent between
        # scenes - see docstring)
        _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
        normal = vt[-1]

    v_axis = up - np.dot(up, normal) * normal
    v_axis = v_axis / np.linalg.norm(v_axis)
    u_axis = np.cross(v_axis, normal)
    u_axis = u_axis / np.linalg.norm(u_axis)

    # project onto the wall's own plane before measuring extent, so a
    # selection with some depth (frame thickness) doesn't inflate width/height
    onto_plane = points - np.outer((points - centroid) @ normal, normal)
    centered = onto_plane - centroid
    u = centered @ u_axis
    v = centered @ v_axis
    width = float(u.max() - u.min())
    height = float(v.max() - v.min())
    center = centroid + ((u.max() + u.min()) / 2) * u_axis + ((v.max() + v.min()) / 2) * v_axis

    return Detection3D(
        category=category,
        center=center,
        u_axis=u_axis,
        v_axis=v_axis,
        normal=normal,
        width=width,
        height=height,
    )


def load_manual_segmentation(path: str | Path, up: np.ndarray | None = None) -> list[Detection3D]:
    """Read a CloudCompare-exported .ply with one `scalar_<name>` column
    per hand-segmented opening and return one Detection3D per opening.

    Each opening's wall normal is taken from the nearest wall plane
    detected on the *whole* point cloud (via `plane_fitting`), not fit
    independently per opening - see `_mask_to_detection` for why.

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

    detections = []
    for field_name in vertex.dtype.names:
        category = _field_category(field_name)
        if category is None:
            continue
        mask = ~np.isnan(vertex[field_name])
        if not np.any(mask):
            continue
        selected = points[mask]
        wall_normal = _nearest_wall_normal(selected.mean(axis=0), walls, pc)
        detections.append(_mask_to_detection(selected, category, scene_up, wall_normal))
    return detections
