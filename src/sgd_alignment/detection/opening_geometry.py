"""Shared geometry helpers for turning a raw 3D point cluster (however it was
obtained - CloudCompare hand selection, or 2D-mask backprojection from
multi-view photos) into a `Detection3D`, given the wall planes of the scene
it sits in.

Used by `manual_segmentation.py` (CloudCompare workflow) and
`multiview_segmentation.py` (COLMAP/DA3 photo workflow), so the wall-normal
orientation logic and per-opening measurement logic only exist once.
"""
from __future__ import annotations

import numpy as np

from sgd_alignment.common.types import Detection3D, PointCloud


def orient_walls_outward(walls, pc: PointCloud, is_outdoor: bool, margin: float = 0.10) -> dict[int, np.ndarray]:
    """Orient each wall's normal outward, computed once per wall (not per
    opening) so every opening on the same physical wall gets the same sign.

    Two heuristics were tried and rejected before this one: PCA on each
    opening's own small point cluster (sign is arbitrary, no consistency
    across openings at all) and "away from the whole scene's centroid" (the
    indoor scene's overall shape and the outdoor scene's overall shape are
    different physical surfaces - interior vs exterior faces - not a
    rotated copy of each other, so their centroids don't relate the way
    this heuristic needs).

    Point density asymmetry across the wall gets the direction right, but
    the RULE FLIPS depending on which side of the wall the scan was taken
    from - this was the actual bug behind a confirmed, systematic ~180
    degree normal mismatch between every indoor/outdoor pair tested:
      - an INDOOR scan sits inside the room, so the room itself (dense,
        richly captured) is on the *interior* side, and the true exterior
        is barely seen at all -> exterior = the side with FEWER points.
      - an OUTDOOR scan sits outside (e.g. in a corridor), so *that*
        space is what gets densely captured, and the room is only
        glimpsed sparsely through the opening -> exterior = the side
        with MORE points, the opposite rule.
    Applying the "fewer points" rule unconditionally to both silently gets
    every outdoor wall's normal backwards.
    """
    oriented = {}
    for idx, wall in enumerate(walls):
        wall_point = pc.points[wall.inlier_indices].mean(axis=0)
        normal = wall.normal
        signed = pc.points @ normal - np.dot(normal, wall_point)
        frac_pos = float((signed > margin).mean())
        frac_neg = float((signed < -margin).mean())
        # indoor: outward = fewer points on that side. outdoor: outward =
        # MORE points on that side (see docstring). Either way, decide
        # which side ("+" or "-") is outward, then flip normal if that's
        # the negative side.
        positive_side_is_outward = (frac_pos <= frac_neg) if not is_outdoor else (frac_pos > frac_neg)
        if not positive_side_is_outward:
            normal = -normal
        oriented[idx] = normal
    return oriented


def nearest_wall_normal(centroid: np.ndarray, walls, oriented_normals: dict[int, np.ndarray]) -> np.ndarray | None:
    if not walls:
        return None
    best_idx = min(range(len(walls)), key=lambda i: abs(walls[i].signed_distance(centroid[None, :])[0]))
    return oriented_normals[best_idx]


def points_to_detection(
    points: np.ndarray,
    category: str,
    up: np.ndarray,
    wall_normal: np.ndarray | None,
) -> Detection3D:
    """Measure width/height of a raw 3D point cluster belonging to one
    opening, in a wall-aligned (u, v) frame, using the *actual detected
    wall's* normal (already reliably outward-oriented per-wall, see
    `orient_walls_outward`) rather than re-deriving orientation from this
    small cluster alone.
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
    # selection with some depth (frame thickness, or backprojection noise)
    # doesn't inflate width/height
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
