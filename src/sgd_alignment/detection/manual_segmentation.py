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
    aggregate_oriented_wall_normal,
    orient_walls_outward,
    points_to_detection,
)
from sgd_alignment.detection.plane_fitting import estimate_up_vector_manhattan, extract_wall_planes

WORLD_UP = np.array([0.0, 0.0, 1.0])
DOOR_FIELD_PREFIX = "scalar_cua_ra_vao"
WINDOW_FIELD_PREFIX = "scalar_cua_so"
GENERIC_DOOR_FIELD_PREFIX = "scalar_cua"  # bare "cua_1"/"Cua_2" naming, no "_ra_vao"/"_so" qualifier
# CloudCompare's "Edit > Scalar Fields > Add constant SF" tool names the new
# field literally "Constant" unless the person renames it - confirmed on
# real data (Q1_outdoor): a second door selection was left with this default
# name, silently dropping a whole opening (the exact same value-0/NaN-
# elsewhere pattern as every recognized "cua_*" field, just unrenamed).
# Assumed door (not window) since nothing in the name itself says otherwise.
UNRENAMED_DEFAULT_FIELD = "scalar_constant"


def _field_category(field_name: str) -> str | None:
    # CloudCompare exports sometimes have a stray "-" instead of "_" in the
    # door field name (e.g. "scalar_cua_ra-vao_2") depending on how the
    # scalar field was typed/renamed by hand - normalize before matching
    # the prefix so one typo doesn't silently drop a whole opening. Also
    # lowercase first: some exports capitalize the field ("scalar_Cua_1"),
    # and the naming convention itself isn't consistent across data
    # collectors - some just write "cua_1"/"cua_2" (bare "door", no
    # "_ra_vao"/"_so" qualifier) instead of the fuller original convention.
    normalized = field_name.replace("-", "_").lower()
    if normalized.startswith(WINDOW_FIELD_PREFIX):
        return "window"
    if normalized.startswith(DOOR_FIELD_PREFIX):
        return "door"
    if normalized == UNRENAMED_DEFAULT_FIELD:
        return "door"
    if normalized.startswith(GENERIC_DOOR_FIELD_PREFIX):
        return "door"
    return None


def load_manual_segmentation(
    path: str | Path,
    is_outdoor: bool,
    up: np.ndarray | None = None,
    camera_positions: np.ndarray | None = None,
    trim_percentile: float = 0.0,
    return_points: bool = False,
) -> list[Detection3D] | tuple[list[Detection3D], list[np.ndarray]]:
    """Read a CloudCompare-exported .ply with one `scalar_<name>` column
    per hand-segmented opening and return one Detection3D per opening.

    `is_outdoor` must say which side of the wall this scan was taken from
    - required to orient each wall's normal correctly, see
    `_orient_walls_outward`.

    `camera_positions` (optional): pass the scanning device's recorded
    `(N, 3)` positions (e.g. from a DA3/COLMAP source's poses) when
    available - forwarded to `orient_walls_outward`, where it replaces
    point-density asymmetry as the primary outward-orientation signal
    (confirmed on real data to resolve cases where density is an
    unreliable near-tie for one specific wall despite being clear-cut for
    every other wall in the same scene). `None` (default) keeps the exact
    prior density-only behavior for callers without pose data.

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

    `trim_percentile` (default 0.0, exact prior behavior): forwarded to
    `points_to_detection` - trims outlier points before measuring
    width/height. Default is a genuine no-op (see that function's
    docstring for why manual selections don't usually need it); exposed
    here as an opt-in for ablation on scenes with noisier hand selections.

    Cross-checks each wall's RANSAC-derived normal against the openings
    it was assigned (2+ sharing the same nearest wall): aggregating those
    openings' own per-cluster normals (`opening_geometry.
    aggregate_oriented_wall_normal`) gives a 2nd, independent estimate
    that doesn't depend on whole-scene wall-plane detection at all - when
    it disagrees with the RANSAC wall's normal by more than
    `wall_normal_agreement_threshold` (same convention as
    `points_to_detection`'s own per-opening PCA fallback), the aggregated
    estimate is trusted instead for every opening in that wall's group.
    Confirmed on real data (`chua_thay`) that a window reveal's inner and
    outer faces can fragment whole-scene wall detection into several
    close, inconsistent candidate planes even after the up-vector fix in
    `plane_fitting.estimate_up_vector_from_camera_rotation`; this
    aggregation is a 2nd line of defense that doesn't depend on getting
    the whole-scene wall list right at all.

    `return_points` (default False, exact prior behavior/return type):
    when True, also returns each opening's own RAW point cluster (the
    selection before any wall-normal attribution or trimming), one per
    returned `Detection3D` in the same order - needed by
    `alignment.refine_with_wall_normal_lock`.
    """
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    pc = PointCloud(points=points)

    scene_up = up if up is not None else estimate_up_vector_manhattan(pc)
    walls = extract_wall_planes(pc, up=scene_up)
    oriented_normals = orient_walls_outward(walls, pc, is_outdoor, camera_positions=camera_positions)

    # pass 1: collect each opening's own selection + which detected wall
    # (if any) it's nearest to, without building the Detection3D yet -
    # grouping by wall needs every opening's cluster gathered first.
    openings = []  # list of (category, selected_points, wall_index_or_None)
    for field_name in vertex.dtype.names:
        category = _field_category(field_name)
        if category is None:
            continue
        mask = ~np.isnan(vertex[field_name])
        if not np.any(mask):
            continue
        selected = points[mask]
        wall_index = None
        if walls:
            centroid = selected.mean(axis=0)
            best_idx = min(range(len(walls)), key=lambda i: abs(walls[i].signed_distance(centroid[None, :])[0]))
            if abs(walls[best_idx].signed_distance(centroid[None, :])[0]) <= 0.15:
                wall_index = best_idx
        openings.append((category, selected, wall_index))

    # pass 2: for each wall with >=2 openings assigned to it, cross-check
    # the RANSAC wall normal against the openings' own aggregated normal;
    # override for that whole group when they disagree.
    wall_normal_override: dict[int, np.ndarray] = {}
    wall_normal_agreement_threshold = 0.85
    groups: dict[int, list[np.ndarray]] = {}
    for _, selected, wall_index in openings:
        if wall_index is not None:
            groups.setdefault(wall_index, []).append(selected)
    for wall_index, clusters in groups.items():
        if len(clusters) < 2:
            continue
        ransac_normal = oriented_normals[wall_index]
        aggregate, reliable = aggregate_oriented_wall_normal(clusters, [ransac_normal] * len(clusters), scene_up)
        if aggregate is None or not reliable:
            continue
        if abs(float(np.dot(aggregate, ransac_normal))) < wall_normal_agreement_threshold:
            wall_normal_override[wall_index] = aggregate

    detections = []
    for category, selected, wall_index in openings:
        if wall_index is not None and wall_index in wall_normal_override:
            wall_normal = wall_normal_override[wall_index]
        else:
            wall_normal = oriented_normals[wall_index] if wall_index is not None else None
        detections.append(points_to_detection(selected, category, scene_up, wall_normal,
                                               trim_percentile=trim_percentile))
    if return_points:
        return detections, [selected for _, selected, _ in openings]
    return detections
