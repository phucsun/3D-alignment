"""Window/door candidate detection on a wall plane (geometric substitute for
the paper's Mask R-CNN based segmentation, Section 3.1).

Without a trained segmentation model (and without reflectance data in the
public dataset), candidates are found directly from the 3D geometry of a
single wall: project the wall's inlier points onto the wall's own local
(u, v) frame, rasterize occupancy, and classify gaps in that occupancy by
which border of the wall's footprint they touch:

- a gap fully enclosed by wall points (touches no border) is a **window**;
- a gap open only at the bottom (floor) border is a **door** (doors reach
  the floor, windows don't - the same rule the paper uses, just applied to
  point-density gaps instead of reflectance);
- a gap touching the left/right/top border is the wall's own boundary or a
  corner, not an opening, and is discarded.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from sgd_alignment.common.types import Detection3D, Plane, PointCloud
from sgd_alignment.detection.laser_image import LaserImage


def wall_local_frame(wall: Plane, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_axis, v_axis) spanning the wall plane, v aligned with `up`."""
    v_axis = up - np.dot(up, wall.normal) * wall.normal
    v_axis = v_axis / np.linalg.norm(v_axis)
    u_axis = np.cross(v_axis, wall.normal)
    u_axis = u_axis / np.linalg.norm(u_axis)
    return u_axis, v_axis


def project_to_wall_uv(
    points: np.ndarray, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    d = points - origin
    return d @ u_axis, d @ v_axis


def _estimate_resolution(u: np.ndarray, v: np.ndarray) -> float:
    area = (u.max() - u.min()) * (v.max() - v.min())
    return float(np.sqrt(area / len(u)))


def _rasterize_occupancy(
    u: np.ndarray, v: np.ndarray, resolution: float
) -> tuple[np.ndarray, float, float]:
    u_min, v_min = u.min(), v.min()
    width = int(np.ceil((u.max() - u_min) / resolution)) + 1
    height = int(np.ceil((v.max() - v_min) / resolution)) + 1
    col = np.clip(np.floor((u - u_min) / resolution).astype(np.int64), 0, width - 1)
    row = np.clip(np.floor((v - v_min) / resolution).astype(np.int64), 0, height - 1)
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[row, col] = True
    return occupancy, u_min, v_min


def _classify_gaps(
    occupancy: np.ndarray,
    min_cells: int,
    max_cells: int,
    floor_row: int | None = None,
) -> list[tuple[str, tuple[slice, slice]]]:
    """Label background components and classify each by border contact.

    `floor_row` is the grid row corresponding to the *true* floor level
    (see `estimate_floor_v_in_wall_frame`), which may differ from row 0 of
    the grid if the wall's own points extend below the real floor (e.g. a
    stairwell/lower area visible through an opening). A gap counts as
    "reaching the floor" only if its row range spans that row, not merely
    the bottom of the (possibly over-extended) grid.
    """
    background = ~occupancy
    # 4-connectivity (default `label` structure): diagonal-only contact
    # between two gaps is treated as NOT connected, matching how a human
    # would visually read the image - 8-connectivity let scattered noise
    # holes chain diagonally into one giant background blob spanning the
    # whole wall, swallowing the real gap into it.
    labels, n_labels = ndimage.label(background)
    results: list[tuple[str, tuple[slice, slice]]] = []
    height = occupancy.shape[0]
    floor_row = 0 if floor_row is None else int(np.clip(floor_row, 0, height - 1))

    for label_id in range(1, n_labels + 1):
        mask = labels == label_id
        n_cells = int(mask.sum())
        if not (min_cells <= n_cells <= max_cells):
            continue

        rows, cols = np.where(mask)
        touches_floor = rows.min() <= floor_row <= rows.max()
        touches_top = rows.max() == height - 1
        touches_left = cols.min() == 0
        touches_right = cols.max() == occupancy.shape[1] - 1

        row_slice = slice(rows.min(), rows.max() + 1)
        col_slice = slice(cols.min(), cols.max() + 1)

        if not (touches_floor or touches_top or touches_left or touches_right):
            results.append(("window", (row_slice, col_slice)))
        elif touches_floor:
            # doors reach the floor - still a door even if it also reaches
            # the ceiling/a side border, which happens for a doorway that
            # is wide open onto another space (no return at any height, so
            # the gap spans the wall's full height, not just door height)
            results.append(("door", (row_slice, col_slice)))
        # else: touches only the top/side border (not the floor) -> wall
        # boundary or corner, discard

    return results


def estimate_floor_v_in_wall_frame(
    pc: PointCloud,
    up: np.ndarray,
    origin: np.ndarray,
    v_axis: np.ndarray,
    floor_percentile: float = 1.0,
) -> float:
    """Estimate the true (global) floor level, expressed in a wall's local v.

    A wall's own observed points may not reach all the way down to the
    floor if the lower part of the wall is entirely occluded by furniture
    - using the wall's own min(v) as the "floor" would then wrongly hide
    a door there. Instead, the floor height is estimated from the *whole*
    point cloud (its low percentile along the global up direction, robust
    to a handful of below-floor noise points) and converted into this
    wall's local v coordinate.
    """
    floor_height_along_up = float(np.percentile(pc.points @ up, floor_percentile))
    origin_height_along_up = float(np.dot(origin, up))
    # v_axis is up with the wall-normal component removed, so it is nearly
    # parallel to up for a near-vertical wall; dot(up, v_axis) rescales the
    # height difference into local v units.
    return (floor_height_along_up - origin_height_along_up) * float(np.dot(up, v_axis))


def _wall_cell_centers(
    wall: Plane,
    up: np.ndarray,
    points: np.ndarray,
    resolution: float,
    v_min_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
    u_axis, v_axis = wall_local_frame(wall, up)
    origin = points.mean(axis=0)
    u, v = project_to_wall_uv(points, origin, u_axis, v_axis)
    u_min = u.min()
    v_min = v.min() if v_min_override is None else min(v.min(), v_min_override)
    width = int(np.ceil((u.max() - u_min) / resolution)) + 1
    height = int(np.ceil((v.max() - v_min) / resolution)) + 1

    row_idx, col_idx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    cell_u = u_min + (col_idx + 0.5) * resolution
    cell_v = v_min + (row_idx + 0.5) * resolution
    cell_points = origin + cell_u[..., None] * u_axis + cell_v[..., None] * v_axis
    return cell_points, u, v, u_min, v_min, u_axis, v_axis


def classify_wall_cells(
    pc: PointCloud,
    wall: Plane,
    up: np.ndarray,
    laser_image: LaserImage,
    resolution: float | None = None,
    range_margin: float = 0.10,
    extend_to_floor: bool = True,
):
    """Classify every cell of a wall's (u, v) grid as wall / open / occluded.

    Resolves the fundamental ambiguity of a bare occupancy grid (empty
    could mean "real opening" or "blocked by furniture in front") by
    checking, from the sensor's own viewpoint, what range was actually
    measured along the ray to each cell versus the range expected if the
    wall were solid there:

    - measured ≈ expected  -> **wall** (solid surface here)
    - measured > expected (or no return at all) -> **open** (the ray passed
      through or beyond the wall plane - a real gap)
    - measured < expected -> **occluded** (something nearer blocked the
      ray; we cannot tell what's behind it, so it must NOT be treated as
      an opening)

    Returns (status_grid, u_min, v_min, resolution, u_axis, v_axis, origin, floor_v)
    where status_grid is int8 with 0=wall, 1=open, 2=occluded, and floor_v
    is the true floor level in this wall's local v (None if not extended).
    """
    points = pc.points[wall.inlier_indices]
    u_axis_tmp, v_axis_tmp = wall_local_frame(wall, up)
    origin_tmp = points.mean(axis=0)
    u_tmp, v_tmp = project_to_wall_uv(points, origin_tmp, u_axis_tmp, v_axis_tmp)
    if resolution is None:
        resolution = _estimate_resolution(u_tmp, v_tmp)

    floor_v = None
    if extend_to_floor:
        floor_v = estimate_floor_v_in_wall_frame(pc, up, origin_tmp, v_axis_tmp)

    cell_points, _, _, u_min, v_min, u_axis, v_axis = _wall_cell_centers(
        wall, up, points, resolution, floor_v
    )
    height, width = cell_points.shape[:2]
    flat_points = cell_points.reshape(-1, 3)

    expected_range = np.linalg.norm(flat_points - laser_image.origin, axis=1)
    measured_range = laser_image.sample_range(flat_points)

    status = np.zeros(len(flat_points), dtype=np.int8)
    is_open = np.isnan(measured_range) | (measured_range > expected_range + range_margin)
    is_occluded = (~np.isnan(measured_range)) & (measured_range < expected_range - range_margin)
    status[is_open] = 1
    status[is_occluded] = 2
    status = status.reshape(height, width)

    return status, u_min, v_min, resolution, u_axis, v_axis, points.mean(axis=0), floor_v


def detect_wall_openings_with_occlusion(
    pc: PointCloud,
    wall: Plane,
    up: np.ndarray,
    laser_image: LaserImage,
    resolution: float | None = None,
    range_margin: float = 0.10,
    closing_iterations: int = 1,
    min_area_m2: float = 0.15,
    max_area_m2: float = 6.0,
    extend_to_floor: bool = True,
) -> list[Detection3D]:
    """Detect window/door candidates, distinguishing real openings from
    furniture occlusion via the sensor's own line-of-sight range (see
    `classify_wall_cells`)."""
    status, u_min, v_min, resolution, u_axis, v_axis, origin, floor_v = classify_wall_cells(
        pc, wall, up, laser_image, resolution, range_margin, extend_to_floor
    )

    not_open = status != 1  # wall or occluded: both treated as non-hole
    if closing_iterations > 0:
        not_open = ndimage.binary_closing(not_open, iterations=closing_iterations)

    cell_area = resolution ** 2
    min_cells = max(1, int(min_area_m2 / cell_area))
    max_cells = max(min_cells, int(max_area_m2 / cell_area))

    floor_row = None
    if floor_v is not None:
        floor_row = int(round((floor_v - v_min) / resolution))

    detections = []
    for category, (row_slice, col_slice) in _classify_gaps(not_open, min_cells, max_cells, floor_row):
        u_lo = u_min + col_slice.start * resolution
        u_hi = u_min + col_slice.stop * resolution
        v_lo = v_min + row_slice.start * resolution
        v_hi = v_min + row_slice.stop * resolution
        center_u = (u_lo + u_hi) / 2
        center_v = (v_lo + v_hi) / 2
        center = origin + center_u * u_axis + center_v * v_axis
        detections.append(
            Detection3D(
                category=category,
                center=center,
                u_axis=u_axis,
                v_axis=v_axis,
                normal=wall.normal,
                width=u_hi - u_lo,
                height=v_hi - v_lo,
            )
        )
    return detections


def detect_wall_openings(
    pc: PointCloud,
    wall: Plane,
    up: np.ndarray,
    resolution: float | None = None,
    closing_iterations: int = 1,
    min_area_m2: float = 0.15,
    max_area_m2: float = 6.0,
) -> list[Detection3D]:
    """Detect window/door candidates as gaps in a wall's point occupancy."""
    points = pc.points[wall.inlier_indices]
    u_axis, v_axis = wall_local_frame(wall, up)
    origin = points.mean(axis=0)
    u, v = project_to_wall_uv(points, origin, u_axis, v_axis)

    if resolution is None:
        resolution = _estimate_resolution(u, v)

    occupancy, u_min, v_min = _rasterize_occupancy(u, v, resolution)
    if closing_iterations > 0:
        occupancy = ndimage.binary_closing(occupancy, iterations=closing_iterations)

    cell_area = resolution ** 2
    min_cells = max(1, int(min_area_m2 / cell_area))
    max_cells = max(min_cells, int(max_area_m2 / cell_area))

    detections = []
    for category, (row_slice, col_slice) in _classify_gaps(occupancy, min_cells, max_cells):
        u_lo = u_min + col_slice.start * resolution
        u_hi = u_min + col_slice.stop * resolution
        v_lo = v_min + row_slice.start * resolution
        v_hi = v_min + row_slice.stop * resolution
        center_u = (u_lo + u_hi) / 2
        center_v = (v_lo + v_hi) / 2
        center = origin + center_u * u_axis + center_v * v_axis
        detections.append(
            Detection3D(
                category=category,
                center=center,
                u_axis=u_axis,
                v_axis=v_axis,
                normal=wall.normal,
                width=u_hi - u_lo,
                height=v_hi - v_lo,
            )
        )
    return detections
