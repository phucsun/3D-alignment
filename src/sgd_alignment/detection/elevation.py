"""Orthographic wall elevation rendering (multi-view 2D projection).

Unlike the panoramic laser image (which needs one fixed sensor origin and
is invalid for a moving/handheld scan), an orthographic projection onto a
wall's own plane only depends on the wall's geometry, not on how it was
scanned. Each wall is rendered as a flat 2D "elevation" image - similar to
a building facade photo - suitable for a 2D detector (e.g. YOLO), with a
simple (u, v) <-> 3D mapping kept for back-projecting any detected pixel
region to its source points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy import ndimage

from sgd_alignment.common.types import Detection3D, Plane, PointCloud
from sgd_alignment.detection.segmentation import (
    _classify_gaps,
    _estimate_resolution,
    project_to_wall_uv,
    wall_local_frame,
)

WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class WallElevation:
    """A flat orthographic rendering of one wall's points."""

    density: np.ndarray  # (H, W) uint8 grayscale, 0=empty .. 255=dense
    u: np.ndarray  # (N,) wall points' u coordinate
    v: np.ndarray  # (N,) wall points' v coordinate
    point_indices: np.ndarray  # (N,) indices into the source PointCloud
    u_min: float
    v_min: float
    resolution: float
    u_axis: np.ndarray
    v_axis: np.ndarray
    origin: np.ndarray
    normal: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.density.shape

    def pixels_to_uv(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        """Convert a pixel-space box (col1,row1,col2,row2) to (u_lo,v_lo,u_hi,v_hi)."""
        u_lo = self.u_min + x1 * self.resolution
        u_hi = self.u_min + x2 * self.resolution
        v_lo = self.v_min + y1 * self.resolution
        v_hi = self.v_min + y2 * self.resolution
        return u_lo, v_lo, u_hi, v_hi

    def points_in_uv_box(self, u_lo: float, v_lo: float, u_hi: float, v_hi: float) -> np.ndarray:
        """Back-project a (u, v) box to the source point indices inside it."""
        mask = (self.u >= u_lo) & (self.u <= u_hi) & (self.v >= v_lo) & (self.v <= v_hi)
        return self.point_indices[mask]

    def to_3d(self, u: float, v: float) -> np.ndarray:
        return self.origin + u * self.u_axis + v * self.v_axis


def render_wall_elevation(
    pc: PointCloud,
    wall: Plane,
    resolution: float | None = None,
) -> WallElevation:
    """Orthographically project a wall's inlier points onto its own plane.

    Assumes the point cloud has already been upright-aligned (world +Z is
    vertical), so the wall's local v axis is simply the vertical direction.
    """
    point_indices = wall.inlier_indices
    points = pc.points[point_indices]
    u_axis, v_axis = wall_local_frame(wall, WORLD_UP)
    origin = points.mean(axis=0)
    u, v = project_to_wall_uv(points, origin, u_axis, v_axis)

    if resolution is None:
        resolution = _estimate_resolution(u, v)

    u_min, v_min = u.min(), v.min()
    width = int(np.ceil((u.max() - u_min) / resolution)) + 1
    height = int(np.ceil((v.max() - v_min) / resolution)) + 1
    col = np.clip(np.floor((u - u_min) / resolution).astype(np.int64), 0, width - 1)
    row = np.clip(np.floor((v - v_min) / resolution).astype(np.int64), 0, height - 1)

    counts = np.zeros((height, width), dtype=np.float64)
    np.add.at(counts, (row, col), 1)
    # log-scale so a handful of stray points don't wash out a solid wall's
    # brightness, then normalize to a standard 0-255 grayscale image
    log_counts = np.log1p(counts)
    density = (log_counts / (log_counts.max() + 1e-9) * 255).astype(np.uint8)

    return WallElevation(
        density=density,
        u=u,
        v=v,
        point_indices=point_indices,
        u_min=u_min,
        v_min=v_min,
        resolution=resolution,
        u_axis=u_axis,
        v_axis=v_axis,
        origin=origin,
        normal=wall.normal,
    )


def detect_wall_openings_orthographic(
    pc: PointCloud,
    wall: Plane,
    resolution: float | None = None,
    closing_size: int = 5,
    closing_iterations: int = 2,
    min_area_m2: float = 0.15,
    max_area_m2: float = 6.0,
    max_door_width_fraction: float = 0.65,
    max_height_m: float | None = 2.2,
    floor_skip_m: float = 0.25,
    door_width_range_m: tuple[float, float] = (0.5, 1.8),
    door_height_range_m: tuple[float, float] = (1.8, 2.3),
) -> list[Detection3D]:
    """Detect window/door candidates as gaps in a wall's orthographic occupancy.

    Same border-touch classification as `segmentation.detect_wall_openings`
    (enclosed gap -> window, floor-touching gap -> door), but computed on
    this sensor-independent orthographic elevation. A stronger morphological
    closing than before is used first, since raw point density here is
    grainy - this merges sampling noise back into the solid wall mass so
    only genuine, sizeable gaps remain.

    A door candidate spanning more than `max_door_width_fraction` of the
    wall's own total width is rejected: a real door never consumes most of
    a wall, so a floor-touching gap that wide is instead a systematic
    near-floor blind spot (e.g. a low cabinet running along the base of the
    wall, or the scan simply not reaching down that far there).

    `max_height_m` crops the grid to just the door-height band above the
    wall's own lowest (floor-level) point before doing anything else. A
    handheld scan tends to be much sparser near the ceiling (held at
    roughly chest/head height while walking), which otherwise creates
    enough scattered holes up there to connect, via diagonal adjacency,
    into one giant background blob spanning the whole wall - swallowing
    the real door gap into it instead of isolating it.

    `floor_skip_m` drops a thin band right at the very bottom before
    labeling. A handheld scan tends to see the floor itself from too
    grazing an angle to get returns within ~20-30cm of it *anywhere* along
    the wall - so that band is empty across its entire width regardless of
    doors, and left in place it silently bridges every gap (including a
    real door) into one wall-spanning background component. Any gap that
    reaches down to this new bottom edge is still treated as floor-
    touching (a real wall surface starts right above the blind band
    everywhere else, so a gap reaching it really does reach the floor).
    """
    elevation = render_wall_elevation(pc, wall, resolution)
    occupancy = elevation.density > 0
    structure = np.ones((closing_size, closing_size))
    closed = ndimage.binary_closing(occupancy, structure=structure, iterations=closing_iterations)

    skip_rows = int(floor_skip_m / elevation.resolution)
    max_row = closed.shape[0]
    if max_height_m is not None:
        # crop only after closing, so closing near the ceiling still has
        # real neighboring context and doesn't create artifacts right at
        # the crop boundary
        max_row = min(max_row, int(max_height_m / elevation.resolution))
    closed = closed[skip_rows:max_row, :]

    cell_area = elevation.resolution ** 2
    min_cells = max(1, int(min_area_m2 / cell_area))
    max_cells = max(min_cells, int(max_area_m2 / cell_area))
    wall_total_width = elevation.shape[1] * elevation.resolution

    detections = []
    for category, (row_slice, col_slice) in _classify_gaps(closed, min_cells, max_cells):
        u_lo, v_lo, u_hi, v_hi = elevation.pixels_to_uv(
            col_slice.start,
            row_slice.start + skip_rows,
            col_slice.stop,
            row_slice.stop + skip_rows,
        )
        width = u_hi - u_lo
        height = v_hi - v_lo
        if category == "door" and width > max_door_width_fraction * wall_total_width:
            continue
        if category == "door" and not (door_width_range_m[0] <= width <= door_width_range_m[1]):
            continue
        if category == "door" and not (door_height_range_m[0] <= height <= door_height_range_m[1]):
            continue
        center = elevation.to_3d((u_lo + u_hi) / 2, (v_lo + v_hi) / 2)
        detections.append(
            Detection3D(
                category=category,
                center=center,
                u_axis=elevation.u_axis,
                v_axis=elevation.v_axis,
                normal=elevation.normal,
                width=width,
                height=v_hi - v_lo,
            )
        )
    return detections


def save_elevation_png(elevation: WallElevation, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(elevation.shape[1] / 100, elevation.shape[0] / 100), dpi=100)
    ax.imshow(elevation.density, origin="lower", cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
