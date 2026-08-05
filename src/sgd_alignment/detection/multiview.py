"""Wall-free multi-view orthographic rendering for 2D detection.

Renders the point cloud from several virtual viewpoints around its vertical
axis (like a camera walking around the room and looking outward at each
wall face-on), independent of any wall-plane extraction step. Each view
keeps a pixel -> point-index map so a 2D detector's output (e.g. YOLO)
can be traced back to 3D points without ever having to know where the
walls are ahead of time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sgd_alignment.common.types import PointCloud

WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class View:
    """One orthographic rendering of the point cloud along `view_dir`."""

    density: np.ndarray  # (H, W) uint8
    index_map: np.ndarray  # (H, W) int64, -1 where empty
    u_axis: np.ndarray  # (3,) image +x direction in 3D
    v_axis: np.ndarray  # (3,) image +y direction in 3D
    view_dir: np.ndarray  # (3,) viewing direction (camera looks along +view_dir)
    origin: np.ndarray  # (3,) 3D point mapped to pixel (0, 0)
    resolution: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.density.shape

    def pixel_to_points(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        sub = self.index_map[y1:y2, x1:x2]
        return sub[sub >= 0]


def render_view(
    pc: PointCloud,
    center: np.ndarray,
    azimuth_deg: float,
    resolution: float = 0.03,
) -> View:
    """Render an orthographic view looking inward from outside the room at
    `azimuth_deg` around `center` (0 deg = looking in the +x direction).

    The camera is conceptually infinitely far away along -view_dir, looking
    toward +view_dir; every point in the room projects onto the image, and
    a z-buffer (nearest `depth` per pixel) keeps only the near wall's
    surface - exactly like a real photo, where far-side points only show
    through wherever the near wall has a genuine gap (a window/door).
    """
    az = np.radians(azimuth_deg)
    view_dir = np.array([np.cos(az), np.sin(az), 0.0])
    u_axis = np.array([-np.sin(az), np.cos(az), 0.0])  # horizontal, perpendicular to view_dir
    v_axis = WORLD_UP.copy()

    d = pc.points - center
    depth = d @ view_dir  # signed distance along viewing direction (for z-buffer)
    u = d @ u_axis
    v = d @ v_axis
    idx = np.arange(len(pc.points))

    u_min, v_min = u.min(), v.min()
    width = int(np.ceil((u.max() - u_min) / resolution)) + 1
    height = int(np.ceil((v.max() - v_min) / resolution)) + 1
    col = np.clip(np.floor((u - u_min) / resolution).astype(np.int64), 0, width - 1)
    row = np.clip(np.floor((v - v_min) / resolution).astype(np.int64), 0, height - 1)
    flat = row * width + col

    n_pixels = height * width
    nearest_depth = np.full(n_pixels, np.inf)
    np.minimum.at(nearest_depth, flat, depth)
    is_nearest = depth <= nearest_depth[flat] + 1e-9
    cand = np.where(is_nearest)[0]
    order = np.argsort(flat[cand], kind="stable")
    sorted_flat = flat[cand][order]
    sorted_idx = idx[cand][order]
    uniq_flat, first_pos = np.unique(sorted_flat, return_index=True)
    best_idx = sorted_idx[first_pos]

    index_map = np.full(n_pixels, -1, dtype=np.int64)
    index_map[uniq_flat] = best_idx
    index_map = index_map.reshape(height, width)

    counts = np.zeros(n_pixels)
    np.add.at(counts, flat, 1)
    log_counts = np.log1p(counts)
    density = (log_counts / (log_counts.max() + 1e-9) * 255).astype(np.uint8).reshape(height, width)

    origin = center + u_min * u_axis + v_min * v_axis
    return View(
        density=density,
        index_map=index_map,
        u_axis=u_axis,
        v_axis=v_axis,
        view_dir=view_dir,
        origin=origin,
        resolution=resolution,
    )


def render_sweep(
    pc: PointCloud,
    center: np.ndarray | None = None,
    n_views: int = 8,
    resolution: float = 0.03,
) -> list[View]:
    """Render `n_views` evenly-spaced views looking inward around `center`."""
    if center is None:
        center = pc.points.mean(axis=0)
    return [
        render_view(pc, center, az)
        for az in np.linspace(0, 360, n_views, endpoint=False)
    ]
