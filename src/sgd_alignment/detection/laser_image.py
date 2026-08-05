"""Panoramic laser image generation from a single-station point cloud (Section 3.1).

The scanning rig sweeps two angles around a fixed optical center: azimuth
(platform rotation) and elevation (the LiDAR's internal scan angle). Each
3D point can therefore be re-projected onto a 2D equirectangular image
indexed by (azimuth, elevation), exactly mirroring how the panoramic image
was produced on the physical rig. This module reconstructs that image from
already-registered xyz + intensity data, and critically keeps a pixel ->
point-index map so that any 2D region found later (segmentation, edge
detection) can be traced back to its originating 3D points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sgd_alignment.common.types import PointCloud

DEFAULT_ORIGIN = np.zeros(3)


@dataclass
class LaserImage:
    """A panoramic range/intensity image plus its pixel -> point-index map."""

    intensity: np.ndarray  # (H, W), NaN where empty
    range_image: np.ndarray  # (H, W), NaN where empty
    index_map: np.ndarray  # (H, W) int64, -1 where empty
    origin: np.ndarray  # (3,) the assumed scan center
    az_min_deg: float
    az_res_deg: float
    el_min_deg: float
    el_res_deg: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.intensity.shape

    def points_in_mask(self, mask: np.ndarray) -> np.ndarray:
        """Back-project a boolean (H, W) pixel mask to source point indices."""
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} does not match image shape {self.shape}")
        hit = mask & (self.index_map >= 0)
        return self.index_map[hit]

    def sample_range(self, points: np.ndarray) -> np.ndarray:
        """Look up the measured range along the ray towards each 3D point.

        Lets any other surface (e.g. a candidate wall-plane cell that may
        not have its own point) be checked against what the sensor actually
        saw in that direction: NaN means no return was recorded at all.
        """
        azimuth, elevation, _ = spherical_coords(points, self.origin)
        height, width = self.shape
        col = np.floor((azimuth - self.az_min_deg) / self.az_res_deg).astype(np.int64)
        row = np.floor((elevation - self.el_min_deg) / self.el_res_deg).astype(np.int64)
        in_bounds = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        result = np.full(len(points), np.nan)
        col_c = np.clip(col, 0, width - 1)
        row_c = np.clip(row, 0, height - 1)
        result[in_bounds] = self.range_image[row_c[in_bounds], col_c[in_bounds]]
        return result


def spherical_coords(points: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (azimuth_deg, elevation_deg, range) of points around origin.

    azimuth is measured in the x-y plane from the +x axis (platform rotation),
    elevation is measured from the x-y plane towards +z (LiDAR scan angle).
    """
    d = points - origin
    r_xy = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    azimuth = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    elevation = np.degrees(np.arctan2(d[:, 2], r_xy))
    rng = np.sqrt(r_xy ** 2 + d[:, 2] ** 2)
    return azimuth, elevation, rng


def estimate_angular_resolution_deg(points: np.ndarray, origin: np.ndarray) -> float:
    """Estimate a square pixel angular size from point density.

    Approximates the average solid angle covered per point (azimuth span
    x elevation span / N) and returns its square root, so that a laser
    image built at this resolution has roughly one point per pixel on
    average rather than being dominated by empty pixels (too fine) or
    losing edge detail (too coarse).
    """
    azimuth, elevation, _ = spherical_coords(points, origin)
    az_span = np.deg2rad(azimuth.max() - azimuth.min())
    el_span = np.deg2rad(elevation.max() - elevation.min())
    pixel_solid_angle = (az_span * el_span) / len(points)
    return float(np.degrees(np.sqrt(pixel_solid_angle)))


def project_to_laser_image(
    pc: PointCloud,
    origin: np.ndarray | None = None,
    resolution_deg: float | None = None,
) -> LaserImage:
    """Rasterize a point cloud into a panoramic (azimuth x elevation) image.

    When multiple points fall in the same pixel (common since the point
    cloud is far denser than one ray per pixel), the nearest-range point
    wins (z-buffer), so glass/void pass-through returns from far walls
    behind a window do not contaminate the near-surface signal used for
    edge detection.
    """
    if origin is None:
        origin = DEFAULT_ORIGIN
    origin = np.asarray(origin, dtype=np.float64)

    azimuth, elevation, rng = spherical_coords(pc.points, origin)

    if resolution_deg is None:
        resolution_deg = estimate_angular_resolution_deg(pc.points, origin)

    az_min, el_min = azimuth.min(), elevation.min()
    width = int(np.ceil((azimuth.max() - az_min) / resolution_deg)) + 1
    height = int(np.ceil((elevation.max() - el_min) / resolution_deg)) + 1

    col = np.floor((azimuth - az_min) / resolution_deg).astype(np.int64)
    row = np.floor((elevation - el_min) / resolution_deg).astype(np.int64)
    col = np.clip(col, 0, width - 1)
    row = np.clip(row, 0, height - 1)
    flat_idx = row * width + col

    n_pixels = height * width
    range_min = np.full(n_pixels, np.inf)
    np.minimum.at(range_min, flat_idx, rng)

    is_nearest = rng <= range_min[flat_idx] + 1e-9
    candidates = np.where(is_nearest)[0]
    order = np.argsort(flat_idx[candidates], kind="stable")
    sorted_flat = flat_idx[candidates][order]
    sorted_points = candidates[order]
    unique_flat, first_pos = np.unique(sorted_flat, return_index=True)
    best_point_idx = sorted_points[first_pos]

    index_map_flat = np.full(n_pixels, -1, dtype=np.int64)
    index_map_flat[unique_flat] = best_point_idx
    index_map = index_map_flat.reshape(height, width)

    range_image = np.full(n_pixels, np.nan)
    range_image[unique_flat] = rng[best_point_idx]
    range_image = range_image.reshape(height, width)

    intensity_image = np.full((height, width), np.nan)
    if pc.intensity is not None:
        intensity_flat = intensity_image.reshape(-1)
        intensity_flat[unique_flat] = pc.intensity[best_point_idx]

    return LaserImage(
        intensity=intensity_image,
        range_image=range_image,
        index_map=index_map,
        origin=origin,
        az_min_deg=float(az_min),
        az_res_deg=resolution_deg,
        el_min_deg=float(el_min),
        el_res_deg=resolution_deg,
    )


def save_laser_image_png(image: LaserImage, out_path, channel: str = "intensity") -> None:
    """Save the intensity or range channel as a grayscale PNG for visual review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = image.intensity if channel == "intensity" else image.range_image
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, origin="lower", cmap="gray", aspect="auto")
    ax.set_xlabel("azimuth bin")
    ax.set_ylabel("elevation bin")
    fig.colorbar(im, ax=ax, label=channel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
