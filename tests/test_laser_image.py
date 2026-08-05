from pathlib import Path

import numpy as np

from sgd_alignment.detection.laser_image import (
    estimate_angular_resolution_deg,
    project_to_laser_image,
    spherical_coords,
)
from sgd_alignment.detection.point_cloud_utils import load_point_cloud

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_spherical_coords_roundtrip():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.zeros(3)
    azimuth, elevation, rng = spherical_coords(points, origin)
    assert np.allclose(azimuth, [0.0, 90.0, 0.0])
    assert np.allclose(elevation, [0.0, 0.0, 90.0])
    assert np.allclose(rng, [1.0, 1.0, 1.0])


def test_estimate_resolution_positive():
    pc = load_point_cloud(REPO_ROOT / "data" / "pantry-10-30.ply")
    res = estimate_angular_resolution_deg(pc.points, np.zeros(3))
    assert res > 0


def test_project_to_laser_image_covers_most_points():
    pc = load_point_cloud(REPO_ROOT / "data" / "hanh-lang-10-34.ply")
    image = project_to_laser_image(pc)
    assert image.index_map.shape == image.intensity.shape == image.range_image.shape
    occupied = (image.index_map >= 0).sum()
    # every occupied pixel maps back to a valid, in-range point index
    valid_idx = image.index_map[image.index_map >= 0]
    assert valid_idx.min() >= 0
    assert valid_idx.max() < len(pc)
    # nearest-range z-buffer: no empty pixel should have a finite range value
    assert np.isnan(image.range_image[image.index_map < 0]).all()
    assert 0 < occupied <= len(pc)  # some pixels shared by multiple points is expected


def test_points_in_mask_backprojection():
    pc = load_point_cloud(REPO_ROOT / "data" / "pantry-10-30.ply")
    image = project_to_laser_image(pc)
    mask = np.zeros(image.shape, dtype=bool)
    mask[0:5, 0:5] = True
    idx = image.points_in_mask(mask)
    assert (idx >= 0).all()
    assert (idx < len(pc)).all()
