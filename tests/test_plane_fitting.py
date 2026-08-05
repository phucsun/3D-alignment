from pathlib import Path

import numpy as np

from sgd_alignment.detection.plane_fitting import estimate_up_vector, extract_wall_planes
from sgd_alignment.detection.point_cloud_utils import load_point_cloud

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "Indoor-Outdoor-Point-Cloud-Dataset-main"


def test_estimate_up_vector_is_unit_and_near_vertical():
    pc = load_point_cloud(DATASET_DIR / "scenario1_room1_indoor.ply")
    up = estimate_up_vector(pc)
    assert np.isclose(np.linalg.norm(up), 1.0)
    # this dataset is well-leveled (unlike the raw single-station scans),
    # so the raw z-axis should already be close to true vertical
    assert up[2] > 0.99


def test_extract_wall_planes_indoor_room_finds_thin_flat_walls():
    pc = load_point_cloud(DATASET_DIR / "scenario1_room1_indoor.ply")
    up = estimate_up_vector(pc)
    walls = extract_wall_planes(pc, up=up)
    assert len(walls) >= 4

    walls_sorted = sorted(walls, key=lambda w: -len(w.inlier_indices))
    for wall in walls_sorted[:4]:
        pts = pc.points[wall.inlier_indices]
        perpendicular_dist = wall.signed_distance(pts)
        # merged walls combine several RANSAC fragments over a large real
        # surface, so some residual waviness is expected - still much
        # smaller than any furniture/clutter plane would show
        assert perpendicular_dist.std() < 0.08
        assert abs(float(np.dot(wall.normal, up))) < 0.3


def test_extract_wall_planes_outdoor_corridor_finds_thin_flat_walls():
    pc = load_point_cloud(DATASET_DIR / "scenario1_room1_outdoor.ply")
    up = estimate_up_vector(pc)
    walls = extract_wall_planes(pc, up=up)
    assert len(walls) >= 2

    walls_sorted = sorted(walls, key=lambda w: -len(w.inlier_indices))
    for wall in walls_sorted[:3]:
        pts = pc.points[wall.inlier_indices]
        perpendicular_dist = wall.signed_distance(pts)
        # merged walls combine several RANSAC fragments over a large real
        # surface, so some residual waviness is expected - still much
        # smaller than any furniture/clutter plane would show
        assert perpendicular_dist.std() < 0.12
