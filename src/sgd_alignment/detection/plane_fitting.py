"""RANSAC-based plane segmentation to extract candidate wall surfaces.

This replaces the paper's deep-learning-based initial segmentation
(Section 3.1) with a purely geometric approach: walls are the dominant
near-vertical planes in the point cloud, extracted by repeatedly fitting
and removing the largest plane. Floor/ceiling and other near-horizontal
planes are filtered out by their normal direction.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from sgd_alignment.common.types import Plane, PointCloud

DEFAULT_RANSAC_SEED = 0


def _segment_plane_ransac(
    points: np.ndarray,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    # open3d's RANSAC has no per-call seed argument; it draws from a global
    # RNG, so pin it before every call for reproducible wall extraction.
    o3d.utility.random.seed(DEFAULT_RANSAC_SEED)
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = o3d_pc.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return np.asarray(plane_model, dtype=np.float64), np.asarray(inliers, dtype=np.int64)


def _extract_planes(
    pc: PointCloud,
    distance_threshold: float,
    min_inliers: int,
    max_planes: int,
    ransac_n: int,
    num_iterations: int,
) -> list[Plane]:
    """Iteratively extract the largest planes, regardless of orientation."""
    remaining_points = pc.points
    remaining_idx = np.arange(len(pc))
    planes: list[Plane] = []

    for _ in range(max_planes):
        if len(remaining_points) < min_inliers:
            break
        model, local_inliers = _segment_plane_ransac(
            remaining_points, distance_threshold, ransac_n, num_iterations
        )
        if len(local_inliers) < min_inliers:
            break

        normal = model[:3] / np.linalg.norm(model[:3])
        global_inliers = remaining_idx[local_inliers]
        planes.append(Plane(normal=normal, d=model[3], inlier_indices=global_inliers))

        keep_mask = np.ones(len(remaining_points), dtype=bool)
        keep_mask[local_inliers] = False
        remaining_points = remaining_points[keep_mask]
        remaining_idx = remaining_idx[keep_mask]

    return planes


def estimate_up_vector(
    pc: PointCloud,
    distance_threshold: float = 0.03,
    min_inliers: int = 3000,
    num_probe_planes: int = 8,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    horizontal_raw_nz: float = 0.7,
) -> np.ndarray:
    """Estimate the true vertical ("up") direction from large horizontal planes.

    The file's raw z-axis is not guaranteed to be exactly vertical (the
    scanning rig may not have been perfectly leveled). Floor/ceiling planes
    are identified using a permissive threshold on the raw z-component of
    their normal, then averaged (sign-aligned) to obtain a more reliable
    up vector than the raw z-axis alone.
    """
    probe_planes = _extract_planes(
        pc, distance_threshold, min_inliers, num_probe_planes, ransac_n, num_iterations
    )
    horizontal = [p for p in probe_planes if abs(p.normal[2]) >= horizontal_raw_nz]
    if not horizontal:
        return np.array([0.0, 0.0, 1.0])

    oriented_normals = np.array(
        [p.normal if p.normal[2] > 0 else -p.normal for p in horizontal]
    )
    up = oriented_normals.mean(axis=0)
    return up / np.linalg.norm(up)


def estimate_up_vector_manhattan(
    pc: PointCloud,
    distance_threshold: float = 0.03,
    min_inliers: int = 2000,
    num_probe_planes: int = 15,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    same_axis_dot: float = 0.9,
) -> np.ndarray:
    """Estimate the true up direction with NO assumption about the file's
    raw axes (needed for handheld/SLAM-merged scans, where the raw frame's
    orientation is arbitrary and can be tilted by any amount - not just a
    small correction like a tripod-mounted scanner would need).

    Assumes a Manhattan-world room (floor/ceiling + two wall directions,
    mutually perpendicular): extracts the largest planes regardless of
    orientation, groups them into up to 3 axis clusters by normal direction
    (a plane and its mirror -normal belong to the same axis), then picks
    the axis along which the *whole point cloud's spatial extent* (max -
    min projection) is smallest - a room or corridor is almost always
    shorter floor-to-ceiling than it is long/wide, so this holds even for
    unusual point distributions.

    An earlier version picked the axis with the greatest total inlier
    point count instead (floor+ceiling are typically the most completely
    captured, least obstructed flat surfaces, unlike walls broken up by
    windows/doors/furniture) - but this fails for a long corridor, where
    the two long wall planes can accumulate more total points than
    floor+ceiling despite not being the vertical axis (confirmed on real
    data: a corridor with height extent 9.1 m still had its wall-normal
    axis carry more RANSAC inlier points than its floor/ceiling axis,
    silently picking the wrong axis as "up"). Spatial extent doesn't have
    that failure mode.
    """
    planes = _extract_planes(pc, distance_threshold, min_inliers, num_probe_planes, ransac_n, num_iterations)
    if not planes:
        return np.array([0.0, 0.0, 1.0])

    planes_sorted = sorted(planes, key=lambda p: -len(p.inlier_indices))
    axes: list[np.ndarray] = []

    for p in planes_sorted:
        if not any(abs(float(np.dot(p.normal, axis))) >= same_axis_dot for axis in axes):
            axes.append(p.normal)

    extents = [(pc.points @ axis).max() - (pc.points @ axis).min() for axis in axes]
    best_axis = axes[int(np.argmin(extents))]
    return best_axis / np.linalg.norm(best_axis)


def estimate_up_vector_from_local_normals(
    pc: PointCloud,
    knn: int = 30,
    horizontal_raw_nz: float = 0.6,
    max_points: int = 150_000,
) -> np.ndarray:
    """Estimate the true up direction from per-point local surface normals.

    Robust to a floor/ceiling that is broken into many small, disconnected
    patches by clutter (so no single RANSAC plane is large enough): normals
    are estimated locally per point via PCA over a k-nearest-neighborhood,
    so floor points scattered between obstacles still contribute. Points
    whose local normal is roughly aligned with the raw z-axis (a permissive
    threshold, since the true tilt is not yet known) are taken as
    floor/ceiling candidates and averaged (sign-aligned) into an up vector.
    """
    points = pc.points
    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points = points[idx]

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(points)
    o3d_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn))
    normals = np.asarray(o3d_pc.normals)

    horizontal_mask = np.abs(normals[:, 2]) >= horizontal_raw_nz
    if not np.any(horizontal_mask):
        return np.array([0.0, 0.0, 1.0])

    oriented = normals[horizontal_mask]
    oriented = np.where(oriented[:, 2:3] > 0, oriented, -oriented)
    up = oriented.mean(axis=0)
    return up / np.linalg.norm(up)


def extract_wall_planes(
    pc: PointCloud,
    distance_threshold: float = 0.03,
    min_inliers: int = 3000,
    max_planes: int = 20,
    verticality_max_dot: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    up: np.ndarray | None = None,
) -> list[Plane]:
    """Iteratively extract near-vertical planes (walls) from a point cloud.

    Repeatedly fits the largest remaining plane and removes its inliers,
    regardless of orientation, so that floor/ceiling planes are consumed
    and don't block the search for walls beneath/above them. A plane is
    kept as a wall candidate only if its normal is mostly perpendicular to
    the true up direction (|normal . up| < verticality_max_dot), since the
    file's z-axis may itself be tilted relative to true vertical.
    """
    if up is None:
        up = estimate_up_vector(pc, distance_threshold, min_inliers, ransac_n=ransac_n, num_iterations=num_iterations)

    all_planes = _extract_planes(pc, distance_threshold, min_inliers, max_planes, ransac_n, num_iterations)
    walls = [p for p in all_planes if abs(float(np.dot(p.normal, up))) < verticality_max_dot]
    return merge_coplanar_planes(walls, pc)


def _refit_plane(pc: PointCloud, inlier_indices: np.ndarray) -> Plane:
    """Least-squares plane fit (via SVD) through a set of points."""
    pts = pc.points[inlier_indices]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[-1]
    d = -float(np.dot(normal, centroid))
    return Plane(normal=normal, d=d, inlier_indices=inlier_indices)


def merge_coplanar_planes(
    planes: list[Plane],
    pc: PointCloud,
    normal_dot_min: float = 0.98,
    d_max_diff: float = 0.15,
) -> list[Plane]:
    """Merge plane fragments that are really the same physical surface.

    Iterative RANSAC can split one real (slightly wavy/noisy) wall into
    several separate fits across iterations, each capturing a different
    subset of its points but with nearly identical normal and offset -
    visually this shows up as one physical wall colored in several
    unrelated colors. Any two planes whose normals are nearly parallel
    (dot >= normal_dot_min) and whose offsets are close (|d1 - d2| <=
    d_max_diff) are merged into one plane, refit by least-squares over the
    union of their points (so the merged plane is the best fit for all of
    its points, not just whichever fragment happened to be picked first).
    """
    remaining = list(planes)
    merged: list[Plane] = []

    while remaining:
        base = remaining.pop(0)
        group_indices = [base.inlier_indices]
        still_remaining = []
        for other in remaining:
            same_plane = (
                abs(float(np.dot(base.normal, other.normal))) >= normal_dot_min
                and abs(base.d - other.d) <= d_max_diff
            )
            if same_plane:
                group_indices.append(other.inlier_indices)
            else:
                still_remaining.append(other)
        remaining = still_remaining
        merged.append(_refit_plane(pc, np.concatenate(group_indices)))

    return merged


def filter_exterior_walls(
    pc: PointCloud,
    walls: list[Plane],
    outward_margin: float = 0.10,
    max_beyond_fraction: float = 0.02,
) -> list[Plane]:
    """Keep only walls that bound the room (nothing lies beyond them).

    Furniture/shelving surfaces are often large, flat and near-vertical
    too, so `extract_wall_planes` alone cannot tell them apart from real
    walls. The distinguishing property is geometric: a true exterior wall
    has (almost) no point cloud beyond it, whereas an interior surface like
    a shelf front has the real wall still visible further out. The wall's
    normal is oriented away from the room centroid, then a wall is kept
    only if the fraction of all points lying beyond it (further out than
    `outward_margin`) is below `max_beyond_fraction`.
    """
    centroid = pc.points.mean(axis=0)
    n_points = len(pc)
    exterior = []
    for wall in walls:
        outward = wall.normal
        wall_point = pc.points[wall.inlier_indices].mean(axis=0)
        if np.dot(outward, wall_point - centroid) < 0:
            outward = -outward

        signed = pc.points @ outward - np.dot(outward, wall_point)
        beyond_fraction = float(np.mean(signed > outward_margin))
        if beyond_fraction <= max_beyond_fraction:
            exterior.append(wall)
    return exterior


def rotation_to_align_up(up: np.ndarray, target: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Return the 3x3 rotation matrix mapping `up` onto `target` (Rodrigues' formula)."""
    up = up / np.linalg.norm(up)
    target = target / np.linalg.norm(target)
    v = np.cross(up, target)
    c = float(np.dot(up, target))
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(target, target)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def align_upright(pc: PointCloud, up: np.ndarray) -> tuple[PointCloud, np.ndarray]:
    """Rotate a point cloud so the estimated `up` direction becomes world +Z.

    After this, the whole model can be worked with in ordinary world
    coordinates (X, Y horizontal, Z vertical) instead of a per-wall local
    frame, which is what multi-view orthographic projection needs.
    Returns (rotated_point_cloud, rotation_matrix_used).
    """
    R = rotation_to_align_up(up)
    rotated_points = pc.points @ R.T
    rotated_normals = pc.normals @ R.T if pc.normals is not None else None
    return PointCloud(points=rotated_points, intensity=pc.intensity, normals=rotated_normals), R
