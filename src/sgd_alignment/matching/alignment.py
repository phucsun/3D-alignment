"""Rigid transform estimation from matched opening correspondences (Section 4).

Once SGD matching (`sgd.py`) has produced correspondences between indoor
and outdoor detected openings, the two point clouds are aligned by finding
the rotation + translation that best maps each matched indoor opening
onto its outdoor counterpart - the classic Kabsch/orthogonal Procrustes
solution via SVD. Per the paper ("the transformation ... is finally
calculated by the SVD method on the corner points of the matched
window/door bounding boxes"), all 8 corners of each matched box are used
as correspondence points, not just the box centers - this also gives 8x
as many points per match for a more robust least-squares fit.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from sgd_alignment.common.types import Detection3D
from sgd_alignment.matching.sgd import PairWeights, SGD, build_sgds, match_sgds


def is_collinear(points: np.ndarray, ratio_threshold: float = 0.05) -> bool:
    """Check whether `points` are (nearly) collinear.

    3 non-collinear points fully determine a 3D rotation; collinear ones
    (e.g. several windows evenly spaced along the *same* wall - a common
    real layout) leave rotation about that line's axis unconstrained, so
    Kabsch can silently return a valid-looking but arbitrary/wrong R there.
    """
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] < 1e-12:
        return True
    return (singular_values[1] / singular_values[0]) < ratio_threshold


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find (R, t) minimizing sum ||R @ src_i + t - dst_i||^2 (Kabsch algorithm).

    Requires at least 3 non-collinear correspondences for a well-
    conditioned 3D rotation; collinear input still returns a result (a
    valid least-squares solution exists) but a warning is raised since the
    rotation about the shared line's axis is effectively unconstrained by
    the data - e.g. 3 windows evenly spaced along one straight wall.
    """
    if len(src) < 3:
        raise ValueError(f"need at least 3 correspondences for a 3D rigid transform, got {len(src)}")
    if is_collinear(src) or is_collinear(dst):
        warnings.warn(
            "matched opening centers are (nearly) collinear - the rotation about "
            "that line's axis is poorly constrained; add a match off that line "
            "(e.g. a door/window on a different wall) before trusting this result",
            stacklevel=2,
        )

    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid

    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_centroid - R @ src_centroid
    return R, t


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ R.T + t


@dataclass
class AlignmentResult:
    R: np.ndarray  # (3,3) rotation
    t: np.ndarray  # (3,) translation, applied as R @ p + t
    matches: list[tuple[int, int, float]]  # (indoor_idx, outdoor_idx, sgd_cost)
    residuals: np.ndarray  # (n_matches,) per-match alignment error after applying (R, t)


def align_indoor_outdoor(
    indoor_detections: list[Detection3D],
    outdoor_detections: list[Detection3D],
    weights: PairWeights | None = None,
    max_cost: float = 5.0,
) -> AlignmentResult:
    """Match openings between an indoor and outdoor scan, then compute the
    rigid transform (indoor -> outdoor frame) from all 8 corners of each
    matched pair of bounding boxes."""
    indoor_sgds: list[SGD] = build_sgds(indoor_detections)
    outdoor_sgds: list[SGD] = build_sgds(outdoor_detections)
    matches = match_sgds(indoor_sgds, outdoor_sgds, weights=weights, max_cost=max_cost)

    if len(matches) < 3:
        raise ValueError(
            f"only {len(matches)} opening(s) matched (need >= 3) - detection or "
            "annotation quality is likely too sparse/wrong for this pair"
        )

    src = np.concatenate([indoor_detections[i].corners() for i, _, _ in matches])
    dst = np.concatenate([outdoor_detections[j].corners() for _, j, _ in matches])
    R, t = estimate_rigid_transform(src, dst)

    center_src = np.array([indoor_detections[i].center for i, _, _ in matches])
    center_dst = np.array([outdoor_detections[j].center for _, j, _ in matches])
    residuals = np.linalg.norm(transform_points(center_src, R, t) - center_dst, axis=1)
    return AlignmentResult(R=R, t=t, matches=matches, residuals=residuals)
