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
from scipy.optimize import linear_sum_assignment

from sgd_alignment.common.types import Detection3D
from sgd_alignment.matching.sgd import INFEASIBLE, PairWeights, SGD, _LARGE_FINITE_COST, build_sgds, match_sgds


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


def estimate_rigid_transform(
    src: np.ndarray, dst: np.ndarray, estimate_scale: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Find (R, t) minimizing sum ||R @ src_i + t - dst_i||^2 (Kabsch algorithm).

    Requires at least 3 non-collinear correspondences for a well-
    conditioned 3D rotation; collinear input still returns a result (a
    valid least-squares solution exists) but a warning is raised since the
    rotation about the shared line's axis is effectively unconstrained by
    the data - e.g. 3 windows evenly spaced along one straight wall.

    `estimate_scale=True` additionally recovers a uniform scale factor
    between src and dst (Umeyama's algorithm, the standard generalization
    of Kabsch to similarity transforms) and folds it into the returned
    `R` (`R = scale * R_pure`) rather than adding a 3rd return value, so
    existing callers that only ever want a rigid (scale=1) transform are
    completely unaffected. `transform_points` needs no change either way,
    since `points @ (scale * R_pure).T == scale * (points @ R_pure.T)`.

    Needed when src and dst come from two independently, arbitrarily
    scaled reconstructions with no shared metric reference - e.g. two
    separate feed-forward depth-model inference runs (each session's
    absolute scale is unrelated to the other's, even though each session
    is internally self-consistent). A pure rotation can never fit such
    data; forcing `estimate_scale=False` there gives a systematically
    biased R that still "looks" like a rotation, which is far more
    dangerous than an outright failure - it should be off by default and
    is on the caller to opt into once dubbed necessary for a given
    source's data.
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
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    if estimate_scale:
        src_spread = float(np.sum(src_c**2))
        if src_spread < 1e-12:
            raise ValueError("source points have ~zero spread - cannot estimate a scale factor")
        scale = float(np.dot(S, np.diag(D))) / src_spread
        R = scale * R

    t = dst_centroid - R @ src_centroid
    return R, t


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ R.T + t


def _reassign_by_geometric_residual(
    indoor_detections: list[Detection3D],
    outdoor_detections: list[Detection3D],
    R: np.ndarray,
    t: np.ndarray,
    max_residual: float | None,
) -> list[tuple[int, int, float]]:
    """One Hungarian re-assignment of ALL (same-category) indoor/outdoor
    pairs by center-to-center distance *after* applying (R, t) - the
    geometric-consensus half of `refine_matches_with_geometric_consensus`.
    """
    n_i, n_o = len(indoor_detections), len(outdoor_detections)
    cost = np.full((n_i, n_o), INFEASIBLE)
    transformed = transform_points(np.array([d.center for d in indoor_detections]), R, t)
    for i, di in enumerate(indoor_detections):
        for j, dj in enumerate(outdoor_detections):
            if di.category != dj.category:
                continue
            residual = float(np.linalg.norm(transformed[i] - dj.center))
            if max_residual is not None and residual > max_residual:
                continue
            cost[i, j] = residual

    valid_rows = ~np.all(np.isinf(cost), axis=1)
    valid_cols = ~np.all(np.isinf(cost), axis=0)
    if not valid_rows.any() or not valid_cols.any():
        return []
    row_ids, col_ids = np.where(valid_rows)[0], np.where(valid_cols)[0]
    sub = cost[np.ix_(valid_rows, valid_cols)]
    finite_sub = np.where(np.isinf(sub), _LARGE_FINITE_COST, sub)
    row_idx, col_idx = linear_sum_assignment(finite_sub)

    return [
        (int(row_ids[r]), int(col_ids[c]), float(sub[r, c]))
        for r, c in zip(row_idx, col_idx) if not np.isinf(sub[r, c])
    ]


def refine_matches_with_geometric_consensus(
    indoor_detections: list[Detection3D],
    outdoor_detections: list[Detection3D],
    matches: list[tuple[int, int, float]],
    estimate_scale: bool = False,
    max_iterations: int = 3,
    max_residual: float | None = None,
) -> tuple[list[tuple[int, int, float]], np.ndarray, np.ndarray]:
    """Geometric-consensus verification/correction pass on top of the SGD
    Hungarian matches (Algorithm 3): fixes correspondences that pure
    per-object descriptor comparison cannot disambiguate - most notably
    several near-identical openings evenly spaced along the same wall,
    where the SGD's *local* relative-geometry description is genuinely
    ambiguous between neighbors (confirmed on real data: 2 of 7 windows
    swapped in a row of evenly-spaced windows, residual ~3m, every other
    match unaffected and under 5cm).

    The idea (like PnP-RANSAC consensus in SfM): fit (R, t) from the
    current matches (dominated by the majority that ARE correct, since
    Kabsch is a least-squares fit), then re-derive correspondences from
    scratch by Hungarian-assigning every same-category indoor/outdoor pair
    by *post-transform geometric distance* alone - a signal pure SGD
    matching doesn't have access to, but which cleanly separates a
    genuinely-matching pair (near-zero residual) from a locally-identical
    but spatially-swapped one (residual on the order of the swap
    distance). Repeats until the assignment stops changing (or
    `max_iterations`), same convergence idea as ICP.

    Returns `(refined_matches, R, t)` - `(matches, None, None)` unchanged
    if there aren't enough matches to fit a transform in the first place
    (`estimate_rigid_transform` needs >= 3).
    """
    if len(matches) < 3:
        return matches, None, None

    current = matches
    R, t = None, None
    for _ in range(max_iterations):
        src = np.concatenate([indoor_detections[i].corners() for i, _, _ in current])
        dst = np.concatenate([outdoor_detections[j].corners() for _, j, _ in current])
        R, t = estimate_rigid_transform(src, dst, estimate_scale=estimate_scale)

        next_matches = _reassign_by_geometric_residual(indoor_detections, outdoor_detections, R, t, max_residual)
        if len(next_matches) < 3:
            break  # consensus collapsed - keep the last valid (current) assignment
        if {(i, j) for i, j, _ in next_matches} == {(i, j) for i, j, _ in current}:
            current = next_matches
            break
        current = next_matches

    return current, R, t


@dataclass
class AlignmentResult:
    R: np.ndarray  # (3,3) rotation (or scale * rotation if estimate_scale=True)
    t: np.ndarray  # (3,) translation, applied as R @ p + t
    matches: list[tuple[int, int, float]]  # (indoor_idx, outdoor_idx, sgd_cost)
    residuals: np.ndarray  # (n_matches,) per-match alignment error after applying (R, t)
    scale: float = 1.0  # recovered indoor->outdoor scale factor; 1.0 unless estimate_scale=True


def align_indoor_outdoor(
    indoor_detections: list[Detection3D],
    outdoor_detections: list[Detection3D],
    weights: PairWeights | None = None,
    max_cost: float = 5.0,
    estimate_scale: bool = False,
    normalize_distance: bool = False,
    refine_with_geometric_consensus: bool = False,
    refine_max_residual: float | None = None,
    use_intrinsic_fallback: bool = False,
) -> AlignmentResult:
    """Match openings between an indoor and outdoor scan, then compute the
    rigid transform (indoor -> outdoor frame) from all 8 corners of each
    matched pair of bounding boxes.

    `estimate_scale`/`normalize_distance` are both off by default (exact
    prior behavior, needed for the already-verified CloudCompare/manual
    real-world-metric datasets). Turn both on together when indoor and
    outdoor come from two independently-scaled reconstructions with no
    shared metric reference (e.g. two separate feed-forward depth-model
    inference runs) - see `estimate_rigid_transform` and
    `sgd.build_sgds` for what each one fixes; `weights.distance_threshold`
    should then be re-tuned as a *relative* (not absolute-meter) value,
    since `normalize_distance` divides every scene's distances by its own
    median inter-opening spacing.

    `refine_with_geometric_consensus=False` by default (exact prior
    behavior). Turn on to additionally run
    `refine_matches_with_geometric_consensus` after the SGD matching -
    fixes correspondences that pure per-object descriptor comparison
    cannot disambiguate (most notably several near-identical openings
    evenly spaced along one wall); see that function's docstring.

    `use_intrinsic_fallback=False` by default (exact prior behavior).
    Turn on to let `sgd.sgdu_distance` fall back to comparing each
    object's own absolute geometry (category + aspect ratio) when a
    scene has too few objects for the normal neighbor-relative
    descriptor to mean anything - see `sgd._intrinsic_cost`. Needed for
    scenes with only 1-2 detected openings, which otherwise can never be
    matched at all regardless of how correct the detection is.
    """
    indoor_sgds: list[SGD] = build_sgds(indoor_detections, normalize_distance=normalize_distance)
    outdoor_sgds: list[SGD] = build_sgds(outdoor_detections, normalize_distance=normalize_distance)
    matches = match_sgds(indoor_sgds, outdoor_sgds, weights=weights, max_cost=max_cost,
                          use_intrinsic_fallback=use_intrinsic_fallback)

    if len(matches) < 1:
        raise ValueError(
            f"only {len(matches)} opening(s) matched (need >= 3) - detection or "
            "annotation quality is likely too sparse/wrong for this pair"
        )

    if refine_with_geometric_consensus:
        matches, _, _ = refine_matches_with_geometric_consensus(
            indoor_detections, outdoor_detections, matches,
            estimate_scale=estimate_scale, max_residual=refine_max_residual,
        )
        if len(matches) < 1:
            raise ValueError("geometric consensus refinement left 0 matches - detection quality likely too poor")

    src = np.concatenate([indoor_detections[i].corners() for i, _, _ in matches])
    dst = np.concatenate([outdoor_detections[j].corners() for _, j, _ in matches])
    R, t = estimate_rigid_transform(src, dst, estimate_scale=estimate_scale)
    scale = float(np.linalg.norm(R, axis=0).mean())

    center_src = np.array([indoor_detections[i].center for i, _, _ in matches])
    center_dst = np.array([outdoor_detections[j].center for _, j, _ in matches])
    residuals = np.linalg.norm(transform_points(center_src, R, t) - center_dst, axis=1)
    return AlignmentResult(R=R, t=t, matches=matches, residuals=residuals, scale=scale)
