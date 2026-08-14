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


def matches_span_single_wall(
    detections: list[Detection3D],
    matched_indices: list[int],
    angle_threshold_deg: float = 20.0,
) -> bool:
    """True if every matched detection's wall normal points in (nearly) the
    same direction - i.e. every match on this side was found on a single
    wall.

    IMPORTANT - what this does NOT mean: Kabsch's rotation is still fully,
    uniquely determined even from a single matched box (confirmed on real
    data - `home_phuc`: one match alone reproduces the exact same ~145
    degree rotation as the full 2-match set, byte-for-byte; a real door's
    distinct width/height/thickness rules out the rotational symmetry that
    would make it genuinely ambiguous). So this is NOT a Kabsch/SVD
    degeneracy check, and an earlier version of this function that framed
    it that way (and an even earlier one that tried to detect it via the
    corner point cloud's own coplanarity ratio) was wrong - confirmed on
    real data: a genuinely single-wall case (`home_phuc`) and a healthy
    multi-wall case (`sc2_room2`) gave the *same* corner-cloud thinness
    ratio, because that ratio is dominated by the door's own physical
    thickness (near-constant across real doors), not by wall diversity.

    What single-wall matches actually lack is INDEPENDENT CROSS-
    VALIDATION: with two separately/casually captured videos, each
    reconstruction's coordinate frame has an arbitrary heading (whatever
    direction the camera happened to start facing) - a large compass-like
    rotation between indoor's and outdoor's frames is entirely expected
    and not itself evidence of a bug. But with matches on only one wall,
    there is no second, differently-oriented wall to confirm that rotation
    is physically correct rather than an artifact of a wrong correspondence
    or a wrong wall-normal convention upstream - the fit looks perfect (low
    residual) either way, because residual alone cannot distinguish them.
    """
    normals = [detections[i].normal for i in matched_indices]
    if len(normals) < 2:
        return True
    reference = normals[0]
    cos_threshold = np.cos(np.radians(angle_threshold_deg))
    return all(abs(float(np.dot(reference, n))) >= cos_threshold for n in normals[1:])


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
            "matched opening corners are (nearly) collinear - the rotation about "
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
    normal_angle_threshold: float | None = None,
) -> list[tuple[int, int, float]]:
    """One Hungarian re-assignment of ALL (same-category) indoor/outdoor
    pairs by center-to-center distance *after* applying (R, t) - the
    geometric-consensus half of `refine_matches_with_geometric_consensus`
    and `ransac_match_with_wall_consensus`.

    `normal_angle_threshold`: additionally reject a pair whose wall
    normals aren't consistent under (R, t) (radians). Only meaningful
    when (R, t) itself comes from a trustworthy hypothesis - see
    `ransac_match_with_wall_consensus`'s docstring for why bootstrapping
    this from an unverified full match set (as an earlier version of this
    project's refinement did) is unsound and was reverted.
    """
    n_i, n_o = len(indoor_detections), len(outdoor_detections)
    cost = np.full((n_i, n_o), INFEASIBLE)
    transformed = transform_points(np.array([d.center for d in indoor_detections]), R, t)

    transformed_normals = None
    if normal_angle_threshold is not None:
        raw = np.array([d.normal for d in indoor_detections]) @ R.T
        transformed_normals = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    for i, di in enumerate(indoor_detections):
        for j, dj in enumerate(outdoor_detections):
            if di.category != dj.category:
                continue
            residual = float(np.linalg.norm(transformed[i] - dj.center))
            if max_residual is not None and residual > max_residual:
                continue
            if transformed_normals is not None:
                cos_angle = np.clip(np.dot(transformed_normals[i], dj.normal), -1.0, 1.0)
                if np.arccos(cos_angle) > normal_angle_threshold:
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


def _filter_degenerate(detections: list[Detection3D], min_dimension: float) -> tuple[list[Detection3D], list[int]]:
    """Drop detections whose width or height is physically implausible
    for ANY door/window (not a per-category "typical size" prior, which
    can't generalize across building types/regions - just a sanity floor
    against outright broken measurements). Confirmed on real data: a
    "door" detection 0.03m wide (clearly a segmentation/backprojection
    artifact, not a real opening) produced a deceptively good-looking fit
    to an unrelated match purely because a near-degenerate box's corners
    barely constrain anything - `refine_matches_with_geometric_consensus`
    even "confirmed" it via low residual. Returns `(kept_detections,
    original_indices)` so callers can map filtered-list positions back to
    indices into the original list.
    """
    kept = [(i, d) for i, d in enumerate(detections) if d.width >= min_dimension and d.height >= min_dimension]
    return [d for _, d in kept], [i for i, _ in kept]


def ransac_match_with_wall_consensus(
    indoor_detections: list[Detection3D],
    outdoor_detections: list[Detection3D],
    matches: list[tuple[int, int, float]],
    estimate_scale: bool = False,
    max_residual: float | None = None,
    normal_angle_threshold: float = np.pi / 6,
    min_sample_size: int = 2,
    max_hypotheses: int = 500,
    rng_seed: int = 0,
) -> tuple[list[tuple[int, int, float]], np.ndarray | None, np.ndarray | None]:
    """Proper hypothesize-and-verify RANSAC on top of the SGD matches,
    checking BOTH position and wall-orientation consensus - the intended
    fix for a real failure mode `refine_matches_with_geometric_consensus`
    cannot handle: several openings genuinely belong to unrelated walls
    (confirmed on real data - a meeting room with 1-2 doors facing a
    shared corridor, plus several other doors/windows facing entirely
    different spaces, on both sides).

    Deliberately NOT built by adding a wall-orientation check into that
    existing iterative refiner: that refiner *fits its hypothesis from
    the very matches it's trying to verify* (dominated by whichever
    matches happened to be in the initial SGD result, right or wrong),
    then checks new candidates against that self-derived hypothesis - a
    genuinely wrong initial batch produces a genuinely wrong hypothesis,
    which then "confirms" itself. Tried exactly this on real data: it
    picked a 0.03m-wide degenerate "door" detection as part of the
    accepted result, because a near-degenerate box's corners barely
    constrain anything and happened to fit the (already wrong) hypothesis
    - low residual, high confidence, completely incorrect. Confirmed
    wrong by visual inspection of the exported aligned model; reverted.

    This version fits each hypothesis from a small (`min_sample_size`,
    default 2) sample drawn from `matches`, checks how many OTHER pairs
    - across ALL same-category indoor/outdoor combinations, not just the
    ones already in `matches` - agree with it on both position and wall
    normal, and keeps whichever hypothesis wins the most agreement
    (standard RANSAC consensus scoring, like PnP-RANSAC in SfM). A
    hypothesis seeded from 2 bad matches simply won't attract much
    agreement from the rest of the data and loses to a better one; no
    single detection (however degenerate) can single-handedly bootstrap
    its own confirmation the way the iterative version could.

    Tries every `min_sample_size`-combination of `matches` if there are
    few enough (`max_hypotheses`), otherwise samples that many at random.
    Returns `(matches, None, None)` unchanged if there are too few
    matches to draw even one sample, or if no hypothesis attracted any
    consensus at all (rather than guessing).
    """
    if len(matches) < min_sample_size:
        return matches, None, None

    from itertools import combinations

    all_samples = list(combinations(range(len(matches)), min_sample_size))
    if len(all_samples) > max_hypotheses:
        rng = np.random.default_rng(rng_seed)
        chosen = rng.choice(len(all_samples), size=max_hypotheses, replace=False)
        all_samples = [all_samples[i] for i in chosen]

    best_inliers: list[tuple[int, int, float]] = []
    best_R, best_t = None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a degenerate minimal sample just scores poorly, no need to warn per-hypothesis
        for sample_idx in all_samples:
            sample = [matches[k] for k in sample_idx]
            src = np.concatenate([indoor_detections[i].corners() for i, _, _ in sample])
            dst = np.concatenate([outdoor_detections[j].corners() for _, j, _ in sample])
            try:
                R, t = estimate_rigid_transform(src, dst, estimate_scale=estimate_scale)
            except ValueError:
                continue
            inliers = _reassign_by_geometric_residual(
                indoor_detections, outdoor_detections, R, t, max_residual, normal_angle_threshold,
            )
            if len(inliers) > len(best_inliers):
                best_inliers, best_R, best_t = inliers, R, t

    if not best_inliers:
        return matches, None, None

    # final polish: refit (R, t) from the *entire* winning consensus set,
    # not just its 2-match seed
    src = np.concatenate([indoor_detections[i].corners() for i, _, _ in best_inliers])
    dst = np.concatenate([outdoor_detections[j].corners() for _, j, _ in best_inliers])
    R, t = estimate_rigid_transform(src, dst, estimate_scale=estimate_scale)
    return best_inliers, R, t


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
    use_ambiguity_check: bool = False,
    ambiguity_ratio_threshold: float = 0.8,
    min_dimension: float | None = None,
    use_ransac_consensus: bool = False,
    ransac_normal_angle_threshold: float = np.pi / 6,
    ransac_max_residual: float | None = None,
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

    `use_ambiguity_check`/`ambiguity_ratio_threshold`: see
    `sgd.match_sgds`/`sgd._ambiguity_ratio` - rejects matches that aren't
    decisively better than the next-best alternative, guarding against
    unrelated same-category "distractor" objects on one side (e.g. extra
    fire-exit doors in a corridor with no indoor counterpart).

    `min_dimension` (`None` = off): drop any detection whose width or
    height falls below this (meters, or scene units) before matching -
    see `_filter_degenerate`. Matched indices in the returned result
    always refer to the original `indoor_detections`/`outdoor_detections`
    lists passed in, regardless of filtering.

    `use_ransac_consensus=False` by default. Turn on to run
    `ransac_match_with_wall_consensus` after SGD matching (and after
    `refine_with_geometric_consensus`, if both are on) - the properly
    hypothesize-and-verify version of consensus checking (small seed
    samples + position AND wall-orientation agreement), the intended fix
    for openings that genuinely belong to unrelated walls; see that
    function's docstring for why it's a separate function rather than an
    extension of `refine_with_geometric_consensus`.
    """
    if min_dimension is not None:
        indoor_filtered, indoor_index_map = _filter_degenerate(indoor_detections, min_dimension)
        outdoor_filtered, outdoor_index_map = _filter_degenerate(outdoor_detections, min_dimension)
    else:
        indoor_filtered, indoor_index_map = indoor_detections, list(range(len(indoor_detections)))
        outdoor_filtered, outdoor_index_map = outdoor_detections, list(range(len(outdoor_detections)))

    indoor_sgds: list[SGD] = build_sgds(indoor_filtered, normalize_distance=normalize_distance)
    outdoor_sgds: list[SGD] = build_sgds(outdoor_filtered, normalize_distance=normalize_distance)
    matches = match_sgds(indoor_sgds, outdoor_sgds, weights=weights, max_cost=max_cost,
                          use_intrinsic_fallback=use_intrinsic_fallback,
                          use_ambiguity_check=use_ambiguity_check,
                          ambiguity_ratio_threshold=ambiguity_ratio_threshold)
    matches = [(indoor_index_map[i], outdoor_index_map[j], c) for i, j, c in matches]

    if len(matches) < 1:
        raise ValueError(
            f"only {len(matches)} opening(s) matched (need >= 1) - detection or "
            "annotation quality is likely too sparse/wrong for this pair"
        )

    if refine_with_geometric_consensus:
        matches, _, _ = refine_matches_with_geometric_consensus(
            indoor_detections, outdoor_detections, matches,
            estimate_scale=estimate_scale, max_residual=refine_max_residual,
        )
        if len(matches) < 1:
            raise ValueError("geometric consensus refinement left 0 matches - detection quality likely too poor")

    if use_ransac_consensus:
        matches, _, _ = ransac_match_with_wall_consensus(
            indoor_detections, outdoor_detections, matches,
            estimate_scale=estimate_scale, max_residual=ransac_max_residual,
            normal_angle_threshold=ransac_normal_angle_threshold,
        )
        if len(matches) < 1:
            raise ValueError("RANSAC wall-consensus left 0 matches - detection quality likely too poor")

    if matches_span_single_wall(
        indoor_detections, [i for i, _, _ in matches]
    ) or matches_span_single_wall(outdoor_detections, [j for _, j, _ in matches]):
        warnings.warn(
            "every matched opening shares the same wall normal on at least one "
            "side - this rotation is still the unique Kabsch-optimal fit (not "
            "mathematically ambiguous), but with only one wall's worth of "
            "matches there is no independent cross-check that it is the "
            "physically correct one rather than an artifact of a wrong "
            "correspondence or wall-normal convention upstream; a low residual "
            "does not rule that out. Add a match on a different wall before "
            "trusting this alignment's orientation.",
            stacklevel=2,
        )

    src = np.concatenate([indoor_detections[i].corners() for i, _, _ in matches])
    dst = np.concatenate([outdoor_detections[j].corners() for _, j, _ in matches])
    R, t = estimate_rigid_transform(src, dst, estimate_scale=estimate_scale)
    scale = float(np.linalg.norm(R, axis=0).mean())

    center_src = np.array([indoor_detections[i].center for i, _, _ in matches])
    center_dst = np.array([outdoor_detections[j].center for _, j, _ in matches])
    residuals = np.linalg.norm(transform_points(center_src, R, t) - center_dst, axis=1)
    return AlignmentResult(R=R, t=t, matches=matches, residuals=residuals, scale=scale)
