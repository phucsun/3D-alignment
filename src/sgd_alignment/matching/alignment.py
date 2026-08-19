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
from sgd_alignment.detection.opening_geometry import aggregate_oriented_wall_normal
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
    best_inlier_residual_sum = float("inf")
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
            # Tie-break on total residual, not just insertion order: 2
            # genuinely different candidate walls in the same scene (e.g. 2
            # separate doors into the same corridor from different rooms)
            # can each attract exactly their own seed pair and nothing else
            # - an honest tie in inlier count with no cross-voting either
            # way. Confirmed on real data (`server`): the higher-cost wall
            # was silently kept every time solely because its 2-match seed
            # happened to be enumerated first in `combinations()`, even
            # though the other wall's seed fit with a much lower residual
            # (and a much lower independent SGD relational cost - see
            # CONTRIBUTIONS.md). Preferring the lower total residual on a
            # tie is a well-defined, order-independent choice; it never
            # overrides a hypothesis with strictly more support.
            inlier_residual_sum = sum(residual for _, _, residual in inliers)
            better = len(inliers) > len(best_inliers) or (
                len(inliers) == len(best_inliers) and inlier_residual_sum < best_inlier_residual_sum
            )
            if better:
                best_inliers, best_R, best_t = inliers, R, t
                best_inlier_residual_sum = inlier_residual_sum

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


def group_by_wall(detections: list[Detection3D], same_wall_dot: float = 0.9) -> list[list[int]]:
    """Group opening detections by which physical wall they're on, using
    each detection's own already-resolved `.normal` (see
    `opening_geometry.points_to_detection`) - no new geometry computed
    here, just a greedy clustering of the normals openings already carry.

    `same_wall_dot`: two detections are on the same wall when their
    normals agree with dot >= this (same DIRECTION, not just parallel -
    deliberately not `abs(dot)`). 2 openings on *opposite* walls of a
    corridor/room (facing away from each other into 2 different spaces)
    have near-antiparallel normals despite being physically close; folding
    them into one group here would repeat the exact sign-unaware mistake
    `plane_fitting.merge_coplanar_planes` had before its fix (see that
    function's docstring) - direction matters, magnitude of the angle
    alone does not.
    """
    groups: list[list[int]] = []
    for idx, d in enumerate(detections):
        for group in groups:
            if float(np.dot(d.normal, detections[group[0]].normal)) >= same_wall_dot:
                group.append(idx)
                break
        else:
            groups.append([idx])
    return groups


@dataclass
class RoomAssignment:
    wall_hub_indices: list[int]  # indices into the original hub_detections list
    result: AlignmentResult  # result.matches' outdoor indices are LOCAL to wall_hub_indices


def align_rooms_to_hub(
    rooms: list[list[Detection3D]],
    hub_detections: list[Detection3D],
    **align_kwargs,
) -> list[RoomAssignment | None]:
    """Align several "room" detection sets to one shared "hub" scene (e.g.
    N rooms each opening onto a single hallway/corridor) - the star
    topology where every room borders the hub but not each other.

    Confirmed on real data that simply calling `align_indoor_outdoor`
    once per room independently, each against the hub's FULL detection
    list, can silently produce a result that is internally consistent
    (low residual, passes its own same-side check) for EVERY room on its
    own, yet globally wrong once combined: 2 rooms' own least-cost wall
    can point at the very same physical wall, and whichever room's SGD
    relational cost happened to look good enough on that shared wall
    "wins" it in its own isolated run - with no way for either run to
    know a second room was competing for the same slot, since
    `align_indoor_outdoor`'s contract is exactly one indoor/outdoor pair
    at a time. This is not a bug in that function; it is a genuinely
    different problem (assigning WHICH wall belongs to WHICH room) that
    only a caller with visibility into every room at once can solve.

    Approach: group `hub_detections` by wall (`group_by_wall`), then try
    every (room, wall) combination through `align_indoor_outdoor`
    restricted to that one wall's detections, recording the mean matched
    SGD cost as that combination's price (`inf` where matching fails
    outright - wrong category, no feasible pair, etc). One Hungarian
    assignment (`scipy.optimize.linear_sum_assignment`) over the resulting
    (room x wall) cost matrix then picks the total-cost-minimizing,
    no-two-rooms-share-a-wall assignment - the same underlying algorithm
    `sgd.py` already uses for opening-to-opening matching, just applied
    one level up, at room-to-wall granularity. Passing each room's FULL
    detection list (not pre-filtered) to every trial also resolves a
    related real issue for free: a room with its own internal decoy
    opening set (e.g. `server`'s 2nd, unrelated door pair on a different
    internal wall - see CONTRIBUTIONS.md) naturally loses on cost to
    whichever of its own opening subsets truly fits a given hub wall, once
    each hub wall is tried in isolation rather than the decoy being able
    to "steal" a hub wall slot in one single whole-hub Hungarian pass.

    `**align_kwargs`: forwarded to every `align_indoor_outdoor` call
    (e.g. `estimate_scale=True, normalize_distance=True,
    use_ransac_consensus=True, use_intrinsic_fallback=True`).

    Returns one `RoomAssignment | None` per room, in `rooms`'s order -
    `None` for a room left unassigned if there are more rooms than walls.
    """
    wall_groups = group_by_wall(hub_detections)
    n_rooms, n_walls = len(rooms), len(wall_groups)

    cost = np.full((n_rooms, n_walls), np.inf)
    cached: dict[tuple[int, int], AlignmentResult] = {}
    for r, room in enumerate(rooms):
        for w, group in enumerate(wall_groups):
            sub_hub = [hub_detections[i] for i in group]
            try:
                result = align_indoor_outdoor(room, sub_hub, **align_kwargs)
            except ValueError:
                continue
            cached[(r, w)] = result
            cost[r, w] = float(np.mean([c for _, _, c in result.matches]))

    finite_cost = np.where(np.isfinite(cost), cost, _LARGE_FINITE_COST)
    row_idx, col_idx = linear_sum_assignment(finite_cost)

    assigned: list[RoomAssignment | None] = [None] * n_rooms
    for r, w in zip(row_idx, col_idx):
        if np.isfinite(cost[r, w]):
            assigned[r] = RoomAssignment(wall_hub_indices=wall_groups[w], result=cached[(r, w)])
    return assigned


def _rotation_between_unit_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The rotation matrix that maps unit vector `source` onto unit
    vector `target` (Rodrigues' rotation formula). Falls back to identity
    (already aligned) or a 180-degree rotation about an arbitrary
    perpendicular axis (exactly opposite - axis choice doesn't matter for
    the correction we use this for) when the 2 vectors are degenerate
    (parallel/antiparallel), since the cross-product axis is undefined
    there.
    """
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    v = np.cross(source, target)
    sin_angle = np.linalg.norm(v)
    cos_angle = float(np.dot(source, target))
    if sin_angle < 1e-10:
        if cos_angle > 0:
            return np.eye(3)
        perpendicular = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, perpendicular)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    # NOTE: K is built from the RAW cross product `v`, not a unit axis
    # (v/sin_angle) - this specific "vector-to-vector rotation" identity
    # requires that (confirmed the bug this fixes: normalizing v before
    # building K here silently produced a NON-orthogonal "rotation"
    # matrix, det=1.25 instead of 1.0, which is what corrupted every
    # real-data test of this function until caught).
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - cos_angle) / (sin_angle ** 2))


def refine_with_wall_normal_lock(
    result: AlignmentResult,
    indoor_detections: list[Detection3D],
    outdoor_detections: list[Detection3D],
    indoor_clusters: list[np.ndarray],
    outdoor_clusters: list[np.ndarray],
    up_indoor: np.ndarray,
    up_outdoor: np.ndarray,
    max_deviation_deg: float = 25.0,
) -> AlignmentResult:
    """Tighten an already-computed `align_indoor_outdoor` result so the
    matched wall's normals become EXACTLY co-directional (`R @ n_indoor
    == n_outdoor`, empirically confirmed - see the note further down on
    why this is PARALLEL rather than the antiparallel relationship a
    naive "2 wall faces point away from each other" intuition suggests).
    `align_indoor_outdoor` itself only fits the matched openings'
    8 corners; with few openings (commonly just 1-2 real doors) that fit
    has little data to average measurement/segmentation noise over, and
    the residual "hơi lệch" (slightly off) look this leaves is exactly
    what motivated this refinement (see CONTRIBUTIONS.md).

    Does NOT change which correspondence was chosen (that's
    `align_indoor_outdoor`/`ransac_match_with_wall_consensus`'s job,
    already reliable at disambiguating between multiple candidate walls -
    see CONTRIBUTIONS.md for why this project's own RANSAC-consensus
    tie-break is kept for that step instead of adopting an alternative
    matcher). Only adjusts rotation - by a small correction pinned at the
    matched openings' own centroid, so translation error isn't introduced
    at the wall itself - and recomputes `t`/`residuals` to match; `scale`
    is unchanged (rotating by an orthogonal correction doesn't change
    `R`'s singular values).

    `indoor_detections`/`outdoor_detections`: the same lists `result` was
    computed from - only used to recompute `residuals` (via their
    `.center`, exactly like `align_indoor_outdoor` itself does) under the
    refined transform, and as the SIGN REFERENCE for the matched walls'
    aggregated normal (see `opening_geometry.aggregate_oriented_wall_normal`)
    - critical, not cosmetic: an earlier version of this function used
    the sign-INVARIANT aggregate directly, silently assuming it already
    pointed "outward" - confirmed on real data that this can be wrong
    roughly half the time (a bare PCA/eigendecomposition normal has no
    inherent sign), which made the correction target `-n_outdoor` when
    the aggregate's arbitrary sign actually meant `+n_outdoor` was
    correct - a ~180-degree misfire that blew up residuals into the
    thousands and flipped the alignment to the wrong side. Sign-aligning
    each side's aggregate to its own `Detection3D.normal` average (already
    reliably oriented by `orient_walls_outward`) before computing the
    correction fixes this.

    `indoor_clusters`/`outdoor_clusters`: EVERY opening's raw point
    cluster for indoor/outdoor respectively (not just the matched ones -
    the aggregation is given only the matched subset internally), in the
    same order/indexing as `indoor_detections`/`outdoor_detections` (see
    `manual_segmentation.load_manual_segmentation`'s `return_points=True`).

    Falls back to returning `result` unchanged (no-op) when either
    side's aggregated wall normal isn't reliable, or the correction this
    would apply exceeds `max_deviation_deg` (deliberately NOT using
    `abs()` on the angle here - see the bug note above; a genuine
    near-180-degree disagreement must be REJECTED, not silently treated
    as "0 degrees off" the way comparing unsigned normals would).
    """
    matched_indoor_clusters = [indoor_clusters[i] for i, _, _ in result.matches]
    matched_outdoor_clusters = [outdoor_clusters[j] for _, j, _ in result.matches]
    matched_indoor_normals = [indoor_detections[i].normal for i, _, _ in result.matches]
    matched_outdoor_normals = [outdoor_detections[j].normal for _, j, _ in result.matches]

    n_indoor, ok_indoor = aggregate_oriented_wall_normal(matched_indoor_clusters, matched_indoor_normals, up_indoor)
    n_outdoor, ok_outdoor = aggregate_oriented_wall_normal(matched_outdoor_clusters, matched_outdoor_normals, up_outdoor)
    if not (ok_indoor and ok_outdoor):
        return result

    rotation_only = result.R / (float(np.linalg.norm(result.R, axis=0).mean()) + 1e-12)
    current_direction = rotation_only @ n_indoor
    current_direction = current_direction / np.linalg.norm(current_direction)
    # PARALLEL, not antiparallel: confirmed on real data (4 independently
    # signed-distance-verified datasets - q1, server+h_server,
    # 310_indoor+h_server, chua_thay) that a correct alignment has
    # `R @ n_indoor` pointing the SAME way as `n_outdoor`, not opposite -
    # an earlier version of this function assumed antiparallel (by loose
    # analogy to 2 wall FACES pointing away from each other) and got
    # ~178-180 degree "corrections" on every one of those already-correct
    # cases, which the (also buggy, at the time) deviation check failed to
    # reject. Both `n_indoor`/`n_outdoor` are sign-aligned to their own
    # side's already-trusted `Detection3D.normal` convention
    # (`orient_walls_outward`), so trust that convention's empirical
    # relationship, not an a priori physical guess.
    target_direction = n_outdoor / np.linalg.norm(n_outdoor)

    # NOT abs() here - both directions are now correctly signed, so a
    # genuine large disagreement (near -1, i.e. R@n_indoor pointing
    # OPPOSITE n_outdoor instead of the same way) must be caught.
    cos_deviation = float(np.clip(np.dot(current_direction, target_direction), -1.0, 1.0))
    deviation_deg = float(np.degrees(np.arccos(cos_deviation)))
    if deviation_deg > max_deviation_deg:
        return result

    correction = _rotation_between_unit_vectors(current_direction, target_direction)

    pivot = np.array([indoor_clusters[i].mean(axis=0) for i, _, _ in result.matches]).mean(axis=0)
    pivot_transformed = transform_points(pivot[None, :], result.R, result.t)[0]

    R_final = correction @ result.R
    t_final = correction @ (result.t - pivot_transformed) + pivot_transformed

    center_src = np.array([indoor_detections[i].center for i, _, _ in result.matches])
    center_dst = np.array([outdoor_detections[j].center for _, j, _ in result.matches])
    residuals = np.linalg.norm(transform_points(center_src, R_final, t_final) - center_dst, axis=1)
    return AlignmentResult(R=R_final, t=t_final, matches=result.matches, residuals=residuals, scale=result.scale)

