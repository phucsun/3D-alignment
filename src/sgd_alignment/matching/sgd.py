"""Semantic-Geometric Descriptor (SGD) construction and matching (Section 3.2, 4).

Verified against Yang et al., "Indoor-Outdoor Point Cloud Alignment Using
Semantic-Geometric Descriptor", Remote Sens. 2022, 14, 5119.

For object i, its SGDU (semantic-geometric descriptor unit) is (S^i, G^i),
where S^i is i's own category and G^i holds one 8-component group {^jG^i}
per other object j in the same scene:

    1. S^j          - j's category
    2. d_i^j        - Euclidean distance between the origins (centers) of
                      i's and j's local coordinates
    3-5. alpha, beta, theta - angles between the vector (center_j -
                      center_i) and the x, y, z axes *of O_i's own local
                      frame* (Section 3.2, Figure 5)
    6-8. rotx, roty, rotz  - Euler angles of the relative rotation that

                      takes O_i's local frame to O_j's local frame


Matching (Section 4) is two-level bipartite assignment, both solved with
the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`, which
natively handles rectangular cost matrices - this plays the role of the
paper's "improved Hungarian algorithm" for unequal set sizes without
needing its brute-force subset search):

  - inner level (Algorithm 2): for one candidate pair (object k in scene
    1, object l in scene 2), the distance between their SGDUs is itself
    the cost of the best assignment between k's (N-1) neighbor-groups and
    l's neighbor-groups - i.e. a full Hungarian match of the two
    "distribution patterns", not just a naive nearest-to-nearest
    comparison. A component exceeding its own threshold, or the two
    neighbor's categories disagreeing, makes that pairing invalid (cost
    infinity); if the whole row/column is invalid the pair can't be
    matched at all (SGDU distance = infinity).
  - outer level (Algorithm 3 / Section 4.2): the same Hungarian matching
    is applied again across all (indoor object, outdoor object) pairs,
    using the inner-level distances as the cost matrix, to get the final
    object correspondences. All-invalid rows/columns are dropped first
    (handles indoor/outdoor detecting different numbers of openings).

NOTE: the paper does not publish numeric values for the lambda weights or
the per-component thresholds (they were presumably tuned empirically on
the authors' data) - the defaults below are reasonable starting points
that need calibrating against real annotated data, not values taken from
the paper itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from sgd_alignment.common.types import Detection3D

EULER_ORDER = "xyz"
INFEASIBLE = float("inf")
_LARGE_FINITE_COST = 1e9


@dataclass
class PairWeights:
    """lambda_1..7 in Equation (2), and their 7 independent rejection
    thresholds - one pair per geometric error term (d, alpha, beta, theta,
    rotx, roty, rotz), matching the paper's Equation (2) exactly rather
    than sharing one weight/threshold across the 3 direction angles or the
    3 rotation angles.
    """

    distance_weight: float = 1.0
    alpha_weight: float = 0.4
    beta_weight: float = 0.4
    theta_weight: float = 0.4
    rotx_weight: float = 0.4
    roty_weight: float = 0.4
    rotz_weight: float = 0.4

    distance_threshold: float = 1.5  # meters
    alpha_threshold: float = np.pi / 3
    beta_threshold: float = np.pi / 3
    theta_threshold: float = np.pi / 3
    rotx_threshold: float = np.pi / 3
    roty_threshold: float = np.pi / 3
    rotz_threshold: float = np.pi / 3

    # used only by the intrinsic-geometry fallback (see sgdu_distance's
    # `use_intrinsic_fallback`) - max relative difference in width/height
    # aspect ratio to still consider 2 isolated objects a candidate match
    intrinsic_aspect_ratio_threshold: float = 0.5

    @property
    def angle_weights(self) -> np.ndarray:
        return np.array([self.alpha_weight, self.beta_weight, self.theta_weight])

    @property
    def angle_thresholds(self) -> np.ndarray:
        return np.array([self.alpha_threshold, self.beta_threshold, self.theta_threshold])

    @property
    def rotation_weights(self) -> np.ndarray:
        return np.array([self.rotx_weight, self.roty_weight, self.rotz_weight])

    @property
    def rotation_thresholds(self) -> np.ndarray:
        return np.array([self.rotx_threshold, self.roty_threshold, self.rotz_threshold])


@dataclass
class PairFeature:
    """The 8-component {^jG^i} group: object i described relative to neighbor j."""

    label: str  # S^j
    distance: float  # d_i^j
    direction_angles: np.ndarray  # (3,) alpha, beta, theta - in O_i's own frame
    relative_euler_angles: np.ndarray  # (3,) rotx, roty, rotz


@dataclass
class SGD:
    """SGDU for one object: its own category plus its PairFeature group G^i
    against every other object in the scene."""

    detection: Detection3D
    neighbors: list[PairFeature] = field(default_factory=list)


def _local_frame(det: Detection3D) -> np.ndarray:
    """3x3 rotation matrix whose columns are (x, y, z) in the paper's own
    convention (Figure 3): x = normal (points outside), y = u_axis (right-
    hand rule from z and x), z = v_axis (points up). Column order matters
    here (not just which 3 vectors are included), since alpha/beta/theta
    and rotx/roty/rotz are indexed against this exact x-y-z ordering.
    """
    return np.stack([det.normal, det.u_axis, det.v_axis], axis=1)


def _pair_feature(i: Detection3D, j: Detection3D, distance_scale: float = 1.0) -> PairFeature:
    offset = j.center - i.center
    raw_distance = float(np.linalg.norm(offset))
    direction = offset / raw_distance if raw_distance > 1e-12 else offset
    distance = raw_distance / distance_scale

    frame_i = _local_frame(i)
    direction_angles = np.arccos(np.clip(direction @ frame_i, -1.0, 1.0))

    frame_j = _local_frame(j)
    r_rel = frame_i.T @ frame_j
    relative_euler_angles = Rotation.from_matrix(r_rel).as_euler(EULER_ORDER)

    return PairFeature(
        label=j.category,
        distance=distance,
        direction_angles=direction_angles,
        relative_euler_angles=relative_euler_angles,
    )


def _scene_distance_scale(detections: list[Detection3D]) -> float:
    """Median pairwise center-to-center distance within one scene - a
    characteristic "unit" of that scene's own layout, used by
    `build_sgds(..., normalize_distance=True)` to make the descriptor's
    distance component scale-invariant.

    Needed when indoor and outdoor come from two independently, arbitrarily
    scaled reconstructions with no shared metric reference (e.g. two
    separate feed-forward depth-model inference runs): comparing raw
    `distance` values between the two sides is then comparing different,
    unrelated units. Dividing every distance by this scene-local median
    turns it into a dimensionless ratio ("how far, relative to how objects
    in *this* scene are typically spaced") that's comparable across scenes
    regardless of their absolute scale.
    """
    if len(detections) < 2:
        return 1.0
    centers = np.array([d.center for d in detections])
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(len(centers), k=1)
    scale = float(np.median(dists[iu]))
    return scale if scale > 1e-9 else 1.0


def build_sgds(detections: list[Detection3D], normalize_distance: bool = False) -> list[SGD]:
    """Build one SGDU per detection: its PairFeature group against every
    other detection in the same scene (order doesn't matter here - the
    inner Hungarian match in `sgdu_distance` doesn't assume any).

    `normalize_distance=False` (default) keeps the exact prior behavior
    (raw metric distances) - needed for the already-verified real-world
    datasets where indoor/outdoor share a common metric scale. See
    `_scene_distance_scale` for what `normalize_distance=True` fixes and
    when it's actually needed.
    """
    scale = _scene_distance_scale(detections) if normalize_distance else 1.0
    sgds = []
    for i, det in enumerate(detections):
        neighbors = [_pair_feature(det, detections[j], scale) for j in range(len(detections)) if j != i]
        sgds.append(SGD(detection=det, neighbors=neighbors))
    return sgds


def _pair_feature_cost(a: PairFeature, b: PairFeature, w: PairWeights) -> float:
    """Equation (2): the cost of matching neighbor-group a to neighbor-group b."""
    if a.label != b.label:
        return INFEASIBLE

    d_e = abs(a.distance - b.distance)
    angle_e = np.abs(a.direction_angles - b.direction_angles)  # (alpha_e, beta_e, theta_e)
    rot_e = np.abs(a.relative_euler_angles - b.relative_euler_angles)  # (rotx_e, roty_e, rotz_e)

    if d_e > w.distance_threshold:
        return INFEASIBLE
    if np.any(angle_e > w.angle_thresholds):
        return INFEASIBLE
    if np.any(rot_e > w.rotation_thresholds):
        return INFEASIBLE

    return (
        w.distance_weight * d_e
        + float(np.dot(w.angle_weights, angle_e))
        + float(np.dot(w.rotation_weights, rot_e))
    )


def _assign_rectangular(cost: np.ndarray) -> tuple[float, int]:
    """Solve a (possibly non-square, possibly-infeasible-entry) assignment
    problem, dropping all-infeasible rows/columns first (as the paper
    does before applying the Hungarian algorithm). Returns (mean cost per
    feasible matched pair, number of feasible matched pairs).

    Mean rather than total: a candidate pair that only manages to match 1
    of 6 possible neighbors isn't more similar than one matching 5 of 6 -
    summing raw cost would perversely favor matching *fewer* neighbors
    (fewer terms added up), since each excluded pair contributes nothing
    to the sum instead of being penalized for not being comparable at all.
    """
    if cost.size == 0:
        return INFEASIBLE, 0

    valid_rows = ~np.all(np.isinf(cost), axis=1)
    valid_cols = ~np.all(np.isinf(cost), axis=0)
    sub = cost[np.ix_(valid_rows, valid_cols)]
    if sub.size == 0:
        return INFEASIBLE, 0

    finite_sub = np.where(np.isinf(sub), _LARGE_FINITE_COST, sub)
    row_idx, col_idx = linear_sum_assignment(finite_sub)

    total_cost = 0.0
    num_matched = 0
    for r, c in zip(row_idx, col_idx):
        if not np.isinf(sub[r, c]):
            total_cost += sub[r, c]
            num_matched += 1

    if num_matched == 0:
        return INFEASIBLE, 0
    return total_cost / num_matched, num_matched


def _regions_compatible(a: Detection3D, b: Detection3D) -> bool:
    """`Detection3D.region` is `None` unless explicitly labeled - compatible
    by default (exact prior behavior, no region information available).
    When BOTH sides carry a label, they must match exactly.

    This is a hard, human-provided constraint, not a geometric heuristic:
    it exists for cases no amount of geometry or imagery can resolve on
    its own - e.g. two architecturally symmetric corridors on opposite
    sides of a building, where a matched pair scores identically well
    under either physical assignment (confirmed: neither RANSAC consensus
    nor a richer descriptor can break a *genuine* symmetry, since the
    ambiguous configurations are, by construction, indistinguishable from
    the sensor data alone). Whoever captured the data already knows which
    physical space is which; `region` just lets that knowledge into the
    matching cost instead of asking the algorithm to reverse-engineer it.
    Labeling every detection in a scene with the same region id (e.g. its
    room name) is also the natural building block for matching MORE than
    2 spaces at once in the future (a labeled graph of regions instead of
    a single indoor/outdoor pair), not just a tie-breaker for this case.
    """
    return a.region is None or b.region is None or a.region == b.region


def _intrinsic_cost(a: Detection3D, b: Detection3D, weights: PairWeights) -> float:
    """Cost between 2 detections using only each one's OWN absolute
    geometric properties - no relation to other objects in the scene.

    This is the only signal available when a scene has too few objects
    for SGD's neighbor-relative descriptor to mean anything (an object
    with 0 neighbors has an empty G^i, so `sgdu_distance` can never match
    it to anything - confirmed on real data: an outdoor scan with only 1
    detected door could never be matched to its indoor counterpart, even
    though the door itself was detected correctly, simply because SGD had
    no relational description to compare). Kabsch/Umeyama can already
    recover a full 6(+1 scale)-DOF transform from a *single* correctly
    identified match (a real door/window's 8 corners span 3 distinct
    dimensions - width, height, thickness - so it isn't rotationally
    symmetric); the missing piece was ever proposing that single match in
    the first place.

    Compares width/height aspect ratio (scale-invariant - indoor/outdoor
    may not share a metric scale yet at matching time, see
    `sgd.build_sgds`'s `normalize_distance`), not absolute size.
    """
    if a.category != b.category or not _regions_compatible(a, b):
        return INFEASIBLE
    if a.height < 1e-9 or b.height < 1e-9:
        return INFEASIBLE
    aspect_a, aspect_b = a.width / a.height, b.width / b.height
    relative_diff = abs(aspect_a - aspect_b) / max(aspect_a, aspect_b, 1e-9)
    if relative_diff > weights.intrinsic_aspect_ratio_threshold:
        return INFEASIBLE
    return relative_diff


def sgdu_distance(
    a: SGD, b: SGD, weights: PairWeights | None = None, use_intrinsic_fallback: bool = False
) -> tuple[float, int]:
    """Algorithm 2: the distance between two SGDUs.

    This is itself the cost of the best assignment between a's neighbor
    groups and b's neighbor groups (the inner Hungarian match) - not a
    naive position-by-position comparison, since neither scene's object
    ordering means anything and the two objects may have different
    numbers of neighbors.

    `use_intrinsic_fallback=False` (default) keeps the exact prior
    behavior: an object with no neighbors (the only object detected in
    its scene) can never be matched. Set `True` to fall back to
    `_intrinsic_cost` specifically in that case - deliberately NOT used
    when both objects merely fail the normal relational comparison
    (wrong geometry is still wrong geometry; the fallback only kicks in
    when the relational comparison had literally nothing to compare).
    """
    if a.detection.category != b.detection.category or not _regions_compatible(a.detection, b.detection):
        return INFEASIBLE, 0

    weights = weights or PairWeights()
    if use_intrinsic_fallback and (not a.neighbors or not b.neighbors):
        return _intrinsic_cost(a.detection, b.detection, weights), 0

    cost = np.array([[_pair_feature_cost(fa, fb, weights) for fb in b.neighbors] for fa in a.neighbors])
    return _assign_rectangular(cost)


def _ambiguity_ratio(costs: np.ndarray, chosen_idx: int) -> float | None:
    """Lowe's-ratio-test-style ambiguity check: how much better is the
    chosen (Hungarian-optimal) cost than the best *alternative* in the
    same row/column? Returns `chosen / next_best` (near 0 = clearly
    distinctive, near 1 = another candidate was almost as good), or
    `None` if there's no finite alternative to compare against (nothing
    to be ambiguous with).
    """
    others = np.delete(costs, chosen_idx)
    others = others[np.isfinite(others)]
    if len(others) == 0:
        return None
    next_best = float(others.min())
    chosen = float(costs[chosen_idx])
    if next_best <= 1e-12:
        return 0.0 if chosen <= 1e-12 else float("inf")
    return chosen / next_best


def match_sgds(
    indoor_sgds: list[SGD],
    outdoor_sgds: list[SGD],
    weights: PairWeights | None = None,
    max_cost: float = 5.0,
    use_intrinsic_fallback: bool = False,
    use_ambiguity_check: bool = False,
    ambiguity_ratio_threshold: float = 0.8,
) -> list[tuple[int, int, float]]:
    """Algorithm 3 (matching phase): match indoor SGDUs to outdoor SGDUs.

    Builds the adjacency matrix of SGDU-to-SGDU distances (Equation 3),
    drops all-infeasible rows/columns, then runs the Hungarian algorithm
    once more at this outer level. Returns (indoor_index, outdoor_index,
    cost) triples for accepted matches - pairs whose cost still exceeds
    `max_cost` after assignment are dropped (no real counterpart existed).

    `use_intrinsic_fallback`: see `sgdu_distance`.

    `use_ambiguity_check=False` by default (exact prior behavior): the
    Hungarian-optimal pair is always accepted regardless of how close the
    *next-best* alternative was. Real risk this misses: a scene can
    contain several same-category objects that have no true counterpart
    at all (confirmed on real data - a corridor scan detected 5 doors,
    only 1 physically corresponding to anything indoors, the other 4
    unrelated fire-exit/utility doors); if one of those distractors
    happens to have a similarly-good relational fit by coincidence,
    nothing currently flags that the "best" match wasn't clearly better
    than a wrong one. Set `True` to reject (not just accept-the-best) any
    match whose cost isn't decisively lower than the next-best
    alternative *on either side* (row and column) - the same principle as
    Lowe's ratio test for SIFT feature matching. Trades some recall
    (a few correct-but-genuinely-ambiguous matches get dropped too) for
    precision (fewer coincidental false accepts) - see
    `refine_matches_with_geometric_consensus` in `alignment.py` for a
    complementary, non-mutually-exclusive fix.
    """
    if not indoor_sgds or not outdoor_sgds:
        return []
    weights = weights or PairWeights()

    cost = np.array([[sgdu_distance(a, b, weights, use_intrinsic_fallback)[0] for b in outdoor_sgds] for a in indoor_sgds])

    valid_rows = ~np.all(np.isinf(cost), axis=1)
    valid_cols = ~np.all(np.isinf(cost), axis=0)
    row_ids = np.where(valid_rows)[0]
    col_ids = np.where(valid_cols)[0]
    sub = cost[np.ix_(valid_rows, valid_cols)]
    if sub.size == 0:
        return []

    finite_sub = np.where(np.isinf(sub), _LARGE_FINITE_COST, sub)
    row_idx, col_idx = linear_sum_assignment(finite_sub)

    matches = []
    for r, c in zip(row_idx, col_idx):
        if np.isinf(sub[r, c]) or sub[r, c] > max_cost:
            continue
        if use_ambiguity_check:
            row_ratio = _ambiguity_ratio(sub[r, :], c)
            col_ratio = _ambiguity_ratio(sub[:, c], r)
            if (row_ratio is not None and row_ratio > ambiguity_ratio_threshold) or \
               (col_ratio is not None and col_ratio > ambiguity_ratio_threshold):
                continue
        matches.append((int(row_ids[r]), int(col_ids[c]), float(sub[r, c])))
    return matches