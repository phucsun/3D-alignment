import numpy as np
import pytest

from sgd_alignment.common.types import Detection3D
from sgd_alignment.matching.alignment import (
    align_indoor_outdoor,
    align_rooms_to_hub,
    estimate_rigid_transform,
    group_by_wall,
    ransac_match_with_wall_consensus,
    refine_matches_with_geometric_consensus,
    transform_points,
)
from sgd_alignment.matching.sgd import PairWeights, build_sgds, match_sgds, sgdu_distance


def _make_detection(category: str, center: np.ndarray, width: float, height: float, region=None) -> Detection3D:
    # right-handed frame (u_axis = cross(v_axis, normal)), matching the
    # convention real annotations use (see annotations.py) - required for
    # relative_euler_angles' rotation-matrix math to be valid
    v_axis = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    return Detection3D(
        category=category,
        center=center,
        u_axis=np.cross(v_axis, normal),
        v_axis=v_axis,
        normal=normal,
        width=width,
        height=height,
        region=region,
    )


def _random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.3, 1.2)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def test_estimate_rigid_transform_recovers_known_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(-5, 5, size=(6, 3))
    R_true = _random_rotation(1)
    t_true = np.array([3.0, -2.0, 1.5])
    dst = transform_points(src, R_true, t_true)

    R, t = estimate_rigid_transform(src, dst)

    assert np.allclose(R, R_true, atol=1e-8)
    assert np.allclose(t, t_true, atol=1e-8)


def test_pair_feature_direction_angles_use_is_own_frame():
    """Section 3.2: alpha/beta/theta are the angles between vector O_i->O_j
    and O_i's OWN local axes - not O_j's."""
    i = _make_detection("door", np.array([0.0, 0.0, 0.0]), 0.9, 2.1)
    # j sits along i's own +normal axis (i's normal is [0,1,0])
    j = _make_detection("window", np.array([0.0, 5.0, 0.0]), 1.0, 1.0)
    # give j a completely different frame so a same-frame bug would show up
    j.normal[:] = [1.0, 0.0, 0.0]
    j.v_axis[:] = [0.0, 1.0, 0.0]
    j.u_axis[:] = np.cross(j.v_axis, j.normal)

    sgds = build_sgds([i, j])
    feature = sgds[0].neighbors[0]  # i's feature relative to j
    # direction (0,1,0) against i's own axes, in the paper's (x=normal,
    # y=u_axis, z=v_axis) order: i's normal=[0,1,0] -> index 0 is 0 rad;
    # i's u_axis=[1,0,0] -> index 1 is pi/2
    assert np.isclose(feature.direction_angles[0], 0.0, atol=1e-10)  # vs i's normal (x)
    assert np.isclose(feature.direction_angles[1], np.pi / 2, atol=1e-10)  # vs i's u_axis (y)


def test_sgdu_distance_matches_neighbors_optimally_regardless_of_order():
    """The inner Hungarian match should find the correct neighbor
    correspondence even when neighbor lists are built in different orders
    and have different lengths - not assume position k lines up with k."""
    # object A has 2 neighbors: a door far away, a window close by
    a_center = np.array([0.0, 0.0, 0.0])
    a = _make_detection("door", a_center, 0.9, 2.1)
    a_neighbor_far_door = _make_detection("door", np.array([10.0, 0.0, 0.0]), 0.9, 2.1)
    a_neighbor_near_window = _make_detection("window", np.array([2.0, 0.0, 0.0]), 1.0, 1.0)

    # object B (candidate match for A) has the SAME two neighbors, physically
    # equivalent, but listed in the opposite order and with an extra one
    b_center = np.array([100.0, 100.0, 0.0])
    b = _make_detection("door", b_center, 0.9, 2.1)
    b_neighbor_near_window = _make_detection("window", b_center + np.array([2.0, 0.0, 0.0]), 1.0, 1.0)
    b_neighbor_far_door = _make_detection("door", b_center + np.array([10.0, 0.0, 0.0]), 0.9, 2.1)
    b_neighbor_extra = _make_detection("window", b_center + np.array([0.0, 20.0, 0.0]), 1.0, 1.0)

    sgds_a = build_sgds([a, a_neighbor_far_door, a_neighbor_near_window])
    sgds_b = build_sgds([b, b_neighbor_near_window, b_neighbor_far_door, b_neighbor_extra])

    cost, num_matched = sgdu_distance(sgds_a[0], sgds_b[0])
    assert num_matched == 2
    assert cost < 0.5  # should find the near-perfect correspondence despite order/count difference


def test_align_indoor_outdoor_recovers_transform_and_correspondence():
    # 5 openings in the "indoor" frame: mix of doors and windows
    indoor_centers = np.array([
        [0.0, 0.0, 1.0],
        [3.0, 0.2, 1.0],
        [0.4, 4.0, 1.2],
        [3.3, 4.7, 1.1],
        [1.5, 2.3, 0.9],
    ])
    categories = ["door", "window", "window", "window", "door"]
    indoor = [
        _make_detection(c, p, 0.9 if c == "door" else 1.1, 2.1 if c == "door" else 1.0)
        for c, p in zip(categories, indoor_centers)
    ]

    R_true = _random_rotation(42)
    t_true = np.array([10.0, -5.0, 2.0])
    # outdoor detections are the SAME physical openings, transformed, and
    # shuffled into a different order (indoor/outdoor detection order has
    # no reason to correspond)
    perm = [3, 1, 4, 0, 2]
    outdoor = []
    for idx in perm:
        c = transform_points(indoor_centers[idx : idx + 1], R_true, t_true)[0]
        d = _make_detection(categories[idx], c, indoor[idx].width, indoor[idx].height)
        d.normal[:] = R_true @ indoor[idx].normal
        d.u_axis[:] = R_true @ indoor[idx].u_axis
        d.v_axis[:] = R_true @ indoor[idx].v_axis
        outdoor.append(d)

    result = align_indoor_outdoor(indoor, outdoor)

    assert len(result.matches) == 5
    assert np.allclose(result.R, R_true, atol=1e-6)
    assert np.allclose(result.t, t_true, atol=1e-6)
    assert np.all(result.residuals < 1e-6)

    # every matched pair must be the correct physical correspondence
    for indoor_idx, outdoor_idx, _cost in result.matches:
        assert perm[indoor_idx] == outdoor_idx


def test_match_sgds_rejects_category_mismatch():
    door = _make_detection("door", np.array([0.0, 0.0, 0.0]), 0.9, 2.1)
    window = _make_detection("window", np.array([0.0, 0.0, 0.0]), 0.9, 2.1)
    sgds_a = build_sgds([door])
    sgds_b = build_sgds([window])
    matches = match_sgds(sgds_a, sgds_b)
    assert matches == []


def test_align_indoor_outdoor_survives_realistic_manual_annotation_noise():
    """Simulate hand-picking error: a human eyeballing the center of an
    empty opening from its surrounding frame, and roughly estimating its
    width/height, will not get exact numbers. Indoor and outdoor are
    annotated independently, so their errors are independent, not
    correlated - this perturbs each side separately.

    All openings here keep the same wall normal (axis-aligned annotation
    is the realistic case: a human typically just picks +/-X or +/-Y, not
    an exact wall-fitted normal), so the rotation/angle components of the
    descriptor carry no discriminating signal in this scenario - matching
    relies on distance + category, same as it would for a human annotator
    who didn't bother computing exact per-wall normals.
    """
    rng = np.random.default_rng(7)

    # 7 openings spread across multiple walls of a ~8m x 6m room (not
    # collinear - a realistic mixed layout, not a single wall of windows)
    true_centers = np.array([
        [0.0, 0.0, 1.2],
        [0.0, 2.5, 1.2],
        [0.0, 5.0, 1.0],   # door, wall x=0
        [4.0, 0.0, 1.2],
        [8.0, 0.0, 1.2],   # wall y=0
        [8.0, 3.0, 1.0],   # door, wall x=8
        [3.0, 6.0, 1.2],   # wall y=6
    ])
    categories = ["window", "window", "door", "window", "window", "door", "window"]
    true_width = [1.1, 1.1, 0.9, 1.1, 1.1, 0.9, 1.1]
    true_height = [1.0, 1.0, 2.1, 1.0, 1.0, 2.1, 1.0]

    def annotate(centers, seed, center_noise=0.15, size_noise_frac=0.2):
        r = np.random.default_rng(seed)
        noisy_centers = centers + r.uniform(-center_noise, center_noise, size=centers.shape)
        dets = []
        for i in range(len(centers)):
            w = true_width[i] * (1 + r.uniform(-size_noise_frac, size_noise_frac))
            h = true_height[i] * (1 + r.uniform(-size_noise_frac, size_noise_frac))
            dets.append(_make_detection(categories[i], noisy_centers[i], w, h))
        return dets

    indoor = annotate(true_centers, seed=1)

    R_true = _random_rotation(99)
    t_true = np.array([20.0, -8.0, 0.5])
    outdoor_true_centers = transform_points(true_centers, R_true, t_true)
    outdoor_full = annotate(outdoor_true_centers, seed=2)
    perm = rng.permutation(len(outdoor_full))
    outdoor = [outdoor_full[i] for i in perm]
    inverse_perm = {int(new): int(old) for new, old in enumerate(perm)}

    # noisy distance costs are bigger than the exact-data tests above, and
    # every opening here shares one wall normal (see docstring), so the
    # angle/rotation thresholds must be permissive enough not to reject
    # everything outright
    weights = PairWeights(
        alpha_threshold=np.pi, beta_threshold=np.pi, theta_threshold=np.pi,
        rotx_threshold=np.pi, roty_threshold=np.pi, rotz_threshold=np.pi,
    )
    result = align_indoor_outdoor(indoor, outdoor, weights=weights, max_cost=1.0)

    assert len(result.matches) == len(true_centers)
    for indoor_idx, outdoor_idx, _cost in result.matches:
        assert inverse_perm[outdoor_idx] == indoor_idx

    # recovered transform should be close to ground truth, though not
    # exact, since it was fit to noisy points - alignment now uses all 8
    # box corners (per the paper), so the +/-20% width/height noise also
    # leaks into the fit, not just the +/-15cm center noise
    rotation_error = np.linalg.norm(result.R - R_true)
    translation_error = np.linalg.norm(result.t - t_true)
    assert rotation_error < 0.2, f"rotation off by {rotation_error}"
    assert translation_error < 0.5, f"translation off by {translation_error}"


def test_align_indoor_outdoor_ignores_unmatched_extra_opening():
    # indoor has an extra door with no outdoor counterpart (e.g. interior door)
    indoor_centers = np.array([
        [0.0, 0.0, 1.0],
        [3.0, 0.0, 1.0],
        [0.0, 4.0, 1.2],
        [5.0, 5.0, 1.0],  # extra, unmatched
    ])
    categories = ["door", "window", "window", "door"]
    indoor = [
        _make_detection(c, p, 0.9 if c == "door" else 1.1, 2.1 if c == "door" else 1.0)
        for c, p in zip(categories, indoor_centers)
    ]

    R_true = np.eye(3)
    t_true = np.array([1.0, 1.0, 0.0])
    outdoor = [
        _make_detection(categories[i], transform_points(indoor_centers[i : i + 1], R_true, t_true)[0],
                         indoor[i].width, indoor[i].height)
        for i in range(3)  # only the first 3 openings exist outdoors
    ]

    result = align_indoor_outdoor(indoor, outdoor)
    assert len(result.matches) == 3
    matched_indoor_idxs = {m[0] for m in result.matches}
    assert 3 not in matched_indoor_idxs


def test_refine_matches_with_geometric_consensus_fixes_local_symmetric_swap():
    """A row of evenly-spaced near-identical windows is exactly where pure
    SGD (local relative-geometry) matching can confuse 2 neighbors -
    confirmed on real data (sc1-room2: 2/7 windows swapped, ~2.96m
    residual, every other match correct and under 5cm - SGD matching
    alone never fixed it).

    Simulate that failure mode directly rather than relying on real data
    in a test: build the TRUE correspondence, then hand
    `refine_matches_with_geometric_consensus` a `matches` list with 2
    same-category entries deliberately swapped (as if SGD had confused
    them) among 5 correct ones. The refiner doesn't get told which ones
    are wrong - it must recover that from (R, t) fit to the majority.
    """
    indoor_centers = np.array([
        [0.0, 3.0, 1.0],   # door, off the line so the whole set isn't collinear
        [2.0, 0.0, 1.0], [4.0, 0.0, 1.0], [6.0, 0.0, 1.0],
        [8.0, 0.0, 1.0], [10.0, 0.0, 1.0], [12.0, 0.0, 1.0],  # 6 evenly-spaced windows
    ])
    categories = ["door", "window", "window", "window", "window", "window", "window"]
    indoor = [_make_detection(c, p, 0.9 if c == "door" else 1.1, 2.1 if c == "door" else 1.0)
              for c, p in zip(categories, indoor_centers)]

    R_true = _random_rotation(7)
    t_true = np.array([5.0, 2.0, 1.0])
    outdoor_centers = transform_points(indoor_centers, R_true, t_true)
    outdoor = [_make_detection(c, p, d.width, d.height) for c, p, d in zip(categories, outdoor_centers, indoor)]
    for d_out, d_in in zip(outdoor, indoor):
        d_out.normal[:] = R_true @ d_in.normal
        d_out.u_axis[:] = R_true @ d_in.u_axis
        d_out.v_axis[:] = R_true @ d_in.v_axis

    wrong_matches = [(i, i, 0.0) for i in range(7)]
    wrong_matches[2] = (2, 5, 0.0)  # simulate SGD confusing 2 evenly-spaced windows
    wrong_matches[5] = (5, 2, 0.0)

    refined, R, t = refine_matches_with_geometric_consensus(indoor, outdoor, wrong_matches)

    assert {i: j for i, j, _ in refined} == {i: i for i in range(7)}
    assert np.allclose(R, R_true, atol=1e-4)
    assert np.allclose(t, t_true, atol=1e-4)


def test_refine_matches_with_geometric_consensus_noop_below_3_matches():
    indoor = [_make_detection("door", np.array([0.0, 0.0, 0.0]), 0.9, 2.1)]
    outdoor = [_make_detection("door", np.array([1.0, 0.0, 0.0]), 0.9, 2.1)]
    matches = [(0, 0, 0.0)]
    refined, R, t = refine_matches_with_geometric_consensus(indoor, outdoor, matches)
    assert refined == matches
    assert R is None and t is None


def test_estimate_rigid_transform_recovers_known_scale():
    """estimate_scale=True (Umeyama) must recover the true scale factor
    between src and dst - needed when they come from two independently,
    arbitrarily scaled reconstructions with no shared metric reference
    (e.g. two separate feed-forward depth-model inference runs)."""
    rng = np.random.default_rng(3)
    src = rng.uniform(-5, 5, size=(6, 3))
    R_true = _random_rotation(4)
    t_true = np.array([2.0, -1.0, 4.0])
    true_scale = 2.4
    dst = transform_points(src, true_scale * R_true, t_true)

    R, t = estimate_rigid_transform(src, dst, estimate_scale=True)
    recovered_scale = np.linalg.norm(R, axis=0).mean()

    assert np.isclose(recovered_scale, true_scale, atol=1e-6)
    assert np.allclose(R / recovered_scale, R_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)

    # without estimate_scale, a pure rotation cannot fit scaled data - the
    # fit is systematically biased (large residual), proving this is
    # actually needed and not just a harmless no-op to add
    R_no_scale, t_no_scale = estimate_rigid_transform(src, dst, estimate_scale=False)
    biased_residual = np.linalg.norm(transform_points(src, R_no_scale, t_no_scale) - dst, axis=1).mean()
    assert biased_residual > 1.0


def test_align_indoor_outdoor_recovers_transform_when_outdoor_is_independently_scaled():
    """Simulates 2 separate feed-forward depth-model inference runs (e.g.
    2 independent DA3 sessions, one for indoor and one for outdoor) with
    no shared metric reference - outdoor's whole coordinate frame is an
    unrelated absolute scale from indoor's. Plain `align_indoor_outdoor`
    (raw metric distances, pure-rotation Kabsch) must fail outright on
    this; `normalize_distance=True` + `estimate_scale=True` must recover
    both the correct correspondences and the true scale/rotation/translation.
    """
    indoor_centers = np.array([
        [0.0, 0.0, 1.0],
        [3.0, 0.2, 1.0],
        [0.4, 4.0, 1.2],
        [3.3, 4.7, 1.1],
        [1.5, 2.3, 0.9],
    ])
    categories = ["door", "window", "window", "window", "door"]
    indoor = [
        _make_detection(c, p, 0.9 if c == "door" else 1.1, 2.1 if c == "door" else 1.0)
        for c, p in zip(categories, indoor_centers)
    ]

    R_true = _random_rotation(11)
    t_true = np.array([50.0, -30.0, 5.0])
    true_scale = 2.4  # outdoor's independent reconstruction has its own unrelated absolute scale
    perm = [3, 1, 4, 0, 2]
    outdoor = []
    for idx in perm:
        c = transform_points(indoor_centers[idx : idx + 1], true_scale * R_true, t_true)[0]
        d = _make_detection(
            categories[idx], c,
            indoor[idx].width * true_scale, indoor[idx].height * true_scale,
        )
        d.normal[:] = R_true @ indoor[idx].normal
        d.u_axis[:] = R_true @ indoor[idx].u_axis
        d.v_axis[:] = R_true @ indoor[idx].v_axis
        outdoor.append(d)

    # plain call (default params) must fail - raw distances differ by
    # ~2.4x between the two sides, exceeding the default (meter-scale)
    # distance_threshold for every candidate pair
    with pytest.raises(ValueError):
        align_indoor_outdoor(indoor, outdoor, max_cost=1.0)

    weights = PairWeights(distance_threshold=0.3)  # now a relative ratio, not meters
    result = align_indoor_outdoor(
        indoor, outdoor, weights=weights, max_cost=1.0,
        estimate_scale=True, normalize_distance=True,
    )

    assert len(result.matches) == 5
    for indoor_idx, outdoor_idx, _cost in result.matches:
        assert perm[indoor_idx] == outdoor_idx
    assert np.isclose(result.scale, true_scale, atol=1e-3)
    assert np.allclose(result.R / result.scale, R_true, atol=1e-4)
    assert np.allclose(result.t, t_true, atol=1e-2)


def test_align_indoor_outdoor_intrinsic_fallback_matches_single_isolated_opening():
    """The exact real failure mode this fallback targets: a scene with
    only 1 detected opening has an empty G^i (no neighbors), so ordinary
    SGD matching can never propose a match for it at all - confirmed on
    real data (a home capture where the outdoor scan only had 1 detected
    door out of 2 physical ones): `align_indoor_outdoor` raised "0
    opening(s) matched" even though the single detection itself was
    perfectly correct.
    """
    indoor = [_make_detection("door", np.array([0.0, 0.0, 1.0]), 0.9, 2.1)]

    R_true = _random_rotation(11)
    t_true = np.array([3.0, -1.0, 0.5])
    outdoor_center = transform_points(indoor[0].center[None, :], R_true, t_true)[0]
    outdoor = [_make_detection("door", outdoor_center, 0.9, 2.1)]
    outdoor[0].normal[:] = R_true @ indoor[0].normal
    outdoor[0].u_axis[:] = R_true @ indoor[0].u_axis
    outdoor[0].v_axis[:] = R_true @ indoor[0].v_axis

    # default: still fails exactly as before (no regression)
    with pytest.raises(ValueError, match="opening"):
        align_indoor_outdoor(indoor, outdoor)

    # with fallback: recovers the single match and the correct transform
    result = align_indoor_outdoor(indoor, outdoor, use_intrinsic_fallback=True)
    assert len(result.matches) == 1
    assert result.matches[0][:2] == (0, 0)
    assert np.allclose(result.R, R_true, atol=1e-6)
    assert np.allclose(result.t, t_true, atol=1e-6)


def test_align_indoor_outdoor_intrinsic_fallback_rejects_wrong_aspect_ratio():
    indoor = [_make_detection("door", np.array([0.0, 0.0, 1.0]), 0.9, 2.1)]
    outdoor = [_make_detection("door", np.array([5.0, 5.0, 1.0]), 2.1, 0.9)]  # swapped w/h -> very different aspect

    with pytest.raises(ValueError, match="opening"):
        align_indoor_outdoor(indoor, outdoor, use_intrinsic_fallback=True)


def test_match_sgds_ambiguity_check_rejects_coincidental_near_tie():
    """Object A (door, index 0) has 2 near-equally-good candidates in the
    other scene - a coincidental tie the plain Hungarian assignment can't
    detect (it just picks whichever is marginally cheaper). Costs are
    deliberately kept away from ~0 (moderate neighbor-distance offsets,
    not a near-perfect match) so the *ratio* between best and second-best
    is meaningfully close to 1, not just their absolute difference small.
    With the ambiguity check off (default), the marginally-cheaper one
    gets accepted outright; with it on, door[0]'s match should be
    rejected as too ambiguous to trust.
    """
    a = _make_detection("door", np.array([0.0, 0.0, 0.0]), 0.9, 2.1)
    a_neighbor = _make_detection("window", np.array([2.0, 0.0, 0.0]), 1.0, 1.0)

    # b1's neighbor is 0.5m further than a's (cost ~0.5); b2's neighbor is
    # 0.55m further (cost ~0.55) - a coincidental near-tie (ratio ~0.91)
    b1 = _make_detection("door", np.array([100.0, 0.0, 0.0]), 0.9, 2.1)
    b1_neighbor = _make_detection("window", np.array([102.5, 0.0, 0.0]), 1.0, 1.0)

    b2 = _make_detection("door", np.array([200.0, 0.0, 0.0]), 0.9, 2.1)
    b2_neighbor = _make_detection("window", np.array([202.55, 0.0, 0.0]), 1.0, 1.0)

    sgds_a = build_sgds([a, a_neighbor])
    sgds_b = build_sgds([b1, b1_neighbor, b2, b2_neighbor])

    matches_default = match_sgds(sgds_a, sgds_b)
    door_match = next(m for m in matches_default if m[0] == 0)
    assert door_match[1] == 0  # accepts the marginally-better b1 regardless

    matches_checked = match_sgds(sgds_a, sgds_b, use_ambiguity_check=True, ambiguity_ratio_threshold=0.8)
    assert not any(m[0] == 0 for m in matches_checked)  # too close to call - rejected rather than guessed


def test_match_sgds_ambiguity_check_keeps_clearly_distinctive_match():
    """A genuinely distinctive match (candidates far apart in cost) must
    still be accepted - the check should only reject *close* calls."""
    a = _make_detection("door", np.array([0.0, 0.0, 0.0]), 0.9, 2.1)
    a_neighbor = _make_detection("window", np.array([2.0, 0.0, 0.0]), 1.0, 1.0)

    b_good = _make_detection("door", np.array([100.0, 0.0, 0.0]), 0.9, 2.1)
    b_good_neighbor = _make_detection("window", np.array([102.0, 0.0, 0.0]), 1.0, 1.0)  # matches distance exactly

    b_bad = _make_detection("door", np.array([200.0, 0.0, 0.0]), 0.9, 2.1)
    b_bad_neighbor = _make_detection("window", np.array([210.0, 0.0, 0.0]), 1.0, 1.0)  # way off (10m vs 2m)

    sgds_a = build_sgds([a, a_neighbor])
    sgds_b = build_sgds([b_good, b_good_neighbor, b_bad, b_bad_neighbor])

    matches = match_sgds(sgds_a, sgds_b, use_ambiguity_check=True, ambiguity_ratio_threshold=0.8)
    door_match = next(m for m in matches if m[0] == 0)
    assert door_match[1] == 0  # matched to b_good (index 0 in sgds_b)


def test_align_indoor_outdoor_min_dimension_drops_degenerate_detection():
    """A near-zero-width detection (segmentation/backprojection artifact,
    not a real opening - confirmed on real data) must not participate in
    matching when `min_dimension` is set, and dropped indices must not
    shift the meaning of surviving matches (they still index into the
    ORIGINAL lists passed in)."""
    indoor = [
        _make_detection("door", np.array([0.0, 0.0, 1.0]), 0.9, 2.1),
        _make_detection("door", np.array([3.0, 0.0, 1.0]), 0.03, 2.1),  # degenerate
        _make_detection("door", np.array([0.0, 4.0, 1.2]), 0.9, 2.1),
    ]
    outdoor = [
        _make_detection("door", np.array([10.0, 0.0, 1.0]), 0.9, 2.1),
        _make_detection("door", np.array([13.0, 0.0, 1.0]), 0.03, 2.1),  # degenerate
        _make_detection("door", np.array([10.0, 4.0, 1.2]), 0.9, 2.1),
    ]

    result = align_indoor_outdoor(indoor, outdoor, min_dimension=0.15)
    matched_indoor = {i for i, _, _ in result.matches}
    matched_outdoor = {j for _, j, _ in result.matches}
    assert 1 not in matched_indoor  # the degenerate detection never entered matching
    assert 1 not in matched_outdoor
    assert matched_indoor == {0, 2}
    assert matched_outdoor == {0, 2}


def test_ransac_match_with_wall_consensus_rejects_degenerate_bad_match():
    """Mirrors the real meeting_room failure this was built to fix: 4
    matches are mutually correct and consistent (same position AND wall
    orientation under one true transform); a 5th is a bad, near-degenerate
    box that happens to sit close in position but belongs to a completely
    unrelated wall. `refine_matches_with_geometric_consensus` (position
    only, bootstrapped from all 5 including the bad one) was confirmed on
    real data to accept exactly this kind of bad match; RANSAC consensus
    (seeded from small samples, scored by how many *other* pairs agree)
    must recover just the 4 correct ones.
    """
    indoor_centers = np.array([
        [0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [0.0, 4.0, 1.2], [3.0, 4.0, 1.1],
    ])
    R_true = _random_rotation(21)
    t_true = np.array([5.0, -2.0, 1.0])
    indoor = [_make_detection("door", c, 0.9, 2.1) for c in indoor_centers]
    outdoor = []
    for i, c in enumerate(indoor_centers):
        oc = transform_points(c[None, :], R_true, t_true)[0]
        d = _make_detection("door", oc, 0.9, 2.1)
        d.normal[:] = R_true @ indoor[i].normal
        d.u_axis[:] = R_true @ indoor[i].u_axis
        d.v_axis[:] = R_true @ indoor[i].v_axis
        outdoor.append(d)

    # 5th pair: near-degenerate width, position transforms correctly, but
    # wall orientation is unrelated to R_true (a genuinely different wall)
    bad_indoor = _make_detection("door", np.array([10.0, 10.0, 1.0]), 0.03, 2.1)
    bad_outdoor_center = transform_points(np.array([[10.0, 10.0, 1.0]]), R_true, t_true)[0]
    bad_outdoor = _make_detection("door", bad_outdoor_center, 0.03, 2.1)
    bad_outdoor.normal[:] = [1.0, 0.0, 0.0]
    bad_outdoor.v_axis[:] = [0.0, 0.0, 1.0]
    bad_outdoor.u_axis[:] = np.cross(bad_outdoor.v_axis, bad_outdoor.normal)

    all_indoor = indoor + [bad_indoor]
    all_outdoor = outdoor + [bad_outdoor]
    matches = [(i, i, 0.0) for i in range(5)]  # as if SGD matching had accepted all 5

    refined, R, t = ransac_match_with_wall_consensus(
        all_indoor, all_outdoor, matches, normal_angle_threshold=np.radians(30),
    )
    assert {(i, j) for i, j, _ in refined} == {(0, 0), (1, 1), (2, 2), (3, 3)}
    assert np.allclose(R, R_true, atol=1e-4)
    assert np.allclose(t, t_true, atol=1e-4)


def test_align_indoor_outdoor_region_label_resolves_symmetric_ambiguity():
    """Mirrors a real architectural-symmetry scenario: 2 outdoor doors are
    geometrically identical (same category, same aspect ratio) - nothing
    in the geometry or intrinsic descriptor can tell them apart, exactly
    the situation discussed as unsolvable by data alone. `region` breaks
    the tie because it's human-provided, not inferred.
    """
    indoor = [_make_detection("door", np.array([0.0, 0.0, 1.0]), 0.9, 2.1, region="wing_A")]

    R_true = _random_rotation(31)
    t_true = np.array([4.0, 1.0, 0.5])
    correct_center = transform_points(indoor[0].center[None, :], R_true, t_true)[0]
    correct = _make_detection("door", correct_center, 0.9, 2.1, region="wing_A")
    correct.normal[:] = R_true @ indoor[0].normal
    correct.u_axis[:] = R_true @ indoor[0].u_axis
    correct.v_axis[:] = R_true @ indoor[0].v_axis

    # a symmetric decoy: identical category/size, plausible position, but
    # explicitly labeled as belonging to a DIFFERENT wing of the building
    decoy = _make_detection("door", np.array([-4.0, -1.0, 0.5]), 0.9, 2.1, region="wing_B")

    outdoor = [decoy, correct]  # decoy listed first so an unconstrained match could pick either

    without_region = [
        _make_detection(d.category, d.center, d.width, d.height) for d in outdoor
    ]
    for i in range(len(outdoor)):
        without_region[i].normal[:] = outdoor[i].normal
        without_region[i].u_axis[:] = outdoor[i].u_axis
        without_region[i].v_axis[:] = outdoor[i].v_axis
    indoor_no_region = [_make_detection(indoor[0].category, indoor[0].center, indoor[0].width, indoor[0].height)]

    # without region info: both candidates are equally valid intrinsic
    # matches (identical aspect ratio) - whichever is cheaper wins, no
    # guarantee it's the physically correct one
    result_unconstrained = align_indoor_outdoor(indoor_no_region, without_region, use_intrinsic_fallback=True)
    assert len(result_unconstrained.matches) == 1  # matches *something* - not necessarily correctly

    # with region info: only wing_A is even feasible, regardless of cost
    result = align_indoor_outdoor(indoor, outdoor, use_intrinsic_fallback=True)
    assert len(result.matches) == 1
    indoor_idx, outdoor_idx, _ = result.matches[0]
    assert outdoor_idx == 1  # the "correct" (wing_A) detection, not the wing_B decoy
    assert np.allclose(result.R, R_true, atol=1e-6)
    assert np.allclose(result.t, t_true, atol=1e-6)


def test_group_by_wall_separates_opposite_facing_normals():
    """A corridor's 2 facing walls have near-antiparallel normals despite
    being physically close - grouping by `abs(dot)` (as an earlier,
    buggy version of `plane_fitting.merge_coplanar_planes` did for actual
    wall planes) would wrongly fuse them. `group_by_wall` must use signed
    dot so openings on opposite walls end up in different groups even
    though their normals are nearly parallel in the ABS sense.
    """
    same_wall_a = _make_detection("door", np.array([0.0, 0.0, 0.0]), 0.9, 2.1)
    same_wall_b = _make_detection("door", np.array([1.0, 0.0, 0.0]), 0.9, 2.1)
    opposite_wall = _make_detection("door", np.array([0.0, 5.0, 0.0]), 0.9, 2.1)
    opposite_wall.normal[:] = -same_wall_a.normal  # near-antiparallel, not same wall
    opposite_wall.u_axis[:] = np.cross(opposite_wall.v_axis, opposite_wall.normal)

    groups = group_by_wall([same_wall_a, same_wall_b, opposite_wall])
    groups_as_sets = [set(g) for g in groups]
    assert {0, 1} in groups_as_sets  # same_wall_a/b grouped together
    assert {2} in groups_as_sets  # opposite_wall kept separate


def test_align_rooms_to_hub_assigns_rooms_to_correct_walls_by_total_cost():
    """Mirrors the real `310_indoor`/`server`/`h_server` bug: 2 rooms each
    open onto the SAME hallway via 2 different physical walls. Naive
    independent-per-room matching (each room run alone against the full
    hub set) was confirmed on real data to be able to produce results
    that are each internally self-consistent yet globally wrong, because
    neither run has visibility into what the other room needs.
    `align_rooms_to_hub` must resolve this with a proper minimum-total-cost
    assignment: both rooms have their own opening spacing designed to
    fit BOTH hub walls somewhat, but only one specific (room, wall)
    pairing per room is the true low-cost match.
    """
    # hub: wall A openings 1.0 apart (normal +x), wall B openings 3.0 apart
    # (normal -x, physically the opposite side of the corridor)
    hub = [
        _make_detection("door", np.array([0.0, 0.0, 0.0]), 0.9, 2.1),
        _make_detection("door", np.array([0.0, 1.0, 0.0]), 0.9, 2.1),
        _make_detection("door", np.array([0.0, 0.0, 5.0]), 0.9, 2.1),
        _make_detection("door", np.array([0.0, 3.0, 5.0]), 0.9, 2.1),
    ]
    for d in hub[2:]:
        d.normal[:] = -hub[0].normal
        d.u_axis[:] = np.cross(d.v_axis, d.normal)

    # room "310": own openings 1.05 apart -> clearly fits hub wall A (1.0) far
    # better than wall B (3.0)
    room_310 = [
        _make_detection("door", np.array([10.0, 0.0, 0.0]), 0.9, 2.1),
        _make_detection("door", np.array([10.0, 1.05, 0.0]), 0.9, 2.1),
    ]
    # room "server": own openings 3.1 apart -> clearly fits hub wall B (3.0)
    # far better than wall A (1.0)
    room_server = [
        _make_detection("door", np.array([-10.0, 0.0, 0.0]), 0.9, 2.1),
        _make_detection("door", np.array([-10.0, 3.1, 0.0]), 0.9, 2.1),
    ]

    wall_groups = group_by_wall(hub)
    assert len(wall_groups) == 2
    wall_a_group, wall_b_group = wall_groups

    assignments = align_rooms_to_hub(
        [room_310, room_server], hub, estimate_scale=True, normalize_distance=True,
    )

    assert len(assignments) == 2
    assert all(a is not None for a in assignments)
    room_310_assignment, room_server_assignment = assignments
    assert set(room_310_assignment.wall_hub_indices) == set(wall_a_group)
    assert set(room_server_assignment.wall_hub_indices) == set(wall_b_group)
    assert room_310_assignment.result.residuals.max() < 0.5
    assert room_server_assignment.result.residuals.max() < 0.5


def test_align_rooms_to_hub_never_assigns_two_rooms_to_the_same_wall():
    """Even when both rooms' own naive per-wall preference (evaluated in
    isolation) would point at the SAME wall, `align_rooms_to_hub` must
    still produce a valid one-to-one assignment (no double-booking) -
    the defining guarantee of the Hungarian assignment step, independent
    of the exact cost values.
    """
    hub = [
        _make_detection("door", np.array([0.0, 0.0, 0.0]), 0.9, 2.1),
        _make_detection("door", np.array([0.0, 1.0, 0.0]), 0.9, 2.1),
        _make_detection("door", np.array([0.0, 0.0, 5.0]), 0.9, 2.1),
        _make_detection("door", np.array([0.0, 1.05, 5.0]), 0.9, 2.1),
    ]
    for d in hub[2:]:
        d.normal[:] = -hub[0].normal
        d.u_axis[:] = np.cross(d.v_axis, d.normal)

    # both rooms have near-identical spacing (~1.0-1.02) - each would
    # independently "prefer" whichever wall is tried, since both walls
    # are almost equally good (1.0 vs 1.05 apart)
    room_a = [
        _make_detection("door", np.array([10.0, 0.0, 0.0]), 0.9, 2.1),
        _make_detection("door", np.array([10.0, 1.0, 0.0]), 0.9, 2.1),
    ]
    room_b = [
        _make_detection("door", np.array([-10.0, 0.0, 0.0]), 0.9, 2.1),
        _make_detection("door", np.array([-10.0, 1.02, 0.0]), 0.9, 2.1),
    ]

    assignments = align_rooms_to_hub([room_a, room_b], hub, estimate_scale=True, normalize_distance=True)
    assert len(assignments) == 2
    assert all(a is not None for a in assignments)

    used_wall_groups = [tuple(sorted(a.wall_hub_indices)) for a in assignments]
    assert used_wall_groups[0] != used_wall_groups[1]  # no double-booking
