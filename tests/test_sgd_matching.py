import numpy as np

from sgd_alignment.common.types import Detection3D
from sgd_alignment.matching.alignment import align_indoor_outdoor, estimate_rigid_transform, transform_points
from sgd_alignment.matching.sgd import PairWeights, build_sgds, match_sgds, sgdu_distance


def _make_detection(category: str, center: np.ndarray, width: float, height: float) -> Detection3D:
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
