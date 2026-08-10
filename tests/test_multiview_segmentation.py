import numpy as np
from scipy.spatial.transform import Rotation

from sgd_alignment.detection.multiview_segmentation import (
    ViewInstance,
    build_detections_from_instances,
    merge_instances,
)
from sgd_alignment.detection.multiview_source import (
    Camera,
    ColmapSource,
    DA3Source,
    Pose,
    backproject_mask_to_world,
    qvec2rotmat,
)


def test_qvec2rotmat_matches_scipy_roundtrip():
    rng = np.random.default_rng(0)
    for seed in range(5):
        R_true = Rotation.from_rotvec(rng.normal(size=3) * 0.8).as_matrix()
        qvec = Rotation.from_matrix(R_true).as_quat()  # scipy: (x,y,z,w)
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])  # -> COLMAP order (w,x,y,z)
        assert np.allclose(qvec2rotmat(qvec), R_true, atol=1e-10)


def test_backproject_mask_to_world_recovers_known_point():
    R = Rotation.from_rotvec([0.1, -0.3, 0.5]).as_matrix()
    t = np.array([1.0, -2.0, 0.5])
    camera = Camera(id=0, model="PINHOLE", width=100, height=100, params=np.array([80.0, 80.0, 50.0, 50.0]))
    pose = Pose(id=0, R=R, t=t, name="v")

    world_pt = np.array([0.3, -0.1, 4.0])
    p_cam = R @ world_pt + t
    K = camera.intrinsics_matrix()
    uvw = K @ p_cam
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]

    depth = np.zeros((100, 100), dtype=np.float32)
    mask = np.zeros((100, 100), dtype=bool)
    yi, xi = int(round(v)), int(round(u))
    depth[yi, xi] = p_cam[2]
    mask[yi, xi] = True

    recovered = backproject_mask_to_world(mask, depth, camera, pose)
    assert recovered.shape == (1, 3)
    assert np.allclose(recovered[0], world_pt, atol=0.05)  # pixel-rounding tolerance


def test_merge_instances_groups_by_proximity_and_category():
    same_door_view_a = {"label": "door", "points": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])}
    same_door_view_b = {"label": "door", "points": np.array([[0.05, 0.02, 0.0]])}  # same physical door, other view
    far_window = {"label": "window", "points": np.array([[5.0, 5.0, 5.0]])}
    near_but_different_category = {"label": "window", "points": np.array([[0.02, 0.0, 0.0]])}

    merged = merge_instances(
        [same_door_view_a, same_door_view_b, far_window, near_but_different_category],
        merge_distance=0.5,
    )

    doors = [c for c in merged if c["label"] == "door"]
    windows = [c for c in merged if c["label"] == "window"]
    assert len(doors) == 1
    assert len(doors[0]["points"]) == 3  # both door views merged
    assert len(windows) == 2  # far window and near-but-different-category window stay separate


def test_colmap_source_reads_text_model_and_looks_up_by_name(tmp_path):
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "cameras.txt").write_text(
        "# comment\n7 PINHOLE 640 480 500.0 500.0 320.0 240.0\n"
    )
    (sparse / "images.txt").write_text(
        "# comment\n"
        "3 1.0 0.0 0.0 0.0 0.0 0.0 0.0 7 frame_0001.jpg\n"
        "\n"
        "9 0.7071 0.0 0.7071 0.0 1.0 2.0 3.0 7 frame_0002.jpg\n"
        "\n"
    )
    (tmp_path / "images").mkdir()
    (tmp_path / "depth").mkdir()

    source = ColmapSource(sparse, tmp_path / "images", tmp_path / "depth")

    assert set(source.view_ids) == {3, 9}
    assert source.name(3) == "frame_0001.jpg"
    assert source.id_by_name("frame_0002.jpg") == 9
    assert source.camera(3).model == "PINHOLE"
    assert np.allclose(source.camera(3).intrinsics_matrix(), [[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    assert np.allclose(source.pose(9).tvec, [1.0, 2.0, 3.0])
    # identity qvec -> identity rotation
    assert np.allclose(source.pose(3).rotation_matrix(), np.eye(3), atol=1e-6)


def test_da3_source_reads_npz(tmp_path):
    n, h, w = 2, 8, 10
    depth = np.full((n, h, w), 3.0, dtype=np.float32)
    extrinsics = np.tile(np.eye(3, 4, dtype=np.float32), (n, 1, 1))
    intrinsics = np.tile(np.array([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]], dtype=np.float32), (n, 1, 1))
    conf = np.ones((n, h, w), dtype=np.float32)
    image = np.zeros((n, h, w, 3), dtype=np.uint8)
    npz_path = tmp_path / "results.npz"
    np.savez(npz_path, depth=depth, extrinsics=extrinsics, intrinsics=intrinsics, conf=conf, image=image)

    source = DA3Source(npz_path)
    assert source.view_ids == [0, 1]
    assert source.depth(0).shape == (h, w)
    assert np.allclose(source.pose(0).rotation_matrix(), np.eye(3))
    assert source.camera(0).intrinsics_matrix().shape == (3, 3)

    points, colors = source.build_scene_point_cloud()
    assert len(points) == n * h * w  # every pixel valid (constant depth, conf=1 everywhere)
    assert colors is not None and colors.shape == (len(points), 3)


def test_build_detections_from_instances_end_to_end_synthetic_scene():
    """Synthetic room: a y=0 wall (with a door-shaped mask on it), a
    z=0 floor and z=3 ceiling (so the scene has 2 distinct plane-normal
    groups and a clear "smallest extent" axis for up-vector estimation) -
    exercises the real, unmodified plane_fitting/opening_geometry code,
    just fed by this module's new backproject+merge glue instead of
    CloudCompare's scalar-field selections.
    """
    rng = np.random.default_rng(0)

    def grid(axis_fixed, value, a_range, b_range, n=90):
        a = rng.uniform(*a_range, size=n * n)
        b = rng.uniform(*b_range, size=n * n)
        pts = np.zeros((n * n, 3))
        axes = [0, 1, 2]
        axes.remove(axis_fixed)
        pts[:, axis_fixed] = value
        pts[:, axes[0]] = a
        pts[:, axes[1]] = b
        return pts

    wall = grid(1, 0.0, (-3, 3), (0, 3))       # y=0 plane (x,z varying)
    floor = grid(2, 0.0, (-3, 3), (0, 6))      # z=0 plane (x,y varying)
    ceiling = grid(2, 3.0, (-3, 3), (0, 6))    # z=3 plane
    scene_points = np.concatenate([wall, floor, ceiling])

    # camera at (0,-4,1.5) looking toward +y (straight at the wall)
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # rows: right, down, forward
    cam_pos = np.array([0.0, -4.0, 1.5])
    t = -R @ cam_pos
    W, H = 640, 480
    fx = fy = 500.0
    cx, cy = W / 2, H / 2

    class FakeSource:
        view_ids = [0]

        def camera(self, view_id):
            return Camera(id=0, model="PINHOLE", width=W, height=H, params=np.array([fx, fy, cx, cy]))

        def pose(self, view_id):
            return Pose(id=0, R=R, t=t, name="v0")

        def depth(self, view_id):
            return np.full((H, W), 4.0, dtype=np.float32)  # wall is frontal -> constant Z-depth

        def build_scene_point_cloud(self):
            return scene_points, None

    # door: x in [-0.5,0.5], z in [0.2,2.3] on the wall -> project to pixel rect analytically
    def project(x, z):
        p_cam = R @ (np.array([x, 0.0, z]) - cam_pos)
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        return u, v

    u0, v0 = project(-0.5, 2.3)
    u1, v1 = project(0.5, 0.2)
    mask = np.zeros((H, W), dtype=bool)
    mask[int(round(min(v0, v1))):int(round(max(v0, v1))), int(round(min(u0, u1))):int(round(max(u0, u1)))] = True

    instances = [ViewInstance(view_id=0, label="door", mask=mask)]
    detections = build_detections_from_instances(FakeSource(), instances, is_outdoor=False)

    assert len(detections) == 1
    d = detections[0]
    assert d.category == "door"
    assert abs(d.width - 1.0) < 0.05
    assert abs(d.height - 2.1) < 0.05
    assert np.allclose(d.center, [0.0, 0.0, 1.25], atol=0.05)
    assert abs(abs(np.dot(d.normal, [0.0, 1.0, 0.0])) - 1.0) < 1e-3
