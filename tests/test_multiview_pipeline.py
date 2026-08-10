import numpy as np
import yaml

from sgd_alignment.detection.segment_2d import DetectedInstance
from sgd_alignment.pipelines.multiview_pipeline import (
    MultiviewPipelineConfig,
    SourceConfig,
    load_pipeline_config,
    run_pipeline,
)


def test_load_pipeline_config_parses_nested_sections(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "source": {"backend": "da3", "npz_path": "some/results.npz"},
        "is_outdoor": True,
        "scale": 2.4,
        "selected_views": [0, 2],
        "output_dir": "out",
        "output_name": "outdoor",
        "segmenter": {"box_threshold": 0.6},
    }))

    config = load_pipeline_config(config_path)

    assert config.source.backend == "da3"
    assert config.source.npz_path == "some/results.npz"
    assert config.is_outdoor is True
    assert config.scale == 2.4
    assert config.selected_views == [0, 2]
    assert config.segmenter.box_threshold == 0.6
    assert config.segmenter.text_threshold == 0.25  # default preserved


class _StubSegmenter:
    """Returns a fixed instance list per view_id, bypassing real GDINO/SAM2
    model loads so this test doesn't need the `segmentation` extra."""

    def __init__(self, per_view_instances: dict[int, list[DetectedInstance]]):
        self._per_view_instances = per_view_instances
        self._next_view_id = iter(per_view_instances.keys())
        self._current = None

    def segment(self, image_rgb):
        # run_pipeline calls source.image(view_id) then segmenter.segment(image),
        # in the same view_id order as source.view_ids - consume matching entry
        view_id = next(self._next_view_id)
        return self._per_view_instances[view_id]


def test_run_pipeline_end_to_end_with_stub_segmenter_and_da3_source(tmp_path):
    """Synthetic DA3 npz with 2 views (a wall-facing view carrying the door
    mask, and a floor-facing view giving the up-vector estimator a 2nd
    plane axis to distinguish from), run through the *real* `run_pipeline`
    (config loading, DA3Source, backproject, merge, wall detection,
    Detection3D construction, .pkl/.ply export) with a stub 2D segmenter
    standing in for Grounding DINO + SAM2.
    """
    W, H = 640, 480
    fx = fy = 500.0
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # view 0: camera at (0,-4,1.5) facing +y, frontal onto the y=0 wall -> constant Z-depth
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    cam0_pos = np.array([0.0, -4.0, 1.5], dtype=np.float32)
    t0 = -R0 @ cam0_pos
    depth0 = np.full((H, W), 4.0, dtype=np.float32)

    # view 1: camera at (0,3,10) facing straight down (-z), frontal onto the z=0 floor
    R1 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    cam1_pos = np.array([0.0, 3.0, 10.0], dtype=np.float32)
    t1 = -R1 @ cam1_pos
    depth1 = np.full((H, W), 10.0, dtype=np.float32)

    depth = np.stack([depth0, depth1])
    extrinsics = np.stack([
        np.concatenate([R0, t0[:, None]], axis=1),
        np.concatenate([R1, t1[:, None]], axis=1),
    ])
    intrinsics = np.stack([K, K])
    conf = np.ones((2, H, W), dtype=np.float32)
    image = np.zeros((2, H, W, 3), dtype=np.uint8)

    npz_path = tmp_path / "results.npz"
    np.savez(npz_path, depth=depth, extrinsics=extrinsics, intrinsics=intrinsics, conf=conf, image=image)

    # door on the wall: x in [-0.5,0.5], z in [0.2,2.3] -> project to view-0 pixels
    def project_view0(x, z):
        p_cam = R0 @ (np.array([x, 0.0, z]) - cam0_pos)
        return fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy

    u0, v0 = project_view0(-0.5, 2.3)
    u1, v1 = project_view0(0.5, 0.2)
    mask = np.zeros((H, W), dtype=bool)
    mask[int(round(min(v0, v1))):int(round(max(v0, v1))), int(round(min(u0, u1))):int(round(max(u0, u1)))] = True

    door_instance = DetectedInstance(label="door", conf=0.9, box=[float(min(u0, u1)), float(min(v0, v1)),
                                                                    float(max(u0, u1)), float(max(v0, v1))], mask=mask)
    stub = _StubSegmenter({0: [door_instance], 1: []})

    config = MultiviewPipelineConfig(
        source=SourceConfig(backend="da3", npz_path=str(npz_path)),
        is_outdoor=False,
        output_dir=str(tmp_path / "outputs"),
        output_name="indoor",
    )

    detections = run_pipeline(config, segmenter=stub)

    assert len(detections) == 1
    d = detections[0]
    assert d.category == "door"
    assert abs(d.width - 1.0) < 0.05
    assert abs(d.height - 2.1) < 0.05
    assert np.allclose(d.center, [0.0, 0.0, 1.25], atol=0.05)

    out_dir = tmp_path / "outputs"
    assert (out_dir / "indoor_detections.pkl").exists()
    assert (out_dir / "indoor_openings.ply").exists()
    assert (out_dir / "indoor_view_0000_segmented.jpg").exists()
