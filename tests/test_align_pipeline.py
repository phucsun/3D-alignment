import pickle

import numpy as np
import yaml

from sgd_alignment.common.types import Detection3D
from sgd_alignment.matching.alignment import transform_points
from sgd_alignment.pipelines.align_pipeline import load_align_config, run_alignment


def _make_detection(category, center, width=1.0, height=2.0):
    v_axis = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    return Detection3D(category=category, center=np.array(center), u_axis=np.cross(v_axis, normal),
                        v_axis=v_axis, normal=normal, width=width, height=height)


def _write_ply(path, points, color):
    from plyfile import PlyData, PlyElement
    colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
    vertex = np.zeros(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def test_load_align_config(tmp_path):
    config_path = tmp_path / "align.yaml"
    config_path.write_text(yaml.safe_dump({
        "indoor_detections": "a.pkl", "indoor_scene_ply": "a.ply",
        "outdoor_detections": "b.pkl", "outdoor_scene_ply": "b.ply",
        "estimate_scale": False,
    }))
    config = load_align_config(config_path)
    assert config.indoor_detections == "a.pkl"
    assert config.estimate_scale is False
    assert config.normalize_distance is True  # default preserved


def test_run_alignment_end_to_end_with_scale_mismatch(tmp_path):
    """Indoor detections/scene at a DIFFERENT (unrelated) internal scale
    than outdoor - the exact scenario `estimate_scale`/`normalize_distance`
    exist to handle (two independent DA3/COLMAP reconstructions)."""
    indoor_centers = np.array([
        [0.0, 0.0, 1.0], [3.0, 0.2, 1.0], [0.4, 4.0, 1.2], [3.3, 4.7, 1.1], [1.5, 2.3, 0.9],
    ])
    categories = ["door", "window", "window", "window", "door"]
    indoor_detections = [_make_detection(c, p) for c, p in zip(categories, indoor_centers)]

    R_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about z
    t_true = np.array([10.0, -5.0, 2.0])
    true_scale = 2.5  # outdoor reconstruction's internal unit is 2.5x indoor's
    outdoor_detections = []
    for idx in range(len(indoor_centers)):
        c = true_scale * (R_true @ indoor_centers[idx]) + t_true
        d = _make_detection(categories[idx], c, width=indoor_detections[idx].width * true_scale,
                             height=indoor_detections[idx].height * true_scale)
        d.normal[:] = R_true @ indoor_detections[idx].normal
        d.u_axis[:] = R_true @ indoor_detections[idx].u_axis
        d.v_axis[:] = R_true @ indoor_detections[idx].v_axis
        outdoor_detections.append(d)

    indoor_pkl, outdoor_pkl = tmp_path / "indoor.pkl", tmp_path / "outdoor.pkl"
    with open(indoor_pkl, "wb") as f:
        pickle.dump(indoor_detections, f)
    with open(outdoor_pkl, "wb") as f:
        pickle.dump(outdoor_detections, f)

    indoor_scene = indoor_centers  # stand-in "scene point cloud" for this test
    outdoor_scene = true_scale * (indoor_centers @ R_true.T) + t_true
    indoor_ply, outdoor_ply = tmp_path / "indoor_scene.ply", tmp_path / "outdoor_scene.ply"
    _write_ply(indoor_ply, indoor_scene, (255, 0, 0))
    _write_ply(outdoor_ply, outdoor_scene, (0, 0, 255))

    from sgd_alignment.pipelines.align_pipeline import AlignPipelineConfig
    config = AlignPipelineConfig(
        indoor_detections=str(indoor_pkl), indoor_scene_ply=str(indoor_ply),
        outdoor_detections=str(outdoor_pkl), outdoor_scene_ply=str(outdoor_ply),
        output_dir=str(tmp_path / "outputs"), output_name="aligned",
        estimate_scale=True, normalize_distance=True,  # indoor/outdoor have different absolute scales here
    )
    result = run_alignment(config)

    assert len(result.matches) == 5
    assert abs(result.scale - true_scale) < 1e-3
    assert np.allclose(result.t, t_true, atol=1e-2)

    aligned = transform_points(indoor_scene, result.R, result.t)
    assert np.allclose(aligned, outdoor_scene, atol=1e-1)

    out_dir = tmp_path / "outputs"
    assert (out_dir / "aligned.ply").exists()
    assert (out_dir / "aligned_matching.log").exists()
