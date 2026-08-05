from pathlib import Path

from sgd_alignment.common.config import load_config
from sgd_alignment.detection.point_cloud_utils import load_point_cloud, project_top_view_png

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "Indoor-Outdoor-Point-Cloud-Dataset-main"


def test_load_point_cloud_scenario1_room1_indoor():
    pc = load_point_cloud(DATASET_DIR / "scenario1_room1_indoor.ply")
    assert len(pc) == 1081334
    assert pc.points.shape == (1081334, 3)
    # public release strips intensity/reflectance, only xyz is present
    assert pc.intensity is None
    assert pc.normals is None


def test_load_point_cloud_scenario1_room1_outdoor():
    pc = load_point_cloud(DATASET_DIR / "scenario1_room1_outdoor.ply")
    assert len(pc) == 654710
    assert pc.points.shape == (654710, 3)


def test_config_resolves_all_dataset_pairs():
    cfg = load_config(REPO_ROOT / "configs" / "detection.yaml")
    for scenario, room, _ in [
        ("scenario1", "room1", None),
        ("scenario1", "room2", None),
        ("scenario2", "room1", None),
        ("scenario2", "room2", None),
        ("scenario2", "room3", None),
        ("scenario2", "room4", None),
        ("scenario2", "room5", None),
    ]:
        for side in ("indoor", "outdoor"):
            name = f"{scenario}_{room}_{side}"
            path = cfg.resolve_point_cloud(name, base_dir=REPO_ROOT)
            assert path.exists(), f"missing {name}"


def test_project_top_view_png(tmp_path):
    pc = load_point_cloud(DATASET_DIR / "scenario1_room1_indoor.ply")
    out = project_top_view_png(pc, tmp_path / "scenario1_room1_indoor_top_view.png")
    assert out.exists()
    assert out.stat().st_size > 0
