"""Primary alignment pipeline for data with camera pose (`results.npz`) -
uses the gravity-locked, camera-verified alignment (`gravity_align.py` +
`robust_align.py`, merged in from a colleague's branch - QuangMinhPham)
instead of this project's own `align_indoor_outdoor` (SGDU + Hungarian +
RANSAC consensus).

Chosen as the PRIMARY method for camera-pose data (2026-08-20, after a
direct real-data comparison - see CONTRIBUTIONS.md) because it does not
depend on whole-scene wall-plane detection (`extract_wall_planes`) at all
for orientation, which sidesteps a real bug in the original pipeline
where window reveals fragment wall detection into multiple close,
inconsistent candidate planes (`chua_thay`) - gravity_align got this case
CONFIDENT and correctly-oriented where the original pipeline (even after
the camera-rotation up-vector fix) needed extra per-opening-normal
aggregation just to match it.

Supports TWO opening sources, both feeding the SAME `align_gravity_camera`:
  - CloudCompare manual segmentation (`openings_from_manual_segmentation`)
  - Auto-detect (Grounding DINO + SAM2, `multiview_pipeline.py`) - reads
    the `<name>_raw_clusters.pkl` that pipeline now saves alongside its
    usual `<name>_detections.pkl`, so no re-detection is needed here.

KNOWN LIMITATION carried over from `robust_align.match_openings` (the
OTHER matcher in that file, not used by this script's `align_gravity_camera`
path, but sharing some helpers): not applicable here - the 3 real bugs
found in `gravity_align.align_gravity_camera`'s OWN candidate search during
the merge (category-mixed hypotheses, missing subset sizes, mean-residual-
only ranking) are already fixed - see CONTRIBUTIONS.md section 11.

Usage:
    python scripts/align_gravity_pipeline.py
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import CameraEvidence, camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import (  # noqa: E402
    OpeningCluster, openings_from_manual_segmentation, transform_points,
)

OpeningLoader = Callable[[], tuple[list[OpeningCluster], list[OpeningCluster], CameraEvidence, CameraEvidence]]


@dataclass
class GravityDatasetEntry:
    name: str
    load_openings: OpeningLoader
    indoor_scene_ply: str  # full colored point cloud to transform + export
    outdoor_scene_ply: str


def _manual_segmentation_loader(indoor_ply: str, indoor_npz: str, outdoor_ply: str, outdoor_npz: str) -> OpeningLoader:
    def _load():
        indoor_clusters = openings_from_manual_segmentation(indoor_ply)
        outdoor_clusters = openings_from_manual_segmentation(outdoor_ply)
        return indoor_clusters, outdoor_clusters, camera_evidence(indoor_npz), camera_evidence(outdoor_npz)

    return _load


def _auto_detect_loader(indoor_pkl: str, indoor_npz: str, outdoor_pkl: str, outdoor_npz: str) -> OpeningLoader:
    """`indoor_pkl`/`outdoor_pkl`: a `<name>_raw_clusters.pkl` saved by
    `multiview_pipeline.run_pipeline` - a `list[OpeningCluster]` built
    directly from the merged Grounding DINO + SAM2 instances, same type
    `openings_from_manual_segmentation` returns for the CloudCompare path,
    so `align_gravity_camera` doesn't need to know which source it came
    from."""

    def _load():
        with open(indoor_pkl, "rb") as f:
            indoor_clusters = pickle.load(f)
        with open(outdoor_pkl, "rb") as f:
            outdoor_clusters = pickle.load(f)
        return indoor_clusters, outdoor_clusters, camera_evidence(indoor_npz), camera_evidence(outdoor_npz)

    return _load


DATASETS: list[GravityDatasetEntry] = [
    GravityDatasetEntry(
        "q1",
        _manual_segmentation_loader(
            "data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz",
            "data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
        "data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply",
    ),
    GravityDatasetEntry(
        "q2",
        _manual_segmentation_loader(
            "data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply", "data/Q2/Q2_indoor/results.npz",
            "data/Q2/Q2_outdoor/Q2_outdoor_points - Cloud.ply", "data/Q2/Q2_outdoor/results.npz"),
        "data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply", "data/Q2/Q2_outdoor/Q2_outdoor_points - Cloud.ply",
    ),
    GravityDatasetEntry(
        "server",
        _manual_segmentation_loader(
            "data/server/server/server_room_points-segment.ply", "data/server/server/results.npz",
            "data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
        "data/server/server/server_room_points-segment.ply",
        "data/server/h_server/h_server_room_points - Cloud - segment.ply",
    ),
    GravityDatasetEntry(
        "room310_hserver",
        _manual_segmentation_loader(
            "data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz",
            "data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
        "data/310_indoor/310_indoor_points - Cloud - segment.ply",
        "data/server/h_server/h_server_room_points - Cloud - segment.ply",
    ),
    GravityDatasetEntry(
        "chua_thay",
        _manual_segmentation_loader(
            "data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply", "data/chua_thay/indoor/results.npz",
            "data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply", "data/chua_thay/outdoor/results.npz"),
        "data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply",
        "data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply",
    ),
    GravityDatasetEntry(
        "q815",
        _auto_detect_loader(
            "outputs/q815/indoor_raw_clusters.pkl", "data/Q815/Q815_indoor/results.npz",
            "outputs/q815/outdoor_raw_clusters.pkl", "data/Q815/Q815_outdoor/results.npz"),
        "outputs/q815/indoor_scene.ply", "outputs/q815/outdoor_scene.ply",
    ),
]


def _read_ply(path: str):
    from plyfile import PlyData

    v = PlyData.read(path)["vertex"].data
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8) if "red" in v.dtype.names else None
    return points, colors


def _write_ply(points: np.ndarray, colors: np.ndarray | None, path: str) -> None:
    from plyfile import PlyData, PlyElement

    if colors is None:
        colors = np.full((len(points), 3), 160, np.uint8)
    vertex = np.zeros(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def run(entry: GravityDatasetEntry, output_dir: str = "outputs/final_aligned") -> None:
    print(f"\n=== {entry.name} ===")
    indoor_clusters, outdoor_clusters, cam_indoor, cam_outdoor = entry.load_openings()
    print(f"  {len(indoor_clusters)} indoor / {len(outdoor_clusters)} outdoor raw opening cluster(s)")
    print(f"  up_consistency: indoor={cam_indoor.up_consistency:.4f} outdoor={cam_outdoor.up_consistency:.4f}")

    result = align_gravity_camera(indoor_clusters, outdoor_clusters, cam_indoor, cam_outdoor)
    print(f"  status={result.status}  reason={result.reason}")
    print(f"  matches={result.matches}  scale={result.s:.4f}")
    print(f"  grav_dot={result.grav_dot:.4f}  wall_normal_locked={result.wall_normal_locked}  "
          f"roll_refined_deg={result.roll_refined_deg:.2f}")
    print(f"  cam_side={result.cam_side}  opening_residual={result.opening_residual:.4f}")

    pts_in, cols_in = _read_ply(entry.indoor_scene_ply)
    pts_out, cols_out = _read_ply(entry.outdoor_scene_ply)
    aligned_in = transform_points(pts_in, result.s, result.R, result.t)
    all_pts = np.concatenate([aligned_in, pts_out])
    all_cols = np.concatenate([
        cols_in if cols_in is not None else np.full((len(pts_in), 3), 160, np.uint8),
        cols_out if cols_out is not None else np.full((len(pts_out), 3), 160, np.uint8),
    ])
    out_path = f"{output_dir}/{entry.name}_aligned.ply"
    _write_ply(all_pts, all_cols, out_path)
    print(f"  saved -> {out_path}")


def main() -> None:
    for entry in DATASETS:
        run(entry)


if __name__ == "__main__":
    main()
