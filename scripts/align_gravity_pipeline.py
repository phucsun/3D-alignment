"""Primary alignment pipeline for CloudCompare manual-segmentation data
that has camera pose (`results.npz` next to each .ply) - uses Minh's
gravity-locked, camera-verified alignment (`gravity_align.py` +
`robust_align.py`) instead of this project's own `align_indoor_outdoor`
(SGDU + Hungarian + RANSAC consensus).

Chosen as the PRIMARY method for this data category (2026-08-20, after a
direct real-data comparison - see CONTRIBUTIONS.md) because it does not
depend on whole-scene wall-plane detection (`extract_wall_planes`) at all
for orientation, which sidesteps a real bug in the original pipeline
where window reveals fragment wall detection into multiple close,
inconsistent candidate planes (`chua_thay`) - gravity_align got this case
CONFIDENT and correctly-oriented where the original pipeline (even after
the camera-rotation up-vector fix) needed extra per-opening-normal
aggregation just to match it.

KNOWN LIMITATION carried over from `robust_align.match_openings`, not
fixed here: when 2 different candidate walls both attract only their own
1-opening seed hypothesis (no cross-voting either way), the ranking can
prefer whichever trivially-perfect single-correspondence fit was
enumerated, over a genuinely-better multi-correspondence one - confirmed
wrong on real data (`server`, picked the known-wrong wall). This script
reports `status`/`reason` from `GravityAlignResult` for every dataset so
a `AMBIGUOUS` or suspicious result is visible, not silently accepted -
cross-check against `outputs/final_aligned/<name>_aligned.ply` (the
original `align_indoor_outdoor` pipeline, still available) if in doubt.

Does NOT (yet) cover the auto-detect (Grounding DINO + SAM2) multiview
pipeline - `multiview_pipeline.py`/`video1`/`meeting_room` are unaffected
and still use `align_indoor_outdoor`.

Usage:
    python scripts/align_gravity_pipeline.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points  # noqa: E402


@dataclass
class GravityDatasetEntry:
    name: str
    indoor_ply: str
    indoor_npz: str
    outdoor_ply: str
    outdoor_npz: str


DATASETS: list[GravityDatasetEntry] = [
    GravityDatasetEntry(
        "q1",
        "data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz",
        "data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz",
    ),
    GravityDatasetEntry(
        "q2",
        "data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply", "data/Q2/Q2_indoor/results.npz",
        "data/Q2/Q2_outdoor/Q2_outdoor_points - Cloud.ply", "data/Q2/Q2_outdoor/results.npz",
    ),
    GravityDatasetEntry(
        "server",
        "data/server/server/server_room_points-segment.ply", "data/server/server/results.npz",
        "data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz",
    ),
    GravityDatasetEntry(
        "room310_hserver",
        "data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz",
        "data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz",
    ),
    GravityDatasetEntry(
        "chua_thay",
        "data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply", "data/chua_thay/indoor/results.npz",
        "data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply", "data/chua_thay/outdoor/results.npz",
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
    indoor_clusters = openings_from_manual_segmentation(entry.indoor_ply)
    outdoor_clusters = openings_from_manual_segmentation(entry.outdoor_ply)
    cam_indoor = camera_evidence(entry.indoor_npz)
    cam_outdoor = camera_evidence(entry.outdoor_npz)
    print(f"  {len(indoor_clusters)} indoor / {len(outdoor_clusters)} outdoor raw opening cluster(s)")
    print(f"  up_consistency: indoor={cam_indoor.up_consistency:.4f} outdoor={cam_outdoor.up_consistency:.4f}")

    result = align_gravity_camera(indoor_clusters, outdoor_clusters, cam_indoor, cam_outdoor)
    print(f"  status={result.status}  reason={result.reason}")
    print(f"  matches={result.matches}  scale={result.s:.4f}")
    print(f"  grav_dot={result.grav_dot:.4f}  wall_normal_locked={result.wall_normal_locked}  "
          f"roll_refined_deg={result.roll_refined_deg:.2f}")
    print(f"  cam_side={result.cam_side}  opening_residual={result.opening_residual:.4f}")

    pts_in, cols_in = _read_ply(entry.indoor_ply)
    pts_out, cols_out = _read_ply(entry.outdoor_ply)
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
