"""Chạy riêng từng bộ NỐI 2 KHÔNG GIAN (như đã test trước ở
`multi_space_scenarios_test.py`) qua `resolve_opening_conflict_graph`
(docs/multi_space_alignment_plan.md, Giai đoạn 3), xuất point cloud đã ghép
ra `outputs/multi_space_pairs/` để kiểm tra bằng mắt.

4 bộ: q1, q2, chua_thay (đã biết đúng từ trước, mục 11 CONTRIBUTIONS.md) +
h_server/server (1 cạnh trong bộ 4-không-gian server, để so sánh).

Usage:
    python scripts/multi_space_pairs_export.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points  # noqa: E402
from sgd_alignment.matching.multi_space_graph import resolve_opening_conflict_graph  # noqa: E402

PAIR_DATASETS = {
    "q1": {
        "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
        "q1_outdoor": ("data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
    },
    "q2": {
        "q2_indoor": ("data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply", "data/Q2/Q2_indoor/results.npz"),
        "q2_outdoor": ("data/Q2/Q2_outdoor/Q2_outdoor_points - Cloud.ply", "data/Q2/Q2_outdoor/results.npz"),
    },
    "chua_thay": {
        "chua_thay_indoor": ("data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply",
                              "data/chua_thay/indoor/results.npz"),
        "chua_thay_outdoor": ("data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply",
                               "data/chua_thay/outdoor/results.npz"),
    },
    "server_pair": {
        "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
        "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    },
}


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


def main() -> None:
    out_dir = Path("outputs/multi_space_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, spaces in PAIR_DATASETS.items():
        print(f"\n=== {label} ===")
        clusters, cams = {}, {}
        for name, (ply, npz) in spaces.items():
            clusters[name] = openings_from_manual_segmentation(ply)
            cams[name] = camera_evidence(npz)
            print(f"  {name}: {len(clusters[name])} opening cluster(s)")

        edges = resolve_opening_conflict_graph(clusters, cams, verbose=True)
        if not edges:
            print("  KHÔNG tìm được cạnh nào - bỏ qua xuất file.")
            continue

        (a, b), edge = next(iter(edges.items()))
        print(f"  -> cạnh chấp nhận: {a}-{b}  dùng {a}{sorted(i for _, i in edge.used_a)} <-> "
              f"{b}{sorted(j for _, j in edge.used_b)}  weight={edge.weight:.3f}")

        ply_a, _ = spaces[a]
        ply_b, _ = spaces[b]
        pts_a, cols_a = _read_ply(ply_a)
        pts_b, cols_b = _read_ply(ply_b)
        if cols_a is None:
            cols_a = np.full((len(pts_a), 3), 160, np.uint8)
        if cols_b is None:
            cols_b = np.full((len(pts_b), 3), 160, np.uint8)

        aligned_a = transform_points(pts_a, edge.s, edge.R, edge.t)
        out_path = out_dir / f"{label}_aligned.ply"
        _write_ply(np.concatenate([aligned_a, pts_b]), np.concatenate([cols_a, cols_b]), str(out_path))
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
