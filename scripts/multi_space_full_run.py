"""Chạy `resolve_opening_conflict_graph` (docs/multi_space_alignment_plan.md,
Giai đoạn 3 - hướng đã verify tốt nhất) trên TOÀN BỘ dữ liệu CloudCompare/
manual-segmentation thật đang có trong `data/` (10 không gian, 5 "building"
khác nhau: server, Q1, Q2, chùa Thầy, connecting_space+310_indoor cùng
server) - KHÔNG cho biết trước bất kỳ thông tin topology/hub nào.

Đây là bài test khắt khe nhất từ trước đến giờ: C(10,2)=45 cặp ứng viên,
đa số HOÀN TOÀN không liên quan (khác building) - kỳ vọng thuật toán tự
tách đúng thành các "cụm liên thông" (connected component) trùng khớp
CHÍNH XÁC với ranh giới building thật, không rò rỉ cạnh giả xuyên building.

Xuất kết quả (point cloud đã ghép, mỗi cụm liên thông 1 file) ra
`outputs/multi_space_full_graph/`.

Usage:
    python scripts/multi_space_full_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points  # noqa: E402
from sgd_alignment.matching.multi_space_graph import resolve_opening_conflict_graph, compose  # noqa: E402

ALL_SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
    "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
    "q1_outdoor": ("data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
    "q2_indoor": ("data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply", "data/Q2/Q2_indoor/results.npz"),
    "q2_outdoor": ("data/Q2/Q2_outdoor/Q2_outdoor_points - Cloud.ply", "data/Q2/Q2_outdoor/results.npz"),
    "chua_thay_indoor": ("data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply",
                          "data/chua_thay/indoor/results.npz"),
    "chua_thay_outdoor": ("data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply",
                           "data/chua_thay/outdoor/results.npz"),
}

# Ranh giới building THẬT (để đối chiếu) - mọi cạnh xuyên nhóm đều là false positive
EXPECTED_GROUPS = [
    {"h_server", "server", "room_310", "connecting_space"},
    {"q1_indoor", "q1_outdoor"},
    {"q2_indoor", "q2_outdoor"},
    {"chua_thay_indoor", "chua_thay_outdoor"},
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


def _find(parent: dict, x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def main() -> None:
    print("=== Đang load toàn bộ 10 không gian ===")
    clusters, cams = {}, {}
    for name, (ply, npz) in ALL_SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
        print(f"  {name}: {len(clusters[name])} opening cluster(s)")

    print(f"\n=== Chạy resolve_opening_conflict_graph trên C({len(ALL_SPACES)},2)="
          f"{len(ALL_SPACES) * (len(ALL_SPACES) - 1) // 2} cặp ứng viên (không biết trước topology) ===")
    edges = resolve_opening_conflict_graph(clusters, cams, verbose=True)

    print(f"\n=== {len(edges)} cạnh được chấp nhận ===")
    parent = {name: name for name in ALL_SPACES}
    for (a, b), edge in edges.items():
        ra, rb = _find(parent, a), _find(parent, b)
        cross_building = not any({a, b} <= g for g in EXPECTED_GROUPS)
        tag = "!! XUYÊN BUILDING (SAI) !!" if cross_building else "trong-building (đúng phạm vi)"
        print(f"  {a}-{b}  weight={edge.weight:.3f}  [{tag}]")
        if ra != rb:
            parent[ra] = rb

    components: dict[str, list[str]] = {}
    for name in ALL_SPACES:
        root = _find(parent, name)
        components.setdefault(root, []).append(name)

    print(f"\n=== {len(components)} cụm liên thông tìm được ===")
    for root, members in components.items():
        expected_group = next((g for g in EXPECTED_GROUPS if set(members) <= g), None)
        status = "ĐÚNG (khớp đúng 1 building thật)" if expected_group and set(members) == expected_group else \
            ("ĐÚNG (subset đúng 1 building)" if expected_group else "!! SAI - lẫn giữa các building !!")
        print(f"  {sorted(members)}  -> {status}")

    # ---- xuất point cloud đã ghép cho mỗi cụm liên thông có >=2 không gian ----
    out_dir = Path("outputs/multi_space_full_graph")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Xuất point cloud đã ghép -> {out_dir} ===")
    for root, members in components.items():
        if len(members) < 2:
            continue
        # dựng cây khung từ các cạnh đã chấp nhận, đặt gốc = members[0]
        root_node = members[0]
        abs_pose = {root_node: (1.0, np.eye(3), np.zeros(3))}
        frontier = {root_node}
        remaining = set(members) - frontier
        while remaining:
            progressed = False
            for (a, b), edge in edges.items():
                if a in frontier and b in remaining:
                    abs_pose[b] = compose(abs_pose[a], (edge.s, edge.R, edge.t))
                    frontier.add(b)
                    remaining.discard(b)
                    progressed = True
                elif b in frontier and a in remaining:
                    from sgd_alignment.matching.multi_space_graph import invert
                    abs_pose[a] = compose(abs_pose[b], invert((edge.s, edge.R, edge.t)))
                    frontier.add(a)
                    remaining.discard(a)
                    progressed = True
            if not progressed:
                break

        all_pts, all_cols = [], []
        for name in members:
            ply, _ = ALL_SPACES[name]
            pts, cols = _read_ply(ply)
            if cols is None:
                cols = np.full((len(pts), 3), 160, np.uint8)
            s, R, t = abs_pose[name]
            aligned = transform_points(pts, s, R, t)
            all_pts.append(aligned)
            all_cols.append(cols)

        out_path = out_dir / f"{'_'.join(sorted(members))}.ply"
        _write_ply(np.concatenate(all_pts), np.concatenate(all_cols), str(out_path))
        print(f"  {sorted(members)} -> {out_path}")


if __name__ == "__main__":
    main()
