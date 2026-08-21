"""Xuất point cloud đã ghép (sau khi chạy Decoupled Pipeline: MWIS+prior ->
GNC-TLS, Giai đoạn 4-5 đã verify hoàn hảo) ra 1 folder để kiểm tra bằng mắt.

5 kịch bản: Q1, Q2, chùa Thầy (N=2 mỗi cái), server building (N=4), và
server+Q1 xuyên-building (N=6, có Layer 1 prior).

Xuất ra outputs/zora_decoupled/.

Usage:
    python scripts/zora_export_aligned.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points  # noqa: E402
from sgd_alignment.matching.multi_space_graph import resolve_opening_conflict_graph  # noqa: E402
from sgd_alignment.zora.differentiable_sim3 import solve_pose_graph_gnc  # noqa: E402
from sgd_alignment.zora.vlm_topology import parse_topology_from_text  # noqa: E402

Q1 = {
    "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
    "q1_outdoor": ("data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
}
Q2 = {
    "q2_indoor": ("data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply", "data/Q2/Q2_indoor/results.npz"),
    "q2_outdoor": ("data/Q2/Q2_outdoor/Q2_outdoor_points - Cloud.ply", "data/Q2/Q2_outdoor/results.npz"),
}
CHUA_THAY = {
    "chua_thay_indoor": ("data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply",
                          "data/chua_thay/indoor/results.npz"),
    "chua_thay_outdoor": ("data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply",
                           "data/chua_thay/outdoor/results.npz"),
}
SERVER_N4 = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}
SERVER_Q1_N6 = dict(SERVER_N4, **Q1)
SERVER_BUILDING = set(SERVER_N4)
Q1_BUILDING = set(Q1)
N6_DESCRIPTION = (
    "Có 2 toà nhà HOÀN TOÀN KHÔNG LIÊN QUAN đến nhau: toà nhà A gồm h_server "
    "(hành lang chung), server, room_310, connecting_space. Toà nhà B (Q1) gồm "
    "q1_indoor và q1_outdoor - toà nhà B KHÔNG hề kề với bất kỳ không gian nào của toà nhà A."
)


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


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def run(label: str, spaces: dict, out_dir: Path, prefix: str, prior_weights=None) -> None:
    print(f"\n=== {label} ===")
    clusters, cams = {}, {}
    for name, (ply, npz) in spaces.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    names = list(spaces)
    root = names[0]

    mwis_edges = resolve_opening_conflict_graph(clusters, cams, prior_weights=prior_weights)
    edges = {e: (c.s, c.R, c.t) for e, c in mwis_edges.items()}
    prior_w = {e: max(c.weight, 1e-3) for e, c in mwis_edges.items()}
    gnc = solve_pose_graph_gnc(edges, names, root=root, edge_weights=prior_w,
                                c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)

    # gom theo CỤM LIÊN THÔNG (union-find trên cạnh đã chấp nhận) - không ép các
    # cụm KHÔNG kề nhau vào chung 1 file (vị trí tương đối giữa chúng không có
    # ý nghĩa vật lý gì, chỉ là tham số tự do của solver).
    parent = {n: n for n in names}
    for e, w in gnc.gnc_weights.items():
        if w > 0.5:
            a, b = e
            ra, rb = _find(parent, a), _find(parent, b)
            if ra != rb:
                parent[ra] = rb
    components: dict[str, list[str]] = {}
    for n in names:
        components.setdefault(_find(parent, n), []).append(n)

    for comp_root, members in components.items():
        if len(members) < 2:
            print(f"  Cụm cô lập (chỉ 1 không gian, không có cạnh nào): {members} - bỏ qua")
            continue
        print(f"  Cụm liên thông: {sorted(members)}")
        # đặt gốc = comp_root (transform tuyệt đối do GNC đã tính so với `root`
        # gốc chung - vẫn dùng được trực tiếp vì mọi node trong CÙNG cụm đã
        # được ghép về 1 hệ toạ độ nhất quán qua các cạnh đã chấp nhận)
        all_pts, all_cols = [], []
        for name in members:
            ply, _ = spaces[name]
            pts, cols = _read_ply(ply)
            if cols is None:
                cols = np.full((len(pts), 3), 160, np.uint8)
            s, R, t = gnc.poses[name]
            s = float(s)
            R = R.numpy() if hasattr(R, "numpy") else R
            t = t.numpy() if hasattr(t, "numpy") else t
            aligned = transform_points(pts, s, R, t) if name != root else pts
            all_pts.append(aligned)
            all_cols.append(cols)

        out_path = out_dir / f"{prefix}_{'_'.join(sorted(members))}.ply"
        _write_ply(np.concatenate(all_pts), np.concatenate(all_cols), str(out_path))
        print(f"  saved -> {out_path}")


def main() -> None:
    out_dir = Path("outputs/zora_decoupled")
    out_dir.mkdir(parents=True, exist_ok=True)

    run("Q1", Q1, out_dir, "q1")
    run("Q2", Q2, out_dir, "q2")
    run("Chùa Thầy", CHUA_THAY, out_dir, "chua_thay")
    run("Server building (N=4)", SERVER_N4, out_dir, "server_n4")

    print("\n=== Server + Q1 (N=6, xuyên-building, có Layer 1 prior) ===")
    names = list(SERVER_Q1_N6)
    mock = []
    for a, b in combinations(names, 2):
        same = ({a, b} <= SERVER_BUILDING) or ({a, b} <= Q1_BUILDING)
        if not same:
            mock.append({"a": a, "b": b, "adjacent": False, "confidence": 0.95})
    soft_graph = parse_topology_from_text(N6_DESCRIPTION, names, mock_response=mock)
    prior_weights = {}
    for a, b in combinations(names, 2):
        adjacent, conf = soft_graph.edge_confidence(a, b)
        if conf <= 0.0:
            prior_weights[(a, b)] = 1.0
        elif adjacent:
            prior_weights[(a, b)] = 1.0 + conf
        else:
            prior_weights[(a, b)] = 1.0 - conf * 0.999
    run("Server+Q1 (N=6)", SERVER_Q1_N6, out_dir, "server_q1_n6", prior_weights=prior_weights)


if __name__ == "__main__":
    main()
