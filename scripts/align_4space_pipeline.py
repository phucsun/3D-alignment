"""Mở rộng Giai đoạn 1 (docs/multi_space_alignment_plan.md) sang 4 không
gian, KHÔNG giả định trước cấu trúc hub/star: chạy `align_gravity_camera`
cho MỌI cặp trong 4 không gian để tự phát hiện cạnh nào thực sự tồn tại
(status != NO_SOLUTION), rồi dựng 1 cây khung (spanning tree, ưu tiên cạnh
CONFIDENT/residual thấp) để đưa cả 4 không gian về chung 1 hệ toạ độ
(gốc = h_server).

4 không gian: server, room_310, h_server (hub cũ), connecting_space (mới).
Không biết trước connecting_space nối với không gian nào - để thuật toán tự
tìm qua toàn bộ cặp, đúng tinh thần "sparse portal-only edge discovery" của
kế hoạch tổng quát hoá N-không-gian.

Usage:
    python scripts/align_4space_pipeline.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points  # noqa: E402

_STATUS_RANK = {"CONFIDENT": 2, "AMBIGUOUS": 1, "NO_SOLUTION": 0}


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


def _compose(T2, T1):
    """T2 ∘ T1: áp T1 trước rồi T2 - (s,R,t) mỗi cái. Trả về Sim(3) tổng hợp."""
    s1, R1, t1 = T1
    s2, R2, t2 = T2
    return s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2


SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}
ROOT = "h_server"


def main() -> None:
    clusters = {}
    cams = {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
        print(f"{name}: {len(clusters[name])} opening cluster(s), up_consistency={cams[name].up_consistency:.4f}")

    # ---- bước 1: khám phá cạnh - chạy align_gravity_camera cho MỌI cặp ----
    print("\n=== Khám phá cạnh (mọi cặp trong 4 không gian) ===")
    edges: dict[tuple[str, str], object] = {}
    names = list(SPACES.keys())
    for a, b in combinations(names, 2):
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        print(f"  {a} -> {b}: status={result.status} residual={result.opening_residual:.4f} "
              f"matches={result.matches} reason={result.reason}")
        if result.status != "NO_SOLUTION":
            edges[(a, b)] = result

    # ---- bước 2: dựng cây khung (Prim, ưu tiên CONFIDENT rồi residual thấp) ----
    placed = {ROOT: (1.0, np.eye(3), np.zeros(3))}  # transform không gian -> hệ ROOT
    remaining = set(names) - {ROOT}
    print(f"\n=== Dựng cây khung từ gốc '{ROOT}' ===")
    while remaining:
        best = None  # (rank, -residual, edge_key, direction)
        for (a, b), result in edges.items():
            if a in placed and b in remaining:
                src, dst = a, b
            elif b in placed and a in remaining:
                src, dst = b, a
            else:
                continue
            key = (_STATUS_RANK[result.status], -result.opening_residual)
            if best is None or key > best[0]:
                best = (key, (a, b), result, src, dst)
        if best is None:
            print(f"  KHÔNG tìm được cạnh nối tới các không gian còn lại: {remaining} - bỏ qua (không đủ dữ liệu)")
            break
        _, (a, b), result, src, dst = best
        # result là align_gravity_camera(a, b) -> transform a vào hệ b (s,R,t áp lên a)
        if src == a:  # cạnh (a,b): a->b transform có sẵn, dst=b, src=a đã placed
            t_src_to_dst = (result.s, result.R, result.t)
        else:  # a là dst (chưa placed), b=src đã placed -> cần transform ngược: b vào hệ a là nghịch đảo a->b
            s_ab, R_ab, t_ab = result.s, result.R, result.t
            s_inv = 1.0 / s_ab
            R_inv = R_ab.T
            t_inv = -s_inv * (R_inv @ t_ab)
            t_src_to_dst = (s_inv, R_inv, t_inv)
        T_dst_to_root = _compose(placed[src], t_src_to_dst)
        placed[dst] = T_dst_to_root
        remaining.discard(dst)
        print(f"  + {dst} <- qua cạnh ({a},{b}) status={result.status} residual={result.opening_residual:.4f}")

    # ---- xuất kết quả ----
    out_dir = Path("outputs/final_aligned")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pts, all_cols = [], []
    for name, (s, R, t) in placed.items():
        ply, _ = SPACES[name]
        pts, cols = _read_ply(ply)
        if cols is None:
            cols = np.full((len(pts), 3), 160, np.uint8)
        aligned = transform_points(pts, s, R, t) if name != ROOT else pts
        all_pts.append(aligned)
        all_cols.append(cols)
        print(f"{name}: s={s:.4f} (transform -> {ROOT})")

    combined_out = out_dir / "space4_all_aligned.ply"
    _write_ply(np.concatenate(all_pts), np.concatenate(all_cols), str(combined_out))
    print(f"\nsaved (4 không gian, những cái đặt được) -> {combined_out}")
    if set(placed.keys()) != set(names):
        missing = set(names) - set(placed.keys())
        print(f"CẢNH BÁO: không đặt được: {missing} - không có cạnh CONFIDENT/AMBIGUOUS nối tới phần còn lại của đồ thị")


if __name__ == "__main__":
    main()
