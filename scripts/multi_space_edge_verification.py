"""Giai đoạn 3 (docs/multi_space_alignment_plan.md), hướng MỚI: kết hợp 3
tín hiệu ĐỘC LẬP để quyết định 1 cạnh (candidate edge) là thật/giả, không
cần biết trước hub, không cần thêm dư thừa đồ thị:

1. Sim(3) pose-graph residual (`multi_space_graph.solve_pose_graph`).
2. p-value thống kê so với null distribution xây từ các cặp KHÔNG GIAN
   THẬT SỰ không liên quan (Q1/Q2/chua_thay - khác toà nhà hoàn toàn).
3. Va chạm thể tích (`edge_verification.voxel_occupancy_overlap`) trên
   point cloud ĐẦY ĐỦ (không chỉ opening).

Test trên đúng bộ 4 không gian đã biết ground-truth (server, room_310,
connecting_space, h_server) - xem 3 tín hiệu kết hợp có loại được nốt cạnh
giả còn sót (room_310-connecting_space) mà chỉ riêng Sim(3) pose-graph
không loại được hay không.

Usage:
    python scripts/multi_space_edge_verification.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points  # noqa: E402
from sgd_alignment.matching.multi_space_graph import solve_pose_graph  # noqa: E402
from sgd_alignment.matching.edge_verification import p_value_from_null, voxel_occupancy_overlap  # noqa: E402

SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}
GROUND_TRUTH_REAL_EDGES = {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space")}

# Không gian THẬT SỰ không liên quan (toà nhà khác hoàn toàn) - dùng để xây
# phân phối null. Không cần is_outdoor đúng tuyệt đối cho mục đích null
# (chỉ cần "không liên quan" là đủ, không cần chính xác vật lý).
UNRELATED_SPACES = {
    "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
    "q1_outdoor": ("data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
    "q2_indoor": ("data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply", "data/Q2/Q2_indoor/results.npz"),
    "q2_outdoor": ("data/Q2/Q2_outdoor/Q2_outdoor_points - Cloud.ply", "data/Q2/Q2_outdoor/results.npz"),
    "chua_thay_indoor": ("data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply",
                          "data/chua_thay/indoor/results.npz"),
    "chua_thay_outdoor": ("data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply",
                           "data/chua_thay/outdoor/results.npz"),
}


def _read_ply(path: str):
    from plyfile import PlyData

    v = PlyData.read(path)["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def main() -> None:
    clusters, cams = {}, {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    names = list(SPACES.keys())

    print("=== Tín hiệu 1: đo transform từng cặp trong 4 không gian ===")
    edges = {}
    edge_weights = {}
    for a, b in combinations(names, 2):
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        edges[(a, b)] = (result.s, result.R, result.t)
        edge_weights[(a, b)] = len(result.matches) / (result.opening_residual + 1e-3)
        print(f"  {a}-{b}: residual={result.opening_residual:.4f} status={result.status}")

    # Cấu hình đã xác nhận TỐT NHẤT (docs/multi_space_alignment_plan.md): KHÔNG
    # chuẩn hoá đơn vị (angle_scale_rad/trans_scale/log_scale_scale = 1.0, tương
    # đương tắt) - lần trước quên truyền, vô tình dùng lại cấu hình mặc định đã
    # xác nhận TỆ HƠN.
    pg = solve_pose_graph(edges, names, root="h_server", edge_weights=edge_weights,
                           robust_loss="soft_l1", f_scale=0.1,
                           angle_scale_rad=1.0, trans_scale=1.0, log_scale_scale=1.0)

    print("\n=== Tín hiệu 2: xây phân phối null từ không gian KHÔNG liên quan (Q1/Q2/chua_thay) ===")
    unrelated_clusters, unrelated_cams = {}, {}
    for name, (ply, npz) in UNRELATED_SPACES.items():
        unrelated_clusters[name] = openings_from_manual_segmentation(ply)
        unrelated_cams[name] = camera_evidence(npz)

    # SỬA quan trọng: KHÔNG bỏ qua NO_SOLUTION nữa - lần chạy trước chỉ giữ lại
    # các cặp "tình cờ trông ổn nhất" (status != NO_SOLUTION), gây selection
    # bias đúng kiểu Selective Inference cảnh báo (p-value tính trên tập đã lọc
    # sẵn sẽ sai). NO_SOLUTION = thất bại hoàn toàn, gán 1 giá trị SENTINEL lớn
    # (1.0, lớn hơn hẳn mọi residual thật quan sát được <0.1) để phản ánh đúng
    # "hầu hết cặp không liên quan thất bại hoàn toàn", không phải bỏ qua.
    NO_SOLUTION_SENTINEL = 1.0
    null_residuals = []
    n_no_solution = 0
    for target in names:
        for u_name in UNRELATED_SPACES:
            result = align_gravity_camera(clusters[target], unrelated_clusters[u_name], cams[target], unrelated_cams[u_name])
            if result.status == "NO_SOLUTION":
                null_residuals.append(NO_SOLUTION_SENTINEL)
                n_no_solution += 1
            else:
                null_residuals.append(result.opening_residual)
    print(f"  {len(null_residuals)} mẫu null ({n_no_solution} là NO_SOLUTION, gán sentinel={NO_SOLUTION_SENTINEL}): "
          f"min={min(null_residuals):.4f} max={max(null_residuals):.4f} mean={np.mean(null_residuals):.4f}")

    print("\n=== Tín hiệu 3: va chạm thể tích (point cloud đầy đủ) ===")
    full_points = {name: _read_ply(ply) for name, (ply, _) in SPACES.items()}

    print("\n=== Kết hợp cả 3 tín hiệu cho từng cạnh ===")
    PG_ANGLE_THRESHOLD_DEG = 5.0
    PVALUE_THRESHOLD = 0.05
    OVERLAP_THRESHOLD = 0.05
    for (a, b), (s, R, t) in edges.items():
        tag = "THẬT" if (a, b) in GROUND_TRUTH_REAL_EDGES else "GIẢ"
        pg_angle = pg.edge_residuals[(a, b)]["angle_deg"]
        pg_ok = pg_angle <= PG_ANGLE_THRESHOLD_DEG

        # residual gốc (đo trực tiếp, không phải sau pose-graph) để so với null
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        pval = p_value_from_null(result.opening_residual, null_residuals)
        pval_ok = pval <= PVALUE_THRESHOLD

        pts_a_in_b = transform_points(full_points[a], s, R, t)
        overlap = voxel_occupancy_overlap(pts_a_in_b, full_points[b], voxel_size=0.15)
        overlap_ok = overlap <= OVERLAP_THRESHOLD

        n_pass = sum([pg_ok, pval_ok, overlap_ok])
        decision = "GIỮ" if n_pass == 3 else "LOẠI"
        print(f"  {a}-{b} [{tag}]: pose-graph={pg_angle:.2f}°({'OK' if pg_ok else 'x'})  "
              f"p-value={pval:.4f}({'OK' if pval_ok else 'x'})  "
              f"overlap-thể-tích={overlap:.4f}({'OK' if overlap_ok else 'x'})  "
              f"-> {n_pass}/3 tín hiệu đồng thuận -> {decision}")

    # In riêng bảng SO SÁNH TƯƠNG ĐỐI (không áp ngưỡng tuỳ ý) để đánh giá xem
    # từng tín hiệu có tách biệt thật/giả ở mức độ NÀO, kể cả khi ngưỡng tuyệt
    # đối đã chọn chưa đúng.
    print("\n=== So sánh tương đối (không áp ngưỡng) - để hiệu chỉnh ngưỡng đúng sau này ===")
    for (a, b) in edges:
        tag = "THẬT" if (a, b) in GROUND_TRUTH_REAL_EDGES else "GIẢ"
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        pval = p_value_from_null(result.opening_residual, null_residuals)
        s, R, t = edges[(a, b)]
        overlap = voxel_occupancy_overlap(transform_points(full_points[a], s, R, t), full_points[b], voxel_size=0.15)
        print(f"  {a}-{b} [{tag}]: pose-graph={pg.edge_residuals[(a, b)]['angle_deg']:.2f}°  "
              f"p-value={pval:.4f}  overlap={overlap:.4f}")


if __name__ == "__main__":
    main()
