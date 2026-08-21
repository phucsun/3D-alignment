"""Giai đoạn 3 (docs/multi_space_alignment_plan.md), thử nghiệm THẬT:
tối ưu Sim(3) pose-graph robust (`multi_space_graph.solve_pose_graph`)
trên 4 không gian - KHÔNG hardcode hub, KHÔNG đếm tam giác rời rạc (cách
`multi_space_cycle_consistency.py` đã thất bại) - xem robust loss có tự
tách được 3 cạnh thật (vào h_server) khỏi 3 cạnh giả (server-room_310,
server-connecting_space, room_310-connecting_space) hay không, chỉ dựa
trên residual sau khi tối ưu TOÀN CỤC.

Usage:
    python scripts/multi_space_robust_pose_graph.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.matching.multi_space_graph import solve_pose_graph  # noqa: E402

SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}
GROUND_TRUTH_REAL_EDGES = {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space")}


def main() -> None:
    clusters, cams = {}, {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)

    names = list(SPACES.keys())
    print("=== Đo transform từng cặp (align_gravity_camera, 1 lần/cặp) ===")
    edges = {}
    edge_weights = {}
    for a, b in combinations(names, 2):
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        edges[(a, b)] = (result.s, result.R, result.t)
        # ĐÃ THỬ dùng cam_dominance/grav_dot/up_consistency làm trọng số -
        # THẤT BẠI, xác nhận bằng số liệu thật: đây là các ngưỡng GATE NHỊ
        # PHÂN bên trong align_gravity_camera (CONFIDENT/AMBIGUOUS nghĩa là
        # đã pass, nên luôn ~0.97-1.0 cho MỌI cạnh kể cả cạnh giả) - gần như
        # không phân biệt được gì, làm pose-graph tệ hơn hẳn bản gốc. Quay
        # lại n_matches/opening_residual (có phân giải thật, dao động rõ
        # rệt 34.6-78.7 giữa các cạnh) - xem docs/multi_space_alignment_plan.md.
        edge_weights[(a, b)] = len(result.matches) / (result.opening_residual + 1e-3)
        tag = "THẬT" if (a, b) in GROUND_TRUTH_REAL_EDGES else "GIẢ (theo ground-truth đã biết)"
        print(f"  {a} -> {b}: status={result.status} residual={result.opening_residual:.4f} "
              f"n_matches={len(result.matches)} weight={edge_weights[(a, b)]:.1f}  [{tag}]")

    for root in ["h_server"]:  # h_server làm gốc chỉ để có hệ quy chiếu chung khi in, KHÔNG dùng làm "biết trước hub"
        # ĐÃ THỬ chuẩn hoá đơn vị (2 độ/5cm/5%, f_scale=1.0) - THẤT BẠI, tệ hơn
        # bản KHÔNG chuẩn hoá (raw units, f_scale=0.1) dù lý thuyết nghe hợp
        # lý hơn. Xác nhận bằng số liệu thật 2 lần chạy khác nhau - quay lại
        # đúng cấu hình gốc đã cho kết quả tốt nhất tìm được (2/3 tách rõ):
        # KHÔNG chuẩn hoá (scale=1.0 cho cả 3 tham số = tương đương tắt).
        print(f"\n=== Giải Sim(3) pose-graph robust (root hiển thị = {root}, robust_loss=soft_l1, "
              "f_scale=0.1, KHÔNG chuẩn hoá đơn vị - cấu hình gốc tốt nhất đã xác nhận) ===")
        pg = solve_pose_graph(edges, names, root=root, edge_weights=edge_weights, robust_loss="soft_l1",
                               f_scale=0.1, angle_scale_rad=1.0, trans_scale=1.0, log_scale_scale=1.0)
        print(f"success={pg.success}")

        print("\nResidual từng cạnh SAU KHI tối ưu toàn cục (không phải đo trực tiếp - đây là độ lệch "
              "giữa transform đo được và transform mà nghiệm toàn cục dự đoán):")
        rows = sorted(pg.edge_residuals.items(), key=lambda kv: kv[1]["angle_deg"])
        for (a, b), err in rows:
            tag = "THẬT" if (a, b) in GROUND_TRUTH_REAL_EDGES else "GIẢ"
            print(f"  {a}-{b} [{tag}]: lệch góc={err['angle_deg']:.2f}° lệch tịnh tiến={err['trans_norm']:.4f} "
                  f"lệch scale(log)={err['scale_dev']:.4f}")

        # đánh giá: cạnh THẬT có residual thấp hơn cạnh GIẢ một cách rõ ràng không?
        real_angles = [pg.edge_residuals[e]["angle_deg"] for e in GROUND_TRUTH_REAL_EDGES]
        fake_angles = [err["angle_deg"] for (a, b), err in pg.edge_residuals.items()
                       if (a, b) not in GROUND_TRUTH_REAL_EDGES]
        print(f"\nTrung bình lệch góc cạnh THẬT: {np.mean(real_angles):.2f}°  "
              f"vs cạnh GIẢ: {np.mean(fake_angles):.2f}°")
        if max(real_angles) < min(fake_angles):
            print("=> TÁCH ĐƯỢC rõ ràng: mọi cạnh thật đều có residual thấp hơn MỌI cạnh giả.")
        else:
            print("=> KHÔNG tách được rõ ràng (còn chồng lấn giữa cạnh thật/giả).")

        # PCM đơn giản (không lặp IRLS - đã xác nhận IRLS làm TỆ HƠN ở quy mô
        # N=4/6 cạnh, xem docs/multi_space_alignment_plan.md): giữ cạnh có
        # residual (so với nghiệm 1-lần ở trên) dưới ngưỡng.
        PCM_THRESHOLD_DEG = 5.0
        kept = {e for e, err in pg.edge_residuals.items() if err["angle_deg"] <= PCM_THRESHOLD_DEG}
        correct_real = GROUND_TRUTH_REAL_EDGES & kept
        wrong_fake_kept = kept - GROUND_TRUTH_REAL_EDGES
        missed_real = GROUND_TRUTH_REAL_EDGES - kept
        print(f"\n=== PCM (ngưỡng {PCM_THRESHOLD_DEG}°, trên nghiệm 1-lần, KHÔNG IRLS) ===")
        print(f"Giữ đúng {len(correct_real)}/3 cạnh thật, giữ NHẦM {len(wrong_fake_kept)} cạnh giả "
              f"{wrong_fake_kept or ''}, bỏ sót {len(missed_real)} cạnh thật {missed_real or ''}")


if __name__ == "__main__":
    main()
