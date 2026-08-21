"""Giai đoạn 3 (docs/zora_design_plan.md) - VERIFY GNC-TLS trên đúng bộ
4-không-gian đã biết kết quả tốt nhất (numpy soft_l1: 3/3 cạnh thật đúng,
1/3 cạnh giả còn sót do giới hạn thông tin thật - room_310-connecting_space).

Kỳ vọng: GNC weight ~1.0 cho 3 cạnh thật, ~0.0 cho 2 cạnh giả rõ ràng
(server-room_310, server-connecting_space) - không kỳ vọng giải quyết được
ca khó (room_310-connecting_space), đó là giới hạn dữ liệu không phải
thuật toán.

Usage:
    python scripts/zora_gnc_verify.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.zora.differentiable_sim3 import solve_pose_graph_gnc  # noqa: E402

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

    edges, edge_weights = {}, {}
    for a, b in combinations(names, 2):
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        edges[(a, b)] = (result.s, result.R, result.t)
        edge_weights[(a, b)] = len(result.matches) / (result.opening_residual + 1e-3)

    print("=== Chạy GNC-TLS (c_deg=5.0, mu_factor=1.4, n_outer=15) ===")
    gnc = solve_pose_graph_gnc(edges, names, root="h_server", edge_weights=edge_weights,
                                c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)

    print(f"mu qua các vòng: {[round(m, 4) for m in gnc.mu_history]}")
    print("\nKết quả cuối:")
    correct = wrong_kept = wrong_dropped = 0
    for e, w in sorted(gnc.gnc_weights.items(), key=lambda kv: -kv[1]):
        tag = "THẬT" if e in GROUND_TRUTH_REAL_EDGES else "GIẢ"
        angle = gnc.edge_residuals[e]["angle_deg"]
        decision = "GIỮ" if w > 0.5 else "LOẠI"
        print(f"  {e[0]}-{e[1]} [{tag}]: GNC weight={w:.4f}  lệch góc={angle:.2f}°  -> {decision}")
        if tag == "THẬT" and w > 0.5:
            correct += 1
        elif tag == "THẬT" and w <= 0.5:
            wrong_dropped += 1
        elif tag == "GIẢ" and w > 0.5:
            wrong_kept += 1

    print(f"\n=> Giữ đúng {correct}/3 cạnh thật, giữ NHẦM {wrong_kept} cạnh giả, "
          f"loại NHẦM {wrong_dropped} cạnh thật")


if __name__ == "__main__":
    main()
