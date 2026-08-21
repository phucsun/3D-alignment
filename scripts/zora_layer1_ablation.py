"""Giai đoạn 4 (docs/zora_design_plan.md) - Ablation chứng minh giá trị
thật của Layer 1 (VLM/text prior) so với Giai đoạn 3 (GNC thuần hình học):

Ghép building `server` (4 không gian) + building `Q1` (2 không gian) thành
1 đồ thị 6 node, C(6,2)=15 cặp. Đã xác nhận thật ở phiên trước (spot-check
xuyên-building): `server<->q1_indoor` cho CONFIDENT, residual=0.0212, 2
match - hình học thuần (kể cả GNC) không có cơ chế nào phân biệt cặp này
với 1 cạnh thật, vì 2 building không dùng chung cửa nào để tạo xung đột.

So sánh:
(a) GNC-TLS thuần hình học (Giai đoạn 3, không Layer 1) - kỳ vọng CÓ THỂ
    giữ nhầm cạnh xuyên-building.
(b) GNC-TLS + prior Layer 1 (mô phỏng: mô tả text nói rõ 2 building tách
    biệt, KHÔNG kề nhau) - kỳ vọng loại đúng cạnh xuyên-building.

Usage:
    python scripts/zora_layer1_ablation.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.zora.differentiable_sim3 import solve_pose_graph_gnc  # noqa: E402
from sgd_alignment.zora.vlm_topology import (  # noqa: E402
    parse_topology_from_text, combine_geometric_and_vlm_prior,
)

SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
    "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
    "q1_outdoor": ("data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
}
SERVER_BUILDING = {"h_server", "server", "room_310", "connecting_space"}
Q1_BUILDING = {"q1_indoor", "q1_outdoor"}
GROUND_TRUTH_REAL_EDGES = {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space"),
                           ("q1_indoor", "q1_outdoor")}

DESCRIPTION = (
    "Có 2 toà nhà HOÀN TOÀN KHÔNG LIÊN QUAN đến nhau: toà nhà A gồm h_server "
    "(hành lang chung), server, room_310, connecting_space (server và room_310 "
    "đều mở cửa vào h_server, connecting_space nối vào cửa cuối hành lang). "
    "Toà nhà B (Q1) gồm q1_indoor và q1_outdoor, kề nhau qua 1 cửa/cửa sổ - "
    "toà nhà B KHÔNG hề kề với bất kỳ không gian nào của toà nhà A."
)


def _mock_llm_response(names: list[str]) -> list[dict]:
    """Mô phỏng đầu ra 1 LLM parse ĐÚNG `DESCRIPTION` ở trên sẽ trả về -
    chưa gọi API thật (quyết định 2026-08-21)."""
    out = []
    for a, b in combinations(names, 2):
        same_building = ({a, b} <= SERVER_BUILDING) or ({a, b} <= Q1_BUILDING)
        if not same_building:
            out.append({"a": a, "b": b, "adjacent": False, "confidence": 0.95})
        # cùng building: không nói cụ thể cặp nào kề cặp nào -> để hình học tự quyết (confidence=0,
        # không thêm vào list - `edge_confidence` mặc định trả (True, 0.0) khi không có trong graph)
    return out


def main() -> None:
    clusters, cams = {}, {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    names = list(SPACES.keys())

    print("=== Đo transform mọi cặp (6 không gian, 2 building) ===")
    edges, geo_weights = {}, {}
    for a, b in combinations(names, 2):
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        if result.status == "NO_SOLUTION":
            continue
        edges[(a, b)] = (result.s, result.R, result.t)
        geo_weights[(a, b)] = len(result.matches) / (result.opening_residual + 1e-3)
        cross = "XUYÊN-BUILDING" if not (({a, b} <= SERVER_BUILDING) or ({a, b} <= Q1_BUILDING)) else "cùng building"
        print(f"  {a}-{b} [{cross}]: status={result.status} residual={result.opening_residual:.4f} "
              f"n_matches={len(result.matches)}")

    def report(label, gnc):
        print(f"\n--- {label} ---")
        wrong_kept_cross_building = []
        for e, w in sorted(gnc.gnc_weights.items(), key=lambda kv: -kv[1]):
            tag = "THẬT" if e in GROUND_TRUTH_REAL_EDGES else \
                ("XUYÊN-BUILDING" if not (({e[0], e[1]} <= SERVER_BUILDING) or ({e[0], e[1]} <= Q1_BUILDING))
                 else "GIẢ cùng-building")
            decision = "GIỮ" if w > 0.5 else "LOẠI"
            print(f"  {e[0]}-{e[1]} [{tag}]: weight={w:.4f} lệch góc={gnc.edge_residuals[e]['angle_deg']:.2f}° -> {decision}")
            if tag == "XUYÊN-BUILDING" and w > 0.5:
                wrong_kept_cross_building.append(e)
        if wrong_kept_cross_building:
            print(f"  !!! GIỮ NHẦM {len(wrong_kept_cross_building)} cạnh xuyên-building: {wrong_kept_cross_building}")
        else:
            print("  Không giữ nhầm cạnh xuyên-building nào.")

    print("\n=== (a) GNC-TLS THUẦN HÌNH HỌC (không Layer 1) ===")
    gnc_no_prior = solve_pose_graph_gnc(edges, names, root="h_server", edge_weights=geo_weights,
                                         c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)
    report("(a) Không Layer 1", gnc_no_prior)

    print("\n=== (b) GNC-TLS + Layer 1 (prior text mô phỏng) ===")
    soft_graph = parse_topology_from_text(DESCRIPTION, names, mock_response=_mock_llm_response(names))
    combined_weights = combine_geometric_and_vlm_prior(geo_weights, soft_graph, contradicted_penalty=0.001)
    gnc_with_prior = solve_pose_graph_gnc(edges, names, root="h_server", edge_weights=combined_weights,
                                           c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)
    report("(b) Có Layer 1", gnc_with_prior)


if __name__ == "__main__":
    main()
