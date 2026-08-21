"""Giai đoạn 4, Hướng 1 CUỐI CÙNG - Decoupled Pipeline (theo đề xuất trực
tiếp của người dùng, 2026-08-21):

- **Bộ não Logic (discrete)**: `resolve_opening_conflict_graph` (Giai đoạn
  1, KHÔNG đổi, dùng `n_matches/residual`) chạy 1 LẦN DUY NHẤT lúc đầu
  (initialization) - giải quyết TRIỆT ĐỂ xung đột "dùng chung cửa" (loại
  xung đột duy nhất mà MWIS xử lý được).
- **Bộ máy Tinh chỉnh (continuous)**: tập cạnh ĐÃ SẠCH xung đột cửa được
  đẩy thẳng vào `solve_pose_graph_gnc` (Giai đoạn 3, KHÔNG đổi) - tối ưu
  R/t + loại nốt nhiễu hình học toàn cục (bao gồm trùng hợp xuyên-building,
  loại xung đột MWIS KHÔNG xử lý được vì không dùng chung cửa nào).

KHÔNG luân phiên - MWIS chạy đúng 1 lần, GNC là bộ tiêu thụ 1 chiều phía
sau, không quay lại gọi MWIS nữa (lý do: đã xác nhận GNC-weight dùng làm
tiêu chí MWIS cho ra quyết định TỆ HƠN n_matches/residual ở 1 số trường
hợp cạnh tranh khó - xem docs/multi_space_alignment_plan.md).

Test cả N=4 (verify không regression) và N=6 (verify sửa được lỗi xuyên-
building).

Usage:
    python scripts/zora_decoupled_pipeline_verify.py [n4|n6|both]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.matching.multi_space_graph import resolve_opening_conflict_graph  # noqa: E402
from sgd_alignment.zora.differentiable_sim3 import solve_pose_graph_gnc  # noqa: E402
from sgd_alignment.zora.vlm_topology import parse_topology_from_text, combine_geometric_and_vlm_prior  # noqa: E402

N4_SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}
N4_REAL = {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space")}

N6_SPACES = dict(N4_SPACES)
N6_SPACES.update({
    "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
    "q1_outdoor": ("data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
})
N6_REAL = N4_REAL | {("q1_indoor", "q1_outdoor")}
SERVER_BUILDING = {"h_server", "server", "room_310", "connecting_space"}
Q1_BUILDING = {"q1_indoor", "q1_outdoor"}
DESCRIPTION = (
    "Có 2 toà nhà HOÀN TOÀN KHÔNG LIÊN QUAN đến nhau: toà nhà A gồm h_server "
    "(hành lang chung), server, room_310, connecting_space. Toà nhà B (Q1) gồm "
    "q1_indoor và q1_outdoor - toà nhà B KHÔNG hề kề với bất kỳ không gian nào của toà nhà A."
)


def _load(spaces):
    clusters, cams = {}, {}
    for name, (ply, npz) in spaces.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    return clusters, cams


def run(label: str, spaces: dict, real_edges: set, root: str = "h_server", use_prior: bool = False) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    clusters, cams = _load(spaces)
    names = list(spaces)

    prior_weights = None
    if use_prior:
        mock = []
        from itertools import combinations
        for a, b in combinations(names, 2):
            same = ({a, b} <= SERVER_BUILDING) or ({a, b} <= Q1_BUILDING)
            if not same:
                mock.append({"a": a, "b": b, "adjacent": False, "confidence": 0.95})
        soft_graph = parse_topology_from_text(DESCRIPTION, names, mock_response=mock)
        # Ở đây cần HỆ SỐ NHÂN (không phải trọng số tuyệt đối như
        # `combine_geometric_and_vlm_prior` trả về cho GNC) - tự tính trực
        # tiếp từ `soft_graph.edge_confidence` cho đúng ngữ cảnh MWIS.
        prior_weights = {}
        for a, b in combinations(names, 2):
            adjacent, conf = soft_graph.edge_confidence(a, b)
            if conf <= 0.0:
                prior_weights[(a, b)] = 1.0
            elif adjacent:
                prior_weights[(a, b)] = 1.0 + conf
            else:
                prior_weights[(a, b)] = 1.0 - conf * (1.0 - 0.001)

    print(f"--- Bước 1: MWIS (Giai đoạn 1, 1 lần{'​ + Layer 1 prior' if use_prior else ''}) ---")
    mwis_edges = resolve_opening_conflict_graph(clusters, cams, prior_weights=prior_weights)
    for e, edge in mwis_edges.items():
        tag = "THẬT" if e in real_edges else "GIẢ"
        print(f"  {e[0]}-{e[1]} [{tag}]: dùng {e[0]}{sorted(i for _, i in edge.used_a)} <-> "
              f"{e[1]}{sorted(j for _, j in edge.used_b)}  weight={edge.weight:.3f}")

    print("\n--- Bước 2: GNC-TLS (Giai đoạn 3, không đổi) trên tập cạnh MWIS đã chốt ---")
    edges = {e: (c.s, c.R, c.t) for e, c in mwis_edges.items()}
    prior_w = {e: max(c.weight, 1e-3) for e, c in mwis_edges.items()}
    gnc = solve_pose_graph_gnc(edges, names, root=root, edge_weights=prior_w,
                                c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)

    correct = wrong = 0
    for e, w in sorted(gnc.gnc_weights.items(), key=lambda kv: -kv[1]):
        tag = "THẬT" if e in real_edges else "GIẢ"
        decision = "GIỮ" if w > 0.5 else "LOẠI"
        print(f"  {e[0]}-{e[1]} [{tag}]: gnc_w={w:.4f} lệch góc={gnc.edge_residuals[e]['angle_deg']:.2f}° -> {decision}")
        if tag == "THẬT" and w > 0.5:
            correct += 1
        elif tag == "GIẢ" and w > 0.5:
            wrong += 1
    missed = real_edges - {e for e, w in gnc.gnc_weights.items() if w > 0.5}
    print(f"\n=> Giữ đúng {correct}/{len(real_edges)} cạnh thật, giữ NHẦM {wrong} cạnh giả, "
          f"bỏ sót {len(missed)} {missed or ''}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("n4", "both"):
        run("N=4 (server building) - verify không regression", N4_SPACES, N4_REAL)
    if mode in ("n6", "both"):
        run("N=6 (server+Q1, xuyên-building) - KHÔNG Layer 1", N6_SPACES, N6_REAL, use_prior=False)
        run("N=6 (server+Q1, xuyên-building) - CÓ Layer 1 (prior đưa vào ngay bước MWIS)",
            N6_SPACES, N6_REAL, use_prior=True)
