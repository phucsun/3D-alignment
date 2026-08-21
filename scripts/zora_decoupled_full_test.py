"""Test Decoupled Pipeline (Giai đoạn 4-5, đã verify hoàn hảo N=4/N=6) trên
TOÀN BỘ dữ liệu CloudCompare/manual-segmentation thật đang có: 10 không
gian, 4 building khác nhau (server, Q1, Q2, chùa Thầy) - C(10,2)=45 cặp
ứng viên, đa số hoàn toàn không liên quan.

Usage:
    python scripts/zora_decoupled_full_test.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.matching.multi_space_graph import resolve_opening_conflict_graph  # noqa: E402
from sgd_alignment.zora.differentiable_sim3 import solve_pose_graph_gnc  # noqa: E402
from sgd_alignment.zora.vlm_topology import parse_topology_from_text  # noqa: E402

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
BUILDINGS = {
    "server": {"h_server", "server", "room_310", "connecting_space"},
    "q1": {"q1_indoor", "q1_outdoor"},
    "q2": {"q2_indoor", "q2_outdoor"},
    "chua_thay": {"chua_thay_indoor", "chua_thay_outdoor"},
}
REAL_EDGES = {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space"),
              ("q1_indoor", "q1_outdoor"), ("q2_indoor", "q2_outdoor"),
              ("chua_thay_indoor", "chua_thay_outdoor")}
DESCRIPTION = (
    "Có 4 toà nhà HOÀN TOÀN KHÔNG LIÊN QUAN đến nhau: "
    "(A) server: h_server (hành lang chung), server, room_310, connecting_space; "
    "(B) Q1: q1_indoor, q1_outdoor; (C) Q2: q2_indoor, q2_outdoor; "
    "(D) chùa Thầy: chua_thay_indoor, chua_thay_outdoor. "
    "KHÔNG có bất kỳ không gian nào của 1 toà nhà kề với không gian của toà nhà khác."
)


def _same_building(a: str, b: str) -> bool:
    return any({a, b} <= members for members in BUILDINGS.values())


def main() -> None:
    print("=== Load 10 không gian, 4 building ===")
    clusters, cams = {}, {}
    for name, (ply, npz) in ALL_SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
        print(f"  {name}: {len(clusters[name])} opening cluster(s)")
    names = list(ALL_SPACES)

    print("\n=== Layer 1: sinh soft graph từ mô tả text (mô phỏng LLM parse đúng) ===")
    mock = []
    for a, b in combinations(names, 2):
        if not _same_building(a, b):
            mock.append({"a": a, "b": b, "adjacent": False, "confidence": 0.95})
    soft_graph = parse_topology_from_text(DESCRIPTION, names, mock_response=mock)

    prior_weights = {}
    for a, b in combinations(names, 2):
        adjacent, conf = soft_graph.edge_confidence(a, b)
        if conf <= 0.0:
            prior_weights[(a, b)] = 1.0
        elif adjacent:
            prior_weights[(a, b)] = 1.0 + conf
        else:
            prior_weights[(a, b)] = 1.0 - conf * 0.999

    print(f"\n=== Bước 1: MWIS + Layer 1 prior (1 lần) - C({len(names)},2)="
          f"{len(names) * (len(names) - 1) // 2} cặp ứng viên ===")
    mwis_edges = resolve_opening_conflict_graph(clusters, cams, prior_weights=prior_weights)
    for e, edge in mwis_edges.items():
        cross = "!!! XUYÊN-BUILDING !!!" if not _same_building(*e) else "trong-building"
        print(f"  {e[0]}-{e[1]} [{cross}]: weight={edge.weight:.3f}")

    print("\n=== Bước 2: GNC-TLS trên tập cạnh MWIS đã chốt ===")
    edges = {e: (c.s, c.R, c.t) for e, c in mwis_edges.items()}
    prior_w2 = {e: max(c.weight, 1e-3) for e, c in mwis_edges.items()}
    gnc = solve_pose_graph_gnc(edges, names, root="h_server", edge_weights=prior_w2,
                                c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)

    correct = wrong = 0
    for e, w in sorted(gnc.gnc_weights.items(), key=lambda kv: -kv[1]):
        tag = "THẬT" if e in REAL_EDGES else "GIẢ"
        decision = "GIỮ" if w > 0.5 else "LOẠI"
        print(f"  {e[0]}-{e[1]} [{tag}]: gnc_w={w:.4f} lệch góc={gnc.edge_residuals[e]['angle_deg']:.2f}° -> {decision}")
        if tag == "THẬT" and w > 0.5:
            correct += 1
        elif tag == "GIẢ" and w > 0.5:
            wrong += 1
    missed = REAL_EDGES - {e for e, w in gnc.gnc_weights.items() if w > 0.5}
    print(f"\n=> Giữ đúng {correct}/{len(REAL_EDGES)} cạnh thật, giữ NHẦM {wrong} cạnh giả, "
          f"bỏ sót {len(missed)} {missed or ''}")


if __name__ == "__main__":
    main()
