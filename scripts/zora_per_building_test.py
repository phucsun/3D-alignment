"""Test Decoupled Pipeline (MWIS -> GNC, Giai đoạn 4-5 đã verify) RIÊNG
từng building (không trộn lẫn) - Q1 và server building đã verify trước đó;
đây là Q2 và chùa Thầy, 2 bộ CHƯA từng chạy qua pipeline mới này.

Usage:
    python scripts/zora_per_building_test.py [q2|chua_thay|both]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.matching.multi_space_graph import resolve_opening_conflict_graph  # noqa: E402
from sgd_alignment.zora.differentiable_sim3 import solve_pose_graph_gnc  # noqa: E402

BUILDINGS = {
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
}


def run(label: str, spaces: dict) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    clusters, cams = {}, {}
    for name, (ply, npz) in spaces.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
        print(f"  {name}: {len(clusters[name])} opening cluster(s)")
    names = list(spaces)
    root = names[0]

    print("--- Bước 1: MWIS ---")
    mwis_edges = resolve_opening_conflict_graph(clusters, cams)
    for e, edge in mwis_edges.items():
        print(f"  {e[0]}-{e[1]}: dùng {e[0]}{sorted(i for _, i in edge.used_a)} <-> "
              f"{e[1]}{sorted(j for _, j in edge.used_b)}  weight={edge.weight:.3f}")

    if not mwis_edges:
        print("  KHÔNG tìm được cạnh nào.")
        return

    print("\n--- Bước 2: GNC-TLS ---")
    edges = {e: (c.s, c.R, c.t) for e, c in mwis_edges.items()}
    prior_w = {e: max(c.weight, 1e-3) for e, c in mwis_edges.items()}
    gnc = solve_pose_graph_gnc(edges, names, root=root, edge_weights=prior_w,
                                c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)
    for e, w in sorted(gnc.gnc_weights.items(), key=lambda kv: -kv[1]):
        decision = "GIỮ" if w > 0.5 else "LOẠI"
        print(f"  {e[0]}-{e[1]}: gnc_w={w:.4f} lệch góc={gnc.edge_residuals[e]['angle_deg']:.2f}° -> {decision}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("q2", "both"):
        run("Q2 (2 không gian)", BUILDINGS["q2"])
    if mode in ("chua_thay", "both"):
        run("Chùa Thầy (2 không gian)", BUILDINGS["chua_thay"])
