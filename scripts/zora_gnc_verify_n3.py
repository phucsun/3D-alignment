"""Giai đoạn 3 - verify chéo GNC-TLS trên kịch bản N=3 (server+room_310+
h_server, không connecting_space) - đã biết đúng từ trước (2/2 đúng, không
có cạnh giả nào để loại). Kỳ vọng GNC-TLS giữ cả 2 cạnh (weight~1.0), không
loại nhầm gì khi không có gì cần loại.

Usage:
    python scripts/zora_gnc_verify_n3.py
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
}
GROUND_TRUTH_REAL_EDGES = {("h_server", "server"), ("h_server", "room_310")}


def main() -> None:
    clusters, cams = {}, {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    names = list(SPACES.keys())

    edges, edge_weights = {}, {}
    for a, b in combinations(names, 2):
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        if result.status == "NO_SOLUTION":
            continue
        edges[(a, b)] = (result.s, result.R, result.t)
        edge_weights[(a, b)] = len(result.matches) / (result.opening_residual + 1e-3)

    print(f"Cạnh đo được: {list(edges.keys())}")
    gnc = solve_pose_graph_gnc(edges, names, root="h_server", edge_weights=edge_weights,
                                c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)

    correct = wrong_dropped = 0
    for e, w in gnc.gnc_weights.items():
        tag = "THẬT" if e in GROUND_TRUTH_REAL_EDGES else "GIẢ"
        decision = "GIỮ" if w > 0.5 else "LOẠI"
        print(f"  {e[0]}-{e[1]} [{tag}]: weight={w:.4f} lệch góc={gnc.edge_residuals[e]['angle_deg']:.2f}° -> {decision}")
        if tag == "THẬT" and w > 0.5:
            correct += 1
        elif tag == "THẬT":
            wrong_dropped += 1

    print(f"=> Giữ đúng {correct}/{len(GROUND_TRUTH_REAL_EDGES)} cạnh thật, loại NHẦM {wrong_dropped}")


if __name__ == "__main__":
    main()
