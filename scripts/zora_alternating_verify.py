"""Giai đoạn 4, Hướng 1 (Alternating Optimization) - VERIFY tuần tự:
1. N=4 (server building) - phải KHÔNG regression so với GNC thuần đã hoàn
   hảo (3/3 thật, 0 giả, 0 bỏ sót).
2. N=6 (server + Q1, xuyên-building) - kỳ vọng sửa được lỗi đã phát hiện ở
   GNC thuần + Layer1-only (giữ nhầm cạnh xuyên-building / bỏ sót cạnh thật).

Usage:
    python scripts/zora_alternating_verify.py [n4|n6|both]
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.matching.multi_space_graph import _best_direction_opening_edge  # noqa: E402
from sgd_alignment.zora.alternating import alternating_gnc_mwis  # noqa: E402
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
SERVER_BUILDING = {"h_server", "server", "room_310", "connecting_space"}
Q1_BUILDING = {"q1_indoor", "q1_outdoor"}
N6_REAL = N4_REAL | {("q1_indoor", "q1_outdoor")}
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


def _report(result, real_edges):
    correct = wrong_kept = 0
    for e in sorted(result.locked, key=lambda e: -result.gnc_weights.get(e, 0.0)):
        tag = "THẬT" if e in real_edges else "GIẢ"
        w = result.gnc_weights.get(e, float("nan"))
        angle = result.edge_residuals[e]["angle_deg"]
        print(f"  {e[0]}-{e[1]} [{tag}]: KHOÁ gnc_w={w:.3f} lệch góc={angle:.2f}°")
        if tag == "THẬT":
            correct += 1
        else:
            wrong_kept += 1
    missed = real_edges - set(result.locked)
    print(f"  -> Giữ đúng {correct}/{len(real_edges)}, giữ NHẦM {wrong_kept} cạnh giả, "
          f"bỏ sót {len(missed)} cạnh thật {missed or ''}, hội tụ sau {result.rounds_used} vòng")


def run_n4():
    print("\n" + "=" * 60 + "\nN=4 (server building) - verify không regression\n" + "=" * 60)
    clusters, cams = _load(N4_SPACES)
    result = alternating_gnc_mwis(clusters, cams, list(N4_SPACES), root="h_server",
                                    max_rounds=8, gnc_outer_per_round=15, gnc_inner_steps=150)
    _report(result, N4_REAL)


def run_n6(use_prior: bool):
    label = "CÓ Layer 1 prior" if use_prior else "KHÔNG Layer 1 (thuần hình học + MWIS luân phiên)"
    print("\n" + "=" * 60 + f"\nN=6 (server+Q1, xuyên-building) - {label}\n" + "=" * 60)
    clusters, cams = _load(N6_SPACES)
    names = list(N6_SPACES)

    prior_weights = None
    if use_prior:
        full_avail = {n: set(range(len(clusters[n]))) for n in names}
        base = {}
        for a, b in combinations(names, 2):
            cand = _best_direction_opening_edge(a, b, clusters, cams, full_avail)
            if cand is not None:
                base[(a, b)] = max(cand.weight, 1e-3)
        mock = []
        for a, b in combinations(names, 2):
            same = ({a, b} <= SERVER_BUILDING) or ({a, b} <= Q1_BUILDING)
            if not same:
                mock.append({"a": a, "b": b, "adjacent": False, "confidence": 0.95})
        soft_graph = parse_topology_from_text(DESCRIPTION, names, mock_response=mock)
        prior_weights = combine_geometric_and_vlm_prior(base, soft_graph, contradicted_penalty=0.001)

    result = alternating_gnc_mwis(clusters, cams, names, root="h_server", prior_weights=prior_weights,
                                    max_rounds=8, gnc_outer_per_round=15, gnc_inner_steps=150)
    _report(result, N6_REAL)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("n4", "both"):
        run_n4()
    if mode in ("n6", "both"):
        run_n6(use_prior=False)
        run_n6(use_prior=True)
