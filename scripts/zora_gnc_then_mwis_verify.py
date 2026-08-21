"""Giai đoạn 4, Hướng 1 - biến thể ĐƠN GIẢN HƠN: chạy GNC-TLS ĐẾN HỘI TỤ
HOÀN TOÀN TRƯỚC (như Giai đoạn 3 nguyên bản, KHÔNG xen kẽ, KHÔNG khoá sớm),
rồi CHỈ 1 LẦN MWIS DUY NHẤT ở cuối trên trọng số GNC đã hội tụ - xác định
xem vấn đề ở bản Alternating trước là do "khoá sớm khi trọng số chưa chín"
hay do bản chất kết hợp GNC+MWIS nói chung.

Usage:
    python scripts/zora_gnc_then_mwis_verify.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.matching.multi_space_graph import _best_direction_opening_edge, _opening_edges_conflict  # noqa: E402
from sgd_alignment.zora.differentiable_sim3 import solve_pose_graph_gnc  # noqa: E402

SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}
REAL_EDGES = {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space")}


def main() -> None:
    clusters, cams = {}, {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    names = list(SPACES.keys())
    full_avail = {n: set(range(len(clusters[n]))) for n in names}

    print("=== Bước 1: tính candidate 1 lần trên FULL opening cho mọi cặp (best-of-2-chiều) ===")
    opening_edges = {}
    for a, b in combinations(names, 2):
        cand = _best_direction_opening_edge(a, b, clusters, cams, full_avail)
        if cand is not None:
            opening_edges[(a, b)] = cand
            print(f"  {a}-{b}: dùng {a}{sorted(i for _, i in cand.used_a)} <-> {b}{sorted(j for _, j in cand.used_b)} "
                  f"weight={cand.weight:.3f}")

    edges = {e: (c.s, c.R, c.t) for e, c in opening_edges.items()}
    prior_w = {e: max(c.weight, 1e-3) for e, c in opening_edges.items()}

    print("\n=== Bước 2: chạy GNC-TLS ĐẾN HỘI TỤ HOÀN TOÀN (như Giai đoạn 3, không xen kẽ) ===")
    gnc = solve_pose_graph_gnc(edges, names, root="h_server", edge_weights=prior_w,
                                c_deg=5.0, mu_factor=1.4, n_outer=15, n_inner_steps=200, lr=0.05)
    for e, w in sorted(gnc.gnc_weights.items(), key=lambda kv: -kv[1]):
        tag = "THẬT" if e in REAL_EDGES else "GIẢ"
        print(f"  {e[0]}-{e[1]} [{tag}]: gnc_w={w:.4f} lệch góc={gnc.edge_residuals[e]['angle_deg']:.2f}°")

    print("\n=== Bước 3: CHỈ 1 LẦN MWIS DUY NHẤT trên trọng số GNC đã hội tụ ===")
    keys = list(opening_edges)
    n = len(keys)
    conflict = [[_opening_edges_conflict(opening_edges[keys[i]], opening_edges[keys[j]]) if i != j else False
                 for j in range(n)] for i in range(n)]
    best_subset, best_w = [], -1.0
    for mask in range(1 << n):
        idx = [i for i in range(n) if mask & (1 << i)]
        if any(conflict[i][j] for i in idx for j in idx if i < j):
            continue
        w = sum(gnc.gnc_weights[keys[i]] for i in idx)
        if w > best_w:
            best_w, best_subset = w, idx
    winners = {keys[i] for i in best_subset}

    print(f"Tập thắng (tổng gnc_w={best_w:.3f}):")
    correct = wrong = 0
    for e in winners:
        tag = "THẬT" if e in REAL_EDGES else "GIẢ"
        correct += tag == "THẬT"
        wrong += tag == "GIẢ"
        print(f"  {e[0]}-{e[1]} [{tag}]  gnc_w={gnc.gnc_weights[e]:.4f}")
    missed = REAL_EDGES - winners
    print(f"\n=> Giữ đúng {correct}/{len(REAL_EDGES)}, giữ NHẦM {wrong} cạnh giả, bỏ sót {len(missed)} {missed or ''}")


if __name__ == "__main__":
    main()
