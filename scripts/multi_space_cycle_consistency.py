"""Giai đoạn 2 (docs/multi_space_alignment_plan.md) - kiểm tra tính nhất
quán chu trình (loop-consistency / PCM) để tự động phát hiện cạnh SAI khi
KHÔNG biết trước topology (không giả định sẵn hub/star).

Đầu vào: 6 cạnh đã khám phá ở `align_4space_pipeline.py` (mọi cặp trong 4
không gian: h_server, server, room_310, connecting_space). Ground-truth đã
biết (xác nhận thủ công bằng mắt): CHỈ 3 cạnh star vào h_server là thật,
3 cạnh còn lại (server-room_310, server-connecting_space,
room_310-connecting_space) là false positive.

Thuật toán: với mọi cách chọn 3/6 cạnh tạo thành 1 CÂY KHUNG (spanning
tree) nối đủ 4 không gian, đặt vị trí tuyệt đối theo cây đó, rồi kiểm tra
3 cạnh CÒN LẠI (không nằm trong cây) - so sánh transform mà cây DỰ ĐOÁN
giữa 2 đầu cạnh đó với transform mà chính cạnh đó đo được trực tiếp. Sai
lệch (loop error) càng nhỏ, cây khung đó càng đáng tin. Chọn cây có tổng
loop-error nhỏ nhất trong số MỌI cây khung khả dĩ - không cần biết trước
đâu là hub.

Usage:
    python scripts/multi_space_cycle_consistency.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402

SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}


def _invert(T):
    s, R, t = T
    s_inv = 1.0 / s
    R_inv = R.T
    t_inv = -s_inv * (R_inv @ t)
    return s_inv, R_inv, t_inv


def _compose(T2, T1):
    """T2 ∘ T1: áp T1 trước rồi T2."""
    s1, R1, t1 = T1
    s2, R2, t2 = T2
    return s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2


def _loop_error(T_xy, T_yz, T_zx) -> dict:
    """Hợp chu trình x->y->z->x, đo độ lệch so với Identity."""
    s, R, t = _compose(T_zx, _compose(T_yz, T_xy))
    scale_dev = abs(np.log(s))  # 0 nếu s=1
    cos_ang = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_ang)))
    trans_norm = float(np.linalg.norm(t))
    return {"scale_dev": scale_dev, "angle_deg": angle_deg, "trans_norm": trans_norm}


def main() -> None:
    clusters, cams = {}, {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)

    names = list(SPACES.keys())
    print("=== Đo transform từng cặp (align_gravity_camera, 1 lần/cặp) ===")
    raw_edges: dict[tuple[str, str], tuple[float, np.ndarray, np.ndarray]] = {}
    edge_info = {}
    for a, b in combinations(names, 2):
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        raw_edges[(a, b)] = (result.s, result.R, result.t)
        edge_info[(a, b)] = (result.status, result.opening_residual, len(result.matches))
        print(f"  {a} -> {b}: status={result.status} residual={result.opening_residual:.4f} "
              f"n_matches={len(result.matches)}")

    def T(a, b):
        """Transform a->b, tự động nghịch đảo nếu chỉ đo được b->a."""
        if (a, b) in raw_edges:
            return raw_edges[(a, b)]
        return _invert(raw_edges[(b, a)])

    all_pairs = list(combinations(names, 2))

    # ---- thử MỌI cách chọn 3/6 cạnh, giữ lại các cây khung hợp lệ (nối đủ 4 node, không chu trình) ----
    print("\n=== Thử mọi cây khung khả dĩ (3 cạnh nối 4 không gian), kiểm tra 3 cạnh dư bằng loop-error ===")
    trees = []
    for tree_edges in combinations(all_pairs, 3):
        nodes_in_tree = set()
        for a, b in tree_edges:
            nodes_in_tree.add(a)
            nodes_in_tree.add(b)
        if len(nodes_in_tree) != 4:
            continue  # không nối đủ 4 không gian
        # kiểm tra là cây thật (không chu trình): union-find đơn giản
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        ok = True
        for a, b in tree_edges:
            ra, rb = find(a), find(b)
            if ra == rb:
                ok = False
                break
            parent[ra] = rb
        if not ok:
            continue
        trees.append(tree_edges)

    results = []
    for tree_edges in trees:
        # đặt vị trí tuyệt đối theo cây (root = node đầu tiên xuất hiện)
        root = tree_edges[0][0]
        abs_pos = {root: (1.0, np.eye(3), np.zeros(3))}
        remaining = list(tree_edges)
        while remaining:
            progressed = False
            for a, b in list(remaining):
                if a in abs_pos and b not in abs_pos:
                    abs_pos[b] = _compose(abs_pos[a], T(a, b))
                    remaining.remove((a, b))
                    progressed = True
                elif b in abs_pos and a not in abs_pos:
                    abs_pos[a] = _compose(abs_pos[b], T(b, a))
                    remaining.remove((a, b))
                    progressed = True
            if not progressed:
                break

        leftover_edges = [p for p in all_pairs if p not in tree_edges]
        total_angle, total_trans, total_scale = 0.0, 0.0, 0.0
        detail = []
        for a, b in leftover_edges:
            # transform mà CÂY dự đoán giữa a và b: T(a->root)^-1 rồi T(root->... );
            # đơn giản: a->b dự đoán = invert(abs_pos[a]) đứng trước abs_pos[b]... ta cần
            # transform mapping a's local frame -> b's local frame:
            # abs_pos[x] = transform x -> root. Vậy a->b dự đoán = invert(abs_pos[b]) ∘ abs_pos[a]
            predicted_ab = _compose(_invert(abs_pos[b]), abs_pos[a])
            measured_ab = T(a, b)
            # loop x=a, qua predicted(a->b), rồi invert(measured a->b) để về lại a -> phải = Identity
            err = _loop_error(predicted_ab, _invert(measured_ab), (1.0, np.eye(3), np.zeros(3)))
            detail.append((a, b, err))
            total_angle += err["angle_deg"]
            total_trans += err["trans_norm"]
            total_scale += err["scale_dev"]
        results.append((tree_edges, total_angle, total_trans, total_scale, detail))

    results.sort(key=lambda r: r[1])  # sắp theo tổng lệch góc (chỉ số chính, dễ diễn giải nhất)

    for tree_edges, total_angle, total_trans, total_scale, detail in results:
        tag = "  <-- GROUND TRUTH (star vào h_server)" if set(tree_edges) == {
            ("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space")} else ""
        print(f"\nCây khung {tree_edges}{tag}")
        print(f"  tổng lệch góc={total_angle:.2f}°  tổng lệch tịnh tiến={total_trans:.4f}  tổng lệch scale(log)={total_scale:.4f}")
        for a, b, err in detail:
            print(f"    cạnh dư kiểm tra {a}-{b}: lệch góc={err['angle_deg']:.2f}° lệch tịnh tiến={err['trans_norm']:.4f} "
                  f"lệch scale(log)={err['scale_dev']:.4f}")


if __name__ == "__main__":
    main()
