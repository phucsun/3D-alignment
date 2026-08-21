"""Giai đoạn 3 (docs/multi_space_alignment_plan.md), hướng SỬA ĐÚNG GỐC:
mỗi cửa vật lý (`(không gian, chỉ số opening)`) chỉ có thể là ranh giới của
ĐÚNG 1 cặp không gian - không được dùng lại cho 2 quan hệ khác nhau.

Thuật toán MWIS + LẶP LOẠI TRỪ ĐẾN HỘI TỤ (tổng quát hoá cơ chế fixed-point
đã thành công ở Giai đoạn 1 "adjacency prior", giờ áp dụng cho TOÀN ĐỒ THỊ,
KHÔNG cần biết trước hub):

1. Mỗi cạnh ứng viên "chiếm dụng" 1 tập `(space, opening_idx)` mỗi đầu (đo
   bằng CẢ 2 CHIỀU (a,b)/(b,a) - `align_gravity_camera` không đối xứng, xác
   nhận thật trên dữ liệu, giữ chiều tốt hơn).
2. Tìm tập cạnh KHÔNG xung đột (không chiếm chung cửa) có tổng trọng số lớn
   nhất (MWIS, duyệt toàn bộ 2^n tập con - n nhỏ nên vét cạn được).
3. "Khoá" các cạnh thắng, loại các opening đã dùng khỏi pool còn trống của
   từng không gian.
4. TÍNH LẠI mọi cạnh CHƯA khoá, chỉ trên phần opening CÒN TRỐNG (không phải
   giữ nguyên candidate cũ) - xác nhận thật: nếu không làm bước này, 1 cạnh
   THẬT thua cuộc vì trùng cửa với cạnh khác sẽ bị loại bỏ vĩnh viễn, dù nó
   có thể tìm ra 1 correspondence ĐÚNG khác nếu chỉ được xét trên phần cửa
   còn lại (vd `h_server-connecting_space` chỉ đúng khi bị cấm dùng
   `h_server[2,3]` đã thuộc về `server`, buộc phải tìm ra `h_server[4]`).
5. Lặp lại bước 2-4 đến khi không còn cạnh mới nào được khoá thêm.

Usage:
    python scripts/multi_space_opening_conflict_graph.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points  # noqa: E402
from sgd_alignment.matching.multi_space_graph import invert  # noqa: E402

SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}
GROUND_TRUTH_REAL_EDGES = {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space")}
STATUS_RANK = {"CONFIDENT": 2, "AMBIGUOUS": 1, "NO_SOLUTION": 0}


@dataclass
class Candidate:
    a: str
    b: str
    s: float
    R: np.ndarray
    t: np.ndarray
    weight: float
    used_a: frozenset
    used_b: frozenset


def _conflicts(c1: Candidate, c2: Candidate) -> bool:
    used1 = c1.used_a | c1.used_b
    used2 = c2.used_a | c2.used_b
    return bool(used1 & used2)


def _best_direction_candidate(a, b, clusters, cams, avail) -> Candidate | None:
    """Tính cạnh (a,b) chỉ trên phần opening CÒN TRỐNG của mỗi bên (`avail`),
    thử cả 2 chiều, giữ chiều tốt hơn, map index cục bộ về index thật."""
    idx_a, idx_b = sorted(avail[a]), sorted(avail[b])
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None
    sub_a = [clusters[a][i] for i in idx_a]
    sub_b = [clusters[b][i] for i in idx_b]

    r_ab = align_gravity_camera(sub_a, sub_b, cams[a], cams[b])
    r_ba = align_gravity_camera(sub_b, sub_a, cams[b], cams[a])
    key_ab = (STATUS_RANK[r_ab.status], -r_ab.opening_residual)
    key_ba = (STATUS_RANK[r_ba.status], -r_ba.opening_residual)

    if r_ab.status == "NO_SOLUTION" and r_ba.status == "NO_SOLUTION":
        return None
    if key_ab >= key_ba:
        result = r_ab
        matches = [(idx_a[i], idx_b[j]) for i, j in result.matches]
        s, R, t = result.s, result.R, result.t
    else:
        result = r_ba
        matches = [(idx_a[j], idx_b[i]) for i, j in result.matches]  # r_ba: A=sub_b,B=sub_a -> (i in b-local,j in a-local)
        s, R, t = invert((result.s, result.R, result.t))

    if not matches:
        return None
    weight = len(matches) - result.opening_residual
    used_a = frozenset((a, i) for i, _ in matches)
    used_b = frozenset((b, j) for _, j in matches)
    return Candidate(a, b, s, R, t, weight, used_a, used_b)


def main() -> None:
    clusters, cams = {}, {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    names = list(SPACES.keys())

    avail = {name: set(range(len(clusters[name]))) for name in names}
    locked_by_pair: dict[tuple[str, str], Candidate] = {}
    all_pairs = list(combinations(names, 2))
    unresolved_pairs = set(all_pairs)

    MAX_ROUNDS = 6
    for round_no in range(1, MAX_ROUNDS + 1):
        print(f"\n=== Vòng {round_no}: tính lại cạnh chưa khoá trên phần opening còn trống ===")
        new_candidates: list[Candidate] = []
        for a, b in list(unresolved_pairs):
            cand = _best_direction_candidate(a, b, clusters, cams, avail)
            if cand is not None:
                tag = "THẬT" if (a, b) in GROUND_TRUTH_REAL_EDGES else "GIẢ"
                print(f"  {a}-{b} [{tag}]: dùng {a}{sorted(i for _, i in cand.used_a)} <-> "
                      f"{b}{sorted(j for _, j in cand.used_b)}  weight={cand.weight:.3f}")
                new_candidates.append(cand)
            else:
                print(f"  {a}-{b}: không tìm được correspondence nào (hết opening hoặc NO_SOLUTION)")

        pool = list(locked_by_pair.values()) + new_candidates
        n = len(pool)
        conflict = [[_conflicts(pool[i], pool[j]) if i != j else False for j in range(n)] for i in range(n)]

        best_subset, best_weight = [], -1.0
        for mask in range(1 << n):
            idx = [i for i in range(n) if mask & (1 << i)]
            if any(conflict[i][j] for i in idx for j in idx if i < j):
                continue
            w = sum(pool[i].weight for i in idx)
            if w > best_weight:
                best_weight, best_subset = w, idx

        winners = [pool[i] for i in best_subset]
        newly_locked = [c for c in winners if (c.a, c.b) not in locked_by_pair]
        print(f"  -> {len(winners)} cạnh trong tập thắng vòng này, {len(newly_locked)} cạnh MỚI được khoá")

        if not newly_locked:
            print("  Hội tụ - không còn cạnh mới nào được khoá thêm.")
            break

        for c in newly_locked:
            for space, i in c.used_a | c.used_b:
                avail[space].discard(i)
            unresolved_pairs.discard((c.a, c.b))
            locked_by_pair[(c.a, c.b)] = c

    print("\n=== KẾT QUẢ CUỐI (sau khi hội tụ) ===")
    correct, wrong = 0, 0
    for c in locked_by_pair.values():
        tag = "THẬT" if (c.a, c.b) in GROUND_TRUTH_REAL_EDGES else "GIẢ"
        correct += tag == "THẬT"
        wrong += tag == "GIẢ"
        print(f"  {c.a}-{c.b} [{tag}]  dùng {c.a}{sorted(i for _, i in c.used_a)} <-> "
              f"{c.b}{sorted(j for _, j in c.used_b)}  weight={c.weight:.3f}")
    missed = GROUND_TRUTH_REAL_EDGES - set(locked_by_pair.keys())
    print(f"\n=> Giữ đúng {correct}/3 cạnh thật, giữ NHẦM {wrong} cạnh giả, bỏ sót {len(missed)} cạnh thật {missed or ''}")


if __name__ == "__main__":
    main()
