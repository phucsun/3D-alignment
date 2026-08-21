"""Kiểm tra `resolve_opening_conflict_graph` (docs/multi_space_alignment_plan.md,
Giai đoạn 3 - hướng đã verify tốt nhất) trên nhiều kịch bản KHÁC NHAU, không
chỉ bộ 4-không-gian đã dùng để phát triển thuật toán - để xác nhận nó tổng
quát hoá đúng, không phải chỉ "học thuộc" đúng 1 bộ dữ liệu:

- 2 kịch bản NỐI 2 (N=2, sanity check): q1 và chua_thay - đã biết đúng từ
  trước (mục 11 CONTRIBUTIONS.md), chỉ có 1 cặp/1 cạnh khả dĩ, không có
  xung đột nào để giải quyết - kỳ vọng thuật toán tự nhiên trả về đúng kết
  quả `align_gravity_camera` gốc (trường hợp suy biến bắt buộc phải đúng).
- 1 kịch bản NỐI 3 (server + room_310 + h_server, KHÔNG có connecting_space)
  - đã biết đúng từ trước (Giai đoạn 1, dùng adjacency prior hardcode hub) -
  giờ chạy KHÔNG cho biết trước ai là hub, xem thuật toán tự suy ra đúng
  không.

Usage:
    python scripts/multi_space_scenarios_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.matching.multi_space_graph import resolve_opening_conflict_graph  # noqa: E402


def _load(spaces: dict[str, tuple[str, str]]):
    clusters, cams = {}, {}
    for name, (ply, npz) in spaces.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    return clusters, cams


def run_scenario(label: str, spaces: dict[str, tuple[str, str]], ground_truth_edges: set[tuple[str, str]]) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    clusters, cams = _load(spaces)
    for name in spaces:
        print(f"  {name}: {len(clusters[name])} opening cluster(s)")

    result = resolve_opening_conflict_graph(clusters, cams, verbose=True)

    print(f"\nKết quả (không cho biết trước topology):")
    correct = wrong = 0
    for (a, b), edge in result.items():
        tag = "THẬT" if (a, b) in ground_truth_edges else "GIẢ"
        correct += tag == "THẬT"
        wrong += tag == "GIẢ"
        print(f"  {a}-{b} [{tag}]  dùng {a}{sorted(i for _, i in edge.used_a)} <-> "
              f"{b}{sorted(j for _, j in edge.used_b)}  weight={edge.weight:.3f}")
    missed = ground_truth_edges - set(result.keys())
    n_gt = len(ground_truth_edges)
    print(f"=> Giữ đúng {correct}/{n_gt} cạnh thật, giữ NHẦM {wrong} cạnh giả, "
          f"bỏ sót {len(missed)} cạnh thật {missed or ''}")


def main() -> None:
    # ---- kịch bản NỐI 2: q1 (đã biết đúng, 1/1 match, residual=0.0) ----
    run_scenario(
        "NỐI 2 - q1 (indoor/outdoor, đã biết đúng từ trước)",
        {
            "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
            "q1_outdoor": ("data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
        },
        {("q1_indoor", "q1_outdoor")},
    )

    # ---- kịch bản NỐI 2: chua_thay (đã biết đúng, 5/5 match, residual=0.0355) ----
    run_scenario(
        "NỐI 2 - chua_thay (indoor/outdoor, đã biết đúng từ trước)",
        {
            "chua_thay_indoor": ("data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply",
                                  "data/chua_thay/indoor/results.npz"),
            "chua_thay_outdoor": ("data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply",
                                   "data/chua_thay/outdoor/results.npz"),
        },
        {("chua_thay_indoor", "chua_thay_outdoor")},
    )

    # ---- kịch bản NỐI 3: server + room_310 + h_server (không có connecting_space) ----
    run_scenario(
        "NỐI 3 - server + room_310 + h_server (KHÔNG cho biết trước ai là hub)",
        {
            "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply",
                         "data/server/h_server/results.npz"),
            "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
            "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
        },
        {("h_server", "server"), ("h_server", "room_310")},
    )


if __name__ == "__main__":
    main()
