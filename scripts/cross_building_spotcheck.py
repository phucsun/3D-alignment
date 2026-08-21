"""Kiểm tra nhanh (không chạy toàn bộ 45 cặp) - chỉ chọn vài cặp XUYÊN
BUILDING có chủ đích (chắc chắn không liên quan) để xem `align_gravity_camera`
có tình cờ chấp nhận nhầm cặp nào không, trước khi quyết định có đáng chạy
toàn bộ hay không.

Usage:
    python scripts/cross_building_spotcheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402

SPACES = {
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
    "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
    "q2_indoor": ("data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply", "data/Q2/Q2_indoor/results.npz"),
    "chua_thay_indoor": ("data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply",
                          "data/chua_thay/indoor/results.npz"),
    "chua_thay_outdoor": ("data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply",
                           "data/chua_thay/outdoor/results.npz"),
}

# Chỉ chọn 4 cặp XUYÊN BUILDING có chủ đích - chắc chắn không liên quan
CROSS_PAIRS = [
    ("server", "q1_indoor"),
    ("room_310", "chua_thay_outdoor"),
    ("connecting_space", "q2_indoor"),
    ("h_server", "chua_thay_indoor"),
]


def main() -> None:
    clusters, cams = {}, {}
    needed = {name for pair in CROSS_PAIRS for name in pair}
    for name in needed:
        ply, npz = SPACES[name]
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
        print(f"{name}: {len(clusters[name])} opening cluster(s)")

    print("\n=== Test 4 cặp XUYÊN BUILDING có chủ đích (chắc chắn không liên quan) ===")
    n_false_positive = 0
    for a, b in CROSS_PAIRS:
        r_ab = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        r_ba = align_gravity_camera(clusters[b], clusters[a], cams[b], cams[a])
        print(f"\n{a} <-> {b}:")
        print(f"  A={a},B={b}: status={r_ab.status} residual={r_ab.opening_residual:.4f} matches={r_ab.matches}")
        print(f"  A={b},B={a}: status={r_ba.status} residual={r_ba.opening_residual:.4f} matches={r_ba.matches}")
        risky = (r_ab.status != "NO_SOLUTION" and len(r_ab.matches) >= 2) or \
                (r_ba.status != "NO_SOLUTION" and len(r_ba.matches) >= 2)
        if risky:
            n_false_positive += 1
            print("  !! RỦI RO: có candidate >=2 match, không phải NO_SOLUTION - có thể bị nhầm là thật !!")
        else:
            print("  OK: không có candidate đáng ngờ (NO_SOLUTION hoặc chỉ 1 match trivial)")

    print(f"\n=== Tổng kết: {n_false_positive}/{len(CROSS_PAIRS)} cặp xuyên-building có rủi ro bị nhầm ===")


if __name__ == "__main__":
    main()
