"""Giai đoạn 1 (docs/zora_design_plan.md) - VERIFY: `zora.run_zora_baseline`
(interface mới, đóng gói lại) phải cho kết quả GIỐNG HỆT
`resolve_opening_conflict_graph` gốc đã verify trước đó (CONTRIBUTIONS.md
mục 12) - không được phép có regression chỉ vì tổ chức lại code.

So khớp với đúng số liệu đã ghi lại: N=2 (q1, chua_thay) 100% đúng; N=3
(server+room_310+h_server) 100% đúng; N=4 (thêm connecting_space) 3/3 thật
+ 1 giả còn sót (đã lý giải nguyên nhân).

Usage:
    python scripts/zora_baseline_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.zora import run_zora_baseline  # noqa: E402

SCENARIOS = {
    "N=2 q1": {
        "spaces": {
            "q1_indoor": ("data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply", "data/Q1/Q1_indoor/results.npz"),
            "q1_outdoor": ("data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply", "data/Q1/Q1_outdoor/results.npz"),
        },
        "expected": {("q1_indoor", "q1_outdoor")},
    },
    "N=2 chua_thay": {
        "spaces": {
            "chua_thay_indoor": ("data/chua_thay/indoor/chua_indoor_points - Cloud - segment - 5 - cua.ply",
                                  "data/chua_thay/indoor/results.npz"),
            "chua_thay_outdoor": ("data/chua_thay/outdoor/chua_outdoor_points - Cloud - segment - 5 - cua.ply",
                                   "data/chua_thay/outdoor/results.npz"),
        },
        "expected": {("chua_thay_indoor", "chua_thay_outdoor")},
    },
    "N=3 server+room_310+h_server": {
        "spaces": {
            "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply",
                         "data/server/h_server/results.npz"),
            "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
            "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
        },
        "expected": {("h_server", "server"), ("h_server", "room_310")},
    },
    "N=4 +connecting_space": {
        "spaces": {
            "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply",
                         "data/server/h_server/results.npz"),
            "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
            "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
            "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                                  "data/connecting_space/results.npz"),
        },
        "expected": {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space")},
        "known_extra_false_positive": {("server", "connecting_space")},
    },
}


def main() -> None:
    all_ok = True
    for label, spec in SCENARIOS.items():
        print(f"\n=== {label} ===")
        clusters, cams = {}, {}
        for name, (ply, npz) in spec["spaces"].items():
            clusters[name] = openings_from_manual_segmentation(ply)
            cams[name] = camera_evidence(npz)

        result = run_zora_baseline(clusters, cams)
        found = set(result.edges.keys())
        expected = spec["expected"]
        known_fp = spec.get("known_extra_false_positive", set())

        correct = found & expected
        missed = expected - found
        unexpected = found - expected - known_fp
        expected_fp_present = found & known_fp

        for (a, b), edge in result.edges.items():
            tag = "THẬT (kỳ vọng)" if (a, b) in expected else \
                ("GIẢ (đã biết, chấp nhận được)" if (a, b) in known_fp else "GIẢ MỚI - CẦN KIỂM TRA!")
            print(f"  {a}-{b} [{tag}]  weight={edge.weight:.3f}")

        ok = (len(correct) == len(expected)) and not missed and not unexpected
        print(f"  used_vlm_prior={result.used_vlm_prior} (phải là False ở baseline)")
        print(f"  => {'KHỚP đúng kết quả đã ghi trước đó' if ok else '!!! LỆCH so với kết quả đã ghi trước đó !!!'}")
        if missed:
            print(f"     thiếu: {missed}")
        if unexpected:
            print(f"     thừa ngoài dự kiến: {unexpected}")
        all_ok = all_ok and ok

    print(f"\n{'=' * 50}")
    print("TẤT CẢ KỊCH BẢN KHỚP - không có regression" if all_ok else "CÓ LỆCH - cần kiểm tra lại trước khi qua Giai đoạn 2")


if __name__ == "__main__":
    main()
