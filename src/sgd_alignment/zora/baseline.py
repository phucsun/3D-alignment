"""Giai đoạn 1 (docs/zora_design_plan.md): "Layer 2 + Exclusivity" ablation
baseline - KHÔNG có Layer 1 (không có VLM/sketch/mô tả ngôn ngữ nào cả).
Tương đương "corruption=100%/không prior nào" trong khung đánh giá ZORA
(Section 3 của proposal: so sánh RTE/RRE/Edge-Rejection-Rate theo % nhiễu
prior - baseline này là điểm neo ở đầu mút "không có prior").

Bọc lại `sgd_alignment.matching.multi_space_graph.resolve_opening_conflict_graph`
(đã verify đúng trên 4 kịch bản dữ liệu thật độc lập - xem CONTRIBUTIONS.md
mục 12) - KHÔNG đổi thuật toán, chỉ đổi tên gọi/interface cho khớp vai trò
kiến trúc ZORA, để dễ so sánh khi Layer 1 được thêm vào ở giai đoạn sau.
"""
from __future__ import annotations

from dataclasses import dataclass

from sgd_alignment.matching.gravity_align import CameraEvidence
from sgd_alignment.matching.multi_space_graph import OpeningEdge, resolve_opening_conflict_graph


@dataclass
class ZoraBaselineConfig:
    max_rounds: int = 6  # số vòng lặp tối đa cho fixed-point loại trừ cửa trùng


@dataclass
class ZoraBaselineResult:
    """`edges`: {(space_a, space_b): OpeningEdge} - các cạnh liên-không-gian
    đã được xác nhận (không dùng chung cửa với cạnh nào khác), suy ra HOÀN
    TOÀN từ hình học (không có prior tôpô Layer 1 nào cả)."""
    edges: dict[tuple[str, str], OpeningEdge]
    used_vlm_prior: bool = False  # luôn False ở baseline này - đối chiếu rõ ràng với Giai đoạn 4/5


def run_zora_baseline(
    clusters: dict[str, list],
    cams: dict[str, CameraEvidence],
    config: ZoraBaselineConfig | None = None,
) -> ZoraBaselineResult:
    """Chạy pipeline ZORA ở chế độ "Layer 2 only, no VLM prior" - đây là
    baseline geometry-only, dùng làm cột mốc so sánh cho mọi cải tiến thêm
    Layer 1 (VLM) ở các giai đoạn sau.

    `clusters`: {tên không gian: list[OpeningCluster]} (xem `robust_align.
    openings_from_manual_segmentation`).
    `cams`: {tên không gian: CameraEvidence} (xem `gravity_align.camera_evidence`).
    """
    config = config or ZoraBaselineConfig()
    edges = resolve_opening_conflict_graph(clusters, cams, max_rounds=config.max_rounds)
    return ZoraBaselineResult(edges=edges, used_vlm_prior=False)
