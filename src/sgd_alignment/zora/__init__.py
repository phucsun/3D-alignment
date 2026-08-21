"""ZORA: Topology-Aware Zero-Overlap 3D Room Assembly (branch `zora-vlm-graph`).

Xem `docs/zora_design_plan.md` cho lộ trình đầy đủ. Package này bắt đầu bằng
`baseline` (Giai đoạn 1) - đóng gói thuật toán "geometry-only, không prior
tôpô" đã verify ở `sgd_alignment.matching.multi_space_graph` thành 1
interface rõ ràng, dùng làm cột mốc so sánh cho mọi giai đoạn sau (Layer 1
VLM prior, differentiable solver, GNC...).
"""
from .baseline import ZoraBaselineConfig, ZoraBaselineResult, run_zora_baseline

__all__ = ["ZoraBaselineConfig", "ZoraBaselineResult", "run_zora_baseline"]
