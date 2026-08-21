"""Giai đoạn 4 (docs/zora_design_plan.md): Layer 1 - sinh "soft graph" tôpô
từ mô tả ngôn ngữ tự nhiên (KHÔNG phải sketch 2D - quyết định 2026-08-21,
để test nhanh bằng text trước, không cần thu thập ảnh sketch).

Bản này CHƯA gọi LLM/VLM thật (chưa cài `anthropic` SDK, chưa cần API key -
quyết định 2026-08-21: xây interface + giả lập đầu ra trước, verify tích
hợp logic xong mới tốn chi phí API thật). `parse_topology_from_text` hiện
nhận `mock_response` (mô phỏng đầu ra 1 LLM PARSE ĐÚNG sẽ trả về) thay vì
gọi API - khi sẵn sàng, chỉ cần thay phần "TODO: gọi LLM thật" bằng lệnh
gọi `anthropic.Anthropic().messages.create(...)` với `PROMPT_TEMPLATE`,
schema đầu ra giữ nguyên.
"""
from __future__ import annotations

from dataclasses import dataclass, field


PROMPT_TEMPLATE = """Bạn là hệ thống đọc mô tả kiến trúc và sinh ra đồ thị tôpô giữa các không gian.

Danh sách không gian: {room_names}

Mô tả của người dùng:
\"\"\"{description}\"\"\"

Với MỖI CẶP không gian trong danh sách trên, xác định:
- "adjacent": có kề nhau (chung tường/cửa) theo mô tả không - true/false
- "confidence": độ tin cậy của bạn về nhận định đó (0.0-1.0) - THẤP nếu mô tả không
  đề cập rõ tới cặp đó (không có nghĩa là chắc chắn KHÔNG kề nhau, chỉ là mô tả không nói gì).

Trả về JSON đúng schema:
{{"edges": [{{"a": "...", "b": "...", "adjacent": true/false, "confidence": 0.0-1.0}}, ...]}}
"""


@dataclass
class SoftEdge:
    a: str
    b: str
    adjacent: bool
    confidence: float  # 0.0 (không chắc/không nói tới) -> 1.0 (rất chắc chắn)


@dataclass
class SoftGraph:
    nodes: list[str]
    edges: list[SoftEdge] = field(default_factory=list)

    def edge_confidence(self, a: str, b: str) -> tuple[bool, float]:
        """Trả về (adjacent, confidence) cho cặp (a,b) - không phân biệt thứ
        tự. Nếu không có trong graph (mô tả không đề cập), coi là
        `adjacent=True, confidence=0.0` (KHÔNG PHỦ ĐỊNH - chỉ là "không biết
        gì", để không tự động loại 1 cặp chỉ vì người dùng quên nhắc tới)."""
        for e in self.edges:
            if {e.a, e.b} == {a, b}:
                return e.adjacent, e.confidence
        return True, 0.0


def parse_topology_from_text(
    description: str,
    room_names: list[str],
    mock_response: list[dict] | None = None,
) -> SoftGraph:
    """`mock_response`: dùng để TEST/verify tích hợp mà KHÔNG gọi LLM thật -
    mô phỏng đầu ra 1 LLM PARSE ĐÚNG sẽ trả về (list các dict
    `{"a","b","adjacent","confidence"}`, đúng schema mô tả ở `PROMPT_TEMPLATE`).

    Khi tích hợp LLM thật (chưa làm - quyết định 2026-08-21): thay thế khối
    `if mock_response is not None` bằng lệnh gọi API thật (dùng
    `PROMPT_TEMPLATE.format(room_names=room_names, description=description)`
    làm prompt), parse JSON trả về theo đúng schema này.
    """
    if mock_response is not None:
        edges = [SoftEdge(a=e["a"], b=e["b"], adjacent=e["adjacent"], confidence=e["confidence"])
                  for e in mock_response]
        return SoftGraph(nodes=list(room_names), edges=edges)

    raise NotImplementedError(
        "Chưa tích hợp gọi LLM thật (quyết định 2026-08-21: xây interface + giả lập trước) - "
        "truyền `mock_response` để test, hoặc tự thêm lệnh gọi anthropic SDK ở đây."
    )


def combine_geometric_and_vlm_prior(
    geometric_weights: dict[tuple[str, str], float],
    soft_graph: SoftGraph,
    contradicted_penalty: float = 0.01,
) -> dict[tuple[str, str], float]:
    """Kết hợp trọng số ưu tiên hình học (`n_matches/residual`, đã có) với
    prior tôpô từ Layer 1 - ĐÂY LÀ ĐIỂM TÍCH HỢP CHÍNH của ZORA (khác GNC
    thuần hình học ở Giai đoạn 3).

    - Nếu Layer 1 KHÔNG nói gì về cặp đó (`confidence=0.0`) -> giữ nguyên
      trọng số hình học (không đổi hành vi, tương đương baseline Giai đoạn 3).
    - Nếu Layer 1 nói "adjacent=True" với confidence cao -> TĂNG trọng số
      (khuyến khích GNC coi là inlier).
    - Nếu Layer 1 nói "adjacent=False" với confidence cao -> ép trọng số về
      gần 0 (`contradicted_penalty`) - ngăn GNC nhận nhầm 1 cặp mà con người
      đã xác nhận KHÔNG liên quan (chính là trường hợp `server<->q1_indoor`
      xuyên-building đã phát hiện rủi ro thật, hình học thuần không tự phân
      biệt được).
    """
    out = {}
    for e, geo_w in geometric_weights.items():
        adjacent, conf = soft_graph.edge_confidence(*e)
        if conf <= 0.0:
            out[e] = geo_w  # Layer 1 không nói gì -> không đổi (an toàn, tương thích ngược)
        elif adjacent:
            out[e] = geo_w * (1.0 + conf)  # củng cố thêm
        else:
            # conf=0 -> không đổi (geo_w); conf=1 -> ép mạnh về geo_w*contradicted_penalty
            out[e] = geo_w * (1.0 - conf * (1.0 - contradicted_penalty))
    return out
