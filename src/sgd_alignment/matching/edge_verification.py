"""3 tín hiệu độc lập để chấp nhận/loại 1 cạnh (candidate edge) trong bài
toán ghép N không gian - docs/multi_space_alignment_plan.md, hướng mới sau
khi Sim(3) pose-graph (multi_space_graph.py) chỉ tách được 2/3, rồi 3/3
cạnh thật nhưng còn giữ nhầm 1 cạnh giả (room_310-connecting_space, không
đủ dư thừa đồ thị để tự phân biệt bằng residual pose-graph một mình).

1. Sim(3) pose-graph residual (multi_space_graph.py, đã có).
2. **p-value thống kê** (mới, mượn ý tưởng Selective Inference của
   CTRL-RANSAC arXiv:2410.15133): xây phân phối null "residual đạt được
   HOÀN TOÀN NGẪU NHIÊN khi 2 không gian KHÔNG hề liên quan" bằng cách chạy
   `align_gravity_camera` giữa không gian ứng viên và nhiều không gian
   THẬT SỰ không liên quan (khác toà nhà hoàn toàn, đã có sẵn trong
   project: Q1/Q2/chua_thay) - so residual quan sát được với phân phối này.
3. **Va chạm thể tích** (mới, mượn từ literature room-layout/BIM: honeybee,
   3D IoU collision check) - sau khi áp transform ứng viên, 2 phòng kề
   nhau thật phải gần như KHÔNG chồng lấn thể tích bên trong (chỉ chung 1
   mặt tường mỏng).

Đây là bản THỬ NGHIỆM/prototype (chưa merge vào `alignment.py`).
"""
from __future__ import annotations

import numpy as np


def p_value_from_null(observed_residual: float, null_residuals: list[float]) -> float:
    """P(null <= observed) - nhỏ nghĩa là hiếm khi nào 2 không gian KHÔNG
    liên quan lại tình cờ khớp tốt (hoặc tốt hơn) mức quan sát được -> bằng
    chứng ủng hộ cạnh này là THẬT. Không có mẫu null hợp lệ -> trả về 1.0
    (không có bằng chứng gì, không tự tin claim gì)."""
    if not null_residuals:
        return 1.0
    arr = np.asarray(null_residuals, dtype=float)
    return float(np.mean(arr <= observed_residual))


def voxel_occupancy_overlap(points_a: np.ndarray, points_b: np.ndarray, voxel_size: float = 0.15) -> float:
    """Tỉ lệ overlap thể tích giữa 2 point cloud ĐÃ ở CHUNG 1 hệ toạ độ
    (điểm A đã transform vào hệ B). Voxel hoá cả 2 (lưới `voxel_size` mét),
    trả về `|voxel_A ∩ voxel_B| / min(|voxel_A|, |voxel_B|)` - 2 phòng kề
    nhau thật (chỉ chung 1 mặt tường) phải cho tỉ lệ này RẤT THẤP; 2 phòng
    "đâm xuyên" nhau (transform sai) cho tỉ lệ cao.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    voxels_a = {tuple(v) for v in np.floor(points_a / voxel_size).astype(np.int64)}
    voxels_b = {tuple(v) for v in np.floor(points_b / voxel_size).astype(np.int64)}
    inter = len(voxels_a & voxels_b)
    denom = min(len(voxels_a), len(voxels_b))
    return float(inter / denom) if denom > 0 else 0.0
