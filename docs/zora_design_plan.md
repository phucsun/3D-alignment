# ZORA: Kế hoạch thiết kế từng bước

Nhánh làm việc: `zora-vlm-graph` (KHÔNG commit vào `main`).

Bối cảnh đầy đủ: [`docs/multi_space_alignment_plan.md`](multi_space_alignment_plan.md) mục 12
(CONTRIBUTIONS.md) — baseline "geometry-only, không prior tôpô" đã verify xong, dùng làm cột mốc
so sánh cho toàn bộ lộ trình dưới đây.

## Ánh xạ kiến trúc ZORA -> code hiện có / cần xây mới

| Thành phần ZORA | Trạng thái | Ghi chú |
|---|---|---|
| Layer 2 (geometric nodes: normal, distance, u_min/u_max) | **Đã có** | `gravity_align._wall_footprint`, `robust_align.build_opening` |
| `L_portal` (0-DOF, center distance + normal ngược) | **Đã có (numpy)** | `gravity_locked_similarity`/`_single`, `_refine_normal_lock` |
| Exclusivity (1 cửa = 1 quan hệ) | **Đã có (rời rạc, MWIS)** | `resolve_opening_conflict_graph` - baseline "no-VLM" |
| `L_containment` (interval containment) | **Chưa có bản differentiable** | Có sẵn logic numpy tương đương (`_wall_footprint` bbox), cần port |
| `L_robustness` + GNC | **Chưa có** | Đã thử IRLS tự viết - thất bại; cần GNC chuẩn |
| Layer 1 (VLM sketch/NL -> soft graph) | **Chưa có gì** | Hoàn toàn mới, rủi ro/effort lớn nhất |
| Benchmark (ScanNet/Matterport3D split + corrupted prior) | **Chưa có** | Cần dữ liệu ngoài, quy mô lớn |

## Lộ trình 6 giai đoạn (làm tuần tự, xác nhận trước mỗi giai đoạn)

### Giai đoạn 1 — Đóng gói baseline "no-VLM" thành ablation chính thức
Bọc `resolve_opening_conflict_graph` thành 1 interface rõ ràng đại diện cho "Layer 2 + exclusivity,
KHÔNG có Layer 1" — đây là điểm neo (anchor) để so sánh mọi cải tiến sau này. Không cần code mới
nhiều, chủ yếu là tổ chức lại cho rõ vai trò trong kiến trúc ZORA.

### Giai đoạn 2 — Differentiable geometry solver (PyTorch), thay thế phần numpy/scipy
Viết lại `L_portal` + `L_containment` bằng PyTorch, tham số hoá SE(3)/Sim(3) qua Lie algebra
(rotvec + log-scale, autograd-compatible - có thể dùng thư viện có sẵn như PyPose/Theseus thay vì tự
viết lại phép toán manifold). Verify: nghiệm PyTorch phải khớp (trong sai số số học) với nghiệm
numpy đã verify đúng ở `align_gravity_camera`/`resolve_opening_conflict_graph` trên CÙNG bộ dữ liệu
thật - không được phép regression.

### Giai đoạn 3 — GNC thay IRLS
Tích hợp Graduated Non-Convexity (dùng implementation có sẵn nếu có, không tự viết lại thuật toán
gốc) làm `L_robustness`. Verify trên đúng bộ 4-không-gian đã biết kết quả tốt nhất (3/3 đúng, 1 cạnh
giả còn sót) - kỳ vọng KHÔNG tệ hơn baseline Giai đoạn 1, lý tưởng là tách nốt được cạnh giả còn sót.

### Giai đoạn 4 — Layer 1: VLM sinh soft graph từ sketch/mô tả ngôn ngữ
Thiết kế mới hoàn toàn - cần quyết định: định dạng input (ảnh sketch tay vẽ? mô tả text tự do?),
chọn VLM (model có sẵn hay gọi API), schema output (room node + edge + confidence). Đây là giai đoạn
rủi ro cao nhất, cần bàn kỹ trước khi code.

### Giai đoạn 5 — Ghép Layer 1 + Layer 2 thành 1 pipeline end-to-end
Soft edge từ VLM làm prior trọng số ban đầu cho GNC/differentiable solver, thay vì trọng số tự tính
từ `n_matches/residual` như baseline.

### Giai đoạn 6 — Benchmark + thực nghiệm
Cắt ScanNet/Matterport3D thành phòng rời rạc, sinh prior tổng hợp với nhiễu 10/30/50%, đo RTE/RRE +
Edge Rejection Rate + Sliding Violation % - so sánh với baseline Giai đoạn 1 (0% prior) và literature
(SGHR, GeoTransformer, PointDSC).

---

**Đề xuất bắt đầu từ Giai đoạn 1** (rẻ, không rủi ro, chỉ tổ chức lại code đã có) rồi sang Giai đoạn 2
(differentiable solver) - đây là nền tảng bắt buộc trước khi làm Layer 1. Xác nhận trước khi tôi code.
