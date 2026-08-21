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

### Giai đoạn 4 — Layer 1: VLM sinh soft graph từ mô tả ngôn ngữ ✅ ĐÃ XONG (kiến trúc Decoupled Pipeline)

**Input đã chọn (2026-08-21)**: mô tả text tự do (không phải sketch 2D) - rẻ, không cần thu thập ảnh,
dùng được LLM có sẵn qua prompt. **Chưa gọi LLM/VLM thật** - `parse_topology_from_text` nhận
`mock_response` mô phỏng đầu ra 1 LLM parse đúng (quyết định 2026-08-21: xây interface + verify tích
hợp logic trước, tránh tốn API khi logic downstream chưa chắc đúng - đã đúng, vì phát hiện được nhiều
vấn đề kiến trúc trước khi cần gọi API thật).

**Kiến trúc cuối cùng (Decoupled Pipeline, đề xuất trực tiếp bởi người dùng, đã verify hoàn hảo)**:
tách `n_matches/residual` (MWIS, quyết định RỜI RẠC) và GNC-TLS (tinh chỉnh LIÊN TỤC) thành 2 vai trò
khác biệt, KHÔNG luân phiên (chạy MWIS đúng 1 lần lúc khởi tạo, GNC là bộ tiêu thụ 1 chiều phía sau):

1. **`resolve_opening_conflict_graph`** (Giai đoạn 1, KHÔNG đổi thuật toán lõi) + `prior_weights`
   (MỚI, tham số thêm vào - nhân trọng số Layer 1 vào ngay bước MWIS, không chỉ dùng cho GNC downstream)
   → giải quyết TRIỆT ĐỂ cả xung đột "dùng chung cửa" (trong-building) LẪN trùng hợp xuyên-building
   (nhờ prior).
2. **`solve_pose_graph_gnc`** (Giai đoạn 3, không đổi) nhận tập cạnh ĐÃ SẠCH từ bước 1 → tối ưu pose +
   loại nốt nhiễu hình học toàn cục còn sót (loại xung đột MWIS không xử lý được: cạnh KHÔNG dùng
   chung cửa với ai nhưng vẫn sai).

**3 hướng đã thử và LOẠI BỎ trước khi tới kiến trúc đúng** (ghi đầy đủ, không giấu thất bại - xem
`docs/multi_space_alignment_plan.md`):
- Layer 1 chỉ áp dụng SAU MWIS (chỉ làm trọng số cho GNC) — cải thiện 1 phần (2/3 → giữ đúng ít cạnh
  giả hơn) nhưng KHÔNG đủ, vì MWIS đã chốt sai từ trước khi GNC kịp chạy.
- Alternating Optimization (xen kẽ GNC-outer/MWIS mỗi vòng, dùng GNC-weight làm tiêu chí MWIS) — bất
  ổn qua 3 cấu hình khác nhau trên CHÍNH N=4 (case đơn giản Giai đoạn 1 đã giải hoàn hảo không-GNC) -
  GNC-weight cho quyết định MWIS TỆ HƠN heuristic `n_matches/residual` cũ ở 1 tình huống cạnh tranh cụ
  thể (2 candidate cùng cần đúng 1 cửa vật lý, 1 đúng 1 trùng hợp).
- GNC-hội-tụ-hoàn-toàn-rồi-1-lần-MWIS với candidate TĨNH (tính 1 lần trên full opening, không lặp lại
  theo Giai đoạn 1) — thất bại vì candidate cho `h_server-connecting_space` SAI ngay từ đầu (thiếu cơ
  chế tính lại trên opening còn trống), không phải lỗi thứ tự GNC/MWIS.

**Kết quả verify cuối cùng** (kiến trúc Decoupled đúng, prior tại MWIS):

| Kịch bản | Không Layer 1 | Có Layer 1 (tại MWIS) |
|---|---|---|
| N=4 (server building, không nhiễu xuyên-building) | 3/3 đúng, 0 giả, 0 bỏ sót (Layer 1 không cần thiết ở đây) | (không test riêng - N=4 không có building khác để cần prior) |
| N=6 (server+Q1, có nhiễu xuyên-building) | 1/4 đúng, giữ NHẦM 2 cạnh xuyên-building, bỏ sót 3 cạnh thật | **4/4 đúng, 0 giả, 0 bỏ sót — HOÀN HẢO** |

### Giai đoạn 5 — Ghép Layer 1 + Layer 2 thành 1 pipeline end-to-end ✅ ĐÃ HOÀN THÀNH CÙNG GIAI ĐOẠN 4
(Kiến trúc Decoupled Pipeline ở Giai đoạn 4 ĐÃ LÀ pipeline end-to-end Layer 1 + Layer 2 hoàn chỉnh -
không cần giai đoạn riêng nữa, gộp lại với Giai đoạn 4 ở trên.)

### Giai đoạn 6 — Benchmark + thực nghiệm
Cắt ScanNet/Matterport3D thành phòng rời rạc, sinh prior tổng hợp với nhiễu 10/30/50%, đo RTE/RRE +
Edge Rejection Rate + Sliding Violation % - so sánh với baseline Giai đoạn 1 (0% prior) và literature
(SGHR, GeoTransformer, PointDSC).

---

**Đề xuất bắt đầu từ Giai đoạn 1** (rẻ, không rủi ro, chỉ tổ chức lại code đã có) rồi sang Giai đoạn 2
(differentiable solver) - đây là nền tảng bắt buộc trước khi làm Layer 1. Xác nhận trước khi tôi code.
