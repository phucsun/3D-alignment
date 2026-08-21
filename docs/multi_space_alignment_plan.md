# Kế hoạch: Tổng quát hoá RGA cho N không gian (đa không gian, topology bất kỳ)

Tài liệu con, phân tích chi tiết từng công trình liên quan: [`related_work/`](related_work/).

## 1. Định vị bài toán — khoảng trống thật sự

Từ 3 nhánh literature đã research (xem 4 file trong `related_work/`), không công trình nào giải quyết
đúng bài toán: **N không gian được tái dựng 3D ĐỘC LẬP (video multi-view riêng, scale có thể khác
nhau, KHÔNG có trajectory/odometry liên tục nối chúng lại), gần như không overlap hình học, chỉ nối
với nhau qua các vật thể semantic đặc thù (door/window) quan sát được từ cả 2 phía**.

| Nhánh | Đại diện | Giả định không khớp với bài toán của mình |
|---|---|---|
| Multi-way point cloud registration | [SGHR (CVPR 2023)](related_work/SGHR_Robust_Multiview_Point_Cloud_Registration_with_Reliable_Pose_Graph_Initialization_and_History_Reweighting.md) | Cần overlap hình học rộng giữa các scan |
| Semantic scene-graph SLAM | [Multi S-Graphs (2024)](related_work/Multi_S-Graphs_An_Efficient_Distributed_Semantic-Relational_Collaborative_SLAM.md) | Cần trajectory/odometry liên tục xuyên qua room |
| Multi-robot/multi-session SLAM merge | [DOOR-SLAM (2020)](related_work/DOOR-SLAM_Distributed_Online_and_Outlier_Resilient_SLAM_for_Robotic_Teams.md), [maplab 2.0 (2022)](related_work/maplab_2.0_A_Modular_and_Multi-Modal_Mapping_Framework.md) | Loop closure dựa trên place-recognition/overlap, không phải qua 1 vật thể hẹp làm cầu nối duy nhất |

**Kết luận novelty**: bài toán của mình là 1 trường hợp mới — "sparse portal-only registration" — cần
mượn **công cụ toán** (pose-graph/motion averaging từ nhánh 1, PCM outlier-rejection từ nhánh 3, kiến
trúc phân tầng từ nhánh 2) nhưng **thay hoàn toàn** cơ chế tạo cạnh (edge construction) đầu vào bằng
engine đã có và đã verify của mình (`gravity_align.align_gravity_camera`).

## 2. Trạng thái hiện tại (baseline cần vượt qua)

`align_rooms_to_hub` (`alignment.py:558-628`):
- ✅ Tổng quát theo **số lượng** room (N bất kỳ).
- ❌ Chỉ 1 topology: star/hub (1 hub trung tâm, N room không biên giới lẫn nhau).
- ❌ Gọi `align_indoor_outdoor` (SGD/Hungarian cũ), **chưa** dùng `align_gravity_camera` (gravity-locked, đã verify tốt hơn — mục 11 `CONTRIBUTIONS.md`).
- ❌ Không xử lý được vòng lặp kín (loop closure) — A-B, B-C, C-A đều biên giới thì không mô hình hoá được quan hệ C-A.
- ❌ Không lan truyền/tối ưu độ bất định (uncertainty) toàn cục — mỗi (room, wall) là 1 lần Kabsch độc lập, không có bước "phân bố lại sai số" như pose-graph thật.
- ❌ Chỉ test bằng dữ liệu tổng hợp (`_make_detection` dựng tay) — **chưa từng chạy thật** trên point cloud 3 không gian có sẵn (`310_indoor`+`server`+`h_server`).

## 3. Kế hoạch — 5 giai đoạn

### Giai đoạn 0 — Baseline thực nghiệm (làm ngay, rẻ, không rủi ro)

Chạy `align_rooms_to_hub` thật (không phải synthetic) trên `310_indoor`+`server`+`h_server` — bộ 3
không gian thật duy nhất đã có sẵn point cloud + camera pose. Mục tiêu: có 1 con số baseline thật để
so sánh với mọi cải tiến sau này, và xác nhận lại (hoặc phủ nhận) giả thuyết ban đầu rằng cách làm cũ
"nhìn tưởng đúng nhưng globally wrong" trên đúng bộ dữ liệu đã truyền cảm hứng viết hàm này.

**Deliverable**: 1 script `scripts/align_hub_pipeline.py` (theo đúng khuôn `align_gravity_pipeline.py`
đã có) + số liệu residual/status thật, không synthetic.

### Giai đoạn 1 — Edge construction: thay engine, KHÔNG cần học gì mới

Thay vì dùng `align_indoor_outdoor` bên trong `align_rooms_to_hub`, mọi cặp không gian ứng viên
`(i, j)` chạy qua `align_gravity_camera(clusters_i, clusters_j, cam_i, cam_j)` — cho ra trực tiếp:

- `status` (CONFIDENT/AMBIGUOUS/NO_SOLUTION) → quyết định cạnh `(i,j)` có tồn tại trong đồ thị không (thay cho "ước lượng overlap bằng neural network" của SGHR — ở đây có sẵn tín hiệu vật lý, không cần train).
- `opening_residual`, `cam_dominance`, `grav_dot`, `up_consistency_A/B` → làm **trọng số/độ tin cậy** của cạnh cho bước tối ưu toàn cục ở Giai đoạn 3 (khắc phục đúng giới hạn (e) đã nêu — hiện tại các con số này bị bỏ phí, chỉ dùng để in log).

**Chi phí**: O(N²) lần gọi `align_gravity_camera` cho N không gian (chấp nhận được vì N thực tế nhỏ,
< 10 không gian/toà nhà) — không cần lọc ứng viên trước bằng heuristic gì thêm ở quy mô này.

### Giai đoạn 2 — Loại cạnh sai bằng PCM (mượn từ DOOR-SLAM)

Sau Giai đoạn 1 có 1 đồ thị thưa (chỉ cạnh CONFIDENT). Trước khi đưa vào tối ưu toàn cục, áp dụng
**PCM (pairwise consistent measurement set maximization)** bản tập trung (centralized), không cần bản
phân tán của DOOR-SLAM: với mọi tập con cạnh tạo thành 1 chu trình (cycle) trong đồ thị, kiểm tra tích
các transform Sim(3) vòng quanh chu trình có gần `Identity` không (đây chính là "loop closure error" —
1 chu trình A→B→C→A đúng vật lý thì `T_AB · T_BC · T_CA ≈ I`). Loại cạnh nào liên tục xuất hiện trong
các chu trình lỗi cao — đây là bước **hoàn toàn không có** trong `align_rooms_to_hub` hiện tại (nó tin
tuyệt đối vào Hungarian, không tự kiểm tra chéo).

### Giai đoạn 3 — Tối ưu toàn cục: Sim(3) pose-graph / motion averaging (mượn từ SGHR + Multi S-Graphs)

Thay cơ chế "Hungarian gán room→wall rồi dừng" bằng 1 bước tối ưu toàn cục thật sự:

- Node: mỗi không gian có 1 pose tuyệt đối `(s_k, R_k, t_k) ∈ Sim(3)` (ẩn số cần giải, gắn 1 không
  gian làm gốc/reference).
- Edge: mỗi cạnh còn lại sau Giai đoạn 2 cho 1 ràng buộc `(s_ij, R_ij, t_ij)` (đã có sẵn từ
  `GravityAlignResult`), trọng số = nghịch đảo `opening_residual` (kết hợp `cam_dominance`).
- Giải bằng IRLS/Gauss-Newton trên đa tạp Sim(3) (mượn khung "history reweighting" của SGHR để robust
  hoá thêm 1 lớp nữa, độc lập với PCM ở Giai đoạn 2) — cho ra pose tuyệt đối nhất quán toàn cục cho
  MỌI không gian cùng lúc, tự động khép sai số vòng lặp nếu có (giải quyết đúng giới hạn (d)).
- **Trường hợp suy biến (degenerate check)**: với N=2 hoặc đúng topology star hiện tại, nghiệm bài
  toán tối ưu này phải quy về CHÍNH XÁC kết quả `align_gravity_camera`/`align_rooms_to_hub` hiện tại —
  đây là điều kiện bắt buộc phải verify trước khi tin bất kỳ kết quả N>2 nào (không được phép có
  regression trên các dataset 2-không-gian đã xác nhận đúng).

### Giai đoạn 4 — Biểu diễn phân tầng (tuỳ chọn, mượn từ Multi S-Graphs)

Nếu dữ liệu thật cho thấy có cấu trúc phân tầng tự nhiên (ví dụ nhiều tầng nhà, mỗi tầng nhiều
phòng), thêm 1 tầng node "Floor" phía trên các node "Space" — không bắt buộc cho bài toán 3-không-gian
hiện tại, nhưng là hướng mở rộng tự nhiên nếu có dữ liệu lớn hơn.

### Giai đoạn 5 — Verify thật

1. Synthetic trước (dựng tay 1 đồ thị có chu trình thật — điều `align_rooms_to_hub` không test được)
   để verify Giai đoạn 2+3 hoạt động đúng về mặt toán học trước khi đụng dữ liệu thật.
2. Real 3-không-gian (`310_indoor`+`server`+`h_server`, Giai đoạn 0 baseline) — so sánh trực tiếp
   trước/sau.
3. Nếu có điều kiện, thu thập thêm 1 bộ dữ liệu thật có **chu trình khép kín** (3+ không gian nối vòng
   tròn) — hiện chưa có trong `data/`, đây là dataset duy nhất thật sự chứng minh được giá trị của
   Giai đoạn 2-3 so với cách làm hub/star hiện tại (vì star topology không có chu trình để test).

## 4. Rủi ro & điểm cần quyết định trước khi code

- **PCM + Sim(3) averaging cần thư viện tối ưu phi tuyến** (Gauss-Newton trên đa tạp) — cân nhắc dùng
  `scipy.optimize.least_squares` tự viết residual (kiểm soát được, nhẹ, khớp phong cách code hiện tại
  toàn bộ tự viết bằng numpy) thay vì kéo thêm dependency lớn (GTSAM/g2o) — cần bạn xác nhận hướng nào.
- Giai đoạn 0-1 rẻ, an toàn, nên làm trước — Giai đoạn 2-3 mới là phần cần thiết kế + review kỹ trước
  khi code (đúng nguyên tắc luôn confirm trước khi đổi thuật toán).

## 5. Tóm tắt 1 câu cho phần Introduction/Related Work của bài báo

*"Trong khi multi-way point cloud registration giả định overlap hình học rộng (SGHR), và semantic
scene-graph SLAM/multi-session merging giả định trajectory liên tục hoặc place-recognition qua
overlap quan sát (Multi S-Graphs, DOOR-SLAM, maplab 2.0), bài toán ghép N không gian độc lập chỉ qua
1 vài vật thể cầu nối hẹp (door/window) chưa được giải quyết trực tiếp ở bất kỳ nhánh nào — RGA mở
rộng khung pose-graph/motion-averaging đã trưởng thành sang chế độ 'sparse portal-only edges', dùng
chính engine gravity-locked semantic-geometric đã verify làm cơ chế tạo cạnh, thay cho neural overlap
estimation hay place-recognition."*
