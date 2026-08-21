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

### Giai đoạn 0 — Baseline thực nghiệm ✅ ĐÃ CHẠY THẬT

Chạy `align_rooms_to_hub` thật (không phải synthetic) trên `310_indoor`+`server`+`h_server`
(script: `scripts/align_hub_pipeline.py`). Kết quả thật:

| Room | Hub opening gán | Matches | Residual |
|---|---|---|---|
| server | wall `[0,1,2]` | `(2,2),(3,1)` | 0.2891 |
| room_310 | wall `[3]` (lẻ, trivial) | `(0,0)` | 0.0 (không có ý nghĩa - 1 điểm luôn fit chính nó) |

Kiểm tra bằng mắt: **`room_310` bị ghép SAI** — đè vào đúng 1 cửa lẻ cuối hành lang không phải của
nó, vì thuật toán gán độc quyền ở cấp **room→CẢ NHÓM TƯỜNG**, không phải room→từng opening. Xác nhận
lại giả thuyết ban đầu: cách làm cũ "nhìn tưởng đúng nhưng globally wrong" — **đúng**, có bằng chứng
thật.

### Giai đoạn 1 — Edge construction: thay engine + sửa granularity ✅ ĐÃ CHẠY THẬT (BASELINE CHÍNH THỨC)

Thay `align_indoor_outdoor` bằng `align_gravity_camera` (script:
`scripts/align_hub_pipeline_gravity.py`). Lần chạy đầu (gán theo NHÓM TƯỜNG như Giai đoạn 0) vẫn tái
hiện đúng lỗi granularity trên. **Sửa granularity**: mỗi room chạy trực tiếp trên TOÀN BỘ hub (không
giới hạn subset theo tường trước), chỉ giải quyết tranh chấp ở cấp **OPENING-INDEX** khi 2 room thật
sự đòi dùng chung 1 opening cụ thể. Kết quả sau khi sửa — **đây là baseline được xác nhận đúng bằng
mắt, dùng làm mốc chính thức cho mọi cải tiến sau này**:

| Room | Matches (room_idx, hub_idx) | Status | Residual |
|---|---|---|---|
| server | `(4,3),(5,2)` | CONFIDENT | 0.0101 |
| room_310 | `(0,0),(1,1)` | AMBIGUOUS (2 cửa khá đối xứng, cảnh báo trung thực) | 0.0201 |

`server` matches **trùng khớp trực tiếp** với ground-truth đã biết trước ở mục 11 CONTRIBUTIONS.md
(`indoor[4]/[5]`), residual giảm 0.2891 → 0.0101 (~28 lần) so với Giai đoạn 0. `room_310` dùng đúng 2
opening hub còn thừa mà `server` không chạm tới — không còn tranh chấp/gán sai.

- `status` (CONFIDENT/AMBIGUOUS/NO_SOLUTION) → quyết định cạnh `(i,j)` có tồn tại trong đồ thị không (thay cho "ước lượng overlap bằng neural network" của SGHR — ở đây có sẵn tín hiệu vật lý, không cần train).
- `opening_residual`, `cam_dominance`, `grav_dot`, `up_consistency_A/B` → làm **trọng số/độ tin cậy** của cạnh cho bước tối ưu toàn cục ở Giai đoạn 3 (khắc phục đúng giới hạn (e) đã nêu — hiện tại các con số này bị bỏ phí, chỉ dùng để in log).

**Chi phí**: O(N²) lần gọi `align_gravity_camera` cho N không gian (chấp nhận được vì N thực tế nhỏ,
< 10 không gian/toà nhà) — không cần lọc ứng viên trước bằng heuristic gì thêm ở quy mô này.

### Giai đoạn 2 — Loại cạnh sai bằng PCM (mượn từ DOOR-SLAM) — ⚠️ ĐÃ THỬ, THẤT BẠI, TẠM DỪNG

**Thử nghiệm thật** (4 không gian: thêm `data/connecting_space` vào bộ 3 đã có, script
`scripts/align_4space_pipeline.py` + `scripts/multi_space_cycle_consistency.py`, không giả định
trước topology): chạy `align_gravity_camera` cho MỌI cặp trong 4 không gian (6 cặp) — phát hiện
**3 cạnh false positive thật** (`server↔room_310`, `server↔connecting_space`, `room_310↔connecting_space`
đều ra CONFIDENT/AMBIGUOUS với residual thấp, dù ground-truth xác nhận bằng mắt là chúng KHÔNG hề kề
nhau — chỉ `h_server` mới là không gian chung, `connecting_space` nối vào đúng cửa cuối hành lang).

Thử fix bằng loop-consistency (PCM): liệt kê toàn bộ 16 cây khung khả dĩ nối 4 không gian bằng 6 cạnh
đo được, chọn cây có tổng loop-error thấp nhất. **Kết quả: THẤT BẠI** — cây ground-truth (star vào
`h_server`) xếp hạng 9/16, không phải thấp nhất; cây có tổng loop-error thấp nhất lại là 1 cây SAI
(dùng 2 trong 3 cạnh giả).

**Nguyên nhân đã phân tích bằng tổ hợp** (không chỉ dựa vào 1 lần chạy): với đúng cấu trúc "1 hub +
N cạnh giả nối chéo giữa các room" (đồ thị đầy đủ Kn), MỌI cạnh — kể cả cạnh THẬT — chỉ xuất hiện
trong các tam giác có lẫn ít nhất 1 cạnh giả (3 cạnh hub tạo thành 1 cây, không có tam giác nào toàn
cạnh thật để tự xác nhận lẫn nhau). Việc "quy trách nhiệm cho cạnh nào" trong 1 tam giác lỗi là mơ hồ
toán học thật sự khi topology thật là star thuần — **không đủ dư thừa (redundancy) hình học để tự
phân biệt, không phải lỗi code**.

**Kết luận (phát hiện âm tính, có giá trị cho phần Discussion/Future Work của bài báo)**: loop-
consistency/PCM thuần geometric KHÔNG đủ khi topology thật là star (không có chu trình toàn-cạnh-thật
nào để đối chiếu). Cần 1 trong 2 hướng bổ sung (chưa code, tạm dừng theo quyết định 2026-08-21):

1. **Adjacency prior từ metadata** (mượn ý tưởng `region` đã có ở mục 4b CONTRIBUTIONS.md, mở rộng
   lên cấp đồ thị N-không-gian) — biết trước cặp nào ĐÁNG thử, không chạy mù toàn bộ C(N,2).
2. **Thêm redundancy hình học thật** (thu thêm dữ liệu tạo chu trình toàn-cạnh-thật) để PCM có đủ tín
   hiệu tự xác nhận.

**Quyết định hiện tại**: dùng kết quả Giai đoạn 1 (3 không gian, opening-level, đã verify đúng) làm
baseline chính thức; việc tổng quát hoá phát hiện topology tự động cho N>3 không gian (không biết
trước hub) tạm gác lại, chờ hướng giải quyết mới.

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
