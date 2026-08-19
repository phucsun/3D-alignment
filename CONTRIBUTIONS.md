# RGA (Robust Geometric-semantic Alignment) — Tổng kết Contribution

So sánh với: Yang et al., "Indoor-Outdoor Point Cloud Alignment Using Semantic-Geometric
Descriptor", Remote Sens. 2022, 14, 5119.

Chú thích trạng thái dùng trong tài liệu này:
- ✅ **Đã cài đặt + đã kiểm chứng** trên dữ liệu thật (test tự động + regression trên nhiều dataset)
- 🔶 **Đã cài đặt, đã kiểm chứng một phần** (infra đúng, không phá kết quả cũ, nhưng chưa đủ bằng chứng để claim "tốt hơn")
- 📝 **Mới ở mức thiết kế/đề xuất** — chưa code, hoặc bị chặn bởi thiếu dữ liệu

Kế thừa duy nhất từ bài gốc: cách biểu diễn một đối tượng qua quan hệ hình học với các đối
tượng lân cận trong cùng không gian (SGDU), và phép gán hai tầng (Hungarian nội bộ + Hungarian
toàn cục) làm cơ chế ghép cặp cơ sở. Mọi phần dưới đây, trừ khi ghi chú khác, là thiết kế/cài đặt
độc lập, không có trong bài gốc.

---

## 1. Xử lý trường hợp scene chỉ có 1 vật thể ✅

**File:** `src/sgd_alignment/matching/sgd.py` (`_intrinsic_cost`, `sgdu_distance(..., use_intrinsic_fallback=True)`)

Hạn chế cấu trúc của bài gốc: một object không có neighbor có SGDU rỗng (G^i = ∅) → không bao
giờ ghép được, dù detect đúng 100%. Xác nhận trên dữ liệu thật: outdoor chỉ detect 1 cửa, không
thể match dù cửa đó đúng là cửa cần tìm.

Giải pháp: nhận ra Kabsch/Umeyama SVD chỉ cần **một** correspondence đúng để khôi phục đầy đủ
6(+1 scale) bậc tự do — vì một khối hộp cửa/sổ thật có 3 chiều phân biệt (width, height,
thickness), không đối xứng quay. Bổ sung `_intrinsic_cost`: so khớp category + tỉ lệ khung hình
bất biến-tỉ-lệ (width/height), chỉ kích hoạt khi object không có neighbor nào để so sánh quan hệ
— không áp dụng tràn lan đè lên matching quan hệ bình thường.

**Kiểm chứng:** `tests/test_sgd_matching.py`, verified trên dữ liệu `home` (1 cửa outdoor).

---

## 2. Kiểm tra độ mơ hồ khi ghép cặp (ambiguity check) ✅

**File:** `src/sgd_alignment/matching/sgd.py` (`_ambiguity_ratio`, `match_sgds(..., use_ambiguity_check=True)`)

Bài gốc không có cơ chế này. Kiểu Lowe's-ratio-test: sau khi Hungarian chọn 1 cặp là tối ưu,
kiểm tra ứng viên tốt-nhì trong cùng hàng/cột có gần bằng ứng viên tốt-nhất không (`ratio =
chosen_cost / next_best_cost`). Nếu quá gần (gần 1), loại match đó thay vì chấp nhận một lựa
chọn không dứt khoát — chống lại các đối tượng "distractor" cùng category không có counterpart
thật ở phía bên kia (ví dụ cửa thoát hiểm phụ chỉ có ở outdoor).

**Kiểm chứng:** `tests/test_sgd_matching.py`.

---

## 3. Lọc detection suy biến (degenerate-detection filter) ✅

**File:** `src/sgd_alignment/matching/alignment.py` (`_filter_degenerate`, `align_indoor_outdoor(..., min_dimension=...)`)

Loại bỏ các detection có kích thước vật lý phi thực tế (ví dụ "cửa" rộng 3cm — lỗi detection,
không phải cửa thật) trước khi đưa vào matching. Cố tình dùng ngưỡng sanity-floor chung, **không**
dùng size-prior theo category (đã cân nhắc và loại bỏ hướng size-prior vì không thực tế khi áp
dụng cho nhiều loại công trình khác nhau).

**Kiểm chứng:** `tests/test_sgd_matching.py`, `tests/test_multiview_segmentation.py`.

---

## 4. Chống đối xứng kiến trúc — đóng góp mạnh nhất ✅

### 4a. RANSAC hypothesize-and-verify wall-consensus

**File:** `src/sgd_alignment/matching/alignment.py` (`ransac_match_with_wall_consensus`,
`align_indoor_outdoor(..., use_ransac_consensus=True)`)

Bài gốc chỉ dùng Hungarian thuần trên descriptor — thất bại có hệ thống khi tòa nhà có đối xứng
kiến trúc thật (ví dụ 2 cánh hành lang giống hệt nhau về hình học cục bộ). Giải pháp: RANSAC
minimal-sample (2-match) hypothesize-and-verify — sinh giả thuyết biến đổi (R,t) từ mẫu tối
thiểu, xác thực trên TOÀN BỘ ứng viên khác bằng **đồng thời** residual vị trí VÀ độ lệch hướng
pháp tuyến tường sau biến đổi, chọn giả thuyết có đồng thuận cao nhất.

Lưu ý lịch sử quan trọng (để giải thích khi bị hỏi "sao không làm đơn giản hơn"): phiên bản đầu
tiên (`refine_matches_with_geometric_consensus` + `normal_angle_threshold`) dùng chính (R,t) hiện
tại để "tự xác nhận" chính nó — một lỗi circular-validation, đã bị phát hiện sai trên dữ liệu
thật (khớp nhầm một detection suy biến rộng 0.03m) và revert hoàn toàn. RANSAC hiện tại được
thiết kế lại để không tự tham chiếu vòng.

**Kiểm chứng:** verified đúng trên `sc1_room2` (loại đúng 2 cặp đối xứng nhầm), `meeting_room`
(door-only rerun 2/2, residual 0.46cm), regression trên toàn bộ 6 dataset.

### 4b. Region-constrained matching

**File:** `src/sgd_alignment/common/types.py` (`Detection3D.region`), `src/sgd_alignment/matching/sgd.py` (`_regions_compatible`)

Giới hạn thông tin luận đã xác lập: nếu 2 cấu hình vật lý thực sự không thể phân biệt được từ dữ
liệu cảm biến sẵn có (đối xứng kiến trúc thật), không có thuật toán nào giải quyết được chỉ bằng
hình học — kể cả RANSAC consensus. Giải pháp: cơ chế ràng buộc cứng `region: str | None`, cho
phép đưa tri thức con người (nhãn "cánh A/cánh B" gán lúc thu thập dữ liệu) vào bài toán. Mặc
định `None` (tương thích ngược 100%, không ảnh hưởng dữ liệu cũ chưa gán nhãn). Đặt nền cho mở
rộng tương lai: ghép nối đa không gian (multi-region registration, tương tự pose-graph SLAM),
không chỉ là tie-breaker cho bài toán 2-không-gian hiện tại.

**Kiểm chứng:** test tổng hợp `test_align_indoor_outdoor_region_label_resolves_symmetric_ambiguity`
— chứng minh trực tiếp: không có `region`, thuật toán chọn nhầm giữa 2 cửa đối xứng hình học
giống hệt nhau; có `region`, luôn chọn đúng, R/t khớp chính xác ground-truth. Regression 0 lỗi
trên toàn bộ dữ liệu thật (vì `region` mặc định `None` ở mọi nơi hiện có).

### 4c. Ý tưởng đã cân nhắc và bác bỏ có lý do (đáng ghi vào Discussion)

Đã cân nhắc dùng visual/CLIP embedding của ảnh cửa để phân biệt 2 phía đối xứng, nhưng bác bỏ vì
lý do cấu trúc: mặt trong và mặt ngoài của MỘT cửa thật thường khác nhau về màu sắc/vật liệu cụ
thể — so sánh appearance cross-scene có thể đẩy sai cả những cặp match ĐÚNG, không chỉ vô dụng
với cặp đối xứng. Xem mục 8 để biết hướng tích hợp có nguyên tắc hơn (đề xuất, chưa cài đặt).

---

## 5. Chuẩn hóa trọng số λ có nguyên tắc — thay vì chọn tay 🔶

**File:** `src/sgd_alignment/matching/weight_calibration.py`

Bài gốc không công bố cách chọn 7 trọng số λ trong hàm cost (Equation 2); code trước đây của
chính project này cũng dùng hằng số chọn tay. Đóng góp: nhận ra `cost = λ·e` là hàm **tuyến
tính** theo trọng số (với neighbor-assignment cố định) → biến việc chọn λ thành bài toán margin
optimization (cùng họ với structured-SVM/LMNN) — ép cost(cặp đúng đã xác nhận) thấp hơn cost(mọi
cặp sai cùng scene) một khoảng margin, giải bằng Nelder-Mead (derivative-free, vì Hungarian rời
rạc không khả vi).

**Đánh giá trung thực (leave-one-scene-out CV trên 24 correspondence, 6 scene):**

| Scene giữ lại để test | Margin ratio — trọng số tay | Margin ratio — trọng số học (LOSO) |
|---|---|---|
| sc1_room2 | 1.489 | 1.489 (không đổi) |
| sc2_room2 | 0.217 | 0.261 (tệ hơn) |
| sc2_room5 | 0.389 | 0.499 (tệ hơn) |
| vkist | 0.115 | 0.129 (tệ hơn) |
| video1 | nan | nan (không đủ ứng viên để so sánh) |
| meeting_room | 0.264 | 0.203 (tốt hơn) |

Kết luận trung thực: 24 mẫu / 7 tham số tự do là quá ít để LOSO khái quát hóa tốt hơn hằng số
chọn tay hiện tại (1/4 fold tốt hơn, 3/4 tệ hơn hoặc không đổi) — **không claim "tốt hơn"**.
Điểm tích cực đã xác nhận: trọng số calibrate trên toàn bộ 6 scene vẫn cho **đúng y hệt** match
set như trọng số mặc định trên cả 6 dataset (không phá bất kỳ kết quả đã xác nhận đúng nào).
Ratio > 1 của sc1_room2 ở cả 2 cách là một xác nhận độc lập, nhất quán với mục 4a: pure SGD cost
không đủ phân biệt cặp đối xứng ở đó — đúng là lý do RANSAC wall-consensus cần thiết.

Trọng số học được **không** được bật mặc định trong pipeline (chủ động, theo yêu cầu) — vẫn dùng
`PairWeights()` hằng số cũ. Đóng góp ở đây là **phương pháp luận + infra tái lập được**, không
phải một bộ trọng số cụ thể tốt hơn.

---

## 6. Ước lượng trục "up" không giả định trước ✅

**File:** `src/sgd_alignment/detection/plane_fitting.py`

Bài gốc giả định hệ tọa độ đã biết trước. RGA ước lượng "up" hoàn toàn từ hình học:

- `estimate_up_vector_manhattan`: RANSAC đa mặt phẳng, nhóm theo hướng normal, chọn trục có
  spatial extent nhỏ nhất (cải tiến so với heuristic "nhiều inlier nhất" trước đó — heuristic cũ
  thất bại trên corridor dài, xác nhận trên dữ liệu thật: corridor cao 9.1m nhưng trục tường vẫn
  có nhiều điểm RANSAC inlier hơn trục floor/ceiling).
- `estimate_up_vector_cross_scene`: giải quyết hạn chế còn lại của heuristic trên (không gian
  hẹp/dài vi phạm giả định "extent nhỏ nhất = up") bằng cách khai thác việc bài toán luôn có 2
  scene — so khớp TOÀN BỘ candidate-axis list giữa indoor/outdoor (không chỉ top-1 mỗi bên), chọn
  cặp đồng thuận cao nhất. Chủ động raise lỗi thay vì đoán khi không scene nào đủ tin cậy.

**Bug quan trọng đã phát hiện và fix:** RANSAC plane-fitting của Open3D không deterministic giữa
các lần chạy process riêng biệt (do RNG toàn cục đa luồng) dù đã seed trước mỗi lần gọi — xác
nhận: cùng 1 file, cùng code, 2 lần chạy cho ra candidate-axis list khác nhau. Đã viết lại RANSAC
plane-fitting hoàn toàn bằng numpy với RNG cục bộ (không phụ thuộc Open3D), có vector hóa batch
để bù tốc độ.

**Kiểm chứng:** 2 lần chạy process độc lập trên `home_phuc` cho kết quả **giống hệt bit-for-bit**;
regression 0 lỗi trên toàn bộ 6 dataset thật (match count giữ nguyên 100%); 42/42 test pass.

---

## 7. Framework đa nguồn dữ liệu thống nhất ✅

**File:** `src/sgd_alignment/detection/manual_segmentation.py` (nhánh LiDAR/CloudCompare),
`src/sgd_alignment/detection/multiview_segmentation.py` (nhánh video đơn mắt)

Một biểu diễn chung (`Detection3D`) nhận đầu vào từ 2 nguồn khác bản chất:

- **LiDAR/máy quét 3D**: point cloud scale mét thật, phân đoạn thủ công (CloudCompare) — độ
  chính xác cao, không cần ước lượng scale.
- **Video đơn mắt**: tái tạo 3D không có scale tuyệt đối (feed-forward depth model), phân đoạn 2D
  tự động đa góc nhìn, backproject + merge instance đa view.

Cơ chế hỗ trợ nhánh video: `estimate_scale` (Umeyama, khôi phục uniform scale factor) +
`normalize_distance` (chuẩn hóa khoảng cách SGD theo median inter-opening spacing của chính
scene đó) — để 2 luồng dữ liệu hoàn toàn khác bản chất vẫn tương thích với cùng một bộ trọng số
ghép cặp.

**Kiểm chứng:** cả 2 nhánh đã chạy end-to-end trên dữ liệu thật (4 dataset CloudCompare, 2
dataset video/DA3).

---

## 8. Visual-style descriptor extension (CLIP similarity) 🔶

**File:** `src/sgd_alignment/detection/clip_visual.py`, `src/sgd_alignment/matching/weight_calibration.py`
(`CalibrationExample.visual_similarity`, `margin_loss_with_visual`, `calibrate_visual_weight`,
`mean_margin_ratio_with_visual`)

Thành phần cost bổ sung dựa trên visual-style similarity (CLIP ViT-B/32), tích hợp vào đúng cơ
chế hiệu chỉnh trọng số ở mục 5: độ tin cậy của tín hiệu (`visual_weight`) được **học từ dữ
liệu** bằng margin-optimization, không gán tay — nếu tín hiệu nhiễu, quy trình tự đưa trọng số về
gần 0.

**Giới hạn dữ liệu thật (quan trọng):** pipeline không lưu liên kết `Detection3D` ↔ ảnh/crop gốc
sau bước backproject 2D→3D, và 4/6 dataset (toàn bộ nhánh CloudCompare) vốn không có ảnh. Trong
số dữ liệu hiện có, **chỉ duy nhất `meeting_room` (door-only rerun) còn giữ ảnh review có overlay
segmentation (viền xanh)** đủ để crop chính xác — `video1` chỉ còn frame gốc không có overlay/bbox
nên không dùng được. Việc gán ảnh đúng cho đúng detection ở `meeting_room` được làm bằng cách đối
chiếu thủ công vị trí 3D của cụm (center) với log số view + kiểm tra trực quan từng ảnh, không
phải suy luận máy móc — vì pipeline gốc không log lại mapping detection↔view.

**Kết quả thật đã chạy** (crop tự động qua color-mask viền xanh → CLIP ViT-B/32 → cosine similarity),
trên 2 correspondence đã xác nhận đúng của `meeting_room`:

| Cặp đúng | Cosine similarity | Xếp hạng so với các ứng viên sai cùng scene |
|---|---|---|
| indoor[1] ↔ outdoor[1] | 0.9304 | **#1/4** (others: 0.9228, 0.9152, 0.8644) |
| indoor[2] ↔ outdoor[0] | 0.9638 | **#1/4** (others: 0.9547, 0.9255, 0.8804) |

Tín hiệu thô đúng hướng ở cả 2 trường hợp sẵn có (n=2, quá nhỏ để kết luận tổng quát).

**Nhưng bài test margin-calibration trên chính `meeting_room` không kết luận được gì thêm** —
lý do minh bạch: matching hình học thuần túy của `meeting_room` vốn đã KHÔNG mơ hồ (margin giữa
cost đúng và cost sai gần nhất ~0.79, margin_target=0.05) → hinge loss = 0 với MỌI giá trị
`visual_weight`, nên `calibrate_visual_weight` không có gradient nào để học — không có "vấn đề"
nào để CLIP giải quyết ở dataset này. Dataset thực sự có mơ hồ hình học thật (`sc1_room2`, margin
ratio >1, mục 5) thì lại không có ảnh để test. Đây là giới hạn dữ liệu, không phải giới hạn
phương pháp — **không claim CLIP cải thiện margin**, chỉ claim tín hiệu thô có xu hướng đúng
hướng trên 2 mẫu ít ỏi.

**Kết luận trung thực để dùng khi viết bài:** hạ tầng đã cài đặt đầy đủ, chạy được thật trên dữ
liệu thật (không phải mô phỏng), nhưng cỡ mẫu (n=2, chỉ 1 dataset) quá nhỏ để khẳng định hiệu quả
— nên trình bày như một cơ chế đã triển khai + bằng chứng sơ bộ tích cực, không phải một kết quả
đã được validate đầy đủ. Cần dataset mới có giữ ảnh gốc + camera pose cho cả 2 phía, và có tình
huống đối xứng/mơ hồ hình học thật, mới đánh giá được đúng giá trị của hướng này.

---

## 9. Hướng tường ra ngoài dùng camera pose thay mật độ điểm ✅

**File:** `src/sgd_alignment/detection/opening_geometry.py` (`orient_walls_outward`),
`src/sgd_alignment/detection/manual_segmentation.py` (`camera_positions` param)

Bug thật phát hiện trên dữ liệu `server`/`home_phuc`: heuristic mật độ điểm 2 bên tường (nhiều
điểm hơn = phía trong) có thể gần như 50/50 cho một bức tường cụ thể (`server`: 0.2996 vs 0.2256,
margin 0.074) dù các tường khác trong CÙNG scene rất dứt khoát (margin 0.78–0.94) — không có tín
hiệu nào cho biết quyết định đó gần như tung đồng xu, khiến indoor/outdoor bị ghép lộn cùng 1 phía
tường mà residual/cost không hề bất thường.

Khi có `results.npz` (camera pose) cho cả 2 phía: vị trí camera so với mặt phẳng tường là **sự
thật hình học cứng**, không phải thống kê — máy quay đứng ở phía nào thì chắc chắn ở phía đó, dùng
làm tín hiệu chính thay mật độ điểm (mật độ vẫn tính để đối chiếu chéo).

**Kiểm chứng bằng phép thử signed-distance độc lập** (không dùng residual/cost, vốn đã được xác
nhận KHÔNG phát hiện được lỗi này): biến đổi centroid 1 scene bằng (R, t) đã tính, so dấu khoảng
cách có dấu tới mặt phẳng cửa đã ghép với dấu của chính centroid scene còn lại — cùng dấu = 2
không gian bị lồng cùng phía (sai), khác dấu = đúng 2 phía đối diện tường. Xác nhận trên `q1`,
`q2`, `server` (đều có camera pose): trước fix ít nhất 1/3 bộ SAI (cùng phía), sau fix cả 3 đều
ĐÚNG (khác phía) — không cần fix riêng cho bộ nào.

**Kiểm chứng:** regression 0 lỗi trên toàn bộ dataset CloudCompare cũ (`camera_positions=None` giữ
nguyên hành vi cũ 100%); `tests/test_sgd_matching.py`.

---

## 10. Bộ mô tả tỉ lệ rộng/cao riêng từng vật thể (aspect_ratio_weight) 🔶

**File:** `src/sgd_alignment/matching/sgd.py` (`PairWeights.aspect_ratio_weight`, `_aspect_ratio_diff`)

Ý tưởng: mô tả quan hệ gốc (SGDU — khoảng cách/góc tới vật thể lân cận) không hề dùng kích
thước/tỉ lệ TUYỆT ĐỐI của chính vật thể đang so khớp. Khi 2 bức tường có khoảng cách giữa các cửa
tình cờ gần giống nhau, về lý thuyết matching có thể chọn nhầm tường dù kích thước cửa khác nhau
rõ rệt — thêm tùy chọn `aspect_ratio_weight` (mặc định `0.0`, tắt) để cộng thêm chênh lệch tỉ lệ
rộng/cao bất biến-tỉ-lệ vào cost quan hệ đã tính, chỉ cộng khi cost quan hệ vẫn hữu hạn (không bao
giờ "cứu" lại một cặp đã bị loại vì sai category/ngưỡng).

**Đã tự kiểm tra lại và RÚT LẠI claim ban đầu:** lần đầu tưởng đã xác nhận `weight=1.0` sửa lỗi
chọn nhầm tường trên `server` (tường sai cost 0.2277 → tường đúng), nhưng đó là do lúc test còn
thiếu `estimate_scale`/`normalize_distance` cho dữ liệu này (dữ liệu `server` tái dựng từ DA3,
giống `q1`/`q2`, KHÔNG phải LiDAR scale mét thật — cần 2 cờ này, xem mục 9). Sau khi bật đúng 2 cờ
đó (bắt buộc cho mọi dữ liệu DA3, không phải tính năng riêng của mục này), tường đúng đã được chọn
kể cả khi `aspect_ratio_weight=0.0` — đã kiểm tra trực tiếp cả 2 giá trị `0.0` và `1.0` cho ra
CÙNG kết quả. Vậy nguyên nhân lỗi ban đầu là thiếu chuẩn hóa scale, không phải thiếu tín hiệu
aspect-ratio — cơ chế này chưa có bằng chứng thật nào chứng minh nó cần thiết.

**Giữ lại vì:** hạ tầng đúng, không đổi hành vi khi tắt (mặc định `0.0`), 0 regression trên toàn
bộ dataset cũ, 32/32 test pass — là một cơ chế phòng ngừa hợp lý (2 tường có khoảng cách object
GẦN GIỐNG nhau NHƯNG kích thước khác hẳn là tình huống có thể xảy ra ở dataset khác), nhưng
**không** trình bày như một kết quả đã kiểm chứng khi viết bài — chỉ nên nhắc là cơ chế đã cài đặt
sẵn, chưa có dữ liệu thật nào cần đến nó.

---

## 11. Ước lượng gravity từ ma trận xoay camera + pipeline gravity-locked cho dữ liệu có camera pose ✅

**File:** `src/sgd_alignment/detection/plane_fitting.py` (`estimate_up_vector_from_camera_rotation`),
`src/sgd_alignment/matching/gravity_align.py`, `src/sgd_alignment/matching/robust_align.py`
(merge có chọn lọc từ nhánh thử nghiệm của cộng sự — QuangMinhPham), `scripts/align_gravity_pipeline.py`

**Ước lượng "lên" tốt hơn:** thay vì PCA trên vị trí camera (`estimate_up_vector_from_camera_trajectory`,
cần nhiều vị trí trải rộng, suy biến khi ít view), lấy trung bình hướng "lên" trực tiếp từ CHÍNH ma
trận xoay mỗi khung hình (`up = -R[1,:]`) — ổn định hơn (hoạt động cả với 1 frame), có sẵn chỉ số
tin cậy `up_consistency`. Đã trở thành mặc định cho toàn bộ pipeline (`multiview_pipeline.py`,
`compare_baseline_vs_rga.py`) — cải thiện residual trên `q1` (0.0396→0.0239) và **tự sửa luôn** bug
tường-hốc-lõm ở `chua_thay` (mục kế tiếp) mà không cần thêm cơ chế nào khác.

**Pipeline gravity-locked (`gravity_align.py`/`robust_align.py`):** thay vì dò tường toàn cảnh
(`extract_wall_planes`) để xác định pháp tuyến, tính pháp tuyến **riêng từng cụm điểm cửa** (PCA cục
bộ, sign-invariant), khóa gravity qua ma trận xoay camera, và dùng vị trí camera để xác nhận 2 phía
đối diện tường. Đã chứng minh **thắng rõ rệt** so với pipeline gốc trên `chua_thay` (kiến trúc chùa,
cửa cánh không chuẩn) — pipeline gốc bị bug tường-hốc-lõm ở cửa sổ có độ sâu (RANSAC tách 1 mặt
tường thật thành nhiều mặt phẳng ứng viên gần nhau, sai hướng); pipeline gravity không phụ thuộc dò
tường toàn cảnh nên né hoàn toàn.

**3 bug thật phát hiện và sửa trong lúc merge** (dữ liệu `server`, đã biết ground-truth từ trước —
`indoor[4]/[5]` mới đúng tường, không phải `indoor[2]/[3]`):
1. Giả thuyết nhiều-cửa (`_add_multi`) chạy **trộn lẫn category** (A có cả window+door, B chỉ có
   door) — mọi tổ hợp đủ độ dài đều dính window, bị loại hết → rơi hẳn về nhánh 1-cửa (luôn "khớp
   hoàn hảo" giả tạo vì 1 điểm luôn fit chính nó). Sửa: nhóm theo category trước khi tạo tổ hợp.
2. Chỉ thử đúng 1 kích thước tổ hợp (bằng `min(len_a, len_b)`) — ép dùng hết toàn bộ cửa 1 bên, dễ
   trộn lẫn 2 tường khác nhau. Sửa: thử mọi kích thước từ 2 đến min.
3. Xếp hạng candidate chỉ dựa **residual trung bình** — 1 cặp rác (gần ngưỡng loại) trộn vào vẫn
   "qua cửa" nếu các cặp còn lại đủ tốt để kéo trung bình xuống. Sửa: xếp theo SỐ LƯỢNG khớp trước
   (không bỏ sót bằng chứng thật), thêm ngưỡng lọc theo cặp TỆ NHẤT (không chỉ trung bình) để chặn
   cặp rác trà trộn.

**Kiểm chứng:** cả `server` và `chua_thay` đều CONFIDENT + đúng ground-truth cùng lúc sau khi sửa cả
3 bug (trước đó sửa bug 1-2 riêng lẻ vẫn làm cái này đúng cái kia sai). `q1` CONFIDENT đúng, `q2` và
`310_indoor+h_server` bị flag AMBIGUOUS trung thực (dữ liệu tự nó kém ổn định/đối xứng thật, không
phải bug). 35/35 test cũ vẫn pass.

**Giới hạn phạm vi:** chỉ áp dụng cho dữ liệu CloudCompare có camera pose (`results.npz`) — dataset
LiDAR thuần (`sc1/sc2_room*`, `vkist`) không có camera pose nên vẫn dùng pipeline gốc
(`align_indoor_outdoor`), không đổi. Luồng auto-detect (`video1`, `meeting_room`, `Q815`) cũng chưa
chuyển sang pipeline này (cần cụm điểm thô, hiện chỉ lưu `Detection3D` đã trừu tượng hóa).

---

## Bảng tổng hợp dataset đã regression-test (sau toàn bộ thay đổi C1–C7)

| Dataset | Nguồn | Match | Residual |
|---|---|---|---|
| sc1_room2 | CloudCompare | 7/7 | 0.03–2.96cm (2 outlier đã biết) |
| sc2_room2 | CloudCompare | 5/5 | 0.5–3.1cm |
| sc2_room5 | CloudCompare | 5/5 | 2.1–5.7cm |
| vkist | CloudCompare | 3/3 | 3.2–8.9cm |
| video1 | Video/DA3 | 2/2 | 1.75cm |
| meeting_room (door-only) | Video/DA3 | 2/2 | 0.46cm |

`home`/`home_phuc` chủ động loại khỏi bảng đánh giá — dữ liệu chưa đủ tốt để khẳng định đúng/sai
của thuật toán (không phải do lỗi thuật toán).

42/42 unit test pass.

---

## Baseline (bài gốc) vs RGA — bằng chứng định lượng trực tiếp

**File:** `scripts/compare_baseline_vs_rga.py` — chạy tự động cả 2 config (baseline = Hungarian
thuần, không RANSAC consensus/ambiguity check/region/intrinsic fallback; RGA = đầy đủ) trên cùng
1 bộ detection cho mỗi dataset, dễ mở rộng thêm dataset mới (chỉ cần thêm 1 `DatasetEntry`).

6 dataset mới bổ sung (2026-08-13, cùng loại CloudCompare/LiDAR đã verify trước đó): `sc1_room1`,
`sc2_room1`, `sc2_room3`, `sc2_room4`, `q1`, `q2`.

| Dataset | Baseline | RGA | Ý nghĩa |
|---|---|---|---|
| sc1_room2 | 7/7, **sai 2 cặp** (residual ~2.97–2.99m, đối xứng nhầm) | 7/7 đúng hết (max 5cm) | RANSAC wall-consensus sửa đối xứng 2 chiều |
| sc2_room4 (mới) | 4/4, **sai xoay vòng 3 object** (residual tới 1.47m) | 4/4 đúng hết (max 3cm) | RANSAC wall-consensus sửa được cả nhầm lẫn phức tạp hơn (xoay vòng, không chỉ đổi chỗ đôi) |
| meeting_room | 3/3, **sai 1 cặp** (residual 0.53–1.12m) | 2/2 đúng (0.46cm) | ambiguity check + RANSAC loại đúng match giả |
| q1 (mới) | **lỗi** (0 match — baseline không có intrinsic fallback, không match nổi scene chỉ 1 object) | **1/1, residual = 0.0** | bằng chứng thật đầu tiên cho C2 trên dữ liệu hoàn toàn mới, không phải dữ liệu đã biết trước |
| sc1_room1, sc2_room1, sc2_room3 (mới) | đúng, khớp RGA | đúng | không có gì để robust hóa — baseline = RGA, đúng như kỳ vọng |
| sc2_room2, sc2_room5, vkist, video1 | đúng, khớp RGA | đúng | không đổi so với trước |
| q2 (mới) | lỗi | lỗi | **lỗi dữ liệu thật**: `Q2_outdoor` chưa được gán nhãn field cửa/sổ nào trong CloudCompare — không phải lỗi thuật toán, cần thu thập/gán nhãn lại |

**2 sửa lỗi tương thích dữ liệu mới phát hiện khi mở rộng dataset** (không đổi hành vi trên dữ
liệu cũ, đã regression-test):
- `manual_segmentation._field_category`: chỉ nhận diện field tên `scalar_cua_ra_vao`/`scalar_cua_so_N`; dữ liệu mới (`q1`, `q2`) dùng quy ước ngắn hơn `scalar_Cua_N`/`scalar_cua_N` (không hậu tố, viết hoa khác) — bị bỏ qua âm thầm trước khi sửa. Đã thêm nhận diện không phân biệt hoa/thường + fallback "cua" trần → door.
- Thông báo lỗi khi 0 match sai nội dung ("need >= 3" trong khi code chỉ chặn ở `< 1`) — đã sửa lại đúng "need >= 1".

**Về `data/meeting-room/meeting_room1`, `meeting_room2` và `data/vkist_308/video1_v1..v3`:** đã
đối chiếu (so ảnh frame đầu) — đây là **dữ liệu gốc (raw video + cache DA3) của `video1`/
`meeting_room` đã test rồi**, chỉ được tổ chức lại vào `data/`, không phải dataset mới cần chạy
lại pipeline detect.

**Về CLIP (mục 8):** không bật trong bất kỳ config baseline/RGA nào ở trên — theo đúng quyết định
giữ nó ở trạng thái tùy chọn, chỉ dùng khi có bằng chứng đủ tốt (hiện chưa có).
