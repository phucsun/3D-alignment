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
