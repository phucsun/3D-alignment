# SGD Alignment — Căn chỉnh đám mây điểm Indoor/Outdoor

Project hiện thực hoá và mở rộng phương pháp **Semantic-Geometric Descriptor (SGD)**
(Yang et al., *Indoor-Outdoor Point Cloud Alignment Using Semantic-Geometric
Descriptor*, Remote Sens. 2022) để tự động ghép nối 2 đám mây điểm 3D (quét từ
bên trong và bên ngoài cùng 1 toà nhà) dựa vào việc phát hiện và đối chiếu vị
trí các cửa ra vào / cửa sổ.

Có 2 nguồn đầu vào được hỗ trợ:

1. **Quét tay bằng CloudCompare** (`manual_segmentation.py`) — chọn điểm thủ công.
2. **Ảnh chụp đa góc (multi-view)** — dựng 3D bằng **COLMAP** hoặc **DA3 (Depth
   Anything 3)**, rồi tự động phát hiện cửa/sổ bằng Grounding DINO + SAM2.

Tài liệu này hướng dẫn chạy luồng thứ 2 (ảnh đa góc) — luồng chính, đầy đủ,
không cần notebook.

---

## 1. Cấu trúc project (phần liên quan đến chạy pipeline)

```
src/sgd_alignment/
├── common/types.py              # PointCloud, Detection3D, Plane
├── detection/
│   ├── plane_fitting.py         # RANSAC tìm mặt tường + trục "lên" (kể cả từ camera pose)
│   ├── opening_geometry.py      # đo width/height/normal của 1 cửa/sổ, hướng tường ra ngoài
│   ├── manual_segmentation.py   # đọc dữ liệu CloudCompare (quét tay)
│   ├── multiview_source.py      # đọc pose/depth từ COLMAP hoặc DA3
│   ├── multiview_segmentation.py# backproject 2D->3D, gộp instance
│   └── segment_2d.py            # Grounding DINO + SAM2
├── matching/
│   ├── sgd.py                   # SGD descriptor + Hungarian 2 tầng
│   └── alignment.py             # Kabsch/Umeyama (SVD) tính R, t, scale
└── pipelines/
    ├── multiview_pipeline.py    # CLI: ảnh -> Detection3D (.pkl)
    └── align_pipeline.py        # CLI: 2 x .pkl -> model 3D đã align

configs/
├── multiview_pipeline.example.yaml   # config mẫu cho bước detection
└── align_pipeline.example.yaml       # config mẫu cho bước align

scripts/
├── compare_baseline_vs_rga.py   # chạy + so sánh baseline gốc vs RGA đầy đủ trên mọi dataset đã đăng ký
└── pick_points.py                # tiện ích chọn điểm tay

tests/    # pytest, không cần dữ liệu thật
```

---

## 2. Cài đặt (chỉ làm 1 lần)

```bash
# Tạo môi trường
conda create -n sgd_alignment python=3.11 -y
conda activate sgd_alignment

# Cài project + toàn bộ dependency (bao gồm torch/transformers/ultralytics
# cho bước detect ảnh 2D)
cd "/Users/phuc/Phúc/VKSIT/3D-aligment"
pip install -e ".[segmentation]"
```

> Nếu chỉ cần chạy matching/test, không cần detect ảnh, có thể cài gọn hơn:
> `pip install -e .` (bỏ qua torch/transformers/ultralytics).

**Kiểm tra cài đặt thành công:**

```bash
python -c "import torch, transformers, ultralytics, cv2, yaml, plyfile; print('OK')"
python -m pytest tests/ -q
```

---

## 3. Chuẩn bị dữ liệu (làm ngoài project)

Project **không tự dựng 3D từ ảnh** — bạn cần tự chạy **COLMAP** hoặc **DA3**
trước, làm **riêng cho cả 2 phía** (indoor và outdoor):

| Backend | Cần chuẩn bị |
|---|---|
| **DA3** (khuyến nghị, đơn giản) | 1 file `results.npz` mỗi phía (đã chứa sẵn `depth`, `extrinsics`, `intrinsics`, `conf`, `image` — không cần ảnh rời) |
| **COLMAP** | `sparse/` (đã `image_undistorter`) + `images/` + `stereo/depth_maps/` (từ `patch_match_stereo`) + `fused.ply` (từ `stereo_fusion`) |

> ⚠️ **Lưu ý quan trọng**: 2 phía indoor/outdoor phải dựng 3D **riêng biệt**
> (2 lần chạy DA3/COLMAP độc lập) nếu ảnh 2 phía không chụp chung 1 lần. Điều
> này khiến 2 bên có thể lệch **tỉ lệ (scale)** với nhau — pipeline đã tự xử
> lý việc này ở bước align (xem mục 6), không cần đo tay.

---

## 4. Tạo file config cho từng phía

Copy config mẫu và sửa lại:

```bash
mkdir -p configs/<ten_scene>
cp configs/multiview_pipeline.example.yaml configs/<ten_scene>/indoor.yaml
cp configs/multiview_pipeline.example.yaml configs/<ten_scene>/outdoor.yaml
```

Mở từng file, sửa các trường sau (còn lại giữ mặc định):

```yaml
source:
  backend: da3
  npz_path: data/<ten_scene>/indoor/results.npz   # đường dẫn file npz

is_outdoor: false        # false cho indoor.yaml, true cho outdoor.yaml
output_dir: outputs/<ten_scene>
output_name: indoor      # "indoor" hoặc "outdoor" — dùng để đặt tên file output
```

**Bảng các trường config quan trọng:**

| Trường | Ý nghĩa | Khi nào cần đổi |
|---|---|---|
| `source.npz_path` | đường dẫn `results.npz` (DA3) | luôn cần |
| `is_outdoor` | phía này là ngoài hay trong nhà | luôn cần, phải đúng |
| `selected_views` | chọn tay 1 số view để segment, `[]` = lấy hết | để `[]` trừ khi muốn lọc bớt ảnh |
| `scale` | hệ số quy đổi đơn vị | để `1.0`, không cần chỉnh tay (xem mục 6) |
| `up` | ép trục "lên" thay vì tự đoán | chỉ điền khi thấy cửa bị đo sai kích thước (xem mục 7) |
| `output_dir`, `output_name` | nơi lưu + tên file kết quả | nên đặt riêng theo từng bộ dữ liệu |

---

## 5. Chạy bước Detection (làm cho từng phía)

```bash
sgd-multiview-pipeline --config configs/<ten_scene>/indoor.yaml
sgd-multiview-pipeline --config configs/<ten_scene>/outdoor.yaml
```

Lần chạy đầu tiên sẽ tự tải model Grounding DINO + SAM2 (mất vài phút, cần mạng).

**Kết quả sinh ra** trong `outputs/<ten_scene>/`:

| File | Nội dung |
|---|---|
| `<name>_detections.pkl` | danh sách cửa/sổ đã phát hiện (dùng cho bước 6) |
| `<name>_scene.ply` | point cloud toàn scene, giữ màu thật |
| `<name>_openings.ply` | point cloud + tô nổi các cửa/sổ đã phát hiện (kiểm tra bằng CloudCompare) |
| `<name>_detection.log` | log chi tiết từng bước |
| `<name>_view_XXXX_segmented.jpg` | ảnh overlay từng view — **nên mở xem** để chắc chắn detect đúng |

---

## 6. Chạy bước Matching + Align (ghép 2 phía)

```bash
cp configs/align_pipeline.example.yaml configs/<ten_scene>/align.yaml
```

Sửa file vừa tạo, trỏ đúng 4 file output của bước 5:

```yaml
indoor_detections: outputs/<ten_scene>/indoor_detections.pkl
indoor_scene_ply: outputs/<ten_scene>/indoor_scene.ply
outdoor_detections: outputs/<ten_scene>/outdoor_detections.pkl
outdoor_scene_ply: outputs/<ten_scene>/outdoor_scene.ply

output_dir: outputs/<ten_scene>
output_name: aligned

estimate_scale: true       # để true nếu 2 phía dựng 3D riêng biệt (mặc định)
normalize_distance: true   # để true cùng với estimate_scale
```

Chạy:

```bash
sgd-align-pipeline --config configs/<ten_scene>/align.yaml
```

**Kết quả:**

| File | Nội dung |
|---|---|
| `aligned.ply` | model 3D cuối cùng — indoor đã xoay/dịch/scale khớp vào outdoor, giữ màu thật |
| `aligned_matching.log` | số cặp khớp, residual từng cặp, scale phục hồi được |

Mở file cuối để xem kết quả:

```bash
open -a CloudCompare "outputs/<ten_scene>/aligned.ply"
```

**Đọc log thế nào là tốt?**
- `N pair(s) matched`: cần **≥ 3 cặp** để tính được phép xoay đáng tin cậy.
- `residual`: sai số từng cặp sau khi align — vài **cm đến vài chục cm** là bình
  thường (đơn vị theo scale đã phục hồi); nếu có cặp lệch hẳn (vài mét) so với
  các cặp còn lại → nhiều khả năng bị khớp nhầm (thường do nhiều cửa/sổ giống
  hệt nhau xếp đều 1 hàng — hạn chế đã biết của phương pháp).

---

## 7. Các lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Cửa/sổ ra kích thước vô lý (vd cao 0.05m) | Trục "lên" tự đoán sai — thường do video quay không đủ sàn/trần | Xem log dòng `up vector (auto-estimated): ...`; nếu 1 phía có up-vector đáng tin cậy, thử dùng `up:` trong config phía còn lại (xem `configs/video1_test/outdoor.yaml` làm ví dụ đã xử lý thật) |
| `only N opening(s) matched (need >= 3)` | 1 phía phát hiện quá ít cửa/sổ (do Grounding DINO bỏ sót, hoặc SGD cần ≥2 vật thể/scene mới có "quan hệ" để so sánh) | Kiểm tra ảnh overlay `*_segmented.jpg`; nếu model bỏ sót cửa thật, cần quay lại rõ hơn hoặc hạ `box_threshold` trong config |
| 2 cửa/sổ giống hệt nhau bị khớp nhầm (hoán đổi vị trí) | Nhiều vật thể cách đều đối xứng trên cùng 1 tường — hạn chế cố hữu của thuật toán descriptor tương đối | Hiện chưa có fix tự động; nhận biết qua residual bất thường cao ở 1-2 cặp |
| File `results.npz` không có key `"image"` | DA3 chạy không xuất ảnh kèm theo | Cần chạy lại DA3 với chế độ có xuất ảnh, hoặc cung cấp `scene_ply` từ nguồn khác |

---

## 8. Chạy test (kiểm tra code không bị hỏng sau khi sửa)

```bash
python -m pytest tests/ -q
```

32 test không cần dữ liệu thật bao phủ: đọc COLMAP/DA3, backproject 2D→3D, gộp
instance, SGD matching, Kabsch/Umeyama alignment (kể cả trường hợp lệch scale)
— chạy nhanh (<15s), không cần GPU. (10 test khác trong `test_plane_fitting.py`
/`test_point_cloud_utils.py`/`test_laser_image.py` cần bộ dataset ngoài
`Indoor-Outdoor-Point-Cloud-Dataset-main/` không đi kèm repo — bỏ qua nếu
không có bộ đó, không phải lỗi do code.)

---

## 9. Luồng dữ liệu CloudCompare (quét tay) + camera pose

Với dữ liệu quét tay bằng CloudCompare (mục 1, cách 1) mà **có kèm camera
pose** (file `results.npz` xuất từ DA3/COLMAP cho cả 2 phía — không bắt buộc,
nhưng nên có nếu có sẵn), `load_manual_segmentation` dùng camera pose làm tín
hiệu chính cho 2 việc, đáng tin cậy hơn hẳn so với chỉ dùng hình học đám mây
điểm:

- **Trục "lên"**: PCA trên quỹ đạo camera (trục có phương sai nhỏ nhất) thay
  vì RANSAC mặt tường — ổn định hơn với video quay ngắn/không đủ góc.
- **Hướng tường ra ngoài**: so vị trí camera với mặt phẳng tường (sự thật vật
  lý cứng — máy quay ở phía nào thì chắc chắn ở phía đó) thay vì chỉ đếm mật
  độ điểm 2 bên (có thể gần như 50/50 với 1 số tường cụ thể dù các tường khác
  trong cùng scene rất rõ ràng).

```python
from sgd_alignment.detection.multiview_source import DA3Source
from sgd_alignment.detection.manual_segmentation import load_manual_segmentation
import numpy as np

def camera_positions(npz_path):
    src = DA3Source(npz_path)
    return np.array([-src.pose(v).R.T @ src.pose(v).t for v in src.view_ids])

cam_in, cam_out = camera_positions("indoor/results.npz"), camera_positions("outdoor/results.npz")
indoor = load_manual_segmentation("indoor.ply", is_outdoor=False, camera_positions=cam_in)
outdoor = load_manual_segmentation("outdoor.ply", is_outdoor=True, camera_positions=cam_out)
```

Không có `results.npz`: bỏ tham số `camera_positions` — hành vi giữ nguyên
như trước (chỉ dùng hình học đám mây điểm), không cần đổi gì khác.

**`PairWeights.aspect_ratio_weight`** (mặc định `0.0`, tắt): về lý thuyết, khi
2 bức tường có khoảng cách giữa các cửa/sổ tình cờ gần giống nhau, chỉ riêng
đặc trưng quan hệ (khoảng cách/góc) có thể chọn nhầm tường — bật trọng số này
cộng thêm chênh lệch tỉ lệ rộng/cao (width/height, không phụ thuộc scale) của
chính từng cửa/sổ vào cost để giúp phân biệt. **Lưu ý:** cơ chế đã cài đặt và
test đơn vị đầy đủ, nhưng **chưa có bằng chứng dữ liệu thật nào cần đến nó**
— trường hợp `server` từng tưởng cần trọng số này thực ra là do thiếu
`estimate_scale`/`normalize_distance` (xem mục 9 và `CONTRIBUTIONS.md` mục
10); sau khi bật đúng 2 cờ đó, `aspect_ratio_weight=0.0` và `=1.0` cho kết
quả giống hệt nhau trên mọi dataset đã có.

```python
from sgd_alignment.matching.alignment import align_indoor_outdoor
from sgd_alignment.matching.sgd import PairWeights

result = align_indoor_outdoor(indoor, outdoor, weights=PairWeights(aspect_ratio_weight=1.0))
```

### Tái tạo kết quả đã có sẵn (mọi dataset đã đăng ký, 1 lệnh)

```bash
python scripts/compare_baseline_vs_rga.py
```

Chạy và in ra số cặp khớp + residual cho **baseline** (Hungarian thuần, đúng
bài gốc Yang et al.) và **RGA** (đầy đủ các cải tiến) trên mọi dataset đã
đăng ký trong `DATASETS` (đầu file `scripts/compare_baseline_vs_rga.py`) —
gồm cả các bộ CloudCompare thường và các bộ có camera pose (`q1`, `q2`,
`server`). Thêm dataset mới bằng cách thêm 1 `DatasetEntry` — dùng
`_load_cloudcompare` (không camera pose) hoặc
`_load_cloudcompare_with_camera_pose` (có `results.npz`).
