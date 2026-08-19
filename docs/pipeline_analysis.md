# Báo cáo phân tích chi tiết Pipeline căn chỉnh Point Cloud Indoor–Outdoor

> Tài liệu này đi qua **toàn bộ** luồng xử lý hiện tại của project, từ ảnh chụp + camera
> pose thô cho tới point cloud indoor/outdoor đã được ghép làm một. Mỗi bước gồm 3 phần:
> **(1) Vấn đề nó giải quyết** — tại sao bước này tồn tại, **(2) Code** — trích nguyên văn
> đoạn xử lý cùng đường dẫn file:dòng, **(3) Giải thích** — công thức/thuật toán hoạt động
> thế nào, kể cả các bước lọc lỗi nhỏ nhất (ngưỡng, NMS, percentile trim...).

## Mục lục

- [0. Sơ đồ tổng thể](#0-sơ-đồ-tổng-thể)
- [Giai đoạn A — Detection (ảnh → Detection3D)](#giai-đoạn-a--detection-ảnh--detection3d)
  - [A0. Orchestration](#a0-orchestration--multiview_pipelinepy)
  - [A1. Đọc nguồn: pose / depth / intrinsics](#a1-đọc-nguồn-pose--depth--intrinsics)
  - [A2. Segmentation 2D: Grounding DINO + SAM2](#a2-segmentation-2d-grounding-dino--sam2)
  - [A3. Backproject mask 2D → cụm điểm 3D](#a3-backproject-mask-2d--cụm-điểm-3d)
  - [A4. Gộp các lượt nhìn cùng 1 vật thể](#a4-gộp-các-lượt-nhìn-cùng-1-vật-thể)
  - [A5. Dựng scene point cloud toàn cảnh](#a5-dựng-scene-point-cloud-toàn-cảnh)
  - [A6. Ước lượng trục "up"](#a6-ước-lượng-trục-up)
  - [A7. Trích mặt phẳng tường (RANSAC)](#a7-trích-mặt-phẳng-tường-ransac)
  - [A8. Định hướng normal tường ra ngoài](#a8-định-hướng-normal-tường-ra-ngoài)
  - [A9. Đo kích thước từng opening → Detection3D](#a9-đo-kích-thước-từng-opening--detection3d)
  - [A10. Xuất kết quả](#a10-xuất-kết-quả)
- [Giai đoạn B — Matching & Alignment](#giai-đoạn-b--matching--alignment)
  - [B0. Orchestration](#b0-orchestration--align_pipelinepy)
  - [B1. SGD: Semantic-Geometric Descriptor](#b1-sgd-semantic-geometric-descriptor)
  - [B2. Hungarian 2 tầng](#b2-hungarian-2-tầng)
  - [B3. Kabsch / Umeyama — tính (R, t, scale)](#b3-kabsch--umeyama--tính-r-t-scale)
  - [B4. Các lớp lọc/tinh chỉnh sau matching](#b4-các-lớp-lọctinh-chỉnh-sau-matching)
  - [B5. Ghép nhiều phòng vào 1 hub](#b5-ghép-nhiều-phòng-vào-1-hub)
  - [B6. Nhánh thay thế "up-free": robust_align.py](#b6-nhánh-thay-thế-up-free-robust_alignpy)
  - [B7. Nhánh gravity-locked: gravity_align.py](#b7-nhánh-gravity-locked-gravity_alignpy)
- [Phụ lục: bảng tổng hợp mọi ngưỡng lọc lỗi](#phụ-lục-bảng-tổng-hợp-mọi-ngưỡng-lọc-lỗi)
- [Phụ lục: cấu trúc dữ liệu dùng chung](#phụ-lục-cấu-trúc-dữ-liệu-dùng-chung)

---

## 0. Sơ đồ tổng thể

```
Ảnh + camera pose (DA3 results.npz | COLMAP sparse+depth)
        │
        ▼
┌─────────────────────────── GIAI ĐOẠN A: DETECTION ───────────────────────────┐
│ A1 đọc pose/depth/K ──► A2 segment 2D (GroundingDINO+SAM2) ──► A3 backproject │
│   ──► A4 gộp theo view ──► A5 scene cloud toàn cảnh ──► A6 ước lượng up       │
│   ──► A7 trích tường (RANSAC) ──► A8 hướng normal ra ngoài                    │
│   ──► A9 đo width/height ──► Detection3D[]                                    │
└────────────────────────────────────────────────────────────────────────────┘
        │                                              │
        ▼ (indoor)                                     ▼ (outdoor)
┌─────────────────────── GIAI ĐOẠN B: MATCHING + ALIGNMENT ───────────────────┐
│  Nhánh chính (paper gốc):                                                    │
│   B1 SGD descriptor ──► B2 Hungarian 2 tầng ──► B3 Kabsch/Umeyama (R,t,scale) │
│      ──► B4 refine (geometric consensus / RANSAC wall consensus)             │
│                                                                                │
│  Nhánh thay thế (up-free, mới hơn):                                          │
│   B6 robust_align: up-free descriptor ──► triangle/2-opening hypotheses      │
│      ──► Hungarian-với-dummy scoring ──► refit lặp                           │
│   B7 gravity_align: khoá trục up từ camera + camera-side voting              │
│      (giải quyết ca chỉ match được đúng 1 tường — rank-1 degeneracy)         │
└────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   Point cloud indoor + outdoor đã ghép, 1 file .ply
```

---

## Giai đoạn A — Detection (ảnh → Detection3D)

### A0. Orchestration — `multiview_pipeline.py`

**Vấn đề giải quyết**: cần 1 điểm vào duy nhất chạy toàn bộ chuỗi A1→A9 theo đúng thứ tự,
đọc config YAML, ghi log ra cả console lẫn file (để chạy headless/không tương tác vẫn có
record đầy đủ), và xuất kết quả ra định dạng người review có thể mở lại (`.pkl` +
`.ply` để soi trực quan).

**Code**: [`multiview_pipeline.py:148-246`](../src/sgd_alignment/pipelines/multiview_pipeline.py#L148-L246) — hàm `run_pipeline`.

```python
def run_pipeline(config: MultiviewPipelineConfig, segmenter=None) -> list[Detection3D]:
    source = _build_source(config.source)                 # A1
    view_ids = _resolve_view_ids(source, config)
    ...
    for view_id in view_ids:
        image = source.image(view_id)
        detected = segmenter.segment(image)                # A2
        for inst in detected:
            instances.append(ViewInstance(view_id=view_id, label=inst.label,
                                           mask=inst.mask, conf=inst.conf))

    raw_instances = backproject_instances(source, instances, scale=config.scale,
                                           depth_range=tuple(config.depth_range),
                                           min_confidence=config.min_confidence)   # A3
    merged = merge_instances(raw_instances, merge_distance=config.merge_distance) # A4
    scene_points, scene_colors = source.build_scene_point_cloud()                 # A5
    scene_points = scene_points * config.scale

    camera_positions = np.array([-source.pose(v).R.T @ source.pose(v).t for v in view_ids])
    camera_rotations = np.array([source.pose(v).R for v in view_ids])

    up, consistency = estimate_up_vector_from_camera_rotation(camera_rotations)   # A6
    if consistency < MIN_UP_CONSISTENCY:
        up = estimate_up_vector_manhattan(PointCloud(points=scene_points))        # A6 fallback

    detections = detections_from_clusters(scene_points, merged, config.is_outdoor, up=up,
                                           trim_percentile=config.trim_percentile,
                                           camera_positions=camera_positions)      # A7,A8,A9
    ...
    return detections
```

**Giải thích**:
- `_build_source` chọn `DA3Source` hoặc `ColmapSource` tuỳ `config.source.backend`.
- `_resolve_view_ids`: nếu người dùng chỉ định `selected_views`/`selected_images` trong
  config thì chỉ xử lý đúng những view đó (đỡ tốn thời gian chạy Grounding DINO/SAM2 trên
  toàn bộ hàng trăm frame); nếu để trống thì lấy hết `source.view_ids`.
- `MIN_UP_CONSISTENCY = 0.90` là **bước lọc lỗi đầu tiên đáng chú ý**: nếu ước lượng up
  từ camera rotation không đủ đồng thuận giữa các frame (rig bị lắc/nghiêng khi quay), tự
  động rơi về phương pháp dự phòng dựa trên hình học point cloud, thay vì tin mù quáng vào
  1 con số không đáng tin.
- Mọi kết quả trung gian (`detections`, `merged` cluster tô màu, scene cloud) đều được
  ghi log + export `.ply`/`.pkl` — mục đích: pipeline chạy headless (không notebook) vẫn
  để lại đủ dấu vết để người review kiểm tra bằng mắt sau đó (`_export_review_ply`).

---

### A1. Đọc nguồn: pose / depth / intrinsics

**Vấn đề giải quyết**: DA3 và COLMAP xuất ra 2 định dạng hoàn toàn khác nhau (`.npz` vs
thư mục text/binary). Toàn bộ 8 bước phía sau cần 1 giao diện duy nhất, không quan tâm
nguồn gốc. Bước này còn thiết lập **convention camera** dùng xuyên suốt cả project — sai ở
đây thì mọi phép tính 3D phía sau sai theo mà không có cách nào phát hiện lại được.

**Code — convention & đọc DA3**: [`multiview_source.py:17-21`](../src/sgd_alignment/detection/multiview_source.py#L17-L21), [`multiview_source.py:308-326`](../src/sgd_alignment/detection/multiview_source.py#L308-L326)

```python
# Camera convention (world -> cam):
#   X_cam = R @ X_world + t
# depth là Z-depth (không phải ray-length) -> unproject là pinhole ngược thuần tuý.

class DA3Source:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self._depth = z["depth"].astype(np.float32)             # (N,H,W)
        extr = z["extrinsics"].astype(np.float64)                # (N,3,4) world->cam
        self._R = extr[:, :3, :3]
        self._t = extr[:, :3, 3]
        self._K = z["intrinsics"].astype(np.float64)              # (N,3,3)
        self._conf = z["conf"] if "conf" in z.files else None
        self._image = z["image"] if "image" in z.files else None
```

**Code — đọc COLMAP (thay thế)**: [`multiview_source.py:145-260`](../src/sgd_alignment/detection/multiview_source.py#L145-L260) — đọc `cameras.bin/.txt`, `images.bin/.txt` (quaternion → ma trận xoay qua `qvec2rotmat`), và depth map dạng binary riêng của COLMAP (`_read_colmap_depth_array`, header ASCII `width&height&channels&` rồi data float32 dạng column-major).

```python
def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])
```

**Giải thích chi tiết**:
- `R` (3×3): hướng xoay camera trong world. `t` (3,) **không phải** vị trí camera — vị trí
  thật (camera center) tính bằng `C = -R^T @ t` (dùng lại nhiều lần ở các bước sau: A6,
  A8, và toàn bộ `gravity_align.py`).
- `depth` là **Z-depth** (khoảng cách theo trục quang trục camera), không phải độ dài tia
  sáng — điều này quyết định công thức unproject ở A3 là 1 phép chia tuyến tính đơn giản
  (pinhole ngược), không phải chuẩn hoá theo norm.
- **Bất đối xứng cần lưu ý**: `conf` (confidence map) chỉ dùng khi build **scene point
  cloud toàn cảnh** (A5), **không** dùng khi backproject mask cửa/cửa sổ (A3) — ở đó chỉ
  lọc theo `depth_range`.
- Với COLMAP, `Camera.intrinsics_matrix()` ([`multiview_source.py:50-65`](../src/sgd_alignment/detection/multiview_source.py#L50-L65)) **chủ động raise lỗi** nếu camera model còn distortion (không phải `PINHOLE`/`SIMPLE_PINHOLE`) — đây là 1 bước lọc lỗi "cứng": thà crash sớm với thông báo rõ ràng ("chạy `colmap image_undistorter` trước") còn hơn âm thầm backproject sai vì bỏ qua distortion.

---

### A2. Segmentation 2D: Grounding DINO + SAM2

**Vấn đề giải quyết**: cần tìm "cửa/cửa sổ nằm ở đâu trong ảnh 2D" trước khi có thể đưa
vào không gian 3D. Đây là bài toán open-vocabulary (không train riêng 1 model chỉ nhận
diện cửa) nên dùng 2 model kết hợp: 1 cái định vị theo ngôn ngữ tự nhiên, 1 cái phân đoạn
chính xác viền.

**Code — khởi tạo & detect box**: [`segment_2d.py:52-127`](../src/sgd_alignment/detection/segment_2d.py#L52-L127)

```python
DEFAULT_TEXT_PROMPT = "door. window."

def _canonical_label(text_label: str) -> str | None:
    label = text_label.lower()
    if "door" in label: return "door"
    if "window" in label: return "window"
    return None

def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1]); area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

class DoorWindowSegmenter:
    def _detect_boxes(self, image_rgb):
        ... # Grounding DINO forward pass
        result = self._gdino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        # giữ box có canonical label khớp "door"/"window"
        ...
        # 1) NMS trong CÙNG 1 nhãn
        for label in np.unique(labels):
            idxs = np.where(labels == label)[0]
            nms_idx = cv2.dnn.NMSBoxes(xywh, scores[idxs].tolist(),
                                        score_threshold=0.0, nms_threshold=self.nms_iou_thres)
            keep.extend(idxs[np.array(nms_idx).flatten()])
        # 2) gộp box trùng GIỮA 2 nhãn, giữ cái confidence cao hơn
        order = np.argsort(scores)[::-1]
        for i in order:
            if all(_iou(boxes[i], boxes[j]) < self.cross_label_iou_thres for j in final):
                final.append(i)
        return boxes[final], scores[final], labels[final]
```

**Code — SAM2 sinh mask**: [`segment_2d.py:129-149`](../src/sgd_alignment/detection/segment_2d.py#L129-L149)

```python
def segment(self, image_rgb):
    boxes, scores, labels = self._detect_boxes(image_rgb)
    sam_result = self._sam.predict(image_bgr, bboxes=boxes, verbose=False)[0]
    masks = sam_result.masks.data.cpu().numpy() if sam_result.masks is not None else []
    for box, label, conf, mask in zip(boxes, labels, scores, masks):
        mask_bool = mask.astype(bool)
        if mask_bool.shape[:2] != (h, w):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        instances.append(DetectedInstance(label=str(label), conf=float(conf),
                                           box=box.tolist(), mask=mask_bool))
    return instances
```

**Giải thích các bước lọc lỗi**:
1. **`box_threshold=0.5` / `text_threshold=0.25`** — 2 ngưỡng riêng của Grounding DINO:
   cái đầu lọc độ tin cậy tổng thể của box, cái sau lọc độ khớp giữa box và từng token
   trong câu prompt.
2. **`_canonical_label`** — Grounding DINO có thể trả câu dài tuỳ ý (vì open-vocabulary),
   nên chỉ cần chứa substring `"door"`/`"window"` là quy về đúng 2 category cố định; nếu
   không khớp gì cả (`None`) thì box đó bị loại thẳng, không tới bước NMS.
3. **NMS trong cùng nhãn (`nms_iou_thres=0.7`)** — dùng `cv2.dnn.NMSBoxes` chuẩn: vì
   prompt có 2 câu con ("door." và "window."), đôi khi model sinh nhiều box gần trùng cho
   cùng 1 vật thể (các cách diễn đạt khác nhau khớp cùng 1 vùng ảnh) → cần lọc trùng trước.
4. **Lọc chéo nhãn (`cross_label_iou_thres=0.7`)** — 1 vật thể có thể vừa được gắn "door"
   vừa "window" (ví dụ cửa kính) → sắp theo confidence giảm dần, giữ box nào rồi loại mọi
   box khác chồng lấn IoU ≥ ngưỡng với nó bất kể khác nhãn.
5. **`INTER_NEAREST` khi resize mask** — vì mask là nhị phân (0/1), nội suy tuyến tính sẽ
   tạo giá trị lưng chừng vô nghĩa ở biên; nearest-neighbor giữ đúng tính chất nhị phân.

Output: `list[DetectedInstance(label, conf, box, mask)]`.

---

### A3. Backproject mask 2D → cụm điểm 3D

**Vấn đề giải quyết**: mask 2D chỉ là "vùng pixel nào là cửa" — vô nghĩa với bài toán ghép
3D nếu không biết pixel đó ở đâu trong không gian thực. Đây là bước đầu tiên thực sự sinh
ra toạ độ 3D.

**Code — công thức unproject**: [`multiview_source.py:94-124`](../src/sgd_alignment/detection/multiview_source.py#L94-L124)

```python
def backproject_mask_to_world(mask, depth, camera, pose,
                               depth_range=(0.05, 30.0)) -> np.ndarray:
    if mask.shape != depth.shape:
        raise ValueError(...)  # bắt buộc mask và depth phải cùng resolution TRƯỚC khi gọi

    valid = mask.astype(bool) & np.isfinite(depth) & (depth > depth_range[0]) & (depth < depth_range[1])
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3))

    K = camera.intrinsics_matrix()
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    d = depth[ys, xs]
    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    points_cam = np.stack([x_cam, y_cam, d], axis=1)

    R, t = pose.rotation_matrix(), pose.tvec
    # X_cam = R @ X_world + t  =>  X_world = R.T @ (X_cam - t)
    return (points_cam - t) @ R
```

**Code — gọi hàng loạt cho mọi instance + lọc confidence**: [`multiview_segmentation.py:87-125`](../src/sgd_alignment/detection/multiview_segmentation.py#L87-L125)

```python
def backproject_instances(source, instances, scale=1.0,
                           depth_range=DEFAULT_DEPTH_RANGE, min_confidence=0.0):
    raw_instances = []
    for inst in instances:
        if inst.conf < min_confidence:          # lọc detection 2D yếu TRƯỚC khi backproject
            continue
        depth = source.depth(inst.view_id)
        camera, pose = source.camera(inst.view_id), source.pose(inst.view_id)
        mask = inst.mask
        if mask.shape != depth.shape:            # khớp resolution nếu 2D detector chạy khác res
            mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        points = backproject_mask_to_world(mask, depth, camera, pose, depth_range=depth_range) * scale
        if len(points) == 0:
            continue
        raw_instances.append({"label": inst.label, "points": points})
    return raw_instances
```

**Giải thích công thức pinhole ngược (ví dụ số)**:

Pixel `(x=700, y=400)`, `depth=d=3.0m`, `fx=fy=900, cx=640, cy=360`:
```
x_cam = (700-640)/900 * 3.0 = 0.2 m
y_cam = (400-360)/900 * 3.0 = 0.133 m
z_cam = 3.0 m
```
Đây là định nghĩa pinhole: tia từ quang tâm qua pixel tạo góc `arctan((x-cx)/fx)` theo
trục ngang, nhân với depth (khoảng cách theo Z) ra được lệch ngang thực.

`(points_cam - t) @ R` là nghịch đảo đại số của `X_cam = R@X_world + t`:
```
X_cam = R@X_world + t
⟹ X_cam - t = R@X_world
⟹ R^T @ (X_cam - t) = X_world     (R trực giao nên R^T = R^{-1})
```
(numpy quy ước điểm là hàng nên `points @ R` ≡ `R^T @ point` cho từng điểm).

**Các bước lọc lỗi**:
- `depth_range=(0.05, 30.0)` — loại pixel có depth phi vật lý (quá gần ống kính = nhiễu,
  quá xa = model depth không đáng tin hoặc là bầu trời/hành lang xa).
- `np.isfinite(depth)` — model depth đôi khi trả `NaN`/`inf` ở vùng không dự đoán được.
- `min_confidence` — loại toàn bộ instance 2D yếu **trước khi tốn công backproject**, tránh
  1 mask nhiễu từ view chéo/mờ làm hỏng cụm điểm 3D sau này.
- `if len(points) == 0: continue` — 1 mask có thể trùng hoàn toàn với vùng depth không hợp
  lệ (ví dụ cửa kính phản chiếu khiến depth sensor fail toàn bộ) → bỏ qua an toàn thay vì
  tạo cụm điểm rỗng gây lỗi ở bước sau.

Output: `list[{"label": str, "points": (M,3)}]` — **1 phần tử cho mỗi lượt nhìn thấy** 1
opening (nếu 1 cửa được 3 ảnh nhìn thấy → 3 phần tử riêng, chưa gộp).

---

### A4. Gộp các lượt nhìn cùng 1 vật thể

**Vấn đề giải quyết**: 1 cửa/cửa sổ vật lý được nhìn thấy từ nhiều ảnh khác nhau sẽ tạo ra
nhiều cụm điểm rời rạc ở bước A3. Cần gộp chúng lại thành 1 cụm điểm duy nhất/vật thể —
càng nhiều view góp điểm, phép đo kích thước càng chính xác.

**Code**: [`multiview_segmentation.py:58-84`](../src/sgd_alignment/detection/multiview_segmentation.py#L58-L84)

```python
def merge_instances(raw_instances: list[dict], merge_distance: float = 0.6) -> list[dict]:
    clusters: list[dict] = []
    for inst in raw_instances:
        centroid = inst["points"].mean(axis=0)
        match = None
        for c in clusters:
            if c["label"] != inst["label"]:
                continue
            c_centroid = np.concatenate(c["points_list"]).mean(axis=0)
            if np.linalg.norm(c_centroid - centroid) < merge_distance:
                match = c
                break
        if match is None:
            clusters.append({"label": inst["label"], "points_list": [inst["points"]]})
        else:
            match["points_list"].append(inst["points"])
    return [{"label": c["label"], "points": np.concatenate(c["points_list"])} for c in clusters]
```

**Giải thích**: thuật toán **greedy 1-pass**, không phải clustering tối ưu toàn cục
(khác DBSCAN/k-means): duyệt tuần tự từng instance, so centroid với centroid "running mean"
của mọi cluster **cùng nhãn** đã tồn tại; nếu khoảng cách Euclid dưới `merge_distance`
(mặc định 0.6m) thì gộp, không thì mở cluster mới.

**Đánh đổi cần biết**: vì centroid của cluster là **running mean** (cập nhật dần mỗi lần
gộp thêm), kết quả **nhạy với thứ tự xử lý** khi nhiều centroid nằm sát ngưỡng
`merge_distance` — đây không phải bug, mà là đánh đổi đơn giản-hoá thay vì cài đặt thuật
toán clustering đầy đủ (chấp nhận được vì số lượng opening trong 1 scene thường nhỏ, vài
đến vài chục).

Output: `list[{"label": str, "points": (N,3)}]` — mỗi phần tử giờ là **1 opening thật**.

---

### A5. Dựng scene point cloud toàn cảnh

**Vấn đề giải quyết**: A6 (up-vector) và A7 (tường) cần nhìn thấy **toàn bộ hình học của
scene** (sàn, trần, tường), không chỉ vùng cửa/cửa sổ — nên cần unproject **mọi** pixel của
**mọi** view (không chỉ vùng mask), rồi gộp lại thành 1 point cloud lớn.

**Code (DA3)**: [`multiview_source.py:359-378`](../src/sgd_alignment/detection/multiview_source.py#L359-L378)

```python
def build_scene_point_cloud(self, conf_percentile: float = 40, max_points: int = 1_500_000):
    pts_list, col_list = [], []
    for i in self.view_ids:
        d = self._depth[i]
        conf_mask = None
        if self._conf is not None:
            thr = np.percentile(self._conf[i], conf_percentile)
            conf_mask = self._conf[i] >= thr
        pts, ys, xs = _unproject_depth_map(d, self._K[i], self._R[i], self._t[i], conf_mask)
        pts_list.append(pts)
        if self._image is not None:
            col_list.append(self._image[i][ys, xs])
    points = np.concatenate(pts_list).astype(np.float64)
    colors = np.concatenate(col_list).astype(np.uint8) if col_list else None
    if len(points) > max_points:
        idx = np.random.RandomState(0).choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None
    return points, colors
```

**Code (COLMAP, thay thế)**: [`multiview_source.py:270-289`](../src/sgd_alignment/detection/multiview_source.py#L270-L289) — đọc trực tiếp file point cloud đã fuse sẵn từ `colmap stereo_fusion` (không tự unproject lại từng depth map), vì bước fusion của COLMAP đã tự làm multi-view consistency filtering tốt hơn 1 phép unproject đơn giản ở đây.

**Giải thích các bước lọc lỗi**:
- **Lọc theo confidence percentile (`conf_percentile=40`)** — loại 40% pixel có độ tin cậy
  thấp nhất của **mỗi view riêng** trước khi gộp vào scene cloud. Đây là điểm khác A3: ở
  A3 (backproject mask cửa) không lọc theo `conf`, chỉ ở A5 (scene cloud dùng để dò
  tường/up) mới lọc — vì scene cloud cần "sạch" để RANSAC tường không bị nhiễu góp phiếu
  sai, còn mask cửa/cửa sổ thường đã nhỏ và conf-filter có thể xoá mất luôn cả cụm điểm.
- **Random subsample có seed cố định (`RandomState(0)`)** khi point cloud vượt 1.5 triệu
  điểm — giữ hiệu năng RANSAC/PCA ở các bước sau trong tầm kiểm soát, seed cố định để kết
  quả **tái lập được** (không đổi giữa 2 lần chạy).

---

### A6. Ước lượng trục "up"

**Vấn đề giải quyết** *(đã phân tích chi tiết ở lượt trao đổi trước — tóm tắt lại)*: raw
point cloud từ DA3/COLMAP không đảm bảo trục Z file trùng trọng lực thật. `up` sai sẽ lan
truyền lỗi qua **toàn bộ chuỗi phía sau**: phân loại sai tường/sàn (A7), sai `u_axis`/
`v_axis` → sai `width`/`height` mỗi opening (A9), sai luôn descriptor dùng để matching
(B1). Vì vậy project đầu tư nhiều phương pháp, có phân cấp ưu tiên rõ ràng.

**Code — phương pháp chính, từ camera rotation**: [`plane_fitting.py:226-263`](../src/sgd_alignment/detection/plane_fitting.py#L226-L263)

```python
def estimate_up_vector_from_camera_rotation(rotations: np.ndarray) -> tuple[np.ndarray, float]:
    # X_cam = R @ X_world + t: hàng 2 của R là trục "xuống" của ảnh trong world -> đảo dấu = "lên"
    up_per_frame = -rotations[:, 1, :]
    up_per_frame = up_per_frame / (np.linalg.norm(up_per_frame, axis=1, keepdims=True) + 1e-12)
    up = up_per_frame.mean(axis=0)
    up = up / (np.linalg.norm(up) + 1e-12)
    consistency = float(np.median(up_per_frame @ up))   # ~1.0 = mọi frame đồng thuận
    return up, consistency
```

**Code — phương pháp dự phòng, RANSAC "Manhattan world"**: [`plane_fitting.py:39-100`](../src/sgd_alignment/detection/plane_fitting.py#L39-L100) (RANSAC lõi) + [`plane_fitting.py:266-309`](../src/sgd_alignment/detection/plane_fitting.py#L266-L309)

```python
def _fit_plane_from_sample(sample: np.ndarray) -> np.ndarray | None:
    if len(sample) == 3:
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
    else:
        centroid = sample.mean(axis=0)
        _, _, vt = np.linalg.svd(sample - centroid, full_matrices=False)
        normal = vt[-1]
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None                          # mẫu suy biến (thẳng hàng/trùng điểm) -> bỏ qua
    normal = normal / norm
    d = -float(np.dot(normal, sample[0]))
    return np.array([normal[0], normal[1], normal[2], d])

def _candidate_up_axes(pc, distance_threshold, min_inliers, num_probe_planes,
                        ransac_n, num_iterations, same_axis_dot):
    planes = _extract_planes(pc, distance_threshold, min_inliers, num_probe_planes,
                              ransac_n, num_iterations)
    planes_sorted = sorted(planes, key=lambda p: -len(p.inlier_indices))
    axes = []
    for p in planes_sorted:
        # gom mặt phẳng có normal gần song song (kể cả ngược dấu) vào cùng 1 trục
        if not any(abs(float(np.dot(p.normal, axis))) >= same_axis_dot for axis in axes):
            axes.append(p.normal)
    # chọn trục có EXTENT (max-min chiếu lên trục) NHỎ NHẤT: phòng luôn thấp hơn là dài/rộng
    extents = [(pc.points @ axis).max() - (pc.points @ axis).min() for axis in axes]
    candidates = sorted(zip(axes, extents), key=lambda pair: pair[1])
    return [(axis / np.linalg.norm(axis), extent) for axis, extent in candidates]
```

**Code — chọn phương pháp trong pipeline**: [`multiview_pipeline.py:203-224`](../src/sgd_alignment/pipelines/multiview_pipeline.py#L203-L224)

```python
MIN_UP_CONSISTENCY = 0.90
up, consistency = estimate_up_vector_from_camera_rotation(camera_rotations)
if consistency >= MIN_UP_CONSISTENCY:
    ... # dùng luôn
else:
    up = estimate_up_vector_manhattan(PointCloud(points=scene_points))   # fallback
```

**Giải thích**:
1. **Phương pháp chính** dựa trên **rotation của từng camera frame** — không phải trung
   bình vị trí (trajectory PCA, cũng có trong file nhưng không còn dùng mặc định) mà là
   trung bình **hướng "lên"** suy ra trực tiếp từ ma trận xoay mỗi frame. Ưu điểm xác nhận
   qua thực nghiệm ("Confirmed on real data" trong docstring): hoạt động dù chỉ 1 frame,
   cho dấu (sign) xác định ngay (không mơ hồ như PCA — PCA chỉ cho trục, không biết chiều
   nào là "lên"), và có **chỉ số tin cậy thật** (`consistency`) đi kèm — không phải đoán mò.
2. **Ngưỡng lọc lỗi `MIN_UP_CONSISTENCY=0.90`**: nếu các frame không đồng thuận đủ (rig bị
   lắc/nghiêng, hoặc capture kiểu 360° khiến "lên" của mỗi frame trỏ lung tung), không tin
   mù quáng vào con số trung bình đó — rơi về phương pháp hình học dự phòng.
3. **Phương pháp dự phòng** (`estimate_up_vector_manhattan`) giả định thế giới kiểu
   "Manhattan" (sàn/trần + 2 hướng tường, vuông góc nhau): RANSAC lặp trích các mặt phẳng
   lớn nhất bất kể hướng (`_extract_planes` — lấy mẫu 3 điểm ngẫu nhiên, fit qua tích có
   hướng hoặc SVD, đếm inlier theo ngưỡng khoảng cách, loại inlier khỏi tập, lặp lại), gom
   theo trục, rồi **chọn trục có extent (max−min hình chiếu) nhỏ nhất** — vì phòng/hành
   lang hầu như luôn thấp hơn là dài/rộng. Đây từng dùng tiêu chí "nhiều điểm inlier nhất"
   nhưng bị đổi vì có trường hợp thực tế: hành lang dài, 2 mặt tường bên tích luỹ nhiều
   điểm RANSAC hơn cả sàn+trần dù không phải trục thẳng đứng.
4. `_fit_plane_from_sample` trả `None` nếu mẫu suy biến (3 điểm thẳng hàng/trùng nhau ra
   normal độ dài ~0) — RANSAC bỏ qua mẫu đó, không crash.

*(File còn có thêm `estimate_up_vector_cross_scene` — dùng khi có 2 scene của cùng toà
nhà, tìm cặp trục ứng viên giữa 2 bên đồng thuận nhau nhất theo thứ tự ưu tiên rank, và
`estimate_up_vector_from_local_normals` — ước lượng qua PCA cục bộ từng điểm, bền hơn khi
sàn/trần bị vật cản chia nhỏ thành nhiều mảnh rời rạc. Cả hai không phải đường mặc định
trong `multiview_pipeline.py` hiện tại nhưng sẵn có cho các use-case khác.)*

---

### A7. Trích mặt phẳng tường (RANSAC)

**Vấn đề giải quyết**: cần biết chính xác **tường thật** của scene ở đâu — không chỉ để vẽ
mà để (a) cho mỗi opening 1 `wall_normal` đáng tin (thay vì tự suy từ cụm điểm nhỏ, dễ
nhiễu) và (b) loại các mặt phẳng lớn không phải tường thật (sàn, trần, mặt tủ/kệ).

**Code — trích + lọc theo verticality**: [`plane_fitting.py:460-484`](../src/sgd_alignment/detection/plane_fitting.py#L460-L484)

```python
def extract_wall_planes(pc, distance_threshold=0.03, min_inliers=3000, max_planes=20,
                         verticality_max_dot=0.3, ransac_n=3, num_iterations=1000, up=None):
    if up is None:
        up = estimate_up_vector(pc, distance_threshold, min_inliers, ...)
    all_planes = _extract_planes(pc, distance_threshold, min_inliers, max_planes,
                                  ransac_n, num_iterations)
    walls = [p for p in all_planes if abs(float(np.dot(p.normal, up))) < verticality_max_dot]
    return merge_coplanar_planes(walls, pc)
```

**Code — gộp mảnh RANSAC của cùng 1 tường vật lý**: [`plane_fitting.py:497-555`](../src/sgd_alignment/detection/plane_fitting.py#L497-L555)

```python
def merge_coplanar_planes(planes, pc, normal_dot_min=0.98, d_max_diff=0.15):
    remaining = list(planes)
    merged = []
    while remaining:
        base = remaining.pop(0)
        group_indices = [base.inlier_indices]
        still_remaining = []
        for other in remaining:
            dot = float(np.dot(base.normal, other.normal))
            # offset check CÓ Ý THỨC VỀ DẤU: normal ngược dấu -> so |d1+d2|, không phải |d1-d2|
            d_diff = abs(base.d - other.d) if dot >= 0 else abs(base.d + other.d)
            same_plane = abs(dot) >= normal_dot_min and d_diff <= d_max_diff
            if same_plane:
                group_indices.append(other.inlier_indices)
            else:
                still_remaining.append(other)
        remaining = still_remaining
        merged.append(_refit_plane(pc, np.concatenate(group_indices)))   # least-squares lại
    return merged
```

**Code — loại tường "không thật" (đồ nội thất phẳng, lớn)**: [`plane_fitting.py:558-588`](../src/sgd_alignment/detection/plane_fitting.py#L558-L588)

```python
def filter_exterior_walls(pc, walls, outward_margin=0.10, max_beyond_fraction=0.02):
    centroid = pc.points.mean(axis=0)
    exterior = []
    for wall in walls:
        outward = wall.normal
        wall_point = pc.points[wall.inlier_indices].mean(axis=0)
        if np.dot(outward, wall_point - centroid) < 0:
            outward = -outward
        signed = pc.points @ outward - np.dot(outward, wall_point)
        beyond_fraction = float(np.mean(signed > outward_margin))   # % điểm NẰM PHÍA SAU wall
        if beyond_fraction <= max_beyond_fraction:
            exterior.append(wall)
    return exterior
```

**Giải thích**:
1. **`_extract_planes`** (lõi RANSAC dùng chung cho cả A6 và A7): lặp `max_planes` vòng,
   mỗi vòng gọi `_segment_plane_ransac` (lấy `num_iterations` mẫu ngẫu nhiên `ransac_n=3`
   điểm, fit mặt phẳng, đếm inlier theo `distance_threshold=3cm`, giữ mô hình thắng, refit
   least-squares trên toàn bộ inlier thắng cuộc), loại các điểm inlier đó khỏi tập, lặp lại
   cho tới khi hết điểm hoặc đạt `max_planes`. **Đáng chú ý**: RANSAC được **tự viết lại
   bằng numpy thuần** thay vì gọi `open3d.segment_plane` — lý do ghi rõ trong docstring
   ([`plane_fitting.py:46-59`](../src/sgd_alignment/detection/plane_fitting.py#L46-L59)):
   Open3D dùng RNG global process-wide và song song hoá đa luồng, khiến **cùng 1 input chạy
   2 lần cho ra 2 kết quả khác nhau** (tie-break phụ thuộc lịch trình luồng) — không tái lập
   được. Bản tự viết dùng `np.random.default_rng(seed)` cục bộ, đơn luồng → xác định 100%.
2. **`verticality_max_dot=0.3`**: giữ mặt phẳng có `|normal·up| < 0.3` (góc với phương ngang
   lớn hơn ~72.5°) là tường; loại sàn/trần (normal gần song song `up`).
3. **`merge_coplanar_planes`** — vì RANSAC lặp có thể cắt 1 tường thật (hơi gợn sóng/nhiễu)
   thành nhiều mặt fit riêng biệt (mỗi mặt bắt 1 phần điểm khác nhau nhưng normal/offset gần
   như nhau). Điều kiện gộp: `|dot(n1,n2)| >= 0.98` **và** offset gần nhau — nhưng offset
   phải so sánh **có ý thức về dấu**: nếu 2 normal cùng dấu thì so `|d1-d2|`, nếu ngược dấu
   (RANSAC có thể fit ra normal với dấu tuỳ ý) thì phải so `|d1+d2|`. Đây là 1 bug thật đã
   xảy ra: hành lang hẹp có 2 tường đối diện, normal gần đối song song **và** `|d|` tình cờ
   gần nhau (`d1=0.251, d2=0.24`) → nếu dùng `|d1-d2|` không điều kiện sẽ gộp nhầm 2 tường
   khác nhau thành 1 mặt phẳng vô nghĩa (least-squares refit trên 2.6 triệu điểm hỗn hợp ra
   normal gần song song trục `up` một cách giả tạo).
4. **`filter_exterior_walls`** (hàm phụ trợ, không nằm trong luồng mặc định của
   `extract_wall_planes` nhưng sẵn có để dùng khi cần lọc gắt hơn): tường thật của phòng
   không có gì "nằm phía sau" nó (nhìn từ trong phòng ra) — còn mặt kệ/tủ phẳng thì phía sau
   nó vẫn còn tường thật xa hơn. Đo % điểm toàn point cloud nằm lệch quá `outward_margin`
   (10cm) về phía "outward" của mặt phẳng; nếu tỷ lệ đó vượt `max_beyond_fraction=2%` thì
   loại — không phải tường bao ngoài thật.

---

### A8. Định hướng normal tường ra ngoài

**Vấn đề giải quyết**: RANSAC chỉ cho normal mặt phẳng, **dấu tuỳ ý** (có thể trỏ vào trong
hoặc ra ngoài phòng, không xác định). Nhưng SGD descriptor và alignment cần normal có
hướng **nhất quán "ra ngoài"** để so sánh 2 scene được — nếu 1 bên trỏ vào, 1 bên trỏ ra,
mọi phép so sánh góc sẽ lệch ~180°.

**Code**: [`opening_geometry.py:17-119`](../src/sgd_alignment/detection/opening_geometry.py#L17-L119)

```python
def orient_walls_outward(walls, pc, is_outdoor, margin=0.10,
                          camera_positions=None, low_confidence_threshold=0.15):
    oriented, low_confidence = {}, []
    camera_centroid = camera_positions.mean(axis=0) if camera_positions is not None else None

    for idx, wall in enumerate(walls):
        wall_point = pc.points[wall.inlier_indices].mean(axis=0)
        normal = wall.normal
        signed = pc.points @ normal - np.dot(normal, wall_point)
        frac_pos = float((signed > margin).mean())
        frac_neg = float((signed < -margin).mean())
        density_margin = abs(frac_pos - frac_neg)
        # indoor: outward = ít điểm hơn. outdoor: outward = NHIỀU điểm hơn (ngược lại!)
        density_positive_is_outward = (frac_pos <= frac_neg) if not is_outdoor else (frac_pos > frac_neg)

        if camera_centroid is not None:
            camera_signed = float(np.dot(camera_centroid - wall_point, normal))
            # indoor: outward = xa nơi camera đứng. outdoor: outward = phía camera đứng
            camera_positive_is_outward = (camera_signed <= 0) if not is_outdoor else (camera_signed > 0)
            positive_side_is_outward = camera_positive_is_outward
            if density_margin >= low_confidence_threshold and density_positive_is_outward != camera_positive_is_outward:
                low_confidence.append(idx)      # 2 tín hiệu bất đồng -> ghi nhận, KHÔNG lặng im
        else:
            positive_side_is_outward = density_positive_is_outward
            if density_margin < low_confidence_threshold:
                low_confidence.append(idx)      # mật độ gần hoà -> quyết định không đáng tin

        if not positive_side_is_outward:
            normal = -normal
        oriented[idx] = normal

    result = _OrientedNormals(oriented); result.low_confidence = low_confidence
    return result
```

**Giải thích**:
1. **Tín hiệu chính (khi có `camera_positions`)**: vị trí camera lúc quay là **sự thật hình
   học cứng**, không phải thống kê — người quay indoor chắc chắn đứng trong phòng, người
   quay outdoor chắc chắn đứng ngoài. Quy tắc đảo chiều tuỳ indoor/outdoor:
   - **Indoor**: outward = phía camera **không** đứng (phòng dày đặc camera, ngoại thất gần
     như không quay tới).
   - **Outdoor**: outward = phía camera **có** đứng (không gian ngoài mới là nơi được quay
     dày đặc, phòng chỉ thấy thấp thoáng qua khung cửa).
2. **Tín hiệu phụ — mật độ điểm 2 phía tường**: cùng quy tắc đảo chiều nhưng dựa vào số
   điểm point cloud rơi về mỗi phía (`frac_pos`, `frac_neg`, cách mặt phẳng ít nhất
   `margin=10cm` để không tính nhiễu sát mặt phẳng). Đây là bug đã được sửa gần đây
   (commit `a435a33` — *"Sửa lỗi hướng tường sai khi mật độ điểm mơ hồ, dùng camera pose
   làm tín hiệu chính khi có"*): áp dụng quy tắc "outward = ít điểm hơn" **không điều kiện**
   cho cả indoor lẫn outdoor từng khiến mọi tường outdoor bị đảo ngược 180°.
3. **`low_confidence_threshold=0.15`** — 2 tình huống được đánh dấu "không chắc":
   - Không có camera signal, và `density_margin < 0.15` (mật độ 2 phía gần hoà — ví dụ
     0.30/0.23 như trong 1 case thực tế, "hơn nhưng không rõ ràng").
   - Có camera signal nhưng **bất đồng** với density signal khi density margin đủ rõ ràng
     (≥0.15) — nghĩa là 2 tín hiệu độc lập mâu thuẫn nhau, đáng ngờ hơn cả trường hợp không
     có tín hiệu nào rõ.
   Quan trọng: hàm **không** loại bỏ các tường "low confidence" này — chỉ ghi lại chỉ số
   (`result.low_confidence`) để caller có thể log cảnh báo, không âm thầm bỏ qua rủi ro.

---

### A9. Đo kích thước từng opening → Detection3D

**Vấn đề giải quyết**: cụm điểm 3D thô (đã gộp ở A4) chỉ là 1 đám mây điểm chưa có hình
dạng "hộp". Cần biến nó thành `Detection3D` có đủ `center, u_axis, v_axis, normal, width,
height` — sẵn sàng cho SGD matching.

**Code**: [`opening_geometry.py:158-254`](../src/sgd_alignment/detection/opening_geometry.py#L158-L254)

```python
def points_to_detection(points, category, up, wall_normal, trim_percentile=0.0,
                         wall_normal_agreement_threshold=0.85) -> Detection3D:
    centroid = points.mean(axis=0)

    def _pca_normal():
        _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
        return vt[-1]

    if wall_normal is None:
        normal = _pca_normal()                       # không tìm thấy tường gần -> tự suy từ cụm điểm
    else:
        pca_normal = _pca_normal()
        if abs(float(np.dot(wall_normal, pca_normal))) < wall_normal_agreement_threshold:
            # tường gần đó KHÔNG thực sự trùng bề mặt cụm điểm (hõm/lệch) -> tin PCA cục bộ hơn
            normal = pca_normal if np.dot(wall_normal, pca_normal) >= 0 else -pca_normal
        else:
            normal = wall_normal

    v_axis = up - np.dot(up, normal) * normal        # chiếu up lên mặt phẳng tường
    v_axis = v_axis / np.linalg.norm(v_axis)
    u_axis = np.cross(v_axis, normal)
    u_axis = u_axis / np.linalg.norm(u_axis)

    # chiếu điểm LÊN mặt phẳng tường trước khi đo (loại độ dày khung cửa/nhiễu backproject)
    onto_plane = points - np.outer((points - centroid) @ normal, normal)
    centered = onto_plane - centroid
    u = centered @ u_axis
    v = centered @ v_axis
    u_lo, u_hi = np.percentile(u, [trim_percentile, 100 - trim_percentile])
    v_lo, v_hi = np.percentile(v, [trim_percentile, 100 - trim_percentile])
    width = float(u_hi - u_lo)
    height = float(v_hi - v_lo)
    center = centroid + ((u_hi + u_lo) / 2) * u_axis + ((v_hi + v_lo) / 2) * v_axis

    return Detection3D(category=category, center=center, u_axis=u_axis, v_axis=v_axis,
                        normal=normal, width=width, height=height)
```

**Tìm tường gần nhất trước đó**: [`opening_geometry.py:132-155`](../src/sgd_alignment/detection/opening_geometry.py#L132-L155)

```python
def nearest_wall_normal(centroid, walls, oriented_normals, max_distance=0.15):
    if not walls:
        return None
    best_idx = min(range(len(walls)), key=lambda i: abs(walls[i].signed_distance(centroid[None,:])[0]))
    if abs(walls[best_idx].signed_distance(centroid[None,:])[0]) > max_distance:
        return None                     # tường gần nhất vẫn quá xa -> đừng gán bừa
    return oriented_normals[best_idx]
```

**Giải thích các bước lọc lỗi / công thức**:
1. **`nearest_wall_normal` với `max_distance=0.15m`** — chỉ gán `wall_normal` của tường
   RANSAC nếu nó thực sự **gần** centroid cụm điểm (dưới 15cm). Ngưỡng này rút ra từ dữ
   liệu thực: mọi opening gán-đúng trong project có khoảng cách 0.01–0.05m, còn 1 case
   thất bại thực tế (scan thiếu vùng, RANSAC không tìm ra tường thật đằng sau 2 cửa
   outdoor) có khoảng cách 0.39–0.40m — ngưỡng 0.15 nằm an toàn giữa 2 vùng đó.
2. **`wall_normal_agreement_threshold=0.85`** — bước lọc tinh tế nhất của hàm này: dù tường
   RANSAC ở rất gần centroid, không có nghĩa cửa **nằm phẳng trên đúng bề mặt** đó (ví dụ
   cửa nằm trong hốc/góc nghiêng). So sánh `wall_normal` với normal tự suy ra từ chính cụm
   điểm (PCA — trục có variance nhỏ nhất của tập điểm) bằng dot product; nếu lệch quá 0.85
   (~32°) thì **bỏ qua tường**, dùng PCA cục bộ (đã align dấu theo `wall_normal` để giữ quy
   ước "ra ngoài"). Case thực tế: 1 tường cách centroid chỉ 0.001m nhưng normal lệch 44°
   so PCA — nếu dùng thẳng `wall_normal` sẽ ra aspect ratio méo (0.93 thay vì ~0.54 đúng).
3. **Công thức trục `v_axis`/`u_axis`** *(đã giải thích ở lượt trước, nhắc lại ngắn gọn)*:
   ```
   v_axis = normalize(up - (up·normal)*normal)   # thành phần của up VUÔNG GÓC với normal
   u_axis = normalize(normal × v_axis)           # hệ trực chuẩn phải tay
   ```
4. **Chiếu điểm lên mặt phẳng tường trước khi đo `width`/`height`** — loại bỏ ảnh hưởng của
   độ dày khung cửa hoặc nhiễu backproject theo hướng `normal` (nếu không chiếu, 1 điểm hơi
   lồi ra ngoài mặt tường sẽ không ảnh hưởng `width`/`height` vì nó chỉ lệch theo `normal`,
   nhưng vẫn tính an toàn hơn khi chiếu tường minh).
5. **`trim_percentile`** — mặc định `0.0` (dùng đúng `min()`/`max()`, không đổi hành vi cũ);
   đặt ví dụ `2.0` để dùng percentile [2,98] thay vì [0,100] — vì `min`/`max` là thống kê
   **kém robust nhất có thể**: 1 điểm nhiễu (mép mask bị lem ở 1 view chéo) đủ để kéo lệch
   `width`/`height` đo được. Chỉ cần thiết cho luồng multi-view (nhiều view độc lập nhiễu
   gộp lại); luồng CloudCompare thủ công không cần vì người dùng đã tự loại điểm sai khi
   chọn tay.

Output: `Detection3D` hoàn chỉnh — kết thúc Giai đoạn A.

---

### A10. Xuất kết quả

**Code**: [`multiview_pipeline.py:230-244`](../src/sgd_alignment/pipelines/multiview_pipeline.py#L230-L244)

```python
detections_path = output_dir / f"{config.output_name}_detections.pkl"
with open(detections_path, "wb") as f:
    pickle.dump(detections, f)

review_path = output_dir / f"{config.output_name}_openings.ply"
_export_review_ply(scene_points, scene_colors, merged, review_path)   # tô màu opening trên scene

scene_path = output_dir / f"{config.output_name}_scene.ply"
_export_review_ply(scene_points, scene_colors, [], scene_path)        # scene gốc, không tô
```

**Giải thích**: `detections` (list `Detection3D`) được lưu `.pkl` — input trực tiếp cho
Giai đoạn B (`align_pipeline.py` đọc lại bằng `pickle.load`). File `.ply` review tô màu
riêng biệt cửa (xanh lá) / cửa sổ (xanh dương) đè lên scene xám — mục đích kiểm tra bằng
mắt (CloudCompare/MeshLab) trước khi tin tưởng chạy tiếp bước matching, vì đây là pipeline
headless không có notebook hiển thị trực tiếp.

---

## Giai đoạn B — Matching & Alignment

### B0. Orchestration — `align_pipeline.py`

**Vấn đề giải quyết**: nối kết quả detection của 2 bên (indoor/outdoor, mỗi bên chạy
`multiview_pipeline.py` riêng) lại, chạy matching+alignment, rồi áp transform lên **toàn bộ
scene point cloud màu** (không chỉ Detection3D) để xuất 1 file `.ply` hoàn chỉnh xem được.

**Code**: [`align_pipeline.py:79-129`](../src/sgd_alignment/pipelines/align_pipeline.py#L79-L129)

```python
def run_alignment(config: AlignPipelineConfig) -> AlignmentResult:
    with open(config.indoor_detections, "rb") as f:
        indoor_detections = pickle.load(f)
    with open(config.outdoor_detections, "rb") as f:
        outdoor_detections = pickle.load(f)

    weights = PairWeights(**config.weights) if config.weights else None
    result = align_indoor_outdoor(
        indoor_detections, outdoor_detections,
        weights=weights, max_cost=config.max_cost,
        estimate_scale=config.estimate_scale, normalize_distance=config.normalize_distance,
    )

    indoor_points, indoor_colors = _read_ply(config.indoor_scene_ply)
    outdoor_points, outdoor_colors = _read_ply(config.outdoor_scene_ply)
    aligned_indoor_points = transform_points(indoor_points, result.R, result.t)

    all_points = np.concatenate([aligned_indoor_points, outdoor_points])
    all_colors = np.concatenate([indoor_colors or fallback_red, outdoor_colors or fallback_blue])
    _write_ply(all_points, all_colors, out_path)
    return result
```

**Giải thích**: `estimate_scale=True` và `normalize_distance=True` là **mặc định của riêng
pipeline này** (khác mặc định `False` của hàm `align_indoor_outdoor` gốc) — vì trường hợp
thường gặp nhất ở đây là indoor/outdoor đến từ 2 lần chạy DA3 **độc lập, không chia sẻ
scale mét thật** (mỗi lần inference feed-forward tự chọn 1 scale nội bộ riêng, không liên
quan gì tới lần kia). Nếu 2 bên vốn cùng 1 reconstruction (đã chung scale), config có thể
tắt cả 2 cờ này.

---

### B1. SGD: Semantic-Geometric Descriptor

**Vấn đề giải quyết**: matching 2 tập object rời rạc (indoor/outdoor) không thể chỉ so
"khoảng cách tuyệt đối" (2 scene có hệ toạ độ, gốc, hướng hoàn toàn khác nhau) — cần 1 mô
tả **tương đối, bất biến với hệ quy chiếu**: "object này đứng ở đâu so với các object khác
trong cùng scene của nó". Đây là hiện thực hoá Section 3.2 của paper gốc.

**Code — hệ trục cục bộ + 1 pair-feature**: [`sgd.py:145-173`](../src/sgd_alignment/matching/sgd.py#L145-L173)

```python
def _local_frame(det: Detection3D) -> np.ndarray:
    # cột = (x,y,z) theo quy ước paper: x=normal, y=u_axis, z=v_axis
    return np.stack([det.normal, det.u_axis, det.v_axis], axis=1)

def _pair_feature(i: Detection3D, j: Detection3D, distance_scale: float = 1.0) -> PairFeature:
    offset = j.center - i.center
    raw_distance = float(np.linalg.norm(offset))
    direction = offset / raw_distance if raw_distance > 1e-12 else offset
    distance = raw_distance / distance_scale

    frame_i = _local_frame(i)
    direction_angles = np.arccos(np.clip(direction @ frame_i, -1.0, 1.0))   # (alpha, beta, theta)

    frame_j = _local_frame(j)
    r_rel = frame_i.T @ frame_j
    relative_euler_angles = Rotation.from_matrix(r_rel).as_euler("xyz")     # (rotx, roty, rotz)

    return PairFeature(label=j.category, distance=distance,
                        direction_angles=direction_angles,
                        relative_euler_angles=relative_euler_angles)
```

**Code — scale-invariant khi 2 scene không chung metric**: [`sgd.py:176-217`](../src/sgd_alignment/matching/sgd.py#L176-L217)

```python
def _scene_distance_scale(detections):
    if len(detections) < 2: return 1.0
    centers = np.array([d.center for d in detections])
    dists = np.linalg.norm(centers[:,None,:] - centers[None,:,:], axis=-1)
    iu = np.triu_indices(len(centers), k=1)
    scale = float(np.median(dists[iu]))                # khoảng cách trung vị giữa mọi cặp object
    return scale if scale > 1e-9 else 1.0

def build_sgds(detections, normalize_distance=False):
    scale = _scene_distance_scale(detections) if normalize_distance else 1.0
    sgds = []
    for i, det in enumerate(detections):
        neighbors = [_pair_feature(det, detections[j], scale) for j in range(len(detections)) if j != i]
        sgds.append(SGD(detection=det, neighbors=neighbors))
    return sgds
```

**Giải thích công thức**:
- **8 thành phần mỗi `PairFeature`**: `S^j` (category), `d_i^j` (khoảng cách center-to-center,
  có thể chia cho `distance_scale` để thành tỷ lệ không đơn vị), `alpha, beta, theta` (góc
  giữa hướng `j-i` với từng trục `normal/u_axis/v_axis` **của chính object i**), và
  `rotx, roty, rotz` (Euler angles thứ tự `xyz` của rotation tương đối `frame_i.T @ frame_j`
  — đưa hệ trục cục bộ của `i` khớp sang hệ trục cục bộ của `j`).
- **Tại sao dùng `arccos` cho góc hướng**: `direction @ frame_i` cho 3 giá trị cosine (vì cả
  `direction` và mỗi cột của `frame_i` đều là unit vector) — `arccos` (đã `clip` vào
  `[-1,1]` để tránh lỗi số học float vượt biên) đổi thành góc thực (radian).
- **`normalize_distance=True`**: chia mọi khoảng cách cho **khoảng cách trung vị giữa mọi
  cặp object trong CHÍNH scene đó** — biến "khoảng cách tuyệt đối theo mét" thành "khoảng
  cách tương đối theo layout riêng của scene" — cần thiết khi 2 scene không chung 1 đơn vị
  mét thật (2 lần inference DA3 độc lập).

---

### B2. Hungarian 2 tầng

**Vấn đề giải quyết**: có 2 lớp bài toán gán cặp lồng nhau — (a) so 2 object cụ thể, cần so
"tập hàng xóm" của object này với "tập hàng xóm" của object kia (không biết trước hàng xóm
nào ứng với hàng xóm nào); (b) sau khi có ma trận chi phí giữa mọi cặp (indoor, outdoor),
cần chọn 1 phép gán tổng thể tối ưu (1 indoor chỉ khớp với tối đa 1 outdoor).

**Code — chi phí 1 cặp pair-feature (Equation 2 của paper)**: [`sgd.py:220-240`](../src/sgd_alignment/matching/sgd.py#L220-L240)

```python
def _pair_feature_cost(a, b, w: PairWeights) -> float:
    if a.label != b.label:
        return INFEASIBLE
    d_e = abs(a.distance - b.distance)
    angle_e = np.abs(a.direction_angles - b.direction_angles)
    rot_e = np.abs(a.relative_euler_angles - b.relative_euler_angles)
    if d_e > w.distance_threshold: return INFEASIBLE
    if np.any(angle_e > w.angle_thresholds): return INFEASIBLE
    if np.any(rot_e > w.rotation_thresholds): return INFEASIBLE
    return w.distance_weight*d_e + float(np.dot(w.angle_weights, angle_e)) + float(np.dot(w.rotation_weights, rot_e))
```

**Code — tầng trong (Algorithm 2), gán "hàng xóm" tối ưu**: [`sgd.py:243-276`](../src/sgd_alignment/matching/sgd.py#L243-L276), [`sgd.py:340-379`](../src/sgd_alignment/matching/sgd.py#L340-L379)

```python
def _assign_rectangular(cost: np.ndarray) -> tuple[float, int]:
    if cost.size == 0: return INFEASIBLE, 0
    valid_rows = ~np.all(np.isinf(cost), axis=1)
    valid_cols = ~np.all(np.isinf(cost), axis=0)
    sub = cost[np.ix_(valid_rows, valid_cols)]
    if sub.size == 0: return INFEASIBLE, 0
    finite_sub = np.where(np.isinf(sub), _LARGE_FINITE_COST, sub)
    row_idx, col_idx = linear_sum_assignment(finite_sub)          # Hungarian (scipy)
    total_cost, num_matched = 0.0, 0
    for r, c in zip(row_idx, col_idx):
        if not np.isinf(sub[r, c]):
            total_cost += sub[r, c]; num_matched += 1
    if num_matched == 0: return INFEASIBLE, 0
    return total_cost / num_matched, num_matched     # TRUNG BÌNH, không phải TỔNG (xem giải thích)

def sgdu_distance(a: SGD, b: SGD, weights=None, use_intrinsic_fallback=False):
    if a.detection.category != b.detection.category or not _regions_compatible(a.detection, b.detection):
        return INFEASIBLE, 0
    cost = np.array([[_pair_feature_cost(fa, fb, weights) for fb in b.neighbors] for fa in a.neighbors])
    relational_cost, num_matched = _assign_rectangular(cost)
    return relational_cost, num_matched
```

**Code — tầng ngoài (Algorithm 3), gán object indoor↔outdoor cuối cùng**: [`sgd.py:401-466`](../src/sgd_alignment/matching/sgd.py#L401-L466)

```python
def match_sgds(indoor_sgds, outdoor_sgds, weights=None, max_cost=5.0, ...):
    cost = np.array([[sgdu_distance(a, b, weights, ...)[0] for b in outdoor_sgds] for a in indoor_sgds])
    valid_rows = ~np.all(np.isinf(cost), axis=1)
    valid_cols = ~np.all(np.isinf(cost), axis=0)
    sub = cost[np.ix_(valid_rows, valid_cols)]
    finite_sub = np.where(np.isinf(sub), _LARGE_FINITE_COST, sub)
    row_idx, col_idx = linear_sum_assignment(finite_sub)
    matches = []
    for r, c in zip(row_idx, col_idx):
        if np.isinf(sub[r, c]) or sub[r, c] > max_cost:
            continue
        # (tuỳ chọn use_ambiguity_check: xem giải thích bên dưới)
        matches.append((int(row_ids[r]), int(col_ids[c]), float(sub[r, c])))
    return matches
```

**Giải thích**:
1. **Tầng trong**: với 1 cặp ứng viên `(object k indoor, object l outdoor)`, không so trực
   tiếp — vì "hàng xóm thứ 3" của `k` chưa chắc ứng với "hàng xóm thứ 3" của `l` (thứ tự
   liệt kê không mang ý nghĩa gì). Xây ma trận chi phí giữa **mọi cặp hàng xóm có thể**, rồi
   giải Hungarian trên chính ma trận đó → cách khớp "tập hàng xóm" tốt nhất, không phải so
   1-1 theo thứ tự liệt kê.
2. **Lấy TRUNG BÌNH chứ không phải TỔNG** ([`sgd.py:251-253`](../src/sgd_alignment/matching/sgd.py#L251-L253) giải thích rõ): nếu lấy tổng, 1 cặp chỉ khớp được 1/6 hàng xóm sẽ có tổng
   chi phí thấp hơn (vì ít số hạng cộng vào) so với cặp khớp được 5/6 hàng xóm dù đúng hơn
   nhiều — **ngược đời**. Lấy trung bình theo số cặp khớp được mới công bằng.
3. **`INFEASIBLE = inf`**: category khác nhau, hoặc bất kỳ thành phần lỗi vượt ngưỡng riêng
   (`distance_threshold=1.5m`, các ngưỡng góc `π/3` mặc định) → cặp đó **không thể** khớp,
   không chỉ là "chi phí cao". Hàng/cột toàn `inf` bị loại trước khi đưa vào Hungarian thật
   (`_LARGE_FINITE_COST=1e9` thay thế `inf` chỉ để `scipy.linear_sum_assignment` không lỗi
   số học, nhưng logic sau đó vẫn kiểm tra lại `np.isinf` trên **giá trị gốc**).
4. **Tầng ngoài** lặp lại đúng cơ chế Hungarian này 1 lần nữa, nhưng trên ma trận
   (indoor object × outdoor object) với chi phí là kết quả tầng trong. `max_cost=5.0` là
   ngưỡng loại cuối cùng: dù Hungarian chọn 1 cặp là "tốt nhất có thể", nếu chi phí tuyệt
   đối vẫn quá cao (không có đối tác thật sự tương ứng) thì loại.
5. **`use_ambiguity_check`** (tắt mặc định) — kiểu Lowe's ratio test của SIFT: so chi phí
   cặp được chọn với chi phí ứng viên **tốt thứ nhì** cùng hàng/cột (`_ambiguity_ratio`); nếu
   tỷ lệ `chosen/next_best > 0.8` (gần bằng nhau) thì từ chối luôn cặp đó — phòng trường hợp
   1 hành lang có nhiều cửa cùng loại nhưng chỉ 1 cái thật sự tương ứng, các cái còn lại là
   "distractor" (cửa thoát hiểm, cửa kho...) tình cờ có mô tả tương đối gần giống.

---

### B3. Kabsch / Umeyama — tính (R, t, scale)

**Vấn đề giải quyết**: sau khi có các cặp `(indoor_idx, outdoor_idx)` khớp, cần tìm 1 phép
biến đổi cứng (rotation + translation, có thể thêm scale) áp lên toàn bộ scene indoor để
nó khớp scene outdoor — bài toán Procrustes trực giao kinh điển.

**Code**: [`alignment.py:82-146`](../src/sgd_alignment/matching/alignment.py#L82-L146)

```python
def is_collinear(points: np.ndarray, ratio_threshold: float = 0.05) -> bool:
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] < 1e-12: return True
    return (singular_values[1] / singular_values[0]) < ratio_threshold

def estimate_rigid_transform(src, dst, estimate_scale=False):
    if len(src) < 3:
        raise ValueError(f"need at least 3 correspondences ..., got {len(src)}")
    if is_collinear(src) or is_collinear(dst):
        warnings.warn("matched opening corners are (nearly) collinear - rotation about "
                       "that line's axis is poorly constrained ...")

    src_centroid, dst_centroid = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_centroid, dst - dst_centroid

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T                        # đảm bảo det(R)=+1 (không lật gương)

    if estimate_scale:
        src_spread = float(np.sum(src_c**2))
        if src_spread < 1e-12:
            raise ValueError("source points have ~zero spread - cannot estimate a scale factor")
        scale = float(np.dot(S, np.diag(D))) / src_spread
        R = scale * R                          # gộp scale vào R luôn, không trả riêng

    t = dst_centroid - R @ src_centroid
    return R, t
```

**Ghép vào `align_indoor_outdoor`, dùng 8 góc hộp thay vì chỉ center**: [`alignment.py:513-521`](../src/sgd_alignment/matching/alignment.py#L513-L521)

```python
src = np.concatenate([indoor_detections[i].corners() for i, _, _ in matches])
dst = np.concatenate([outdoor_detections[j].corners() for _, j, _ in matches])
R, t = estimate_rigid_transform(src, dst, estimate_scale=estimate_scale)
scale = float(np.linalg.norm(R, axis=0).mean())
```

**Giải thích**:
1. **Kabsch (SVD)**: `H = src_c^T @ dst_c` (ma trận hiệp phương sai chéo giữa 2 tập điểm đã
   center); `SVD(H) = U·S·V^T`; nghiệm rotation tối ưu (theo nghĩa least-squares) là
   `R = V·D·U^T`. `D = diag(1,1,det(V·U^T))` **chỉ để đảm bảo `det(R)=+1`** — nếu không có
   bước này, khi 2 tập điểm gần suy biến (đối xứng gương), SVD thô có thể ra 1 ma trận với
   `det=-1` (phép quay + lật gương, không phải rotation vật lý thật).
2. **`is_collinear`** (bước lọc cảnh báo, không chặn): dùng tỷ lệ 2 singular value đầu của
   tập điểm đã center — nếu điểm gần như nằm trên 1 đường thẳng (ví dụ nhiều cửa sổ dàn
   đều trên cùng 1 tường), rotation quanh trục đường thẳng đó **không được ràng buộc đủ**
   bởi dữ liệu (Kabsch vẫn ra 1 nghiệm hợp lệ toán học, nhưng có thể không phải nghiệm vật
   lý đúng). Cảnh báo (`warnings.warn`), không raise lỗi — để người dùng tự quyết định có
   tin kết quả hay không.
3. **`estimate_scale` (Umeyama)**: `scale = dot(S, diag(D)) / Σ|src_centered|²`. Cần thiết
   khi indoor/outdoor tới từ 2 lần inference DA3 độc lập, mỗi lần tự chọn 1 đơn vị chiều dài
   nội bộ khác nhau — pure rotation (scale=1 ép buộc) sẽ cho `R` **trông giống** rotation
   hợp lệ nhưng thực chất bị lệch hệ thống (biased), nguy hiểm hơn cả việc fail rõ ràng.
4. **Dùng 8 góc hộp (`corners()`) thay vì chỉ `center`** — đúng theo paper gốc: mỗi cặp match
   đóng góp 8 điểm tương ứng thay vì 1, vừa cho nhiều dữ liệu hơn để least-squares ổn định
   hơn, vừa tự nhiên ràng buộc chặt rotation hơn (vì 1 hộp cửa/cửa sổ có 3 chiều riêng biệt
   — rộng, cao, dày — không đối xứng xoay).
5. **`len(src) < 3` → raise lỗi cứng**: 3 điểm không thẳng hàng là điều kiện tối thiểu để
   xác định 1 rotation 3D duy nhất — dưới 3 điểm, bài toán vô nghiệm về mặt toán học, không
   có cách "đoán thêm".

---

### B4. Các lớp lọc/tinh chỉnh sau matching

Đây là các bước **tuỳ chọn** (tắt mặc định, bật qua flag trong `align_indoor_outdoor`), xử
lý các trường hợp mà so sánh descriptor tĩnh (B1/B2) không đủ sức phân biệt.

#### B4.1 — Lọc detection phi vật lý

**Code**: [`alignment.py:258-272`](../src/sgd_alignment/matching/alignment.py#L258-L272)

```python
def _filter_degenerate(detections, min_dimension):
    kept = [(i, d) for i, d in enumerate(detections) if d.width >= min_dimension and d.height >= min_dimension]
    return [d for _, d in kept], [i for i, _ in kept]
```
Loại các `Detection3D` có `width`/`height` quá nhỏ (ví dụ 0.03m — rõ ràng là lỗi
segmentation/backprojection, không phải cửa thật nào). Lý do cần bước này: 1 hộp gần suy
biến có 8 góc gần trùng nhau, khiến `estimate_rigid_transform` "fit" rất dễ (residual thấp
giả tạo) dù correspondence hoàn toàn sai — case thực tế đã quan sát: 1 "door" rộng 0.03m
từng khiến `refine_matches_with_geometric_consensus` (bên dưới) "xác nhận" nhầm 1 cặp match
sai vì residual thấp.

#### B4.2 — Tinh chỉnh lặp theo đồng thuận hình học (giống ICP)

**Code**: [`alignment.py:148-255`](../src/sgd_alignment/matching/alignment.py#L148-L255)

```python
def refine_matches_with_geometric_consensus(indoor_detections, outdoor_detections, matches,
                                             estimate_scale=False, max_iterations=3, max_residual=None):
    if len(matches) < 3:
        return matches, None, None
    current = matches
    for _ in range(max_iterations):
        src = np.concatenate([indoor_detections[i].corners() for i,_,_ in current])
        dst = np.concatenate([outdoor_detections[j].corners() for _,j,_ in current])
        R, t = estimate_rigid_transform(src, dst, estimate_scale=estimate_scale)
        next_matches = _reassign_by_geometric_residual(indoor_detections, outdoor_detections, R, t, max_residual)
        if len(next_matches) < 3:
            break                       # đồng thuận sụp đổ -> giữ kết quả hợp lệ cuối cùng
        if {(i,j) for i,j,_ in next_matches} == {(i,j) for i,j,_ in current}:
            current = next_matches; break   # hội tụ -> dừng
        current = next_matches
    return current, R, t
```

Ý tưởng như **ICP/PnP-RANSAC consensus**: fit `(R,t)` từ tập match hiện tại (chi phối bởi
phần đúng đa số, vì Kabsch là least-squares) → **gán lại từ đầu** mọi cặp cùng category
theo khoảng cách hình học sau biến đổi (`_reassign_by_geometric_residual`, [`alignment.py:148`](../src/sgd_alignment/matching/alignment.py#L148)) → lặp tới khi hội tụ. Sửa được trường hợp SGD gán nhầm 2 cửa
sổ giống hệt nhau trên cùng 1 hàng (mô tả tương đối cục bộ mơ hồ giữa các hàng xóm giống
nhau), vì lúc này có thêm tín hiệu "khoảng cách tuyệt đối sau khi xoay" mà bản thân SGD
không có.

#### B4.3 — RANSAC hypothesize-and-verify đầy đủ (mạnh hơn B4.2)

**Code**: [`alignment.py:275-380`](../src/sgd_alignment/matching/alignment.py#L275-L380)

```python
def ransac_match_with_wall_consensus(indoor_detections, outdoor_detections, matches,
                                      estimate_scale=False, max_residual=None,
                                      normal_angle_threshold=np.pi/6, min_sample_size=2,
                                      max_hypotheses=500, rng_seed=0):
    all_samples = list(combinations(range(len(matches)), min_sample_size))
    if len(all_samples) > max_hypotheses:
        rng = np.random.default_rng(rng_seed)
        chosen = rng.choice(len(all_samples), size=max_hypotheses, replace=False)
        all_samples = [all_samples[i] for i in chosen]

    best_inliers, best_inlier_residual_sum = [], float("inf")
    for sample_idx in all_samples:
        sample = [matches[k] for k in sample_idx]
        src = ...; dst = ...
        R, t = estimate_rigid_transform(src, dst, estimate_scale=estimate_scale)
        inliers = _reassign_by_geometric_residual(indoor_detections, outdoor_detections,
                                                    R, t, max_residual, normal_angle_threshold)
        inlier_residual_sum = sum(residual for _,_,residual in inliers)
        better = len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and inlier_residual_sum < best_inlier_residual_sum)
        if better:
            best_inliers, best_R, best_t = inliers, R, t
            best_inlier_residual_sum = inlier_residual_sum
    ...
    # polish cuối: refit trên TOÀN BỘ tập consensus thắng cuộc, không chỉ 2-match hạt giống
```

**Khác biệt cốt lõi với B4.2** (giải thích trong chính docstring, [`alignment.py:293-306`](../src/sgd_alignment/matching/alignment.py#L293-L306)): B4.2 tự fit hypothesis từ **chính** tập match nó
đang cố xác minh (nếu tập ban đầu sai đa số, nó tự "xác nhận" cái sai của chính mình). B4.3
là RANSAC thật: lấy mẫu **rất nhỏ** (2 match) làm hạt giống, kiểm tra bao nhiêu cặp KHÁC
(không chỉ trong `matches` ban đầu — toàn bộ tổ hợp cùng category) đồng thuận với hypothesis
đó cả về **vị trí lẫn hướng normal tường** (`normal_angle_threshold=π/6=30°`), chọn
hypothesis nào thu được nhiều phiếu nhất. Giải quyết case thực tế: 1 phòng có 1-2 cửa
thông ra hành lang chung, cộng thêm vài cửa/cửa sổ khác thông ra không gian hoàn toàn khác
— B4.2 dễ bị "thao túng" bởi 1 detection gần-suy-biến tình cờ khớp giả với hypothesis sai
ngay từ đầu; B4.3 không cho phép 1 detection đơn lẻ tự "bootstrap" xác nhận chính nó.
Tie-break khi 2 hypothesis có cùng số inlier: chọn cái có **tổng residual thấp hơn**
(không phải thứ tự liệt kê ngẫu nhiên) — tránh phụ thuộc vào thứ tự `combinations()` sinh ra.

#### B4.4 — Cảnh báo "chỉ khớp trên 1 tường"

**Code**: [`alignment.py:40-79`](../src/sgd_alignment/matching/alignment.py#L40-L79), [`alignment.py:498-511`](../src/sgd_alignment/matching/alignment.py#L498-L511)

```python
def matches_span_single_wall(detections, matched_indices, angle_threshold_deg=20.0):
    normals = [detections[i].normal for i in matched_indices]
    if len(normals) < 2: return True
    reference = normals[0]
    cos_threshold = np.cos(np.radians(angle_threshold_deg))
    return all(abs(float(np.dot(reference, n))) >= cos_threshold for n in normals[1:])
```

Đây **không phải** kiểm tra suy biến toán học (Kabsch vẫn ra nghiệm duy nhất từ dù chỉ 1
match — xác nhận thực nghiệm: 1 match tái tạo đúng byte-for-byte cùng 1 rotation ~145° như
khi dùng đủ 2 match). Mà là kiểm tra **thiếu cross-validation độc lập**: khi mọi match chỉ
nằm trên 1 tường, không có "tường thứ 2" theo hướng khác để xác nhận rotation tìm được là
**đúng vật lý** chứ không phải sản phẩm của 1 correspondence sai hoặc quy ước normal sai từ
trước — residual thấp không loại trừ được khả năng đó. Chỉ `warnings.warn`, không chặn.

---

### B5. Ghép nhiều phòng vào 1 hub

**Vấn đề giải quyết**: khi có **N phòng** cùng mở ra 1 hành lang/hub chung (topology hình
sao), không thể chạy `align_indoor_outdoor` độc lập từng phòng — 2 phòng có thể cùng "chấm"
1 tường hub làm đối tác tốt nhất trong lần chạy riêng của mình, không ai biết phòng kia
cũng đang cạnh tranh đúng chỗ đó.

**Code**: [`alignment.py:557-628`](../src/sgd_alignment/matching/alignment.py#L557-L628)

```python
def align_rooms_to_hub(rooms, hub_detections, **align_kwargs):
    wall_groups = group_by_wall(hub_detections)     # gom opening hub theo tường vật lý
    n_rooms, n_walls = len(rooms), len(wall_groups)
    cost = np.full((n_rooms, n_walls), np.inf)
    cached = {}
    for r, room in enumerate(rooms):
        for w, group in enumerate(wall_groups):
            sub_hub = [hub_detections[i] for i in group]
            try:
                result = align_indoor_outdoor(room, sub_hub, **align_kwargs)
            except ValueError:
                continue
            cached[(r, w)] = result
            cost[r, w] = float(np.mean([c for _,_,c in result.matches]))

    finite_cost = np.where(np.isfinite(cost), cost, _LARGE_FINITE_COST)
    row_idx, col_idx = linear_sum_assignment(finite_cost)   # Hungarian LẦN THỨ 3, ở tầng room<->wall
    ...
```

**Giải thích**: gom `hub_detections` theo tường (`group_by_wall`, [`alignment.py:524-548`](../src/sgd_alignment/matching/alignment.py#L524-L548) —
gộp opening có `normal` gần **cùng hướng**, `dot >= 0.9`, chú ý là cùng **chiều** chứ không
chỉ song song — 2 tường đối diện của hành lang có normal gần đối song song, không được gộp
chung). Sau đó **thử mọi tổ hợp (phòng, tường hub)** qua `align_indoor_outdoor` bị giới hạn
chỉ với tường đó, ghi lại chi phí trung bình các match. 1 Hungarian ở tầng cao nhất
(room × wall) chọn phép gán tổng-chi-phí-nhỏ-nhất, đảm bảo **không có 2 phòng nào giành
cùng 1 tường**. Bonus miễn phí: 1 phòng có sẵn "cửa giả" nội bộ (thông ra 1 không gian khác
không phải hub) tự động thua chi phí khi so với tường hub thật, vì mỗi lần thử chỉ giới hạn
đúng 1 tường — không có cơ hội "cửa giả" đó cướp mất 1 slot tường hub khác.

---

### B6. Nhánh thay thế "up-free": `robust_align.py`

**Vấn đề giải quyết**: toàn bộ nhánh B1–B5 phụ thuộc `Detection3D.u_axis/v_axis/normal` —
mà các trục này lại phụ thuộc `up` đã ước lượng ở A6. Nếu `up` sai (case khó — scene ít đặc
trưng hình học), mọi thứ phía trên sai theo mà không có cách tự phát hiện. `robust_align.py`
thiết kế lại matching **không tin bất kỳ đại lượng nào phụ thuộc up** — tính lại descriptor
trực tiếp từ cụm điểm thô, "sign-free" (không giả định trước dấu/hướng).

**Code — descriptor up-free 1 opening**: [`robust_align.py:81-145`](../src/sgd_alignment/matching/robust_align.py#L81-L145)

```python
def _robust_plane(P, iters=3, trim_pct=85.0):
    """Fit mặt phẳng, lặp trim outlier theo khoảng cách trực giao."""
    keep = P; center = keep.mean(0)
    for _ in range(iters):
        Q = keep - center
        _, sing, Vt = np.linalg.svd(Q, full_matrices=False)
        normal = Vt[2]
        d = np.abs(Q @ normal)
        thr = max(np.percentile(d, trim_pct), 1e-9)
        m = d <= thr
        if m.sum() < max(30, 0.3*len(keep)): break
        keep = keep[m]; center = keep.mean(0)
    Q = keep - center
    _, sing, Vt = np.linalg.svd(Q, full_matrices=False)
    return center, Vt, sing, keep

def _min_area_rectangle(P2, n_ang=90):
    """Quét 90 góc trong [0,90 độ), với mỗi góc đo bbox theo percentile[2,98],
    chọn góc có DIỆN TÍCH nhỏ nhất -> hình chữ nhật bao tối thiểu."""
    best = None
    for k in range(n_ang):
        ang = (np.pi/2.0) * k / n_ang
        ca, sa = np.cos(ang), np.sin(ang)
        x = P2[:,0]*ca + P2[:,1]*sa
        y = -P2[:,0]*sa + P2[:,1]*ca
        x2, x98 = np.percentile(x,2), np.percentile(x,98)
        y2, y98 = np.percentile(y,2), np.percentile(y,98)
        area = (x98-x2)*(y98-y2)
        if best is None or area < best[0]:
            cx, cy = 0.5*(x2+x98), 0.5*(y2+y98)
            center2 = np.array([cx*ca-cy*sa, cx*sa+cy*ca])
            best = (area, center2, tuple(sorted((abs(x98-x2), abs(y98-y2)))))
    return best[1], best[2]

def build_opening(cluster, min_points=150) -> RobustOpening | None:
    P = cluster.points[np.isfinite(cluster.points).all(1)]
    if len(P) < min_points: return None
    center0, Vt, sing, kept = _robust_plane(P)
    e1, e2, e3 = Vt[0], Vt[1], Vt[2]
    s = sing / (sing[0] + 1e-12)
    planarity = float(s[1] - s[2]); linearity = float(1.0 - s[1])
    normal_reliable = (s[2] < 0.30) and (s[1] > 0.12)   # mảng phẳng thật, không phải 1 đường thẳng
    Q = kept - center0
    P2 = np.stack([Q @ e1, Q @ e2], axis=1)
    center2, extents = _min_area_rectangle(P2)
    center = center0 + center2[0]*e1 + center2[1]*e2
    size_reliable = normal_reliable and extents[0] > 1e-4
    return RobustOpening(category=cluster.category, center=center,
                          normal_line=e3/(np.linalg.norm(e3)+1e-12),   # KHÔNG ĐỊNH HƯỚNG (n ≡ -n)
                          extents=extents, ...)
```

**Giải thích**:
1. **`_robust_plane`**: fit mặt phẳng qua SVD, **lặp trim outlier** theo phần trăm khoảng
   cách trực giao lớn nhất (`trim_pct=85` — giữ 85% điểm gần mặt phẳng nhất mỗi vòng) —
   khác `_extract_planes` (RANSAC ngẫu nhiên) ở chỗ đây là fit trực tiếp + trim dần, phù hợp
   khi cụm điểm nhỏ (1 opening, không phải cả scene) và không cần robust tới mức RANSAC đầy
   đủ. Điều kiện dừng sớm: nếu trim làm mất quá nhiều điểm (`< max(30, 30%)` còn lại) thì
   dừng lặp — tránh trim quá tay tới mức không còn đủ điểm để fit tin cậy.
2. **`_min_area_rectangle`**: quét 90 góc trong `[0°, 90°)` (không cần quét hết 360° vì hình
   chữ nhật đối xứng 90°), với mỗi góc đo bbox theo trục xoay đó bằng percentile [2,98] (robust
   với outlier), chọn góc cho **diện tích nhỏ nhất** — đây là bài toán "minimum bounding
   rectangle" cổ điển, giải xấp xỉ bằng lưới góc thay vì thuật toán "rotating calipers" chính
   xác (đủ tốt vì opening là hình chữ nhật gần đều, không cần chính xác tuyệt đối).
3. **`extents` sắp theo `(nhỏ, lớn)` — không gọi là width/height**: đây là điểm khác biệt cốt
   lõi so với `Detection3D` — vì **chưa biết `up`**, không thể biết cạnh nào là "rộng" cạnh
   nào là "cao" theo nghĩa vật lý; chỉ biết 2 cạnh có độ dài bao nhiêu. Việc so khớp giữa 2
   scene dựa trên `extents` do đó **tự động bất biến với việc `up` có bị nhầm 90° hay không**.
4. **`normal_line` là "đường không định hướng"** (`n ≡ -n`, không tự ý chọn dấu) — khác hẳn
   `Detection3D.normal` (đã cố định hướng "ra ngoài" ở A8). Đây là điểm mấu chốt của toàn bộ
   file: mọi phép so sánh về sau (`_score`, `_matched_wall_normal`...) đều dùng `abs(dot(...))`
   thay vì `dot(...)` trực tiếp — không bao giờ giả định trước dấu nào đúng.
5. **`normal_reliable = (s[2] < 0.30) and (s[1] > 0.12)`**: dùng tỉ lệ singular value đã
   chuẩn hoá (`s = sing/sing[0]`) để phân biệt 3 dạng suy biến: mảng phẳng thật (2 singular
   value đầu lớn, cái 3 nhỏ) vs. 1 đường thẳng/cạnh mỏng (chỉ 1 singular value lớn — `s[1]`
   cũng nhỏ) vs. 1 khối 3D đặc (`s[2]` không nhỏ). Chỉ khi thực sự là mảng phẳng mới tin
   `normal_line` để dùng làm bằng chứng hướng tường ở các bước sau.

**Code — Umeyama (giống B3 nhưng có scale + có thể cho phép gương)**: [`robust_align.py:151-165`](../src/sgd_alignment/matching/robust_align.py#L151-L165)

```python
def umeyama(src, dst, allow_reflection=False):
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (Xs**2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / (var_s + 1e-12))
    t = mu_d - s * (R @ mu_s)
    return s, R, t
```
Công thức Umeyama chuẩn (SVD của hiệp phương sai chéo, scale = `trace(D·S)/var(src)`).
Tham số `allow_reflection` **chỉ dùng cho chẩn đoán** ở bước reflection-check phía dưới,
không dùng để tạo transform thật.

**Code — sinh hypothesis (tam giác 3 điểm hoặc 2 điểm + normal)**: [`robust_align.py:347-421`](../src/sgd_alignment/matching/robust_align.py#L347-L421)

```python
def _triangle_bad(pts, min_height_ratio=0.12):
    """Loại tam giác gần suy biến (3 điểm gần thẳng hàng): tỷ lệ chiều cao/cạnh dài nhất quá nhỏ."""
    e = [np.linalg.norm(pts[i]-pts[j]) for i,j in [(0,1),(0,2),(1,2)]]
    longest = max(e)
    if longest < 1e-6: return True
    area = 0.5 * np.linalg.norm(np.cross(pts[1]-pts[0], pts[2]-pts[0]))
    altitude = 2*area/longest
    return (altitude/longest) < min_height_ratio

def _triangle_hypotheses(A, B, tau_scene, T_center, log_scale_tol=0.35):
    hyps = []
    for ia in itertools.combinations(range(len(A)), 3):        # mọi tam giác 3 điểm bên A
        if _triangle_bad(cA[list(ia)]): continue
        for ib in itertools.permutations(range(len(B)), 3):    # mọi hoán vị 3 điểm bên B
            if any(A[ia[k]].category != B[ib[k]].category for k in range(3)): continue
            if _triangle_bad(cB[list(ib)]): continue
            # kiểm tra 3 cạnh tam giác có tỷ lệ SCALE nhất quán (log-scale) trước khi fit
            ls = [np.log(||cB[p]-cB[q]|| / ||cA[p]-cA[q]||) for (p,q) in [(0,1),(0,2),(1,2)]]
            if (max(ls) - min(ls)) > log_scale_tol: continue
            s, R, t = umeyama(cA[list(ia)], cB[list(ib)])
            if s > 1e-6: hyps.append((s, R, t))
    return hyps
```

**Giải thích**: vì số lượng opening trong 1 scene thường nhỏ (vài đến vài chục), có thể
**vét cạn gần hết** không gian hypothesis thay vì phải tối ưu thống kê phức tạp:
1. Với mỗi tổ hợp 3 điểm bên A (không thẳng hàng, kiểm bằng `_triangle_bad`) và mỗi hoán vị
   3 điểm bên B cùng category tương ứng, kiểm tra **3 cạnh tam giác có cùng tỷ lệ scale**
   trước khi tốn công chạy Umeyama (lọc rẻ trước khi lọc đắt — 1 phép log-subtract rẻ hơn
   nhiều so với SVD 3×3, dù SVD 3×3 vốn đã rẻ, nhưng số tổ hợp có thể lớn).
2. Nếu không tam giác nào khả thi (ví dụ scene chỉ có 2 opening), fallback
   `_two_opening_hypotheses` ([`robust_align.py:376-410`](../src/sgd_alignment/matching/robust_align.py#L376-L410)): 2 center xác định baseline (scale +
   2 bậc tự do rotation), quét **roll quanh trục baseline** (72 mẫu góc đều nhau trong
   `[0°,360°)`), tìm local minima của tổng sai lệch normal — mỗi local minimum là 1
   hypothesis riêng (vì roll có thể có nhiều nghiệm cục bộ hợp lý).

**Code — chấm điểm hypothesis bằng Hungarian có "dummy node"**: [`robust_align.py:297-341`](../src/sgd_alignment/matching/robust_align.py#L297-L341)

```python
def _score(A, B, s, R, t, T_center, T_angle=np.deg2rad(25), T_logsize=0.45, hard_normal=True):
    nA, nB = len(A), len(B)
    cA = transform_points(np.array([o.center for o in A]), s, R, t)
    cost = np.full((nA, nB), 1e6)
    for i in range(nA):
        for j in range(nB):
            if A[i].category != B[j].category: continue
            cres = np.linalg.norm(cA[i] - B[j].center) / T_center
            if cres >= 1.0: continue                 # GATE CỨNG: center phải đủ gần
            if hard_normal and A[i].normal_reliable and B[j].normal_reliable:
                dot = abs(float(np.dot(R @ A[i].normal_line, B[j].normal_line)))
                if np.arccos(np.clip(dot,0,1)) / T_angle >= 1.0: continue   # GATE CỨNG: normal
            tie = 0.0
            if A[i].normal_reliable and B[j].normal_reliable:
                tie += 0.05 * (np.arccos(np.clip(dot,0,1)) / T_angle)
            if A[i].size_reliable and B[j].size_reliable:
                la = np.log(np.array(A[i].extents)*s + 1e-9); lb = np.log(np.array(B[j].extents)+1e-9)
                tie += 0.02 * min(float(np.max(np.abs(la-lb))) / T_logsize, 2.0)
            cost[i, j] = cres + tie
    NOMATCH = 1.3
    cost[cost >= NOMATCH] = 1e6
    size = nA + nB
    C = np.full((size, size), 1e6)
    C[:nA, :nB] = cost
    for i in range(nA): C[i, nB+i] = NOMATCH      # mỗi object A có 1 lựa chọn "không khớp ai"
    for j in range(nB): C[nA+j, j] = NOMATCH      # mỗi object B cũng vậy
    C[nA:, nB:] = 0.0                             # dummy<->dummy: miễn phí, không phạt gì
    row, col = linear_sum_assignment(C)
    return [(r, c, float(cost[r,c])) for r,c in zip(row,col) if r<nA and c<nB and cost[r,c] < 1e6]
```

**Giải thích kỹ thuật "dummy node"**: Hungarian chuẩn **bắt buộc** gán đủ mọi hàng với mọi
cột (perfect matching trên ma trận vuông) — không có khái niệm "không gán". Nhưng thực tế
2 scene có thể có **số lượng opening khác nhau**, hoặc 1 opening bên A không có đối tác thật
bên B. Kỹ thuật chuẩn để xử lý: mở rộng ma trận `nA×nB` thành `(nA+nB)×(nA+nB)`, thêm 1
"lựa chọn không khớp" giá `NOMATCH=1.3` cố định cho mỗi object, và 1 khối `dummy×dummy` giá
0 để Hungarian luôn có nghiệm hợp lệ. Kết quả: Hungarian tự do chọn "không khớp" cho object
nào không có đối tác đủ tốt (`cost >= NOMATCH`), thay vì bị ép gán bừa. Cost chính là
**center residual đã chuẩn hoá** (chia `T_center` — 1 "đơn vị khoảng cách đặc trưng" của
scene B, xem bên dưới); phần `tie` (0.05× lệch góc normal + 0.02× lệch log-size) chỉ là
tiebreaker nhỏ, **không phải gate** — 2 gate cứng thực sự là `cres < 1.0` và (nếu
`hard_normal=True`) góc normal `< T_angle=25°`.

**Code — vòng lặp chính, refit, và fallback nới lỏng**: [`robust_align.py:424-496`](../src/sgd_alignment/matching/robust_align.py#L424-L496)

```python
def match_openings(clusters_A, clusters_B, tau_scene=0.18, size_floor_frac=0.5, ...):
    A = [build_opening(c) for c in clusters_A if ... is not None]
    B = [build_opening(c) for c in clusters_B if ... is not None]
    if len(A) < 2 or len(B) < 2:
        return AlignResult(..., "NO_SOLUTION", reason=f"too few valid openings ...")

    L_B = _median_pairwise(cB)                          # đơn vị "khoảng cách đặc trưng" của scene B
    med_ext_B = median(extents trung bình mọi opening B)
    T_center = max(tau_scene * L_B, size_floor_frac * med_ext_B, 1e-4)

    hyps = _triangle_hypotheses(A, B, tau_scene, T_center) or _two_opening_hypotheses(A, B)
    scored = [(len(m), -mean_residual, s, R, t, m) for (s,R,t) in hyps for m in [_score(A,B,s,R,t,T_center)]]
    scored.sort(reverse=True)                            # ưu tiên NHIỀU inlier nhất, rồi residual thấp nhất
    n_in, _, s, R, t, matches = scored[0]

    matches, s, R, t = _refit(matches, s, R, t)           # lặp tối đa 3 vòng: refit Umeyama trên inlier hiện tại

    if n_in < 3:      # fallback: bỏ gate normal, chấp nhận kết quả yếu hơn thay vì NO_SOLUTION thẳng
        be = [(len(_score(A,B,s2,R2,t2,T_center,hard_normal=False)), s2,R2,t2, ...) for (s2,R2,t2) in hyps]
        be.sort(reverse=True)
        if be and be[0][0] > n_in:
            _, s, R, t, matches = be[0]
            matches, s, R, t = _refit(matches, s, R, t)
```

**Giải thích**:
1. **`T_center`** — ngưỡng "đủ gần để coi là 1 opening" — **luôn tính từ scene B cố định**
   (không phụ thuộc assignment đang xét), lấy max giữa 2 mốc: tỷ lệ `tau_scene=0.18` của
   khoảng cách trung vị giữa mọi cặp opening B, hoặc phân nửa (`size_floor_frac=0.5`) kích
   thước trung bình 1 opening B — mốc thứ 2 đảm bảo `T_center` không nhỏ hơn kích thước vật
   lý thật của 1 cửa (nếu chỉ có 2-3 opening rất gần nhau, mốc thứ nhất có thể ra số quá
   nhỏ phi thực tế).
2. **Ưu tiên xếp hạng `(n_in, -residual)`**: hypothesis nào thu được nhiều inlier nhất thắng
   trước — càng nhiều object đồng thuận với 1 giả thuyết transform càng đáng tin (giống tiêu
   chí RANSAC chuẩn); hoà số inlier thì chọn residual trung bình thấp hơn.
3. **`_refit`** lặp tối đa 3 vòng: từ tập inlier hiện tại → Umeyama lại (hoặc chỉ refit
   scale+translation nếu center thẳng hàng — `_refit_scale_t`, vì rotation không thể xác
   định từ dữ liệu thẳng hàng) → `_score` lại để cập nhật tập inlier → lặp. Đúng tinh thần
   ICP: mỗi vòng làm chặt thêm cả transform lẫn tập inlier.
4. **Fallback nới gate normal** khi `n_in < 3`: đây là triết lý "best-effort" — thay vì trả
   `NO_SOLUTION` cứng nhắc khi bằng chứng hơi yếu, thử lại toàn bộ hypothesis với
   `hard_normal=False` (bỏ gate góc normal, chỉ giữ gate center), và nếu kết quả **thực sự
   tốt hơn** (`> n_in`) thì dùng — nhưng kết quả cuối vẫn bị đánh dấu `LOW_CONFIDENCE` (xem
   phần phân loại status bên dưới), không giả vờ là `CONFIDENT`.

**Code — chẩn đoán chirality/reflection & phân loại status cuối**: [`robust_align.py:549-639`](../src/sgd_alignment/matching/robust_align.py#L549-L639)

```python
if n_in >= 3:
    s_r, R_r, t_r = umeyama(srcs, dsts, allow_reflection=True)   # cho phép cả nghiệm "gương"
    err_proper = mean(||transform(srcs, s,R,t) - dsts||)
    err_impro  = mean(||transform(srcs, s_r,R_r,t_r) - dsts||)
    rank = matrix_rank(centered_srcs)
    chirality_observable = rank >= 3 and abs(err_proper - err_impro) > 0.1*T_center
    reflection_preferred = chirality_observable and det(R_r) < 0 and err_impro < err_proper - 1e-9

single_wall = min(walls_A, walls_B) < 2       # số hướng tường ĐỘC LẬP mà các match trải qua
decisive = (n_in - second_inliers) >= 1       # có "ứng viên thứ 2" gần bằng không

if n_in >= 3 and walls_A >= 2 and walls_B >= 2 and decisive and not reflection_preferred:
    status = "CONFIDENT"
elif n_in >= 1 and single_wall:
    status = "AMBIGUOUS"; provisional = True   # xem B7 - đây chính là ca gravity_align.py giải quyết
elif n_in >= 1:
    status = "LOW_CONFIDENCE"
else:
    status = "NO_SOLUTION"
```

**Giải thích**:
- **Chirality check**: so sánh fit "đúng" (rotation thuần, `det=+1`) với fit "cho phép
  gương" (`det` bất kỳ, bằng cách bật `S[2,2]=-1` có điều kiện trong `umeyama`) trên **cùng**
  tập match cuối cùng. Nếu fit-gương tốt hơn **đáng kể** (`err_impro < err_proper` và
  `chirality_observable` — tức dữ liệu đủ hạng 3D để phân biệt được 2 khả năng này, không
  phải trường hợp thoái hoá) thì đây là dấu hiệu mạnh: correspondence hiện tại **sai** (đúng
  vật lý không thể có phép "lật gương" giữa 2 scan cùng 1 công trình thật).
- **`single_wall`**: đếm số **hướng tường độc lập** (không phải số opening) mà tập match đi
  qua, dùng `_num_wall_dirs` (gộp normal gần cùng hướng, `dot_thr=0.9`, chỉ tính opening có
  `normal_reliable`). Nếu dưới 2 hướng độc lập ở **1 trong 2 phía** → roll quanh trục pháp
  tuyến tường đó **không quan sát được** từ riêng dữ liệu opening (đây chính là bài toán
  "rank-1 degeneracy" mà `gravity_align.py` được viết ra để giải — xem B7).
- **4 mức status**: `CONFIDENT` (đủ 3 điều kiện: ≥3 inlier, ≥2 hướng tường độc lập mỗi bên,
  quyết đoán hơn ứng viên thứ 2, và không có dấu hiệu gương) → `AMBIGUOUS` (rank-1, transform
  chỉ mang tính "provisional/preview", không authoritative) → `LOW_CONFIDENCE` (ít match hoặc
  cạnh tranh sít sao hoặc nghi ngờ gương) → `NO_SOLUTION`.

---

### B7. Nhánh gravity-locked: `gravity_align.py`

**Vấn đề giải quyết**: giải quyết đúng ca `AMBIGUOUS/rank-1` của B6 — khi mọi opening khớp
được chỉ nằm trên **1 tường chung**, roll quanh pháp tuyến tường đó (và với 2 cửa nằm ngang
hàng, cả "cửa nào bên nào" + "trên/dưới") **không quan sát được** chỉ từ hình học opening.
File này bổ sung 2 tín hiệu **độc lập, đáng tin** lấy thẳng từ dữ liệu camera gốc để phá vỡ
sự mơ hồ đó.

**Code — 2 tín hiệu camera lấy từ `.npz`**: [`gravity_align.py:43-64`](../src/sgd_alignment/matching/gravity_align.py#L43-L64)

```python
@dataclass
class CameraEvidence:
    centers: np.ndarray     # (N,3) vị trí camera trong world
    up: np.ndarray          # (3,) gravity trung bình, hướng lên trần
    up_consistency: float   # median cos(up từng frame, up trung bình)
    forwards: np.ndarray

def camera_evidence(npz_path: str) -> CameraEvidence:
    Z = np.load(npz_path)
    EX = Z["extrinsics"].astype(np.float64)
    R, t = EX[:,:3,:3], EX[:,:3,3]
    centers = np.einsum("nij,nj->ni", np.transpose(R,(0,2,1)), -t)   # C = -R^T @ t, cho MỌI frame
    up_per = -R[:,1,:]; up_per /= norm(up_per, axis=1, keepdims=True)+1e-12
    up = up_per.mean(0); up /= norm(up)+1e-12
    consistency = median(up_per @ up)
    fwd = R[:,2,:]; fwd /= norm(fwd, axis=1, keepdims=True)+1e-12
    return CameraEvidence(centers=centers, up=up, up_consistency=consistency, forwards=fwd)
```

Đây chính là công thức đã gặp ở A1/A6 (`C=-R^T@t`, `up=-R[1,:]`), áp dụng **vector hoá**
cho mọi frame cùng lúc (`np.einsum`) thay vì lặp per-frame — cùng 1 công thức, chỉ khác chỗ
đặt để phục vụ trực tiếp bài toán gravity-lock.

**Code — khoá rotation quanh gravity, giải bằng số phức**: [`gravity_align.py:70-109`](../src/sgd_alignment/matching/gravity_align.py#L70-L109)

```python
def _basis_from_up(up):
    """Hệ trực chuẩn [e1, e2, up] -- nhân với x đưa up của x về +Z."""
    w = up / (norm(up)+1e-12)
    a = [1,0,0] if abs(w[0]) < 0.9 else [0,1,0]
    e1 = cross(a, w); e1 /= norm(e1)+1e-12
    e2 = cross(w, e1)
    return np.stack([e1, e2, w], axis=0)

def gravity_locked_similarity(srcs, dsts, up_src, up_dst):
    Ua, Ub = _basis_from_up(up_src), _basis_from_up(up_dst)
    a, b = srcs @ Ua.T, dsts @ Ub.T                 # đổi sang hệ chuẩn hoá, up -> +Z
    axy, az = a[:,:2], a[:,2]; bxy, bz = b[:,:2], b[:,2]
    ca, cb = axy.mean(0), bxy.mean(0)
    A0, B0 = axy-ca, bxy-cb
    za = A0[:,0] + 1j*A0[:,1]; zb = B0[:,0] + 1j*B0[:,1]     # biểu diễn (x,y) dạng số phức
    w = (za.conj() * zb).sum() / ((za.conj()*za).real.sum() + 1e-12)
    s, phi = float(abs(w)), float(np.angle(w))               # |w|=scale, arg(w)=góc xoay
    Rz = [[cos(phi),-sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]]
    tz = mean(bz - s*az)
    txy = cb - s*(Rz[:2,:2] @ ca)
    R = Ub.T @ Rz @ Ua
    t = Ub.T @ [txy[0], txy[1], tz]
    return s, R, t, horiz_span
```

**Giải thích công thức "least-squares phức"**: khi đã khoá trục up trùng nhau ở cả 2 scene
(bằng cách đổi cơ sở `Ua`, `Ub`), bài toán "tìm rotation+scale tối ưu quanh trục up" tương
đương bài toán **2D**: tìm `(scale, phase)` biến đổi tập điểm phức `za` khớp `zb` tốt nhất.
Với số phức, phép "nhân với `w = scale·e^{iφ}`" **chính là** phép xoay `φ` + scale `s` cùng
lúc — nên bài toán trở thành least-squares tuyến tính trên số phức có nghiệm đóng:
```
w* = argmin_w Σ|w·za_k - zb_k|²  =  Σ(za_k* · zb_k) / Σ(za_k* · za_k)
```
(đạo hàm theo `w*`, cho bằng 0 — giống hệt least-squares thực, chỉ khác trên trường số
phức). Đây là cách gọn nhất để giải **đồng thời** rotation 2D + scale mà không cần SVD.
Thành phần Z (chiều cao) được xử lý tách riêng: cùng 1 scale `s` (đồng nhất/uniform) nhưng
translation `tz` giải bằng trung bình đơn giản (vì đã biết rotation quanh trục up không ảnh
hưởng thành phần Z).

**Code — 1 opening duy nhất vẫn giải được (2 candidate azimuth)**: [`gravity_align.py:112-130`](../src/sgd_alignment/matching/gravity_align.py#L112-L130)

```python
def gravity_locked_single(oA, oB, up_src, up_dst):
    s = mean(oB.extents) / (mean(oA.extents) + 1e-9)          # tỷ lệ kích thước -> scale
    nA, nB = Ua@oA.normal_line, Ub@oB.normal_line              # normal về hệ chuẩn hoá
    aA, aB = arctan2(nA[1],nA[0]), arctan2(nB[1],nB[0])        # góc phương vị của normal
    out = []
    for flip in (0.0, np.pi):           # normal KHÔNG ĐỊNH HƯỚNG -> 2 khả năng góc cách nhau 180°
        phi = aB - aA + flip
        Rz = [[cos(phi),-sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]]
        R = Ub.T @ Rz @ Ua
        t = oB.center - s*(R @ oA.center)
        out.append((s, R, t))
    return out
```

Với **chỉ 1 opening chung**, vẫn đủ ràng buộc: gravity cố định trục dọc, **kích thước**
(extents) cho scale, **normal** (dù không định hướng) cho góc phương vị **sai khác đúng 1
trong 2 giá trị cách nhau 180°** (vì `n ≡ -n`), **center** cho translation. Trả về **cả 2**
candidate — bước sau (camera-side) sẽ chọn cái đúng.

**Code — loại bỏ detection trùng (VD: 1 cửa kính bị gắn cả nhãn "door" lẫn "window")**: [`gravity_align.py:174-211`](../src/sgd_alignment/matching/gravity_align.py#L174-L211)

```python
def _dedupe_openings(clusters, up, iom_thr=0.55, ang_thr_deg=15.0):
    """Quyết định gộp dựa trên độ CHỒNG LẤN FOOTPRINT (intersection-over-smaller-area)
    trên mặt phẳng tường, cộng normal gần song song + độ sâu gần bằng nhau."""
    info = [_wall_footprint(c.points, up) for c in clusters]   # (normal, center, u, v, bbox2d)
    keep = [True]*len(clusters)
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            ... # nếu góc normal < 15 độ VÀ IoM >= 0.55 VÀ chênh lệch độ sâu < 30% cạnh nhỏ hơn:
            ...   # giữ footprint LỚN HƠN (đầy đủ hơn), loại cái nhỏ hơn
    return [clusters[k] for k in range(len(clusters)) if keep[k]]
```

**Vấn đề thực tế nó giải quyết**: cửa kính (glass door) thường bị Grounding DINO gắn **cả
2 nhãn** "door" và "window" cùng lúc (đều đúng theo nghĩa ngôn ngữ) → A4 (`merge_instances`)
không gộp được vì merge theo **cùng nhãn**, ra 2 `Detection3D` riêng biệt cho cùng 1 lỗ hổng
vật lý — nếu không lọc, matching sẽ coi đó là 2 opening độc lập, sai số lượng đối tượng.
`_dedupe_openings` dùng tiêu chí hình học (chồng lấn footprint trên **cùng mặt phẳng
tường**, không phải trùng box 2D) để nhận ra 2 detection này thực chất là 1 vật thể.

**Code — tính "wall normal chung" từ nhiều opening đã match (đáng tin hơn gravity)**: [`gravity_align.py:214-238`](../src/sgd_alignment/matching/gravity_align.py#L214-L238)

```python
def _agg_wall_normal(ops, idxs, up):
    """Gộp normal của các opening đã match thành 1 wall normal chung, dùng
    ma trận outer-product (sign-invariant: n và -n cho CÙNG 1 ma trận)."""
    ns = [o.normal_line for i in idxs for o in [ops[i]] if o.normal_reliable
          and abs(dot(o.normal_line, up)) < 0.34]     # phải đủ "đứng" (lệch <~20 độ so phương ngang)
    if not ns: return None, False
    Q = sum(np.outer(n, n) for n in ns)                # ma trận 3x3 sign-invariant
    w, V = np.linalg.eigh(Q)
    n_agg = V[:, -1]                                    # eigenvector ứng eigenvalue lớn nhất
    resid = [degrees(arccos(clip(abs(dot(n, n_agg)),0,1))) for n in ns]
    reliable = len(ns) >= 1 and median(resid) < 8.0
    return n_agg, reliable
```

**Giải thích kỹ thuật "ma trận outer-product sign-invariant"**: muốn lấy trung bình của
nhiều vector **không định hướng** (`n` và `-n` là cùng 1 hướng vật lý) — cộng thẳng vector
lại sẽ sai (`n + (-n) = 0`, triệt tiêu nhau dù cả 2 đều "đúng"). Giải pháp chuẩn: xây ma
trận `Q = Σ nᵢnᵢᵀ` (outer product tự triệt tiêu vấn đề dấu vì `(-n)(-n)ᵀ = nnᵀ`), rồi lấy
**eigenvector ứng với eigenvalue lớn nhất** của `Q` làm hướng trung bình — đây là bài toán
PCA cho dữ liệu "đường không định hướng" (tương tự cách tính trục chính của 1 tập đường
thẳng, không phải tập điểm).

**Code — khoá rotation chặt hơn bằng ràng buộc vật lý "2 mặt tường đối song song"**: [`gravity_align.py:241-275`](../src/sgd_alignment/matching/gravity_align.py#L241-L275)

```python
def _refine_normal_lock(A, B, pi, s, R, t, upA, upB, max_dev_deg=25.0):
    """2 KHÔNG GIAN GẶP NHAU ĐÚNG 1 MẶT PHẲNG -> R @ nA phải = -nB (đối song song).
    Wall normal là trục CHÍNH (tin hơn gravity hơi lệch), gravity là trục PHỤ (cố định roll)."""
    nA, okA = _agg_wall_normal(A, idxA, upA); nB, okB = _agg_wall_normal(B, idxB, upB)
    if not (okA and okB): return R, t, False
    gA = uA - (uA@nA)*nA; gA /= norm(gA)+1e-12          # thành phần gravity VUÔNG GÓC với normal
    FA = np.stack([nA, gA, cross(nA,gA)], axis=1)        # hệ trục A: [normal, gravity_perp, thứ 3]
    best = None
    for sn in (-1.0, 1.0):                                # thử cả 2 dấu của nB (chưa định hướng)
        FB = np.stack([sn*nB, gB, cross(sn*nB,gB)], axis=1)
        Rn = FB @ FA.T                                    # rotation đưa hệ A khớp hệ B
        U,_,Vt = svd(Rn); Rn = U @ diag([1,1,sign(det(U@Vt))]) @ Vt   # ép về rotation thuần
        ang = arccos(clip((trace(Rn@R.T)-1)/2, -1, 1))    # góc lệch giữa Rn và R gốc
        if best is None or ang < best[0]: best = (ang, Rn)
    ang, Rn = best
    if degrees(ang) > max_dev_deg: return R, t, False     # lệch quá xa R gốc -> không tin, giữ nguyên
    t_new = median(cB - s*(cA @ Rn.T), axis=0)             # refit translation robust (median, không mean)
    return Rn, t_new, True
```

**Giải thích**: đây là bước "khoá chặt" — biết chắc về mặt vật lý rằng normal tường của A
sau khi xoay phải **đối song song đúng** với normal tường của B (2 phòng gặp nhau ở đúng 1
mặt tường chung). Xây 2 hệ trục trực chuẩn `FA`, `FB` với **normal làm trục chính** (không
phải gravity — vì normal đo trực tiếp từ hình học mặt tường, đáng tin hơn gravity vốn có
thể lệch nhẹ do rig nghiêng), gravity chỉ dùng để cố định trục phụ (roll quanh normal).
`Rn = FB @ FA^T` là rotation đưa hệ A khớp hệ B theo đúng 2 trục đó. Vì `nB` chưa định
hướng, thử **cả 2 dấu**, chọn dấu nào cho `Rn` **gần với `R` gốc nhất** (giữ nguyên "phe"
mà candidate ban đầu đã chọn — quyết định cuối "đúng phe nào" nhường lại cho camera-side ở
bước sau). Có "van an toàn" `max_dev_deg=25°`: nếu khoá theo normal kéo rotation lệch quá xa
so với ước lượng ban đầu (từ center+gravity), coi là **wall normal không đáng tin cho case
này**, giữ nguyên `R` gốc thay vì tin mù quáng.

**Code — tinh chỉnh roll để khớp opening trong mặt phẳng**: [`gravity_align.py:278-316`](../src/sgd_alignment/matching/gravity_align.py#L278-L316)

```python
def _refine_roll_openings(A, B, pi, s, R, t, wall_n, max_deg=12.0, n_samples=97):
    """Rotation QUANH pháp tuyến tường giữ nguyên tính đồng phẳng (không đánh đổi gì) ->
    tối ưu tự do để khớp center các opening TRONG mặt phẳng tường."""
    def _inplane_res(Rn):
        d = s*(cA @ Rn.T) - cB
        d = d - d.mean(0)                      # bất biến với translation
        d = d - outer(d @ n, n)                # chỉ giữ thành phần TRONG mặt phẳng tường
        return mean(norm(d, axis=1))

    res0 = _inplane_res(R)
    best = (res0, 0.0, R)
    for th in linspace(-max_deg, max_deg, n_samples):    # quét vét cạn +-12 độ, 97 mẫu
        Rn = _axis_rotation(n, radians(th)) @ R
        res = _inplane_res(Rn)
        if res < best[0]: best = (res, th, Rn)
    if best[0] >= 0.5 * res0:                  # cải thiện KHÔNG đáng kể -> không commit (tránh overfit)
        Rn, roll = R, 0.0
    else:
        Rn, roll = best[2], best[1]
    tn = median(cB - s*(cA @ Rn.T), axis=0)
    return Rn, tn, roll
```

**Giải thích**: sau khi khoá normal (bước trước), vẫn còn 1 bậc tự do "miễn phí" — xoay
quanh chính trục pháp tuyến tường **không phá vỡ** tính đồng phẳng vừa khoá (đồng phẳng chỉ
phụ thuộc hướng normal, không phụ thuộc roll quanh nó) — nên có thể tự do tối ưu roll này để
khớp vị trí các opening **trong mặt phẳng tường** sát hơn, "miễn phí" về mặt vật lý. Quét vét
cạn (không cần gradient descent vì miền tìm kiếm nhỏ, hàm mục tiêu không nhất thiết lồi) trong
±12° với 97 mẫu. **Ngưỡng chặn overfit** (`best[0] >= 0.5*res0`): nếu residual vốn đã thấp ở
roll=0 (đáy hàm mục tiêu "nông"), 1 roll lớn có thể chỉ đang fit nhiễu ngẫu nhiên của vài
điểm center — không commit, giữ roll=0 để tránh **làm nghiêng sàn nhà** một cách giả tạo
(xoay quanh normal có ảnh hưởng tới trục thẳng đứng).

**Code — điểm số "camera 2 phía đối diện tường"**: [`gravity_align.py:136-151`](../src/sgd_alignment/matching/gravity_align.py#L136-L151)

```python
def camera_side_score(cams_A_t, cams_B, wall_p, wall_n):
    dA = (cams_A_t - wall_p) @ wall_n; dB = (cams_B - wall_p) @ wall_n
    mA, mB = median(dA), median(dB)
    domA, domB = abs(mean(sign(dA))), abs(mean(sign(dB)))     # độ "1 phía" của mỗi tập camera
    opposite = sign(mA) * sign(mB) < 0
    return (1.0 if opposite else -1.0), min(domA, domB), mA, mB
```

**Giải thích**: đúng như đã nói ở phần đầu file — 2 phòng thật gặp nhau ở 1 tường thì camera
quay 2 bên chắc chắn đứng **2 phía đối diện** của tường đó. Dùng **median** (không phải
mean) cho robust với outlier vị trí camera; `dominance` đo mức độ "camera có thực sự dồn hẳn
về 1 phía" (gần 1 = rõ ràng, gần 0 = camera rải đều 2 phía — tín hiệu không đáng tin, ví dụ
camera đi lại qua khung cửa nhiều lần).

**Code — hàm chính, chọn candidate thắng cuộc**: [`gravity_align.py:338-512`](../src/sgd_alignment/matching/gravity_align.py#L338-L512) (rút gọn)

```python
def align_gravity_camera(clusters_A, clusters_B, cam_A, cam_B, tau_scene=0.18):
    clusters_A = _dedupe_openings(clusters_A, cam_A.up)   # gộp door+window trùng
    clusters_B = _dedupe_openings(clusters_B, cam_B.up)
    A = [build_opening(c) for c in clusters_A ...]; B = [...]        # tái dùng robust_align.build_opening
    ...
    if n_min >= 2:
        for mọi hoán vị match (nhỏ hơn <-> tập con lớn hơn):
            s, R, t, span = gravity_locked_similarity(...)
            R, t, locked = _refine_normal_lock(...)                  # khoá theo normal tường
            cand.append({...res, size_err, norm_err...})
    if n_min == 1 or not cand:
        for mọi cặp (i,j) cùng category, normal_reliable cả 2 bên:
            for (s,R,t) in gravity_locked_single(...):               # 2 candidate azimuth
                cand.append({...})

    for c in cand:
        side, dom, mA, mB = camera_side_score(...)
        c["side"], c["dom"] = side, dom
        c["grav"] = dot(c["R"]@upA, upB)
        c["geom"] = c["res"]/T_center + c["size_err"] + c["norm_err"]

    SIZE_TOL, NORM_TOL = 0.35, 0.15
    valid = [c for c in cand if c["side"]>0 and c["size_err"]<=SIZE_TOL
             and c["norm_err"]<=NORM_TOL and c["dom"]>=0.5]
    valid.sort(key=lambda c: c["geom"])

    if valid:
        top = valid[0]
        status = "CONFIDENT"
        if len(valid)>=2 and (valid[1]["geom"]-valid[0]["geom"]) < 0.25:
            status = "AMBIGUOUS"     # 2 candidate gần hoà -> có thể là 2 opening đối xứng thật
    else:
        top = min(cand, key=lambda c: c["geom"]); status = "AMBIGUOUS"
        # reason cụ thể: "không candidate nào camera đối diện" / "không khớp size" / "bằng chứng yếu"

    # roll refine + depth offset (khớp mặt tường "khít", không hở/chồng) áp thêm lên top
    ...
    GRAV_MIN = 0.9
    if status=="CONFIDENT" and min(cam_A.up_consistency, cam_B.up_consistency) < GRAV_MIN:
        status = "AMBIGUOUS"        # gravity chính nó không ổn định -> không dám nhận CONFIDENT
    return GravityAlignResult(...)
```

**Giải thích logic chọn candidate cuối cùng**:
1. **3 tiêu chí lọc "physically valid" (`valid`)**: `side>0` (camera 2 bên **phải** đối
   diện — đây là **gate cứng**, không phải điểm cộng), `size_err<=0.35` (kích thước opening
   khớp dưới cùng 1 scale — vì như đã giải thích ở lượt trước, 2 center bất kỳ luôn "fit"
   được về mặt toán học, chỉ có **kích thước** mới thực sự phân biệt được tường đúng),
   `norm_err<=0.15` (~32°, normal sau xoay phải gần khớp), `dom>=0.5` (camera đủ tập trung
   1 phía, không rải đều đáng ngờ).
2. **Xếp hạng trong nhóm valid bằng `geom`** = residual chuẩn hoá + size error + normal
   error — không xếp theo score đơn lẻ nào mà cộng gộp 3 tín hiệu hình học độc lập.
3. **`AMBIGUOUS` khi 2 candidate top gần hoà (`< 0.25`)** — trung thực báo cáo trường hợp
   thực sự mơ hồ (ví dụ 2 cặp cửa sổ đối xứng hoàn toàn giống nhau) thay vì tự tin chọn bừa
   1 cái.
4. **Khi không candidate nào "valid"**: vẫn trả về **best-effort guess** (theo `geom` thấp
   nhất trong toàn bộ, không chỉ nhóm valid) nhưng đánh dấu rõ `AMBIGUOUS` kèm lý do cụ thể
   (không có tín hiệu camera-side, hoặc không khớp size, hoặc bằng chứng yếu) — không bao
   giờ trả `NO_SOLUTION` im lặng nếu còn có 1 candidate hình học nào đó để tham khảo.
5. **Guard cuối `GRAV_MIN=0.9`**: dù mọi tiêu chí hình học đều đạt `CONFIDENT`, nếu chính
   `up_consistency` của 1 trong 2 camera (đo ở A6) dưới 0.9, hạ xuống `AMBIGUOUS` — vì toàn
   bộ phương pháp gravity-lock **dựa trên giả định gravity đáng tin**; nếu nền tảng đó tự nó
   đã lung lay thì không có kết quả downstream nào xứng đáng nhãn "CONFIDENT".

---

## Phụ lục: bảng tổng hợp mọi ngưỡng lọc lỗi

| Ngưỡng | Giá trị mặc định | File:dòng | Ý nghĩa |
|---|---|---|---|
| `box_threshold` | 0.5 | segment_2d.py:63 | Độ tin cậy tối thiểu box Grounding DINO |
| `text_threshold` | 0.25 | segment_2d.py:64 | Độ khớp token-text tối thiểu |
| `nms_iou_thres` | 0.7 | segment_2d.py:65 | NMS trong cùng nhãn |
| `cross_label_iou_thres` | 0.7 | segment_2d.py:66 | Lọc trùng giữa 2 nhãn door/window |
| `depth_range` | (0.05, 30.0) m | multiview_source.py:39 | Loại depth phi vật lý |
| `min_confidence` | 0.0 (tắt) | multiview_segmentation.py:92 | Loại detection 2D yếu trước backproject |
| `merge_distance` | 0.6 m | multiview_segmentation.py:59 | Gộp các lượt nhìn cùng 1 opening |
| `conf_percentile` | 40 | multiview_source.py:359 | Loại 40% pixel confidence thấp nhất khi build scene cloud |
| `MIN_UP_CONSISTENCY` | 0.90 | multiview_pipeline.py:201 | Ngưỡng tin phương pháp up từ camera rotation |
| `distance_threshold` (RANSAC) | 0.03 m | plane_fitting.py:139 | Ngưỡng inlier mặt phẳng |
| `verticality_max_dot` | 0.3 | plane_fitting.py:465 | Lọc tường (vuông góc up) khỏi sàn/trần |
| `normal_dot_min` / `d_max_diff` | 0.98 / 0.15 | plane_fitting.py:500-501 | Gộp mảnh RANSAC cùng 1 tường |
| `low_confidence_threshold` | 0.15 | opening_geometry.py:23 | Mật độ 2 phía tường gần hoà -> không chắc |
| `max_distance` (nearest wall) | 0.15 m | opening_geometry.py:133 | Tường quá xa -> không gán |
| `wall_normal_agreement_threshold` | 0.85 | opening_geometry.py:164 | Tường gần nhưng lệch hình dạng cụm điểm -> dùng PCA |
| `trim_percentile` | 0.0 (tắt) | opening_geometry.py:163 | Đo width/height robust hơn min/max |
| `distance_threshold` (SGD) | 1.5 m | sgd.py:80 | Ngưỡng khả thi thành phần khoảng cách |
| `*_threshold` (góc, SGD) | π/3 | sgd.py:81-86 | Ngưỡng khả thi các thành phần góc |
| `max_cost` (SGD) | 5.0 | sgd.py:405 | Ngưỡng loại match cuối cùng |
| `ambiguity_ratio_threshold` | 0.8 | sgd.py:408 | Lowe's-ratio-test, tắt mặc định |
| `is_collinear ratio_threshold` | 0.05 | alignment.py:25 | Cảnh báo rotation thiếu ràng buộc |
| `min_dimension` | None (tắt) | alignment.py:404 | Loại detection kích thước phi vật lý |
| `normal_angle_threshold` (RANSAC wall) | π/6 (30°) | alignment.py:281 | Đồng thuận hướng tường |
| `angle_threshold_deg` (single wall) | 20° | alignment.py:43 | Cảnh báo mọi match cùng 1 tường |
| `min_points` (robust_align) | 150 | robust_align.py:123 | Tối thiểu điểm để tin 1 opening |
| `trim_pct` (robust plane) | 85 | robust_align.py:81 | Trim outlier khi fit mặt phẳng |
| `normal_reliable` | s[2]<0.30 & s[1]>0.12 | robust_align.py:134 | Phân biệt mảng phẳng thật với đường/khối |
| `T_angle` / `T_logsize` | 25° / 0.45 | robust_align.py:298 | Gate góc normal / tie-break kích thước |
| `SIZE_TOL` / `NORM_TOL` (gravity) | 0.35 / 0.15 | gravity_align.py:431-432 | Lọc candidate "physically valid" |
| `GRAV_MIN` | 0.9 | gravity_align.py:495 | Guard: không nhận CONFIDENT nếu gravity không ổn định |
| `iom_thr` / `ang_thr_deg` (dedupe) | 0.55 / 15° | gravity_align.py:174 | Gộp door+window trùng vật lý |

---

## Phụ lục: cấu trúc dữ liệu dùng chung

**Code**: [`common/types.py`](../src/sgd_alignment/common/types.py)

```python
@dataclass
class PointCloud:
    points: np.ndarray            # (N,3) float64
    intensity: np.ndarray | None  # (N,) phản xạ LiDAR (nếu có)
    normals: np.ndarray | None    # (N,3)

@dataclass
class Plane:
    normal: np.ndarray            # (3,) unit, phương trình: normal.x + d = 0
    d: float
    inlier_indices: np.ndarray
    def signed_distance(self, points): return points @ self.normal + self.d

@dataclass
class Detection3D:
    category: str                 # "door" | "window"
    center: np.ndarray            # (3,)
    u_axis: np.ndarray            # (3,) unit, chiều rộng
    v_axis: np.ndarray            # (3,) unit, chiều cao (song song up)
    normal: np.ndarray            # (3,) unit, hướng ra ngoài
    width: float
    height: float
    thickness: float = 0.05
    region: str | None = None     # nhãn không gian tuỳ chọn (server/hallway/...)

    def corners(self) -> np.ndarray:
        """8 góc hộp mỏng, tổ hợp mọi dấu (±u, ±v, ±normal)."""
        hu, hv, ht = self.width/2, self.height/2, self.thickness/2
        return np.array([self.center + su*hu*self.u_axis + sv*hv*self.v_axis + sn*ht*self.normal
                          for su in (-1,1) for sv in (-1,1) for sn in (-1,1)])
```

`Detection3D` là **hợp đồng dữ liệu** nối 2 giai đoạn A và B — mọi output của Giai đoạn A
(dù từ luồng multi-view hay CloudCompare thủ công) đều phải quy về đúng cấu trúc này để
Giai đoạn B dùng được mà không cần biết nguồn gốc.
