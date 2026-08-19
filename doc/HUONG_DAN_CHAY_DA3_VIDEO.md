# Hướng dẫn chạy DA3 với dữ liệu VIDEO

Quy trình biến **1 video** thành **`results.npz`** (depth + pose + intrinsics + conf + image) —
dùng làm input cho notebook segment cửa — và **point cloud `.ply`** để xem.

## Sơ đồ tổng quát & chỗ chạy

```
VIDEO.mp4
   │  [1] ffmpeg trích frame        ── chạy ở MÁY (Git Bash)
   ▼
data/<TÊN>/*.jpg  (nhiều ảnh)
   │  [2] thêm mount + deploy       ── MÁY (đẩy code lên Modal)
   │  [3] spawn job                 ── kích hoạt chạy trên MODAL (A100 cloud)
   ▼
DA3-GIANT-1.1 chạy inference        ── chạy trên MODAL (GPU A100-80GB)
   │  [4] fetch kết quả             ── MÁY tải về
   ▼
workspace/<TÊN>/results.npz + depth_vis/
   │  [5] make_ply.py               ── MÁY (Python 3.11)
   ▼
workspace/<TÊN>/<TÊN>_points.ply    ── mở CloudCompare
```

| Bước | Chạy ở đâu | Công cụ |
|---|---|---|
| 1. Trích frame | **MÁY** | ffmpeg |
| 2–3. Deploy + spawn | **MÁY → MODAL** | modal |
| DA3 inference | **MODAL (cloud A100)** | — |
| 4. Fetch | **MÁY** | modal |
| 5. Tạo .ply | **MÁY** | python 3.11 |

> **Lưu ý 2 phiên bản Python:**
> - Bước Modal (deploy/spawn/fetch/stop) dùng `python` mặc định (Python 3.13, nơi cài `modal`).
> - Bước tạo `.ply` dùng **Python 3.11** (`C:/Users/pminh/AppData/Local/Programs/Python/Python311/python.exe`) vì có `plyfile`.

---

## Chuẩn bị 1 lần (đã có sẵn, chỉ kiểm tra)

```bash
# Trong Git Bash, tại thư mục D:/Vkist/DA3
which ffmpeg          # phải ra đường dẫn ffmpeg
modal profile current # phải ra tên profile (đã đăng nhập Modal)
```

File cần có sẵn trong repo (đã có): `modal_export_3dgs.py`, `deploy_retry.sh`, `make_ply.py`.

Ví dụ xuyên suốt: video `data/hanh_lang_dai/IMG_0636.MOV`, đặt tên bộ là **`hanh_lang_dai`**.

---

## Bước 1 — Trích frame từ video (MÁY)

```bash
cd /d/Vkist/DA3

# Đặt tên bộ dữ liệu
NAME=hanh_lang_dai
VIDEO="data/hanh_lang_dai/IMG_0636.MOV"

# Xoá & tạo thư mục frame
rm -rf data/$NAME
mkdir -p data/$NAME

# Trích frame ở 0.7 fps (1 ảnh mỗi ~1.4 giây)
ffmpeg -loglevel error -i "$VIDEO" -vf fps=0.7 -q:v 2 "data/$NAME/${NAME}_%04d.jpg"

# Đếm số frame
ls data/$NAME/*.jpg | wc -l
```

**Chọn `fps` thế nào** (mục tiêu 2 frame liền chồng lấp ~70–80%):

| Loại quay | fps nên dùng | Ghi chú |
|---|---|---|
| Quay tay đi bộ trong nhà | **0.7 – 1.2** | 1 ảnh mỗi ~0.8–1.4s |
| Đi thẳng hành lang dài | **0.5 – 0.7** + nên chia đoạn | overlap kém, dễ nén tỉ lệ |
| Video ngắn (<40s) | **1.2 – 1.5** | để đủ ~50 frame |

**Số frame nên nằm trong khả năng GPU** (xem bảng ở cuối). Với A100-80GB: **~60–120 frame** là hợp lý.
Muốn ra ~N frame từ video dài `T` giây: `fps = N / T`.

---

## Bước 2 — Thêm mount vào `modal_export_3dgs.py` (MÁY)

Mở `modal_export_3dgs.py`, tìm khối `.add_local_dir(...)` (gần dòng ~59–75), **thêm 1 dòng** cho bộ mới:

```python
    .add_local_dir(os.path.join(REPO_ROOT, "data", "hanh_lang_dai"), "/root/data/hanh_lang_dai")
```

> Dòng này bảo Modal đẩy thư mục `data/hanh_lang_dai` lên container thành `/root/data/hanh_lang_dai`.

---

## Bước 3 — Deploy lên Modal (MÁY → MODAL)

```bash
cd /d/Vkist/DA3
: > /tmp/da3_deploy.log            # xoá log cũ
bash deploy_retry.sh              # tự thử lại tới khi deploy được (mạng chập chờn vẫn ổn)

# Xem log tới khi thấy "View Deployment"
grep -a "View Deployment" /tmp/da3_deploy.log
```

> `deploy_retry.sh` build image + đăng ký hàm `export_gs` lên Modal. Image đã cache nên thường nhanh.

---

## Bước 4 — Chạy DA3 trên Modal (spawn) và lấy kết quả (fetch)

### 4a. Spawn (kích hoạt job chạy trên cloud)

```bash
cd /d/Vkist/DA3
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8     # tránh lỗi ký tự trên Windows

python -c "
import modal
f = modal.Function.from_name('da3-3dgs-room','export_gs')
# tham số: (model, process_res, gs_views_interval, max_views, data_dir, ref_view_strategy, exports)
c = f.spawn('depth-anything/DA3-GIANT-1.1', 700, None, None,
            '/root/data/hanh_lang_dai', 'saddle_balanced', 'npz,depth_vis')
open(r'C:/temp/callid.txt','w').write(c.object_id)
print('SPAWNED call_id =', c.object_id)
"
```

**Giải thích 7 tham số của `export_gs`:**

| Tham số | Ví dụ | Nghĩa |
|---|---|---|
| model | `depth-anything/DA3-GIANT-1.1` | Model DA3 (bản GIANT, chất lượng cao nhất) |
| process_res | `700` | Độ phân giải xử lý (700 chi tiết vừa; 504 nhẹ; 1024 nét hơn nhưng nặng) |
| gs_views_interval | `None` | Chỉ dùng cho 3DGS, để `None` |
| max_views | `None` | Giới hạn số view; `None` = dùng hết. Đặt số (vd `60`) nếu muốn cắt bớt/tránh OOM |
| data_dir | `/root/data/hanh_lang_dai` | Thư mục frame trên Modal (khớp mount ở Bước 2) |
| ref_view_strategy | `saddle_balanced` | Cách chọn view tham chiếu, để mặc định |
| exports | `npz,depth_vis` | Xuất gì. `npz` = kết quả chính; `depth_vis` = ảnh depth. (Còn `colmap`, `gs` nếu cần) |

### 4b. Fetch (tải kết quả về máy)

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

python -c "
import modal, os, zipfile
cid = open(r'C:/temp/callid.txt').read().strip()
out = 'workspace/hanh_lang_dai'; os.makedirs(out, exist_ok=True)
p = modal.FunctionCall.from_id(cid).get(timeout=1200)   # chờ job xong rồi tải
for n, d in p.items():
    open(os.path.join(out, n), 'wb').write(d)
zipfile.ZipFile(os.path.join(out, 'depth_vis.zip')).extractall(out)
print('saved:', {n: round(len(d)/1e6, 1) for n, d in p.items()}, 'MB')
"
```

Sau bước này có:
- `workspace/hanh_lang_dai/results.npz` ← **input cho notebook segment cửa**
- `workspace/hanh_lang_dai/depth_vis/*.jpg` ← ảnh depth để xem

---

## Bước 5 — Tạo point cloud `.ply` để xem (MÁY, Python 3.11)

```bash
PY311="/c/Users/pminh/AppData/Local/Programs/Python/Python311/python.exe"

"$PY311" make_ply.py \
    workspace/hanh_lang_dai/results.npz \
    workspace/hanh_lang_dai/hanh_lang_dai_points.ply
```

→ Mở file `.ply` bằng **CloudCompare** (File → Open) để xem point cloud màu thật.

---

## Bước 6 — Dừng Modal cho khỏi tốn phí (MÁY)

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
APP=$(modal app list 2>/dev/null | grep -a da3-3dgs | grep -a deployed | grep -aoE 'ap-[A-Za-z0-9]+' | head -1)
[ -n "$APP" ] && modal app stop "$APP" --yes
echo "stopped $APP"
```

> GPU chỉ tính tiền khi job đang chạy. Sau khi fetch xong, dừng app cho sạch.

---

## Chạy NHIỀU video một lúc (song song)

Modal chạy mỗi job trên 1 container riêng → spawn nhiều cái, chúng chạy song song:

```bash
# Sau khi đã trích frame + thêm mount + deploy cho tất cả các bộ:
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python -c "
import modal, json
f = modal.Function.from_name('da3-3dgs-room','export_gs')
ids = {}
for name in ['Q1_indoor','Q1_outdoor','Q2_indoor']:      # danh sách các bộ
    c = f.spawn('depth-anything/DA3-GIANT-1.1', 700, None, None,
                f'/root/data/{name}', 'saddle_balanced', 'npz,depth_vis')
    ids[name] = c.object_id; print(name, c.object_id)
open(r'C:/temp/ids.json','w').write(json.dumps(ids))
"
# rồi fetch lần lượt từng call_id trong ids.json (giống Bước 4b)
```

---

## Bảng tham chiếu nhanh

### Giới hạn số view theo GPU (ở `process_res=700`)
| GPU (trong `modal_export_3dgs.py`, dòng `gpu=...`) | Số view tối đa an toàn |
|---|---|
| `A10G` (24GB) | ~35–40 |
| `A100-40GB` | ~60–75 |
| **`A100-80GB`** (đang dùng) | **~100–130** |

### Các định dạng `exports` khác
| Giá trị | Ra gì |
|---|---|
| `npz` | `results.npz` (depth+pose+intrinsics+conf+image) — **cần cho notebook** |
| `depth_vis` | ảnh depth tô màu |
| `colmap` | bộ file COLMAP (`cameras/images/points3D.bin`) — mở COLMAP GUI |
| `gs` | `room_3dgs.ply` (3D Gaussian Splatting) |
| Kết hợp | ngăn cách bởi dấu phẩy, vd `npz,depth_vis,colmap` |

### Đổi GPU / model / độ phân giải
- **GPU:** sửa dòng `gpu="A100-80GB"` trong `modal_export_3dgs.py` rồi deploy lại.
- **Model:** `depth-anything/DA3-GIANT-1.1` (khuyến nghị). Bản nhẹ hơn có `DA3NESTED-GIANT-LARGE-1.1`.
- **Độ phân giải:** tham số `process_res` khi spawn (504 nhẹ / 700 vừa / 1024 nét).

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `CUDA out of memory` | Quá nhiều view so với GPU | Giảm `max_views` (vd `60`), hoặc dùng `A100-80GB`, hoặc giảm `process_res` |
| `'charmap' codec can't encode` | Console Windows | Luôn đặt `export PYTHONUTF8=1 PYTHONIOENCODING=utf-8` trước lệnh modal |
| Deploy rớt giữa chừng | Mạng chập chờn | `deploy_retry.sh` tự thử lại; cứ chạy lại |
| Fetch treo lâu | Job chưa chạy xong | `.get()` sẽ tự chờ; job A100 thường ~2–4 phút |
| Point cloud bị "nén"/dẹp (hành lang dài) | DA3 kém với đi thẳng dài | Chia video thành nhiều đoạn ngắn (30–40s) chạy riêng |

---

## Tóm tắt "chép–dán" (1 bộ)

```bash
cd /d/Vkist/DA3
NAME=hanh_lang_dai
VIDEO="data/hanh_lang_dai/IMG_0636.MOV"
PY311="/c/Users/pminh/AppData/Local/Programs/Python/Python311/python.exe"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

# 1. Trích frame
rm -rf data/$NAME; mkdir -p data/$NAME
ffmpeg -loglevel error -i "$VIDEO" -vf fps=0.7 -q:v 2 "data/$NAME/${NAME}_%04d.jpg"
ls data/$NAME/*.jpg | wc -l

# 2. (TAY) thêm dòng .add_local_dir(... "data/$NAME" ... "/root/data/$NAME") vào modal_export_3dgs.py

# 3. Deploy
: > /tmp/da3_deploy.log; bash deploy_retry.sh

# 4a. Spawn
python -c "import modal; f=modal.Function.from_name('da3-3dgs-room','export_gs'); c=f.spawn('depth-anything/DA3-GIANT-1.1',700,None,None,'/root/data/$NAME','saddle_balanced','npz,depth_vis'); open(r'C:/temp/callid.txt','w').write(c.object_id); print(c.object_id)"

# 4b. Fetch
python -c "import modal,os,zipfile; cid=open(r'C:/temp/callid.txt').read().strip(); out='workspace/$NAME'; os.makedirs(out,exist_ok=True); p=modal.FunctionCall.from_id(cid).get(timeout=1200); [open(os.path.join(out,n),'wb').write(d) for n,d in p.items()]; zipfile.ZipFile(os.path.join(out,'depth_vis.zip')).extractall(out); print('done')"

# 5. Point cloud
"$PY311" make_ply.py workspace/$NAME/results.npz workspace/$NAME/${NAME}_points.ply

# 6. Dừng Modal
APP=$(modal app list 2>/dev/null | grep -a da3-3dgs | grep -a deployed | grep -aoE 'ap-[A-Za-z0-9]+' | head -1); [ -n "$APP" ] && modal app stop "$APP" --yes
```
