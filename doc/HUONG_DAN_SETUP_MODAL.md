# Hướng dẫn setup Modal để chạy một chương trình (ví dụ DA3)

## 0. Modal là gì
Dịch vụ **cho thuê GPU/máy cloud kiểu serverless**. Bạn viết code Python bình thường, **đánh dấu hàm nào chạy trên cloud** (chọn loại GPU), Modal tự: dựng máy → cài môi trường → chạy → trả kết quả về. **Tính tiền theo giây** hàm chạy (không chạy = không mất tiền).

Ý tưởng cốt lõi: **cả chương trình gói trong 1 file `.py`** gồm 3 thứ — **App**, **Image** (môi trường), **Function** (hàm chạy trên cloud).

---

## 1. Chuẩn bị 1 lần (account + cài + đăng nhập)

```bash
# B1. Tạo tài khoản: vào https://modal.com đăng ký (free có credit)

# B2. Cài thư viện modal (Python)
pip install modal

# B3. Đăng nhập (mở browser xác thực, tự tạo file ~/.modal.toml chứa token)
modal setup

# B4. Kiểm tra đã đăng nhập
modal profile current      # ra tên profile là OK
```

---

## 2. File Modal nhỏ nhất (hello GPU) — hiểu 3 khối

Tạo file `hello.py`:

```python
import modal

# (1) APP: đặt tên chương trình
app = modal.App("hello-gpu")

# (2) IMAGE: môi trường chạy (OS + thư viện). Modal build 1 lần rồi cache.
image = modal.Image.debian_slim().pip_install("torch")

# (3) FUNCTION: hàm chạy TRÊN CLOUD. gpu=... chọn card.
@app.function(image=image, gpu="A10G")
def kiem_tra_gpu():
    import torch
    return f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}"

# Điểm vào chạy ở MÁY bạn, gọi hàm cloud bằng .remote()
@app.local_entrypoint()
def main():
    print(kiem_tra_gpu.remote())   # .remote() = chạy trên cloud
```

Chạy:
```bash
modal run hello.py
```
Modal sẽ build image (lần đầu), thuê 1 GPU A10G, chạy hàm, in ra tên GPU. Xong tự tắt máy.

> **Chốt:** `def kiem_tra_gpu()` là code bình thường, nhưng nhờ `@app.function(gpu=...)` + gọi `.remote()` nên nó **chạy trên GPU cloud** thay vì máy bạn.

---

## 3. Cấu trúc đầy đủ (giải thích qua file DA3 `modal_export_3dgs.py`)

```python
import os, modal
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

app = modal.App("da3-3dgs-room")                    # (1) APP

# Ổ đĩa bền để cache model (khỏi tải lại 2GB mỗi lần)
hf_cache = modal.Volume.from_name("da3-hf-cache", create_if_missing=True)

# (2) IMAGE — dựng môi trường theo từng lớp (mỗi .xxx là 1 lớp, cache riêng)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", ...)     # cài gói hệ điều hành
    .pip_install("torch==2.5.1", ...)                # cài thư viện python
    .env({"PYTHONPATH": "/opt/da3/src"})             # biến môi trường
    .add_local_dir("src", "/opt/da3/src")            # ĐẨY code/dữ liệu lên container
    .add_local_dir("da3_frames", "/root/frames")     # đẩy frame ảnh lên
)

# (3) FUNCTION — chạy trên cloud
@app.function(
    image=image,                                     # dùng môi trường trên
    gpu="A100-80GB",                                 # chọn GPU
    volumes={"/root/.cache/huggingface": hf_cache},  # gắn ổ cache
    timeout=3600,                                    # tối đa 1 giờ
)
def export_gs(model_repo, process_res, ...):
    import torch
    from depth_anything_3.api import DepthAnything3   # import trong hàm (chỉ có trên cloud)
    model = DepthAnything3.from_pretrained(model_repo).to("cuda")
    ...
    return payload                                    # trả kết quả (bytes) về
```

**Ý nghĩa từng phần:**
| Khối | Vai trò |
|---|---|
| `App` | Tên + gom mọi thứ của chương trình |
| `Image` | "Máy ảo" chứa OS + thư viện + code. Build 1 lần, **cache** → lần sau nhanh |
| `.apt_install / .pip_install` | Cài gói (như `apt install`, `pip install` nhưng trên cloud) |
| `.add_local_dir` | **Đẩy thư mục ở máy bạn lên container** (code, ảnh input) |
| `Volume` | Ổ đĩa **bền** giữa các lần chạy (cache model, lưu kết quả lớn) |
| `@app.function(gpu=...)` | Đánh dấu hàm chạy trên cloud + chọn GPU |
| `import ... trong hàm` | Thư viện chỉ có trên container → import **bên trong** hàm, không phải đầu file |

---

## 4. Ba cách chạy

| Lệnh | Dùng khi | Đặc điểm |
|---|---|---|
| `modal run file.py` | Test nhanh 1 lần | Chạy xong tự tắt (ephemeral) |
| `modal deploy file.py` | Chạy nhiều lần | App "thường trú", gọi hàm bao nhiêu lần cũng được |
| `.spawn()` (sau deploy) | Chạy nền | Bắn job rồi lấy kết quả sau (dùng cho mạng chập chờn) |

**Gọi hàm sau khi deploy** (từ máy bạn):
```python
import modal
f = modal.Function.from_name("da3-3dgs-room", "export_gs")   # lấy hàm đã deploy
call = f.spawn(arg1, arg2, ...)          # chạy nền, trả về call_id
ket_qua = call.get()                     # chờ xong, lấy kết quả về
```

---

## 5. Đưa dữ liệu VÀO và lấy kết quả RA

**Vào:**
- `.add_local_dir("data", "/root/data")` — đẩy thư mục ảnh/code lên (build-time).
- Truyền **tham số** khi gọi hàm: `f.spawn("model_name", 700, ...)`.
- Hoặc upload lên **Volume**.

**Ra:**
- Hàm `return` giá trị (số, chuỗi, **bytes** file) → máy bạn nhận qua `call.get()`.
- Hoặc ghi vào **Volume** rồi `modal volume get` tải về.

Ví dụ trả file về (như DA3):
```python
@app.function(...)
def run():
    ...
    return {"results.npz": open("out.npz","rb").read()}   # trả bytes

# máy bạn:
data = call.get()
open("workspace/results.npz","wb").write(data["results.npz"])
```

---

## 6. Volume — cache model & lưu kết quả

```python
vol = modal.Volume.from_name("ten-vol", create_if_missing=True)

@app.function(volumes={"/root/cache": vol})   # gắn vào đường dẫn trong container
def f():
    # ghi/đọc /root/cache như thư mục thường
    vol.commit()                              # lưu lại thay đổi

# ở máy: modal volume ls ten-vol / modal volume get ten-vol <path> <local>
```
Dùng để: cache model (khỏi tải lại), lưu output lớn.

---

## 7. Checklist tạo 1 chương trình Modal mới

1. `pip install modal` + `modal setup` (1 lần).
2. Tạo file `chuong_trinh.py`:
   - `app = modal.App("ten")`
   - `image = modal.Image....pip_install(...)` — liệt kê thư viện cần.
   - `@app.function(image=image, gpu="...")` cho hàm nặng.
   - Import thư viện nặng **bên trong** hàm.
3. Test: `modal run chuong_trinh.py`.
4. Khi ổn: `modal deploy chuong_trinh.py`.
5. Gọi hàm bằng `Function.from_name(...).spawn(...)` + `.get()`.
6. Dọn: `modal app stop <id>`.

---

## 8. Các lệnh Modal hay dùng

```bash
modal setup                         # đăng nhập
modal profile current               # xem profile
modal run file.py                   # chạy 1 lần
modal deploy file.py                # deploy app
modal app list                      # liệt kê app (State, Tasks)
modal app stop <ap-xxxx> --yes      # dừng app
modal app logs <ten-app>            # xem log
modal volume list                   # liệt kê volume
modal volume ls <vol> <path>        # xem file trong volume
modal volume get <vol> <path> <local>   # tải file từ volume về
```

---

## 9. Lỗi hay gặp (Windows)

| Lỗi | Cách xử lý |
|---|---|
| `'charmap' codec can't encode` | `export PYTHONUTF8=1 PYTHONIOENCODING=utf-8` trước lệnh modal (Git Bash) |
| Deploy rớt giữa chừng (heartbeat) | Chạy lại; hoặc dùng vòng lặp retry (như `deploy_retry.sh`) |
| `CUDA out of memory` | Chọn GPU to hơn (`A100-80GB`) hoặc giảm khối lượng dữ liệu |
| Thư viện "no module" trên cloud | Thêm vào `.pip_install(...)` trong image rồi deploy lại |

> **Tóm tắt:** 1 file Python = App + Image (môi trường) + Function (hàm cloud). `modal run` để test, `modal deploy` + `spawn/get` để dùng thật. Dữ liệu vào bằng `add_local_dir`/tham số, ra bằng `return` hoặc Volume.
