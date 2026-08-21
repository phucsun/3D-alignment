# SGHR — Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting (CVPR 2023)

**Nguồn:** [arXiv:2304.00467](https://arxiv.org/abs/2304.00467) · [GitHub](https://github.com/WHU-USI3DV/SGHR)

## Bài toán giải quyết

Đăng ký (registration) N scan điểm 3D thành 1 hệ toạ độ chung, khi việc đăng ký từng cặp (pairwise) trực tiếp không đủ tin cậy để dùng làm input cho bước tối ưu toàn cục — pose graph "đủ N×N cạnh" (exhaustive pairwise) chứa quá nhiều cạnh sai, làm hỏng bước motion averaging phía sau.

## Input / Output

- **Input**: N point cloud scan (thường overlap đáng kể với nhau, ví dụ nhiều góc quét quanh 1 toà nhà).
- **Output**: Pose tuyệt đối `(R, t)` nhất quán toàn cục cho mỗi scan.

## Phương pháp — 4 bước

1. **Ước lượng overlap bằng neural network**: dự đoán xác suất 2 scan có overlap đáng kể, cho MỌI cặp `(i, j)`, không cần đăng ký thật.
2. **Khởi tạo pose graph thưa**: chỉ giữ lại cạnh giữa các cặp có overlap dự đoán cao — giảm ~70% số phép đăng ký cặp cần chạy so với exhaustive, đồng thời đồ thị "sạch" hơn (ít cạnh sai) ngay từ đầu.
3. **History reweighting (IRLS)**: trong vòng lặp tối ưu, mỗi cạnh có 1 trọng số được cập nhật dựa trên "lịch sử" đóng góp residual của nó qua các vòng lặp trước — cạnh outlier bị hạ trọng số dần thay vì loại cứng ngay từ đầu (robust hơn ngưỡng cứng).
4. **Motion/rotation averaging**: từ pose graph đã làm sạch + trọng số, giải bài toán tối ưu tuyến tính hoá (IRLS) ra pose tuyệt đối nhất quán cho toàn bộ N scan cùng lúc.

## Giả định cốt lõi

Có thể học 1 mạng neural ước lượng overlap **dựa trên đặc trưng hình học đám mây điểm thô** (feature embedding kiểu learned descriptor, ví dụ FCGF/Predator) — ngầm giả định các scan có phần **overlap hình học thực sự** (cùng bề mặt vật lý được quét từ nhiều góc), đủ lớn để mạng học được tín hiệu.

## Kết quả đã công bố

+11% registration recall trên 3DMatch, -13% lỗi đăng ký trên ScanNet, -70% số phép đăng ký cặp cần chạy. Chỉ kiểm chứng trên benchmark chuẩn (3DMatch, ScanNet) — chưa có thảo luận rõ về tổng quát hoá ngoài các bộ dữ liệu này.

## Đối chiếu với bài toán của mình

| Khía cạnh | SGHR | Bài toán N-không-gian của mình |
|---|---|---|
| Overlap giữa các "scan" | Bắt buộc, đáng kể (cùng bề mặt vật lý nhìn từ nhiều góc) | **Gần như KHÔNG** — 2 không gian chỉ chung đúng vài cửa/cửa sổ, phần còn lại hoàn toàn không overlap |
| Cách xác định cạnh đồ thị | Neural network học overlap từ feature hình học thô | Đã có sẵn, **không cần học**: chạy `align_gravity_camera` từng cặp, dùng chính `status` (CONFIDENT/AMBIGUOUS/NO_SOLUTION) + `opening_residual`/`cam_dominance`/`up_consistency` làm tín hiệu "cạnh này có tin được không" — đây là tín hiệu vật lý trực tiếp, không phải xác suất học |
| Bước tối ưu toàn cục (motion averaging) | IRLS + history reweighting trên pose graph rotation | **Ý tưởng dùng được trực tiếp**: một khi đã có N(N-1)/2 cạnh Sim(3) ứng viên (mỗi cạnh 1 `GravityAlignResult`), có thể áp y hệt sơ đồ IRLS/history-reweighting này để giải pose tuyệt đối cho N không gian cùng lúc — thay vì Hungarian rời rạc (gán cứng "room nào vào wall nào") như `align_rooms_to_hub` hiện tại |
| Loại phép biến đổi | SE(3) (rotation + translation, không có scale — cùng máy quét, cùng đơn vị) | Cần **Sim(3)** (thêm scale `s`) vì mỗi không gian có thể tái dựng độc lập, scale khác nhau — cần mở rộng công thức averaging từ SO(3)/SE(3) sang Sim(3) |

**Điểm mượn được cho kế hoạch**: kiến trúc 4 bước (ước lượng độ tin cậy cạnh → đồ thị thưa → reweighting robust → global averaging) là khung sườn tốt, nhưng bước 1 (ước lượng overlap bằng neural network) **thay hoàn toàn** bằng cơ chế đã verify (gravity-locked matching engine của mình) — không cần train thêm model nào, vì bài toán của mình có tín hiệu vật lý trực tiếp (camera pose, gravity) mà SGHR không có.
