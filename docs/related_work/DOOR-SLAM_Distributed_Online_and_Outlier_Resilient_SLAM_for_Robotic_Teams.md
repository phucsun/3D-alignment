# DOOR-SLAM: Distributed, Online, and Outlier Resilient SLAM for Robotic Teams (IEEE RA-L, 2020)

**Nguồn:** [arXiv:1909.12198](https://arxiv.org/abs/1909.12198)

> Lưu ý: tên "DOOR" ở đây là viết tắt (**D**istributed, **O**nline, **O**utlier-**R**esilient), không liên quan gì đến "door/cửa" trong bài toán của mình — trùng tên tình cờ.

## Bài toán giải quyết

Nhiều robot cùng SLAM, cần phát hiện **loop closure liên-robot** (robot A nhận ra đã đi qua chỗ robot B từng đi) để hợp nhất trajectory thành 1 bản đồ chung — nhưng place-recognition liên-robot vốn có nhiều outlier (nhận nhầm 2 chỗ khác nhau là 1), và cách xử lý cũ (đặt ngưỡng rất chặt để tránh outlier) lại bỏ sót quá nhiều loop closure đúng.

## Kiến trúc — 2 module

1. **Front-end phân tán**: mỗi robot tự phát hiện loop closure với robot khác chỉ qua trao đổi **descriptor nén** (không gửi sensor thô) — tiết kiệm băng thông, hoạt động qua giao tiếp peer-to-peer, không cần kết nối đầy đủ (full connectivity) giữa mọi cặp robot.
2. **Back-end**: pose graph optimizer + thuật toán **PCM (Pairwise Consistent Measurement set maximization)**, phân tán — với MỌI tập loop-closure ứng viên nhận được, tìm tập con lớn nhất **nhất quán lẫn nhau về mặt hình học** (2 loop closure "đồng thuận" nếu cùng suy ra 1 phép biến đổi tương thích), loại bỏ phần còn lại như outlier — không cần biết trước ngưỡng threshold cứng nào.

## Giả định

- Robot có khả năng giao tiếp peer-to-peer (không cần trung tâm).
- Vẫn cần **overlap cảm biến** đủ để place-recognition front-end hoạt động (nhận diện lại 1 vị trí).
- Environment không nhất thiết có cấu trúc phòng/tường rõ ràng (khác S-Graphs) — hoạt động cả ngoài trời, hầm mỏ.

## Đối chiếu với bài toán của mình

| Khía cạnh | DOOR-SLAM | Bài toán N-không-gian của mình |
|---|---|---|
| Cách phát hiện "cạnh" giữa 2 map | Place recognition (nhận lại 1 vị trí đã đi qua) | Semantic-geometric matching qua portal — đã có sẵn |
| Cơ chế loại outlier ở cạnh | **PCM**: giữ tập cạnh lớn nhất tự nhất quán lẫn nhau (không cần ground-truth, không cần threshold cứng) | **Ý tưởng mượn được rất trực tiếp**: hiện tại `align_gravity_camera` tự lọc outlier bên trong 1 cặp (không gian A-B), nhưng KHÔNG có cơ chế kiểm tra chéo giữa nhiều cặp — nếu có N không gian và tính transform cho mọi cặp có thể, PCM là cách tự nhiên để phát hiện "cặp (i,j) này cho ra transform MÂU THUẪN với các cặp khác" (ví dụ lỗi tích luỹ hoặc match sai) mà không cần ground-truth |
| Kiến trúc giao tiếp | Peer-to-peer, phân tán, online (robot thật, ràng buộc băng thông) | Bài toán của mình offline, có toàn quyền truy cập mọi dữ liệu cùng lúc — **không cần** ràng buộc phân tán/peer-to-peer của DOOR-SLAM, có thể dùng bản batch/centralized đơn giản hơn của cùng ý tưởng PCM |

**Điểm mượn được cho kế hoạch**: cơ chế **PCM (pairwise consistent measurement set maximization)** là công cụ toán học đúng cho bước "phát hiện/loại cạnh sai trong đồ thị nhiều-không-gian" mà `align_rooms_to_hub` hiện tại hoàn toàn chưa có (nó tin tưởng tuyệt đối vào cost matrix từ Hungarian, không tự kiểm tra tính nhất quán chéo giữa các assignment). Không cần bản phân tán (distributed) của DOOR-SLAM — chỉ cần bản toán học lõi (maximum consistent subset), chạy tập trung (centralized) vì bài toán của mình không có ràng buộc robot thật/băng thông.
