# maplab 2.0 — A Modular and Multi-Modal Mapping Framework (IEEE RA-L, 2022)

**Nguồn:** [arXiv:2212.00654](https://arxiv.org/abs/2212.00654) · [GitHub](https://github.com/ethz-asl/maplab)

## Bài toán giải quyết

Xây 1 framework SLAM đa robot/đa phiên (multi-session) mô-đun hoá, dễ tích hợp module mới (kể cả deep-learning) — không phải 1 thuật toán đăng ký cụ thể, mà là 1 **kiến trúc hệ thống** cho phép nhiều robot/nhiều lần chạy được hợp nhất thành 1 bản đồ.

## Kiến trúc — Mapping Server làm hub trung tâm

Mỗi robot gửi **submap** (1 đoạn factor graph cục bộ của chính nó) lên "Mapping Server". Server:
1. Tối ưu cục bộ từng submap nhận được.
2. Đánh giá chất lượng feature của từng submap.
3. Tìm loop closure — **cả bên trong 1 trajectory (nội-robot) lẫn giữa các trajectory khác nhau (liên-robot)**.
4. Có 1 module **"semantic object-based loop closure"** — dùng đối tượng semantic (không chỉ feature điểm ảnh thô) làm tín hiệu loop closure liên-robot.

Đã verify thực tế trên use-case ~10km, 23 mission (multi-session thật, quy mô lớn).

## Giả định

- Mỗi robot vẫn tự chạy SLAM/odometry liên tục để tạo ra submap của mình.
- Loop closure (kể cả bản "semantic object-based") vẫn dựa trên nhận diện lại **đối tượng/vị trí đã thấy trước đó với overlap quan sát đủ lớn**, không phải khớp qua 1 vật thể hẹp là cầu nối duy nhất giữa 2 vùng hoàn toàn tách biệt.

## Đối chiếu với bài toán của mình

| Khía cạnh | maplab 2.0 | Bài toán N-không-gian của mình |
|---|---|---|
| Vai trò "hub" | Mapping Server — hạ tầng hệ thống (nhận submap, tối ưu, lưu trữ), không phải 1 không gian vật lý | `align_rooms_to_hub` hiện tại: "hub" là 1 KHÔNG GIAN VẬT LÝ thật (hành lang) — khác khái niệm hoàn toàn, dễ nhầm lẫn nếu chỉ đọc tên |
| Semantic object-based loop closure | Có, nhưng dùng object bất kỳ nhận diện lại được (ví dụ 1 cái bàn, 1 biển hiệu) qua nhiều lần thấy | Bài của mình dùng ĐÚNG 1 loại object có ý nghĩa cấu trúc đặc biệt: door/window — luôn nằm ở **ranh giới** giữa 2 không gian, quan sát được từ CẢ 2 phía cùng lúc (không phải "thấy lại sau", mà "thấy đồng thời từ 2 góc độc lập") |
| Mô-đun hoá framework | Điểm mạnh chính được nhấn — dễ cắm module mới | Gợi ý kiến trúc phần mềm tốt cho việc triển khai: nên tách rõ "module tạo cạnh" (đã có, `align_gravity_camera`) khỏi "module tối ưu toàn cục" (chưa có, cần xây) — giống cách maplab tách Mapping Server khỏi front-end từng robot |

**Điểm mượn được cho kế hoạch**: chủ yếu là bài học **kiến trúc phần mềm** (tách front-end tạo cạnh / back-end tối ưu toàn cục thành 2 module độc lập, dễ thay thế) hơn là thuật toán cụ thể — maplab 2.0 không đưa ra công thức toán mới nào giải quyết trực tiếp bài toán "portal-only, no-overlap" của mình, nhưng xác nhận rằng kiến trúc "1 điểm hợp nhất trung tâm nhận nhiều submap độc lập" là mô hình đã được triển khai thực tế ở quy mô lớn (10km, 23 mission) — tăng độ tin cậy khả thi cho hướng thiết kế tương tự.
