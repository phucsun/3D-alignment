# Multi S-Graphs: An Efficient Distributed Semantic-Relational Collaborative SLAM (IEEE RA-L, 2024)

**Nguồn:** [arXiv:2401.05152](https://arxiv.org/pdf/2401.05152) — mở rộng đa-robot của nền tảng [S-Graphs 2.0](https://arxiv.org/abs/2502.18044) (single-robot).

## Bài toán giải quyết

Nhiều robot cùng khám phá 1 toà nhà (mỗi robot tự chạy SLAM online, có trajectory/odometry liên tục riêng) — cần hợp nhất bản đồ của các robot thành 1 bản đồ toàn cục nhất quán, kể cả khi 2 robot chưa từng "gặp nhau" trực tiếp (không có overlap cảm biến trực tiếp tại cùng thời điểm).

## Kiến trúc cốt lõi — 4 tầng phân cấp (S-Graphs)

`Keyframes (pose robot, SE(3)) → Walls (mặt phẳng, RANSAC từ LiDAR) → Rooms (2 tường = hành lang, 4 tường = phòng) → Floors (1 node trung tâm mỗi tầng nhà)`.

Toàn bộ 4 tầng nằm chung **1 pose graph tối ưu được** (factor graph), tối ưu đồng thời — không phải 4 bước tách rời.

## Cơ chế hợp nhất đa-robot — "room-based descriptor"

Thay vì so khớp trực tiếp keyframe-to-keyframe (cần overlap cảm biến), Multi S-Graphs so khớp ở **tầng Room**: mỗi room có 1 descriptor riêng (hình học tường bao + quan hệ liên kết) — 2 robot ở 2 vị trí khác nhau, miễn cùng đi qua đúng 1 room vật lý, sẽ tự nhận diện được loop closure liên-robot **qua descriptor room này, không cần trao đổi dữ liệu cảm biến thô** (tiết kiệm băng thông, giải bài toán "kidnapped robot" khi robot mới join mạng).

## Giả định bắt buộc

1. **Cần trajectory/odometry liên tục** của từng robot — room/wall được trích xuất TRONG lúc robot di chuyển qua, không phải từ 1 bản dựng 3D tĩnh độc lập.
2. Môi trường phải **có cấu trúc phòng/tường rõ ràng** (Manhattan-world), room được định nghĩa cứng bằng 2 hoặc 4 mặt phẳng tường.
3. Việc ghép nối 2 robot **vẫn cần chúng đi qua CÙNG 1 room vật lý** (dù không cùng lúc) — không giải quyết trường hợp 2 không gian chỉ liên kết qua 1 cửa/cửa sổ mà bản thân 2 room đó chưa bao giờ được cùng 1 robot quan sát trọn vẹn.

## Đối chiếu với bài toán của mình

| Khía cạnh | Multi S-Graphs | Bài toán N-không-gian của mình |
|---|---|---|
| Cách có được dữ liệu mỗi không gian | Trajectory robot liên tục, online SLAM | Video multi-view độc lập, tái dựng OFFLINE riêng biệt từng không gian — **không có trajectory nối liền 2 không gian** |
| Cạnh nối 2 không gian được xác lập bằng | Cùng đi qua 1 room vật lý (bên trong room đó) | Quan sát được **CÙNG 1 vật thể (cửa/cửa sổ) từ 2 PHÍA khác nhau** — 2 không gian không hề "đi vào nhau", chỉ nhìn thấy 1 khe hở chung |
| Biểu diễn phân cấp (room/wall/floor) | Có sẵn, tối ưu đồng thời trong 1 factor graph | **Ý tưởng mượn được trực tiếp**: coi mỗi "Space" là 1 node cấp cao (tương đương Room), các opening đã match là "liên kết liên-node" — có thể xây 1 factor graph tương tự, nhưng cạnh không phải "cùng room" mà là "cùng aperture" |
| Loại transform tối ưu | SE(3) trong 1 factor graph GTSAM/g2o chuẩn | Cần Sim(3) (có scale) vì các không gian tái dựng độc lập scale khác nhau |

**Điểm mượn được cho kế hoạch**: ý tưởng "1 node cấp cao đại diện cho cả 1 không gian, tối ưu trong CÙNG 1 pose graph với các node khác" (thay vì xử lý từng cặp rồi ghép tay như hiện tại) là đúng hướng tổng quát hoá cần làm. Khác biệt quan trọng cần điều chỉnh: cạnh của mình được xác lập bằng **matching semantic-geometric qua portal** (đã có sẵn, đã verify), không phải bằng trajectory/room-descriptor như S-Graphs — nên phần "front-end tạo cạnh" của S-Graphs không dùng được, chỉ dùng được phần "back-end: tối ưu factor graph phân cấp".
