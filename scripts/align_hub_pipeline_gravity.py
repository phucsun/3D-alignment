"""Giai đoạn 1 (docs/multi_space_alignment_plan.md): thay cơ chế tạo cạnh
của `align_rooms_to_hub` (SGD/Hungarian cũ, xem `align_hub_pipeline.py` -
baseline Giai đoạn 0) bằng `gravity_align.align_gravity_camera` - engine
gravity-locked đã verify tốt hơn ở bài toán 2-không-gian (mục 11
CONTRIBUTIONS.md).

Bản THỬ NGHIỆM/prototype (chưa merge vào `alignment.py`).

SỬA quan trọng so với lần chạy đầu: KHÔNG gán độc quyền ở cấp "1 room = 1
NHÓM TƯỜNG" nữa - xác nhận thật trên dữ liệu (`server` thắng nhóm tường có
4 opening nhưng chỉ dùng hết 2/4, 2 opening còn thừa lẽ ra là của
`room_310` thì lại bị cấm dùng vì cả nhóm đã "thuộc về" server, đẩy
room_310 sang nhầm 1 cửa lẻ khác ở cuối hành lang). Sửa: mỗi room chạy
`align_gravity_camera` trực tiếp trên TOÀN BỘ hub_clusters (không giới hạn
subset theo tường trước) - bản thân hàm đã tự tìm subset khớp tốt nhất.
Chỉ khi 2 room THẬT SỰ đòi dùng chung đúng 1 hub-opening-index mới cần giải
quyết tranh chấp (ở cấp OPENING, không phải cấp TƯỜNG).

Giai đoạn 2 (Adjacency prior, docs/multi_space_alignment_plan.md): sau khi
loop-consistency/PCM thuần geometric thất bại khi không biết trước topology
(4 không gian, mọi cặp chạy mù - 3 cạnh false positive), quay lại dùng
ĐÚNG tri thức đã biết trước (giống cơ chế `region` ở mục 4b
CONTRIBUTIONS.md, áp ở cấp đồ thị): `h_server` LÀ hub đã biết chắc chắn -
CHỈ test room->hub, không chạy mù các cặp room-room. Thêm
`connecting_space` làm room thứ 3 (kỳ vọng khớp đúng hub opening index 4 -
"cửa cuối hành lang" - đúng cái mà `room_310` từng bị gán nhầm vào trước
khi sửa granularity).

Usage:
    python scripts/align_hub_pipeline_gravity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points  # noqa: E402

_STATUS_RANK = {"CONFIDENT": 2, "AMBIGUOUS": 1, "NO_SOLUTION": 0}


def _read_ply(path: str):
    from plyfile import PlyData

    v = PlyData.read(path)["vertex"].data
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8) if "red" in v.dtype.names else None
    return points, colors


def _write_ply(points: np.ndarray, colors: np.ndarray | None, path: str) -> None:
    from plyfile import PlyData, PlyElement

    if colors is None:
        colors = np.full((len(points), 3), 160, np.uint8)
    vertex = np.zeros(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def _better(a, b) -> bool:
    """`a` (GravityAlignResult) tốt hơn `b` không - status trước, residual sau."""
    if _STATUS_RANK[a.status] != _STATUS_RANK[b.status]:
        return _STATUS_RANK[a.status] > _STATUS_RANK[b.status]
    return a.opening_residual < b.opening_residual


HUB_PLY = "data/server/h_server/h_server_room_points - Cloud - segment.ply"
HUB_NPZ = "data/server/h_server/results.npz"

ROOM_SERVER_PLY = "data/server/server/server_room_points-segment.ply"
ROOM_SERVER_NPZ = "data/server/server/results.npz"

ROOM_310_PLY = "data/310_indoor/310_indoor_points - Cloud - segment.ply"
ROOM_310_NPZ = "data/310_indoor/results.npz"

CONNECTING_PLY = "data/connecting_space/connecting_space_points - Cloud-segmented.ply"
CONNECTING_NPZ = "data/connecting_space/results.npz"


def main() -> None:
    hub_clusters = openings_from_manual_segmentation(HUB_PLY)
    cam_hub = camera_evidence(HUB_NPZ)

    rooms = {
        "server": (openings_from_manual_segmentation(ROOM_SERVER_PLY), camera_evidence(ROOM_SERVER_NPZ), ROOM_SERVER_PLY),
        "room_310": (openings_from_manual_segmentation(ROOM_310_PLY), camera_evidence(ROOM_310_NPZ), ROOM_310_PLY),
        "connecting_space": (openings_from_manual_segmentation(CONNECTING_PLY), camera_evidence(CONNECTING_NPZ), CONNECTING_PLY),
    }
    room_names = list(rooms.keys())

    print(f"hub (h_server): {len(hub_clusters)} opening cluster(s), up_consistency={cam_hub.up_consistency:.4f}")
    for name, (clusters, cam, _) in rooms.items():
        print(f"room {name}: {len(clusters)} opening cluster(s), up_consistency={cam.up_consistency:.4f}")

    # ---- bước 1: mỗi room chạy trực tiếp trên TOÀN BỘ hub, không giới hạn subset trước ----
    available_hub_idx = {name: list(range(len(hub_clusters))) for name in room_names}
    results = {}
    for name in room_names:
        room_clusters, cam_room, _ = rooms[name]
        sub_hub = [hub_clusters[i] for i in available_hub_idx[name]]
        result = align_gravity_camera(room_clusters, sub_hub, cam_room, cam_hub)
        # matches' j-index hiện là local (index trong sub_hub) - map lại về hub index thật
        idx_map = available_hub_idx[name]
        result.matches = [(i, idx_map[j]) for i, j in result.matches]
        results[name] = result
        print(f"\n[{name}] status={result.status} residual={result.opening_residual:.4f} "
              f"matches(room_idx,hub_idx)={result.matches} reason={result.reason}")

    # ---- bước 2: phát hiện + giải quyết tranh chấp Ở CẤP OPENING, LẶP đến khi hết
    # tranh chấp (fixed-point) - loại bỏ 1 vòng duy nhất không đủ: room thua bị loại
    # opening rồi chạy lại CÓ THỂ tìm ra match mới, lại tranh chấp với 1 room KHÁC
    # chưa từng bị đụng tới ở vòng trước (xác nhận thật: room_310 thắng {0,1} ở vòng
    # đầu, connecting_space bị loại {0,3} rồi chạy lại lại vô tình chọn trúng hub-idx
    # 1 - tranh chấp MỚI với room_310 mà 1 vòng duy nhất không phát hiện ra).
    MAX_ROUNDS = 10
    for round_no in range(MAX_ROUNDS):
        hub_idx_owner: dict[int, str] = {}
        conflicts: set[int] = set()
        for name, result in results.items():
            for _, hub_idx in result.matches:
                if hub_idx in hub_idx_owner and hub_idx_owner[hub_idx] != name:
                    conflicts.add(hub_idx)
                else:
                    hub_idx_owner[hub_idx] = name

        if not conflicts:
            if round_no == 0:
                print("\nKhông có tranh chấp opening-index nào - mọi room dùng tập opening rời nhau, đều hợp lệ.")
            else:
                print(f"\nHết tranh chấp sau {round_no} vòng giải quyết.")
            break

        print(f"\n=== Vòng {round_no + 1}: tranh chấp opening-index {conflicts} - giải quyết theo status/residual ===")
        losers = set()
        for hub_idx in conflicts:
            claimants = [n for n, r in results.items() if hub_idx in {h for _, h in r.matches}]
            best = max(claimants, key=lambda n: (_STATUS_RANK[results[n].status], -results[n].opening_residual))
            for n in claimants:
                if n != best:
                    losers.add(n)
        for name in losers:
            # loại khỏi candidate mọi hub-index đang tranh chấp mà room này KHÔNG phải chủ thắng
            available_hub_idx[name] = [i for i in available_hub_idx[name]
                                        if i not in conflicts or hub_idx_owner.get(i) == name]
            room_clusters, cam_room, _ = rooms[name]
            sub_hub = [hub_clusters[i] for i in available_hub_idx[name]]
            idx_map = available_hub_idx[name]
            result = align_gravity_camera(room_clusters, sub_hub, cam_room, cam_hub)
            result.matches = [(i, idx_map[j]) for i, j in result.matches]
            results[name] = result
            print(f"  [{name}] (loại opening tranh chấp) -> status={result.status} "
                  f"residual={result.opening_residual:.4f} matches={result.matches} reason={result.reason}")
    else:
        print(f"\nCẢNH BÁO: vẫn còn tranh chấp sau {MAX_ROUNDS} vòng - dữ liệu có thể không đủ opening "
              "để tách 2 room, hoặc thuật toán dao động không hội tụ.")

    # ---- xuất kết quả ----
    out_dir = Path("outputs/final_aligned")
    out_dir.mkdir(parents=True, exist_ok=True)
    hub_pts, hub_cols = _read_ply(HUB_PLY)
    if hub_cols is None:
        hub_cols = np.full((len(hub_pts), 3), 160, np.uint8)
    all_pts, all_cols = [hub_pts], [hub_cols]

    print("\n=== Kết quả cuối ===")
    for name, result in results.items():
        print(f"{name}: status={result.status} residual={result.opening_residual:.4f} "
              f"scale={result.s:.4f} matches(room_idx,hub_idx)={result.matches}")
        if result.status == "NO_SOLUTION":
            continue
        _, _, ply_path = rooms[name]
        pts, cols = _read_ply(ply_path)
        if cols is None:
            cols = np.full((len(pts), 3), 160, np.uint8)
        aligned = transform_points(pts, result.s, result.R, result.t)

        pair_out = out_dir / f"hubadjgrav_{name}_vs_hub.ply"
        _write_ply(np.concatenate([aligned, hub_pts]), np.concatenate([cols, hub_cols]), str(pair_out))
        print(f"  saved -> {pair_out}")

        all_pts.append(aligned)
        all_cols.append(cols)

    combined_out = out_dir / "hubadjgrav_all_aligned.ply"
    _write_ply(np.concatenate(all_pts), np.concatenate(all_cols), str(combined_out))
    print(f"\nsaved (cả 3 không gian) -> {combined_out}")


if __name__ == "__main__":
    main()
