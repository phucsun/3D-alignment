"""Giai đoạn 0 (docs/multi_space_alignment_plan.md): baseline THẬT cho
`align_rooms_to_hub` (alignment.py) trên bộ 3 không gian có sẵn - trước đây
hàm này chỉ được test bằng dữ liệu tổng hợp (`_make_detection` dựng tay
trong tests/test_sgd_matching.py), chưa từng chạy trên point cloud thật.

Hub = h_server (hành lang chung). Room = server, room_310 - cả 2 đều mở
cửa/cửa sổ vào h_server, KHÔNG biên giới trực tiếp với nhau (đúng star
topology mà align_rooms_to_hub giả định).

Dùng pipeline SGD/Hungarian cũ (align_indoor_outdoor bên trong
align_rooms_to_hub) - CHƯA phải bản đã nối gravity engine (đó là Giai đoạn 1
của kế hoạch), đây chỉ là số liệu baseline để so sánh sau này.

Usage:
    python scripts/align_hub_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.detection.manual_segmentation import load_manual_segmentation  # noqa: E402
from sgd_alignment.matching.alignment import align_rooms_to_hub, group_by_wall, transform_points  # noqa: E402


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


def _camera_positions(npz_path: str) -> np.ndarray:
    Z = np.load(npz_path)
    EX = Z["extrinsics"].astype(np.float64)  # (N,3,4) world2cam
    R, t = EX[:, :3, :3], EX[:, :3, 3]
    return np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)


HUB_PLY = "data/server/h_server/h_server_room_points - Cloud - segment.ply"
HUB_NPZ = "data/server/h_server/results.npz"

ROOM_SERVER_PLY = "data/server/server/server_room_points-segment.ply"
ROOM_SERVER_NPZ = "data/server/server/results.npz"

ROOM_310_PLY = "data/310_indoor/310_indoor_points - Cloud - segment.ply"
ROOM_310_NPZ = "data/310_indoor/results.npz"


def main() -> None:
    hub = load_manual_segmentation(HUB_PLY, is_outdoor=True, camera_positions=_camera_positions(HUB_NPZ))
    room_server = load_manual_segmentation(ROOM_SERVER_PLY, is_outdoor=False,
                                            camera_positions=_camera_positions(ROOM_SERVER_NPZ))
    room_310 = load_manual_segmentation(ROOM_310_PLY, is_outdoor=False,
                                         camera_positions=_camera_positions(ROOM_310_NPZ))

    print(f"hub (h_server): {len(hub)} opening(s)")
    print(f"room server: {len(room_server)} opening(s)")
    print(f"room 310: {len(room_310)} opening(s)")

    wall_groups = group_by_wall(hub)
    print(f"\nhub grouped into {len(wall_groups)} wall(s): {wall_groups}")

    assignments = align_rooms_to_hub(
        [room_server, room_310], hub,
        estimate_scale=True, normalize_distance=True,
        use_ransac_consensus=True, use_intrinsic_fallback=True,
    )

    room_plys = {"server": ROOM_SERVER_PLY, "room_310": ROOM_310_PLY}
    hub_pts, hub_cols = _read_ply(HUB_PLY)
    if hub_cols is None:
        hub_cols = np.full((len(hub_pts), 3), 160, np.uint8)

    out_dir = Path("outputs/final_aligned")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pts, all_cols = [hub_pts], [hub_cols]
    for name, assignment in zip(["server", "room_310"], assignments):
        print(f"\n=== room: {name} ===")
        if assignment is None:
            print("  KHÔNG được gán vào tường hub nào (assignment=None)")
            continue
        print(f"  gán vào hub opening index(es): {assignment.wall_hub_indices}")
        print(f"  matches: {assignment.result.matches}")
        print(f"  residuals: {np.round(assignment.result.residuals, 4)}")
        print(f"  scale: {assignment.result.scale:.4f}")

        pts, cols = _read_ply(room_plys[name])
        if cols is None:
            cols = np.full((len(pts), 3), 160, np.uint8)
        aligned = transform_points(pts, assignment.result.R, assignment.result.t)

        # export riêng từng cặp (room + hub) để dễ kiểm tra tách biệt bằng mắt
        pair_out = out_dir / f"hub3_{name}_vs_hub.ply"
        _write_ply(np.concatenate([aligned, hub_pts]), np.concatenate([cols, hub_cols]), str(pair_out))
        print(f"  saved -> {pair_out}")

        all_pts.append(aligned)
        all_cols.append(cols)

    combined_out = out_dir / "hub3_all_aligned.ply"
    _write_ply(np.concatenate(all_pts), np.concatenate(all_cols), str(combined_out))
    print(f"\nsaved (cả 3 không gian) -> {combined_out}")


if __name__ == "__main__":
    main()
