"""Interactive point picking to help manually annotate windows/doors.

Usage:
    python scripts/pick_points.py data/Indoor-Outdoor-Point-Cloud-Dataset-main/scenario1_room1_indoor.ply

Controls in the window that opens:
    - Shift + left-click to pick a point (it gets highlighted).
    - Pick 3 points per opening, in this order:
        1) the CENTER of the window/door
        2) a point at one edge along the wall (to measure width)
        3) a point at the top or bottom edge (to measure height)
    - Press 'Q' to close the window when done with a scene.

The script then prints, for every group of 3 picked points, a ready-to-paste
YAML block (center/width/height/normal - normal is estimated from the
nearest wall plane automatically) for configs/annotations/<scene>.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgd_alignment.detection.plane_fitting import estimate_up_vector_manhattan, extract_wall_planes, filter_exterior_walls
from sgd_alignment.detection.point_cloud_utils import load_point_cloud


def pick_points(o3d_pc: o3d.geometry.PointCloud) -> list[int]:
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Shift+click to pick points, then press Q")
    vis.add_geometry(o3d_pc)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def nearest_wall_normal(walls, point: np.ndarray) -> np.ndarray:
    best_wall = min(walls, key=lambda w: abs(w.signed_distance(point[None, :])[0]))
    return best_wall.normal


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/pick_points.py <path_to_ply>")
        sys.exit(1)

    ply_path = sys.argv[1]
    pc = load_point_cloud(ply_path)
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc.points)
    if pc.intensity is not None:
        v = (pc.intensity - pc.intensity.min()) / (pc.intensity.max() - pc.intensity.min() + 1e-9)
        o3d_pc.colors = o3d.utility.Vector3dVector(np.stack([v, v, v], axis=1))

    print("Estimating walls (for normal direction only, not required for picking)...")
    up = estimate_up_vector_manhattan(pc)
    walls = extract_wall_planes(pc, up=up)
    walls = filter_exterior_walls(pc, walls)
    print(f"Found {len(walls)} exterior walls.")

    print("\nOpening 3D view. Shift+click to pick points, 'Q' to finish.\n")
    picked = pick_points(o3d_pc)
    if len(picked) == 0:
        print("No points picked.")
        return

    print(f"\nPicked {len(picked)} points.")
    for group_start in range(0, len(picked) - len(picked) % 3, 3):
        idx_center, idx_edge_w, idx_edge_h = picked[group_start:group_start + 3]
        center = pc.points[idx_center]
        width = 2 * np.linalg.norm(pc.points[idx_edge_w] - center)
        height = 2 * np.linalg.norm(pc.points[idx_edge_h] - center)
        normal = nearest_wall_normal(walls, center) if walls else np.array([1.0, 0.0, 0.0])

        print("  - category: door  # or window - EDIT ME")
        print(f"    center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"    width: {width:.2f}")
        print(f"    height: {height:.2f}")
        print(f"    normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

    leftover = len(picked) % 3
    if leftover:
        print(f"\n({leftover} leftover picked point(s) ignored - need groups of 3: center, width-edge, height-edge)")


if __name__ == "__main__":
    main()
