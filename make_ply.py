"""Dựng point cloud xyz+rgb từ results.npz của DA3.

Cách chạy (Python 3.11 - có numpy + plyfile):
    python make_ply.py <results.npz> <output.ply> [max_points]

Ví dụ:
    python make_ply.py workspace/hanh_lang_dai/results.npz workspace/hanh_lang_dai/hanh_lang_dai_points.ply
"""
import sys
import numpy as np
from plyfile import PlyData, PlyElement

CONF_PERCENTILE = 40      # bỏ 40% pixel kém tin cậy nhất mỗi view
DEPTH_MIN, DEPTH_MAX = 0.05, 30.0
OUTLIER_PCT = 99.5        # bỏ điểm xa tâm quá ngưỡng này


def main():
    npz_path = sys.argv[1]
    out_path = sys.argv[2]
    max_points = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000

    Z = np.load(npz_path)
    D, EX, IN, CF, IM = Z["depth"], Z["extrinsics"], Z["intrinsics"], Z["conf"], Z["image"]

    pts, cols = [], []
    for i in range(len(D)):
        d = D[i]; K = IN[i]; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        thr = np.percentile(CF[i], CONF_PERCENTILE)
        vy, vx = np.nonzero(np.isfinite(d) & (d > DEPTH_MIN) & (d < DEPTH_MAX) & (CF[i] >= thr))
        dd = d[vy, vx]
        Xc = np.stack([(vx - cx) / fx * dd, (vy - cy) / fy * dd, dd], axis=1)
        pts.append((Xc - EX[i][:3, 3]) @ EX[i][:3, :3])   # camera -> world
        cols.append(IM[i][vy, vx])
    P = np.concatenate(pts).astype(np.float32)
    C = np.concatenate(cols).astype(np.uint8)

    # bỏ outlier xa
    dist = np.linalg.norm(P - np.median(P, 0), axis=1)
    keep = dist < np.percentile(dist, OUTLIER_PCT)
    P, C = P[keep], C[keep]

    # rút gọn cho nhẹ
    if len(P) > max_points:
        idx = np.random.RandomState(0).choice(len(P), max_points, replace=False)
        P, C = P[idx], C[idx]

    v = np.zeros(len(P), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = P[:, 0], P[:, 1], P[:, 2]
    v["red"], v["green"], v["blue"] = C[:, 0], C[:, 1], C[:, 2]
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(out_path)
    print(f"WROTE {out_path}  |  {len(P)} points  |  {len(D)} views")


if __name__ == "__main__":
    main()
