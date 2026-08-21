"""Giai đoạn 2 (docs/zora_design_plan.md) - VERIFY: `differentiable_sim3.
solve_pose_graph_torch` (PyTorch/autograd) phải cho ra pattern residual
NHẤT QUÁN với `multi_space_graph.solve_pose_graph` (numpy/scipy) đã verify
đúng trước đó trên đúng bộ 4-không-gian (kết quả tốt nhất từng có: cạnh
thật ~0.08-2.37°, 2/3 cạnh giả tách rõ ~25-156°, 1 cạnh giả còn mơ hồ ~1.25°
- xem CONTRIBUTIONS.md mục 12 / docs/multi_space_alignment_plan.md).

KHÔNG kỳ vọng khớp bit-for-bit (optimizer khác nhau: scipy trust-region vs
Adam) - chỉ cần cùng PATTERN phân biệt thật/giả.

Usage:
    python scripts/zora_differentiable_verify.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera  # noqa: E402
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation  # noqa: E402
from sgd_alignment.matching.multi_space_graph import solve_pose_graph  # noqa: E402
from sgd_alignment.zora.differentiable_sim3 import solve_pose_graph_torch  # noqa: E402

SPACES = {
    "h_server": ("data/server/h_server/h_server_room_points - Cloud - segment.ply", "data/server/h_server/results.npz"),
    "server": ("data/server/server/server_room_points-segment.ply", "data/server/server/results.npz"),
    "room_310": ("data/310_indoor/310_indoor_points - Cloud - segment.ply", "data/310_indoor/results.npz"),
    "connecting_space": ("data/connecting_space/connecting_space_points - Cloud-segmented.ply",
                          "data/connecting_space/results.npz"),
}
GROUND_TRUTH_REAL_EDGES = {("h_server", "server"), ("h_server", "room_310"), ("h_server", "connecting_space")}


def main() -> None:
    clusters, cams = {}, {}
    for name, (ply, npz) in SPACES.items():
        clusters[name] = openings_from_manual_segmentation(ply)
        cams[name] = camera_evidence(npz)
    names = list(SPACES.keys())

    print("=== Đo transform từng cặp (giống hệt input đã dùng cho numpy solver trước đó) ===")
    edges, edge_weights = {}, {}
    for a, b in combinations(names, 2):
        result = align_gravity_camera(clusters[a], clusters[b], cams[a], cams[b])
        edges[(a, b)] = (result.s, result.R, result.t)
        edge_weights[(a, b)] = len(result.matches) / (result.opening_residual + 1e-3)
        print(f"  {a}-{b}: status={result.status} residual={result.opening_residual:.4f}")

    print("\n=== Numpy (scipy.least_squares, soft_l1, f_scale=0.1 - cấu hình tốt nhất đã xác nhận) ===")
    pg_np = solve_pose_graph(edges, names, root="h_server", edge_weights=edge_weights,
                              robust_loss="soft_l1", f_scale=0.1,
                              angle_scale_rad=1.0, trans_scale=1.0, log_scale_scale=1.0)
    for e, err in sorted(pg_np.edge_residuals.items(), key=lambda kv: kv[1]["angle_deg"]):
        tag = "THẬT" if e in GROUND_TRUTH_REAL_EDGES else "GIẢ"
        print(f"  {e[0]}-{e[1]} [{tag}]: lệch góc={err['angle_deg']:.2f}°")

    print("\n=== PyTorch (Adam + Huber loss) ===")
    pg_torch = solve_pose_graph_torch(edges, names, root="h_server", edge_weights=edge_weights,
                                       n_steps=800, lr=0.05, huber_delta=1.0)
    print(f"  loss ban đầu={pg_torch.loss_history[0]:.4f} -> loss cuối={pg_torch.loss_history[-1]:.4f}")
    for e, err in sorted(pg_torch.edge_residuals.items(), key=lambda kv: kv[1]["angle_deg"]):
        tag = "THẬT" if e in GROUND_TRUTH_REAL_EDGES else "GIẢ"
        print(f"  {e[0]}-{e[1]} [{tag}]: lệch góc={err['angle_deg']:.2f}°")

    print("\n=== So sánh pattern: cả 2 solver có tách real/fake theo cùng hướng không? ===")
    real_np = [pg_np.edge_residuals[e]["angle_deg"] for e in GROUND_TRUTH_REAL_EDGES]
    fake_np = [v["angle_deg"] for e, v in pg_np.edge_residuals.items() if e not in GROUND_TRUTH_REAL_EDGES]
    real_torch = [pg_torch.edge_residuals[e]["angle_deg"] for e in GROUND_TRUTH_REAL_EDGES]
    fake_torch = [v["angle_deg"] for e, v in pg_torch.edge_residuals.items() if e not in GROUND_TRUTH_REAL_EDGES]
    print(f"  numpy:  real mean={sum(real_np)/3:.2f}°  fake mean={sum(fake_np)/3:.2f}°")
    print(f"  torch:  real mean={sum(real_torch)/3:.2f}°  fake mean={sum(fake_torch)/3:.2f}°")
    consistent = (sum(real_np)/3 < sum(fake_np)/3) == (sum(real_torch)/3 < sum(fake_torch)/3)
    print(f"  => {'NHẤT QUÁN (cùng hướng phân biệt real<fake)' if consistent else '!!! KHÔNG NHẤT QUÁN !!!'}")


if __name__ == "__main__":
    main()
