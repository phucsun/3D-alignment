"""Giai đoạn 2 (docs/zora_design_plan.md): port Sim(3) pose-graph solver
(`sgd_alignment.matching.multi_space_graph.solve_pose_graph`) sang PyTorch
thuần (autograd), KHÔNG dùng thư viện Lie-group ngoài (PyPose/Theseus) -
tự viết để dễ kiểm soát/debug, khớp phong cách tự viết bằng numpy hiện có.

Công thức Sim(3) giữ NGUYÊN quy ước với bản numpy (`compose`, `invert`) -
chỉ đổi kiểu dữ liệu numpy -> torch.Tensor autograd-compatible, và đổi
solver từ `scipy.optimize.least_squares` -> `torch.optim` (Adam/LBFGS).

Tham số hoá mỗi node (trừ root cố định Identity): 7 chiều
`[log_s, rotvec(3), t(3)]` - y hệt `_pose_to_params`/`_params_to_pose` bên
numpy, chỉ khác là `rotvec -> R` dùng công thức Rodrigues viết tay bằng
torch (không dùng `scipy.spatial.transform.Rotation`, không autograd được).

Đây là bản THỬ NGHIỆM/prototype (chưa merge vào `alignment.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

Sim3T = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (s scalar, R (3,3), t (3,))


def rotvec_to_R(rv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rodrigues, autograd-compatible (không dùng scipy). `rv`: (3,)."""
    theta = torch.linalg.norm(rv)
    K = torch.zeros(3, 3, dtype=rv.dtype, device=rv.device)
    K[0, 1], K[0, 2] = -rv[2], rv[1]
    K[1, 0], K[1, 2] = rv[2], -rv[0]
    K[2, 0], K[2, 1] = -rv[1], rv[0]
    I = torch.eye(3, dtype=rv.dtype, device=rv.device)
    # tránh chia 0 khi theta~0 (identity) - dùng khai triển Taylor bậc nhỏ ẩn qua where
    safe_theta = torch.clamp(theta, min=eps)
    A = torch.sin(safe_theta) / safe_theta
    B = (1 - torch.cos(safe_theta)) / (safe_theta ** 2)
    return I + A * K + B * (K @ K)


def compose(T2: Sim3T, T1: Sim3T) -> Sim3T:
    """T2 ∘ T1: áp T1 trước rồi T2 - y hệt quy ước bản numpy."""
    s1, R1, t1 = T1
    s2, R2, t2 = T2
    return s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2


def invert(T: Sim3T) -> Sim3T:
    s, R, t = T
    s_inv = 1.0 / s
    R_inv = R.T
    return s_inv, R_inv, -s_inv * (R_inv @ t)


def R_log_map(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log-map SO(3) -> rotvec (3,), ỔN ĐỊNH toàn miền (dùng `atan2`, không
    `arccos` - tránh đạo hàm vô hạn ở góc 0/180°). Thay cho xấp xỉ góc-nhỏ
    `R - I` (bị bão hoà/nén sai với góc lệch lớn - xác nhận gây bất ổn thật
    khi verify: cạnh có góc lệch lớn hội tụ kém hẳn so với numpy).

    `v = [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]/2 = sin(theta)*axis`;
    `theta = atan2(||v||, (trace(R)-1)/2)` ổn định mọi theta (trừ đúng
    theta=180° - hiếm, chấp nhận được). `logmap = theta * axis = v *
    (theta/sin(theta))`, xấp xỉ Taylor khi theta~0 để tránh chia 0.
    """
    v = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    v_norm = torch.linalg.norm(v)
    cos_theta = torch.clamp((torch.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.atan2(v_norm, cos_theta)
    # theta/sin(theta) -> 1 khi theta->0; dùng sinc ổn định qua where
    safe_v_norm = torch.clamp(v_norm, min=eps)
    scale = theta / safe_v_norm
    return v * scale


def params_to_pose(p: torch.Tensor) -> Sim3T:
    """p = [log_s, rotvec(3), t(3)] (7,) -> (s, R, t)."""
    log_s, rv, t = p[0], p[1:4], p[4:7]
    return torch.exp(log_s), rotvec_to_R(rv), t


@dataclass
class TorchPoseGraphResult:
    poses: dict[str, Sim3T]
    edge_residuals: dict[tuple[str, str], dict]
    loss_history: list[float]


def solve_pose_graph_torch(
    edges: dict[tuple[str, str], tuple[float, "np.ndarray", "np.ndarray"]],
    names: list[str],
    root: str,
    edge_weights: dict[tuple[str, str], float] | None = None,
    n_steps: int = 500,
    lr: float = 0.05,
    huber_delta: float = 1.0,
) -> TorchPoseGraphResult:
    """Tương đương `multi_space_graph.solve_pose_graph` nhưng bằng PyTorch
    autograd + `torch.optim.Adam`, robust loss = Huber (thay `soft_l1` của
    scipy - cùng họ robust loss, sẵn có trong `torch.nn.functional`).
    """
    import numpy as np
    import torch.nn.functional as F

    free_nodes = [n for n in names if n != root]
    node_idx = {n: i for i, n in enumerate(free_nodes)}

    edges_t = {k: (torch.tensor(float(s)), torch.tensor(R, dtype=torch.float64),
                   torch.tensor(t, dtype=torch.float64))
               for k, (s, R, t) in edges.items()}
    weights = edge_weights or {e: 1.0 for e in edges}

    params = torch.zeros(len(free_nodes) * 7, dtype=torch.float64, requires_grad=True)

    def get_pose(name: str) -> Sim3T:
        if name == root:
            return (torch.tensor(1.0, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                    torch.zeros(3, dtype=torch.float64))
        i = node_idx[name]
        return params_to_pose(params[i * 7:(i + 1) * 7])

    optimizer = torch.optim.Adam([params], lr=lr)
    loss_history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float64)
        for (a, b), measured_ab in edges_t.items():
            Ta, Tb = get_pose(a), get_pose(b)
            predicted_ab = compose(invert(Tb), Ta)
            s_err, R_err, t_err = compose(invert(measured_ab), predicted_ab)
            log_s_err = torch.log(s_err)
            rot_err = R_log_map(R_err)  # rotvec (3,) - log-map ổn định, không bão hoà như Frobenius
            w = weights.get((a, b), 1.0)
            edge_residual = w * torch.cat([log_s_err.reshape(1), rot_err, t_err])
            total_loss = total_loss + F.huber_loss(edge_residual, torch.zeros_like(edge_residual),
                                                    delta=huber_delta, reduction="sum")
        total_loss.backward()
        optimizer.step()
        loss_history.append(float(total_loss.detach()))

    poses = {root: (torch.tensor(1.0, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                     torch.zeros(3, dtype=torch.float64))}
    for n in free_nodes:
        poses[n] = get_pose(n)

    edge_residuals = {}
    with torch.no_grad():
        for (a, b), measured_ab in edges_t.items():
            Ta, Tb = poses[a], poses[b]
            predicted_ab = compose(invert(Tb), Ta)
            s_err, R_err, t_err = compose(invert(measured_ab), predicted_ab)
            cos_ang = torch.clamp((torch.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            edge_residuals[(a, b)] = {
                "angle_deg": float(torch.rad2deg(torch.arccos(cos_ang))),
                "trans_norm": float(torch.linalg.norm(t_err)),
                "scale_dev": float(abs(torch.log(s_err))),
            }

    return TorchPoseGraphResult(poses=poses, edge_residuals=edge_residuals, loss_history=loss_history)


# --------------------------------------------------------------------------------------
# Giai đoạn 3 (docs/zora_design_plan.md): GNC-TLS (Yang, Antonante, Tzoumas, Carlone,
# "Graduated Non-Convexity for Robust Spatial Perception", 2020) - THAY cho IRLS tự viết
# đã THẤT BẠI (bất ổn, làm mất cạnh thật ở quy mô N nhỏ - xem
# docs/multi_space_alignment_plan.md). Khác biệt cốt lõi với IRLS tự chế trước đó: GNC có
# LỊCH TRÌNH ANNEALING có công thức toán học rõ ràng (không phải decay tuỳ ý chọn tay),
# bắt đầu từ 1 surrogate GẦN LỒI (mu nhỏ theo công thức khởi tạo chuẩn) rồi tăng dần độ
# non-convex - continuation giúp tránh local minima mà IRLS tự viết mắc phải.
#
# Dùng scalar "góc lệch" (radian, đã xác nhận là chiều thông tin phân biệt thật/giả tốt
# nhất trên dữ liệu này - không mix đơn vị góc/tịnh tiến/scale như đã thử và thất bại
# trước đó) làm residual cho GNC threshold; pose vẫn tối ưu trên đầy đủ 7 chiều.
# --------------------------------------------------------------------------------------
@dataclass
class GNCResult:
    poses: dict[str, Sim3T]
    edge_residuals: dict[tuple[str, str], dict]
    gnc_weights: dict[tuple[str, str], float]  # ~1.0 = inlier (giữ), ~0.0 = outlier (loại)
    mu_history: list[float]


def _gnc_tls_weight(r2: float, mu: float, c2: float) -> float:
    """Công thức weight update GNC-TLS gốc (Yang et al. 2020, eq. 16)."""
    if r2 <= (mu / (mu + 1.0)) * c2:
        return 1.0
    if r2 >= ((mu + 1.0) / mu) * c2:
        return 0.0
    return float((c2 * mu * (mu + 1.0) / r2) ** 0.5 - mu)


def solve_pose_graph_gnc(
    edges: dict[tuple[str, str], tuple[float, "np.ndarray", "np.ndarray"]],
    names: list[str],
    root: str,
    edge_weights: dict[tuple[str, str], float] | None = None,
    c_deg: float = 5.0,
    mu_factor: float = 1.4,
    n_outer: int = 15,
    n_inner_steps: int = 200,
    lr: float = 0.05,
) -> GNCResult:
    """GNC-TLS trên residual góc (radian). `c_deg`: ngưỡng phân biệt inlier/
    outlier (độ) - cùng ý nghĩa với ngưỡng PCM=5° đã dùng ở bản numpy, để so
    sánh trực tiếp được. `edge_weights`: trọng số ưu tiên tiên nghiệm (như
    `n_matches/residual`) - nhân THÊM vào trọng số GNC (2 loại trọng số độc
    lập: 1 cái nói "tin đến đâu nếu đúng", 1 cái GNC tự học "có đúng không").
    """
    import numpy as np
    import torch.nn.functional as F  # noqa: F401 (giữ import cho nhất quán, không dùng huber ở đây)

    free_nodes = [n for n in names if n != root]
    node_idx = {n: i for i, n in enumerate(free_nodes)}
    edges_t = {k: (torch.tensor(float(s)), torch.tensor(R, dtype=torch.float64),
                   torch.tensor(t, dtype=torch.float64))
               for k, (s, R, t) in edges.items()}
    prior_w = edge_weights or {e: 1.0 for e in edges}
    c2 = (c_deg * 3.141592653589793 / 180.0) ** 2

    def get_pose(params, name):
        if name == root:
            return (torch.tensor(1.0, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                    torch.zeros(3, dtype=torch.float64))
        i = node_idx[name]
        return params_to_pose(params[i * 7:(i + 1) * 7])

    def edge_angle_sq(params, a, b):
        Ta, Tb = get_pose(params, a), get_pose(params, b)
        predicted_ab = compose(invert(Tb), Ta)
        _, R_err, _ = compose(invert(edges_t[(a, b)]), predicted_ab)
        return torch.sum(R_log_map(R_err) ** 2)

    def solve_weighted(combined_w: dict, params_init: torch.Tensor, n_steps: int) -> torch.Tensor:
        params = params_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, dtype=torch.float64)
            for (a, b) in edges_t:
                Ta, Tb = get_pose(params, a), get_pose(params, b)
                predicted_ab = compose(invert(Tb), Ta)
                s_err, R_err, t_err = compose(invert(edges_t[(a, b)]), predicted_ab)
                res = torch.cat([torch.log(s_err).reshape(1), R_log_map(R_err), t_err])
                loss = loss + combined_w[(a, b)] * torch.sum(res ** 2)
            loss.backward()
            optimizer.step()
        return params.detach()

    # ---- vòng 0: giải KHÔNG trọng số GNC (chỉ prior_w) để có residual khởi tạo ----
    params = torch.zeros(len(free_nodes) * 7, dtype=torch.float64)
    params = solve_weighted(prior_w, params, n_inner_steps)
    r2 = {e: float(edge_angle_sq(params, *e)) for e in edges_t}

    max_r2 = max(r2.values())
    mu = c2 / (2 * max_r2 - c2) if max_r2 > c2 / 2 else 1e6  # công thức khởi tạo chuẩn GNC (đảm bảo gần lồi)
    mu = max(mu, 1e-6)
    mu_history = [mu]
    gnc_w = {e: 1.0 for e in edges_t}

    for _ in range(n_outer):
        for e in edges_t:
            gnc_w[e] = _gnc_tls_weight(r2[e], mu, c2)
        combined_w = {e: prior_w.get(e, 1.0) * gnc_w[e] for e in edges_t}
        params = solve_weighted(combined_w, params, n_inner_steps)
        r2 = {e: float(edge_angle_sq(params, *e)) for e in edges_t}
        mu *= mu_factor
        mu_history.append(mu)

    poses = {root: (torch.tensor(1.0, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                     torch.zeros(3, dtype=torch.float64))}
    for n in free_nodes:
        poses[n] = get_pose(params, n)

    edge_residuals = {}
    with torch.no_grad():
        for (a, b), measured_ab in edges_t.items():
            Ta, Tb = poses[a], poses[b]
            predicted_ab = compose(invert(Tb), Ta)
            s_err, R_err, t_err = compose(invert(measured_ab), predicted_ab)
            cos_ang = torch.clamp((torch.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            edge_residuals[(a, b)] = {
                "angle_deg": float(torch.rad2deg(torch.arccos(cos_ang))),
                "trans_norm": float(torch.linalg.norm(t_err)),
                "scale_dev": float(abs(torch.log(s_err))),
            }

    return GNCResult(poses=poses, edge_residuals=edge_residuals, gnc_weights=gnc_w, mu_history=mu_history)
