"""Giai đoạn 3 (docs/multi_space_alignment_plan.md): tối ưu Sim(3) pose-graph
toàn cục cho N không gian, dùng robust loss thay cho "history reweighting"
tự viết của SGHR (arXiv:2304.00467) - mọi cạnh (mọi cặp không gian đã chạy
qua `gravity_align.align_gravity_camera`) cùng tham gia MỘT bài toán bình
phương tối thiểu chung; cạnh SAI (false positive - 2 không gian không hề kề
nhau nhưng thuật toán ghép opening "khớp" tình cờ) tự động bị hạ trọng số
qua residual cao sau khi hội tụ, KHÔNG cần biết trước ai là hub và KHÔNG
cần đếm tam giác rời rạc (cách cũ - `multi_space_cycle_consistency.py` -
đã CHỨNG MINH THẤT BẠI trên chính bộ dữ liệu này: topology thật là star,
không có tam giác nào toàn cạnh thật để tự đối chiếu, nên đếm tam giác cô
lập thiếu thông tin phân biệt cạnh thật/giả).

Đây là bản THỬ NGHIỆM/prototype (chưa merge vào `alignment.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .gravity_align import align_gravity_camera


Sim3 = tuple[float, np.ndarray, np.ndarray]  # (s, R (3,3), t (3,))


def _rotvec_to_R(rv: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rv).as_matrix()


def _R_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def compose(T2: Sim3, T1: Sim3) -> Sim3:
    """T2 ∘ T1: áp T1 trước rồi T2."""
    s1, R1, t1 = T1
    s2, R2, t2 = T2
    return s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2


def invert(T: Sim3) -> Sim3:
    s, R, t = T
    s_inv = 1.0 / s
    R_inv = R.T
    return s_inv, R_inv, -s_inv * (R_inv @ t)


def _params_to_pose(p: np.ndarray) -> Sim3:
    """p = [log_s, rotvec(3), t(3)] (7,) -> (s, R, t)."""
    log_s, rv, t = p[0], p[1:4], p[4:7]
    return float(np.exp(log_s)), _rotvec_to_R(rv), t


def _pose_to_params(T: Sim3) -> np.ndarray:
    s, R, t = T
    return np.concatenate([[np.log(s)], _R_to_rotvec(R), t])


@dataclass
class PoseGraphResult:
    poses: dict[str, Sim3]              # node -> transform (node's local frame -> world/root frame)
    edge_residuals: dict[tuple[str, str], dict]  # (a,b) -> {"angle_deg", "trans_norm", "scale_dev"}
    success: bool


def solve_pose_graph(
    edges: dict[tuple[str, str], Sim3],
    names: list[str],
    root: str,
    edge_weights: dict[tuple[str, str], float] | None = None,
    robust_loss: str = "soft_l1",
    f_scale: float = 1.0,
    angle_scale_rad: float = np.radians(2.0),
    trans_scale: float = 0.05,
    log_scale_scale: float = 0.05,
) -> PoseGraphResult:
    """Giải pose tuyệt đối (node -> root) cho mọi node bằng least-squares
    robust trên TOÀN BỘ cạnh cùng lúc.

    `edges[(a, b)]` = transform ĐO ĐƯỢC a->b (từ `align_gravity_camera(a,
    b)` hoặc nghịch đảo nếu chỉ đo `(b, a)`).
    `edge_weights`: trọng số ưu tiên mỗi cạnh (vd cam_dominance/grav_dot/
    up_consistency, xem `_edge_confidence_weight`) - nhân thêm vào robust
    loss, KHÔNG thay thế nó.

    `angle_scale_rad`/`trans_scale`/`log_scale_scale`: CHUẨN HOÁ ĐƠN VỊ
    trước khi ghép 3 thành phần (góc [radian], tịnh tiến [mét], scale
    [không thứ nguyên]) thành 1 vector residual duy nhất - nếu không chuẩn
    hoá, thành phần có độ lớn tuyệt đối lớn nhất (thường là tịnh tiến, có
    thể tới vài mét) sẽ áp đảo quyết định "cạnh nào là outlier" của
    `soft_l1`/`f_scale`, bất kể sai số góc/scale thực ra tốt hay tệ - xác
    nhận đây là nguyên nhân khiến bản đầu tiên (không chuẩn hoá) cho kết
    quả không ổn định. Mỗi thành phần chia cho "1 đơn vị sai số chấp nhận
    được" của riêng nó (mặc định: 2° góc, 5cm tịnh tiến, 5% scale) rồi mới
    ghép chung - để `f_scale=1.0` có ý nghĩa nhất quán trên mọi thành phần:
    "sai số bằng đúng 1 lần ngưỡng chấp nhận được thì bắt đầu bị giảm ảnh
    hưởng".

    `robust_loss`/`f_scale`: forwarded to `scipy.optimize.least_squares`.
    """
    free_nodes = [n for n in names if n != root]
    node_idx = {n: i for i, n in enumerate(free_nodes)}
    n_free = len(free_nodes)

    # khởi tạo: BFS/greedy qua cạnh có sẵn từ root, fallback Identity nếu cô lập
    init_poses: dict[str, Sim3] = {root: (1.0, np.eye(3), np.zeros(3))}
    frontier = {root}
    remaining = set(free_nodes)
    while remaining:
        progressed = False
        for a, b in edges:
            if a in frontier and b in remaining:
                init_poses[b] = compose(init_poses[a], edges[(a, b)])
                frontier.add(b)
                remaining.discard(b)
                progressed = True
            elif b in frontier and a in remaining:
                init_poses[a] = compose(init_poses[b], invert(edges[(a, b)]))
                frontier.add(a)
                remaining.discard(a)
                progressed = True
        if not progressed:
            for n in remaining:
                init_poses[n] = (1.0, np.eye(3), np.zeros(3))
            break

    x0 = np.concatenate([_pose_to_params(init_poses[n]) for n in free_nodes])

    def _get_pose(x: np.ndarray, name: str) -> Sim3:
        if name == root:
            return (1.0, np.eye(3), np.zeros(3))
        i = node_idx[name]
        return _params_to_pose(x[i * 7:(i + 1) * 7])

    def _residuals(x: np.ndarray) -> np.ndarray:
        out = []
        for (a, b), measured_ab in edges.items():
            Ta, Tb = _get_pose(x, a), _get_pose(x, b)
            # a->b dự đoán: local_a --Ta--> world --Tb^-1--> local_b
            predicted_ab = compose(invert(Tb), Ta)
            s_err, R_err, t_err = compose(invert(measured_ab), predicted_ab)
            w = 1.0 if edge_weights is None else edge_weights.get((a, b), 1.0)
            # chuẩn hoá đơn vị TRƯỚC KHI nhân trọng số/ghép vào 1 vector chung
            log_s_n = np.log(s_err) / log_scale_scale
            rotvec_n = _R_to_rotvec(R_err) / angle_scale_rad
            t_n = t_err / trans_scale
            out.extend([w * log_s_n, *(w * rotvec_n), *(w * t_n)])
        return np.array(out)

    result = least_squares(_residuals, x0, loss=robust_loss, f_scale=f_scale, method="trf")

    poses = {root: (1.0, np.eye(3), np.zeros(3))}
    for n in free_nodes:
        poses[n] = _get_pose(result.x, n)

    edge_residuals = {}
    for (a, b), measured_ab in edges.items():
        Ta, Tb = poses[a], poses[b]
        predicted_ab = compose(invert(Tb), Ta)
        s_err, R_err, t_err = compose(invert(measured_ab), predicted_ab)
        cos_ang = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        edge_residuals[(a, b)] = {
            "angle_deg": float(np.degrees(np.arccos(cos_ang))),
            "trans_norm": float(np.linalg.norm(t_err)),
            "scale_dev": float(abs(np.log(s_err))),
        }

    return PoseGraphResult(poses=poses, edge_residuals=edge_residuals, success=result.success)


@dataclass
class IRLSResult:
    poses: dict[str, Sim3]
    edge_residuals: dict[tuple[str, str], dict]
    edge_trust: dict[tuple[str, str], float]        # trọng số cuối cùng, luỹ tích qua mọi vòng
    history: list[dict[tuple[str, str], float]]      # trọng số sau mỗi vòng, để vẽ đồ thị hội tụ
    kept_edges: set[tuple[str, str]]                 # PCM: tập cạnh nhất quán tối đa với nghiệm cuối


def solve_pose_graph_irls(
    edges: dict[tuple[str, str], Sim3],
    names: list[str],
    root: str,
    edge_weights: dict[tuple[str, str], float] | None = None,
    n_iters: int = 8,
    angle_scale_deg: float = 3.0,
    pcm_angle_threshold_deg: float = 5.0,
) -> IRLSResult:
    """History reweighting thật (mượn SGHR arXiv:2304.00467) - KHÁC
    `solve_pose_graph` (1 lần gọi `least_squares(loss=robust)`, không có
    "lịch sử"): mỗi vòng, giải lại toàn bộ pose-graph với trọng số HIỆN TẠI,
    rồi CẬP NHẬT LUỸ TÍCH trọng số mỗi cạnh dựa trên residual của chính nó ở
    vòng đó (Cauchy-style: `w *= 1 / (1 + (angle/scale)^2)`) - cạnh liên tục
    lệch nhiều vòng liên tiếp bị đè trọng số xuống theo cấp số nhân (hiệu
    ứng "trí nhớ"/hysteresis mà 1 lần robust-loss không có), trong khi cạnh
    thật (residual luôn thấp) gần như giữ nguyên trọng số ban đầu.

    Sau khi hội tụ, áp thêm PCM (mượn DOOR-SLAM arXiv:1909.12198, bản batch/
    centralized, không cần bản phân tán): với nghiệm pose CUỐI CÙNG đã tối
    ưu (không phải đo trực tiếp), tìm tập cạnh lớn nhất mà MỌI cặp trong đó
    đều có residual (so với nghiệm cuối) dưới `pcm_angle_threshold_deg` -
    quyết định nhị phân accept/reject cuối cùng, tách biệt khỏi trọng số
    liên tục dùng để giải pose-graph.
    """
    # chuẩn hoá trọng số ban đầu (mean=1) - trọng số thô (vd 40-80 từ n_matches/residual)
    # có scale tuỳ ý, không nên nhân trực tiếp vào residual đã chuẩn hoá của least_squares
    raw = dict(edge_weights) if edge_weights is not None else {e: 1.0 for e in edges}
    mean_w = np.mean(list(raw.values())) or 1.0
    weights = {e: w / mean_w for e, w in raw.items()}

    history: list[dict[tuple[str, str], float]] = []
    pg = None
    WEIGHT_FLOOR = 0.05  # không cho trọng số sụp về 0 tuyệt đối - tránh 1 cạnh bị "khai tử" quá sớm
    for _ in range(n_iters):
        # GIỮ robust loss (soft_l1) ở MỌI vòng - history reweighting chỉ CỘNG THÊM 1 lớp
        # "trí nhớ" lên trên, không thay thế cơ chế robust đã ổn định từ solve_pose_graph
        pg = solve_pose_graph(edges, names, root, edge_weights=weights, robust_loss="soft_l1", f_scale=0.1)
        history.append(dict(weights))
        for e, err in pg.edge_residuals.items():
            decay = 1.0 / (1.0 + (err["angle_deg"] / angle_scale_deg) ** 2)
            weights[e] = max(WEIGHT_FLOOR, weights[e] * decay)

    # PCM: giữ cạnh có residual (so với nghiệm CUỐI) dưới ngưỡng - đơn giản
    # hoá thành "consistency graph": 1 cạnh gốc được giữ nếu residual của
    # chính nó, đo trên nghiệm cuối cùng (đã được đa số cạnh tốt chi phối),
    # đủ nhỏ.
    kept_edges = {e for e, err in pg.edge_residuals.items() if err["angle_deg"] <= pcm_angle_threshold_deg}

    return IRLSResult(poses=pg.poses, edge_residuals=pg.edge_residuals, edge_trust=weights,
                       history=history, kept_edges=kept_edges)


# --------------------------------------------------------------------------------------
# Opening-conflict graph (Giai đoạn 3, hướng đã verify TỐT NHẤT: 3/3 cạnh thật, không cần
# biết trước hub) - mỗi cửa vật lý chỉ được dùng cho ĐÚNG 1 quan hệ; MWIS + lặp loại trừ
# đến hội tụ. Xem scripts/multi_space_opening_conflict_graph.py cho bối cảnh đầy đủ.
# --------------------------------------------------------------------------------------
_STATUS_RANK = {"CONFIDENT": 2, "AMBIGUOUS": 1, "NO_SOLUTION": 0}


@dataclass
class OpeningEdge:
    a: str
    b: str
    s: float
    R: np.ndarray
    t: np.ndarray
    weight: float
    used_a: frozenset  # {(a, opening_idx), ...}
    used_b: frozenset  # {(b, opening_idx), ...}


def _opening_edges_conflict(c1: OpeningEdge, c2: OpeningEdge) -> bool:
    return bool((c1.used_a | c1.used_b) & (c2.used_a | c2.used_b))


def _best_direction_opening_edge(a, b, clusters, cams, avail) -> OpeningEdge | None:
    """Tính cạnh (a,b) chỉ trên phần opening CÒN TRỐNG (`avail`) của mỗi bên -
    thử CẢ 2 CHIỀU `(a,b)`/`(b,a)` (align_gravity_camera không đối xứng, xác
    nhận thật trên dữ liệu server/h_server: 1 chiều cho residual 0.0674/3 cặp,
    chiều kia cho 0.0101/2 cặp) - giữ chiều tốt hơn."""
    idx_a, idx_b = sorted(avail[a]), sorted(avail[b])
    if not idx_a or not idx_b:
        return None
    sub_a = [clusters[a][i] for i in idx_a]
    sub_b = [clusters[b][i] for i in idx_b]

    r_ab = align_gravity_camera(sub_a, sub_b, cams[a], cams[b])
    r_ba = align_gravity_camera(sub_b, sub_a, cams[b], cams[a])
    key_ab = (_STATUS_RANK[r_ab.status], -r_ab.opening_residual)
    key_ba = (_STATUS_RANK[r_ba.status], -r_ba.opening_residual)

    if r_ab.status == "NO_SOLUTION" and r_ba.status == "NO_SOLUTION":
        return None
    if key_ab >= key_ba:
        result = r_ab
        matches = [(idx_a[i], idx_b[j]) for i, j in result.matches]
        s, R, t = result.s, result.R, result.t
    else:
        result = r_ba  # A=sub_b, B=sub_a -> match (i local-b, j local-a)
        matches = [(idx_a[j], idx_b[i]) for i, j in result.matches]
        s, R, t = invert((result.s, result.R, result.t))

    if not matches:
        return None
    weight = len(matches) - result.opening_residual
    used_a = frozenset((a, i) for i, _ in matches)
    used_b = frozenset((b, j) for _, j in matches)
    return OpeningEdge(a, b, s, R, t, weight, used_a, used_b)


def resolve_opening_conflict_graph(
    clusters: dict[str, list],
    cams: dict,
    max_rounds: int = 6,
    verbose: bool = False,
    prior_weights: dict[tuple[str, str], float] | None = None,
) -> dict[tuple[str, str], OpeningEdge]:
    """Tìm tập cạnh (space, space) KHÔNG dùng lại cửa của nhau, tổng trọng số
    lớn nhất, KHÔNG cần biết trước topology/hub - mỗi vòng: tính lại mọi cạnh
    chưa khoá chỉ trên phần opening còn trống, chọn MWIS (vét cạn 2^n tập
    con - n nhỏ nên đủ nhanh), khoá cạnh thắng, lặp đến hội tụ.

    `clusters`: {tên không gian: list[OpeningCluster]}.
    `cams`: {tên không gian: CameraEvidence}.
    `prior_weights`: NHÂN vào trọng số nội tại (`n_matches - residual`) của
    mỗi candidate TRƯỚC KHI đưa vào MWIS - dùng để tiêm prior tôpô từ Layer 1
    (VLM/text) NGAY TỪ BƯỚC MWIS, không chỉ downstream cho GNC. Xác nhận
    thật (docs/multi_space_alignment_plan.md): ở quy mô N≥6 có nhiễu xuyên-
    building, `n_matches/residual` một mình có thể xếp hạng 1 cạnh giả xuyên-
    building CAO HƠN cạnh thật, khiến MWIS loại nhầm cạnh thật ngay từ vòng
    đầu - GNC downstream không cứu được vì không có input đúng để bắt đầu.
    `None` (mặc định) = không đổi hành vi cũ.
    Trả về {(a,b): OpeningEdge} cho các cạnh đã được xác nhận (không xung đột).
    """
    names = list(clusters.keys())
    avail = {name: set(range(len(clusters[name]))) for name in names}
    locked_by_pair: dict[tuple[str, str], OpeningEdge] = {}
    unresolved_pairs = set(combinations(names, 2))

    for round_no in range(1, max_rounds + 1):
        new_candidates: list[OpeningEdge] = []
        for a, b in list(unresolved_pairs):
            cand = _best_direction_opening_edge(a, b, clusters, cams, avail)
            if cand is not None:
                if prior_weights is not None:
                    cand.weight *= prior_weights.get((a, b), 1.0)
                new_candidates.append(cand)
                if verbose:
                    print(f"  [vòng {round_no}] {a}-{b}: dùng {a}{sorted(i for _, i in cand.used_a)} <-> "
                          f"{b}{sorted(j for _, j in cand.used_b)}  weight={cand.weight:.3f}")

        pool = list(locked_by_pair.values()) + new_candidates
        n = len(pool)
        conflict = [[_opening_edges_conflict(pool[i], pool[j]) if i != j else False for j in range(n)]
                    for i in range(n)]

        best_subset, best_weight = [], -1.0
        for mask in range(1 << n):
            idx = [i for i in range(n) if mask & (1 << i)]
            if any(conflict[i][j] for i in idx for j in idx if i < j):
                continue
            w = sum(pool[i].weight for i in idx)
            if w > best_weight:
                best_weight, best_subset = w, idx

        winners = [pool[i] for i in best_subset]
        newly_locked = [c for c in winners if (c.a, c.b) not in locked_by_pair]
        if not newly_locked:
            break
        for c in newly_locked:
            for space, i in c.used_a | c.used_b:
                avail[space].discard(i)
            unresolved_pairs.discard((c.a, c.b))
            locked_by_pair[(c.a, c.b)] = c

    return locked_by_pair
