"""Giai đoạn 4, Hướng 1 (Alternating Optimization) - xen kẽ 2 pha, ĐÚNG cách
(bản sửa lỗi regression phát hiện thật ở N=4: bản đầu tiên chỉ tính
candidate 1 lần trên toàn bộ opening rồi tắt/bật cả cạnh - làm mất đúng cơ
chế "tính lại trên opening còn trống" của Giai đoạn 1, khiến 1 cạnh thật
(`h_server-connecting_space`) bị kẹt vĩnh viễn ở candidate SAI thay vì tìm
ra candidate ĐÚNG khi opening xung đột đã bị loại).

1. **Pha rời rạc (opening pool)**: mọi cặp CHƯA được khoá được TÍNH LẠI
   candidate trên phần opening CÒN TRỐNG (`avail`) - y hệt Giai đoạn 1
   (`_best_direction_opening_edge`).
2. **Pha liên tục (GNC)**: chạy vài vòng GNC-TLS trên tập candidate hiện
   tại (cạnh đã khoá + cạnh chưa khoá vừa tính lại) để có trọng số liên tục
   tốt hơn `n_matches/residual` tĩnh.
3. **Pha rời rạc (MWIS)**: trong số các cạnh CHƯA khoá, tìm tập không xung
   đột lẫn nhau (và tự động không xung đột với cạnh đã khoá, vì `avail` đã
   loại openings của chúng) có tổng trọng số GNC lớn nhất - KHOÁ các cạnh
   thắng (loại opening đã dùng khỏi `avail`).
4. Lặp lại đến khi không còn cạnh mới nào được khoá (hội tụ).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch

from sgd_alignment.matching.gravity_align import CameraEvidence
from sgd_alignment.matching.multi_space_graph import _best_direction_opening_edge, _opening_edges_conflict
from sgd_alignment.zora.differentiable_sim3 import (
    Sim3T, compose, invert, params_to_pose, R_log_map, _gnc_tls_weight,
)


@dataclass
class AlternatingResult:
    poses: dict[str, Sim3T]
    edge_residuals: dict[tuple[str, str], dict]
    gnc_weights: dict[tuple[str, str], float]
    locked: dict[tuple[str, str], bool]        # True = đã khoá (chấp nhận), chỉ các cạnh này trong output cuối
    rounds_used: int


def alternating_gnc_mwis(
    clusters: dict[str, list],
    cams: dict[str, CameraEvidence],
    names: list[str],
    root: str,
    prior_weights: dict[tuple[str, str], float] | None = None,
    max_rounds: int = 8,
    gnc_outer_per_round: int = 3,
    gnc_inner_steps: int = 150,
    lr: float = 0.05,
    c_deg: float = 5.0,
    mu_factor: float = 1.4,
) -> AlternatingResult:
    avail = {n: set(range(len(clusters[n]))) for n in names}
    locked_pairs: set[tuple[str, str]] = set()
    locked_edge: dict[tuple[str, str], object] = {}   # (a,b) -> OpeningEdge đã khoá
    all_pairs = list(combinations(names, 2))
    unresolved = set(all_pairs)

    free_nodes = [n for n in names if n != root]
    node_idx = {n: i for i, n in enumerate(free_nodes)}
    c2 = (c_deg * 3.141592653589793 / 180.0) ** 2

    def get_pose(params, name):
        if name == root:
            return (torch.tensor(1.0, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                    torch.zeros(3, dtype=torch.float64))
        i = node_idx[name]
        return params_to_pose(params[i * 7:(i + 1) * 7])

    def solve_weighted(edges_t, combined_w, params_init, n_steps):
        params = params_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, dtype=torch.float64)
            for e, measured_ab in edges_t.items():
                w = combined_w.get(e, 0.0)
                if w <= 0.0:
                    continue
                a, b = e
                Ta, Tb = get_pose(params, a), get_pose(params, b)
                predicted_ab = compose(invert(Tb), Ta)
                s_err, R_err, t_err = compose(invert(measured_ab), predicted_ab)
                res = torch.cat([torch.log(s_err).reshape(1), R_log_map(R_err), t_err])
                loss = loss + w * torch.sum(res ** 2)
            loss.backward()
            optimizer.step()
        return params.detach()

    def angle_sq(edges_t, params, e):
        a, b = e
        Ta, Tb = get_pose(params, a), get_pose(params, b)
        predicted_ab = compose(invert(Tb), Ta)
        _, R_err, _ = compose(invert(edges_t[e]), predicted_ab)
        return torch.sum(R_log_map(R_err) ** 2)

    params = torch.zeros(len(free_nodes) * 7, dtype=torch.float64)
    gnc_w_all: dict[tuple[str, str], float] = {}
    rounds_used = 0

    for round_no in range(max_rounds):
        rounds_used = round_no + 1

        # ---- pha 1: tính lại candidate cho mọi cặp CHƯA khoá, trên opening còn trống ----
        current_candidates = dict(locked_edge)
        for a, b in list(unresolved):
            cand = _best_direction_opening_edge(a, b, clusters, cams, avail)
            if cand is not None:
                current_candidates[(a, b)] = cand

        if not current_candidates:
            break

        edges_t = {e: (torch.tensor(float(c.s)), torch.tensor(c.R, dtype=torch.float64),
                       torch.tensor(c.t, dtype=torch.float64))
                   for e, c in current_candidates.items()}
        base_prior = dict(prior_weights) if prior_weights is not None else {}
        for e, c in current_candidates.items():
            base_prior.setdefault(e, max(c.weight, 1e-3))
        gnc_w = {e: (1.0 if e in locked_pairs else gnc_w_all.get(e, 1.0)) for e in current_candidates}

        # ---- pha 2: vài vòng GNC-TLS trên tập candidate hiện tại (đã khoá + mới tính lại) ----
        combined = {e: base_prior[e] * gnc_w[e] for e in current_candidates}
        params = solve_weighted(edges_t, combined, params, gnc_inner_steps)
        r2 = {e: float(angle_sq(edges_t, params, e)) for e in current_candidates if e not in locked_pairs}

        if r2:
            max_r2 = max(r2.values())
            mu = c2 / (2 * max_r2 - c2) if max_r2 > c2 / 2 else 1e6
            mu = max(mu, 1e-6)
            for _ in range(gnc_outer_per_round):
                for e in r2:
                    gnc_w[e] = _gnc_tls_weight(r2[e], mu, c2)
                combined = {e: base_prior[e] * gnc_w[e] for e in current_candidates}
                params = solve_weighted(edges_t, combined, params, gnc_inner_steps)
                r2 = {e: float(angle_sq(edges_t, params, e)) for e in r2}
                mu *= mu_factor

        gnc_w_all.update(gnc_w)

        # ---- pha 3: MWIS chỉ trong số cạnh CHƯA khoá (đã tự loại trừ xung đột với cạnh đã
        # khoá nhờ `avail` bị thu hẹp) - khoá cạnh thắng ----
        unresolved_candidates = {e: current_candidates[e] for e in unresolved if e in current_candidates}
        keys = list(unresolved_candidates)
        n = len(keys)
        conflict = [[_opening_edges_conflict(unresolved_candidates[keys[i]], unresolved_candidates[keys[j]])
                     if i != j else False for j in range(n)] for i in range(n)]
        best_subset, best_w = [], -1.0
        for mask in range(1 << n):
            idx = [i for i in range(n) if mask & (1 << i)]
            if any(conflict[i][j] for i in idx for j in idx if i < j):
                continue
            w = sum(gnc_w[keys[i]] for i in idx)
            if w > best_w:
                best_w, best_subset = w, idx
        winners = {keys[i] for i in best_subset}

        if not winners:
            break
        for e in winners:
            c = unresolved_candidates[e]
            for space, i in c.used_a | c.used_b:
                avail[space].discard(i)
            locked_pairs.add(e)
            locked_edge[e] = c
            unresolved.discard(e)

    poses = {root: (torch.tensor(1.0, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                     torch.zeros(3, dtype=torch.float64))}
    for n in free_nodes:
        poses[n] = get_pose(params, n)

    final_edges_t = {e: (torch.tensor(float(c.s)), torch.tensor(c.R, dtype=torch.float64),
                          torch.tensor(c.t, dtype=torch.float64))
                      for e, c in locked_edge.items()}
    edge_residuals = {}
    locked_out = {}
    with torch.no_grad():
        for (a, b), measured_ab in final_edges_t.items():
            Ta, Tb = poses[a], poses[b]
            predicted_ab = compose(invert(Tb), Ta)
            s_err, R_err, t_err = compose(invert(measured_ab), predicted_ab)
            cos_ang = torch.clamp((torch.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            edge_residuals[(a, b)] = {
                "angle_deg": float(torch.rad2deg(torch.arccos(cos_ang))),
                "trans_norm": float(torch.linalg.norm(t_err)),
                "scale_dev": float(abs(torch.log(s_err))),
            }
            locked_out[(a, b)] = True

    return AlternatingResult(poses=poses, edge_residuals=edge_residuals, gnc_weights=gnc_w_all,
                              locked=locked_out, rounds_used=rounds_used)
