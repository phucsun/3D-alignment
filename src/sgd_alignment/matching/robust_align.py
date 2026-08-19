"""Robust, up-free, sign-invariant opening matcher for two independent reconstructions.

Design (co-reviewed): each scene is an arbitrary rotation AND arbitrary uniform scale
of the world, so we recover a SIMILARITY transform (s, R, t) mapping A -> B. We do NOT
trust any up-derived quantity (width/height/u_axis/v_axis can be corrupted by a wrong
"up"). Per opening we recompute up-free, sign-free geometry from the raw point cluster:
  - center      : robust min-area-rectangle center in the fitted plane
  - normal_line : plane normal as an UNORIENTED line (n == -n)
  - extents     : the two rectangle side lengths, sorted ascending (up-free size)
Matching is near-exhaustive hypothesize-and-verify (openings are few): triangle
hypotheses (3 non-collinear centers -> Umeyama-with-scale) plus a 2-opening + normal
fallback, scored by partial bipartite consensus using only sign-invariant residuals.
Reflection (chirality) and observability are reported explicitly rather than hidden.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------------------
@dataclass
class OpeningCluster:
    """Raw input: the 3D points of ONE opening + its category. No up-derived fields."""
    category: str
    points: np.ndarray            # (N, 3)
    source_index: int | None = None


@dataclass
class RobustOpening:
    category: str
    center: np.ndarray            # (3,) rectangle center
    normal_line: np.ndarray       # (3,) unit, UNORIENTED
    extents: tuple[float, float]  # (small, large), up-free
    point_count: int
    planarity: float
    linearity: float
    normal_reliable: bool
    size_reliable: bool
    source_index: int


@dataclass
class AlignResult:
    s: float
    R: np.ndarray                 # (3,3) PURE rotation, det = +1
    t: np.ndarray                 # (3,)
    matches: list[tuple[int, int, float]]  # (idx_A, idx_B, residual)
    status: str                   # CONFIDENT | AMBIGUOUS | LOW_CONFIDENCE | NO_SOLUTION
    reason: str = ""
    # AMBIGUOUS (rank-1 single-wall): (s,R,t) is only ONE of several equally-valid
    # candidates and is NOT authoritative -- use it for preview only, opt-in via `provisional`.
    provisional: bool = False
    candidates: list = field(default_factory=list)   # list of (s, R, t) discrete alternatives
    n_inliers: int = 0
    mean_residual: float = float("nan")
    reflection_preferred: bool = False
    chirality_observable: bool = False
    second_best_inliers: int = 0
    # opposite-side (non-overlap) disambiguation of a degenerate single-wall roll
    side_decided_roll: bool = False
    side_score: float = float("nan")     # -D_A*D_B ; >0 opposite sides (good), <0 same side
    side_dominance_A: float = float("nan")
    side_dominance_B: float = float("nan")


def transform_points(P: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * (P @ R.T) + t


# --------------------------------------------------------------------------------------
# 1. Up-free descriptor from a point cluster
# --------------------------------------------------------------------------------------
def _robust_plane(P: np.ndarray, iters: int = 3, trim_pct: float = 85.0):
    """Fit a plane, trimming orthogonal-distance outliers. Returns (center, Vt, sing)."""
    keep = P
    center = keep.mean(0)
    Vt = np.eye(3)
    sing = np.ones(3)
    for _ in range(iters):
        Q = keep - center
        _, sing, Vt = np.linalg.svd(Q, full_matrices=False)
        normal = Vt[2]
        d = np.abs(Q @ normal)
        thr = max(np.percentile(d, trim_pct), 1e-9)
        m = d <= thr
        if m.sum() < max(30, 0.3 * len(keep)):
            break
        keep = keep[m]
        center = keep.mean(0)
    Q = keep - center
    _, sing, Vt = np.linalg.svd(Q, full_matrices=False)
    return center, Vt, sing, keep


def _min_area_rectangle(P2: np.ndarray, n_ang: int = 90):
    """Robust min-area oriented rectangle of 2D points. Returns (center2, (a,b) sorted)."""
    best = None
    for k in range(n_ang):
        ang = (np.pi / 2.0) * k / n_ang
        ca, sa = np.cos(ang), np.sin(ang)
        x = P2[:, 0] * ca + P2[:, 1] * sa
        y = -P2[:, 0] * sa + P2[:, 1] * ca
        x2, x98 = np.percentile(x, 2), np.percentile(x, 98)
        y2, y98 = np.percentile(y, 2), np.percentile(y, 98)
        wx, wy = x98 - x2, y98 - y2
        area = wx * wy
        if best is None or area < best[0]:
            cx, cy = 0.5 * (x2 + x98), 0.5 * (y2 + y98)
            # rotate rectangle-frame center back into the P2 basis
            center2 = np.array([cx * ca - cy * sa, cx * sa + cy * ca])
            best = (area, center2, tuple(sorted((abs(wx), abs(wy)))))
    return best[1], best[2]


def build_opening(cluster: OpeningCluster, min_points: int = 150) -> RobustOpening | None:
    P = np.asarray(cluster.points, dtype=np.float64)
    P = P[np.isfinite(P).all(1)]
    if len(P) < min_points:
        return None
    center0, Vt, sing, kept = _robust_plane(P)
    e1, e2, e3 = Vt[0], Vt[1], Vt[2]
    s = sing / (sing[0] + 1e-12)
    # planarity: 3rd axis small; linearity: 2nd axis small
    planarity = float(s[1] - s[2])
    linearity = float(1.0 - s[1])
    normal_reliable = (s[2] < 0.30) and (s[1] > 0.12)   # planar patch, not a line
    Q = kept - center0
    P2 = np.stack([Q @ e1, Q @ e2], axis=1)
    center2, extents = _min_area_rectangle(P2)
    center = center0 + center2[0] * e1 + center2[1] * e2
    size_reliable = normal_reliable and extents[0] > 1e-4
    return RobustOpening(
        category=cluster.category, center=center, normal_line=e3 / (np.linalg.norm(e3) + 1e-12),
        extents=extents, point_count=len(P), planarity=planarity, linearity=linearity,
        normal_reliable=normal_reliable, size_reliable=size_reliable,
        source_index=cluster.source_index if cluster.source_index is not None else -1,
    )


# --------------------------------------------------------------------------------------
# 2. Similarity estimation (Umeyama with scale; PURE R, explicit s)
# --------------------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray, allow_reflection: bool = False):
    """Minimize ||dst - (s R src + t)||. Returns (s, R, t). R proper unless allow_reflection."""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / (var_s + 1e-12))
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def _rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimal rotation mapping unit u -> unit v."""
    u = u / (np.linalg.norm(u) + 1e-12); v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.dot(u, v))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        # 180deg about any axis perpendicular to u
        a = np.cross(u, [1, 0, 0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(u, [0, 1, 0])
        a /= np.linalg.norm(a)
        return 2 * np.outer(a, a) - np.eye(3)
    ax = np.cross(u, v)   # UNNORMALIZED: the (I + K + K@K/(1+c)) formula needs the raw cross
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def _axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# --------------------------------------------------------------------------------------
# 3. Scene thresholds (fixed, from B; never assignment-dependent)
# --------------------------------------------------------------------------------------
def _median_pairwise(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return 1.0
    d = [np.linalg.norm(centers[i] - centers[j])
         for i, j in itertools.combinations(range(len(centers)), 2)]
    d = [x for x in d if x > 1e-9]
    return float(np.median(d)) if d else 1.0


def _center_rank_baseline(centers: np.ndarray, tol: float):
    """rank (1 = collinear) + baseline (point, unit axis) via PCA of matched centers."""
    p = centers.mean(0)
    Q = centers - p
    if len(centers) < 2:
        return 0, p, np.array([1.0, 0, 0])
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    rank = int(np.sum(S > tol))
    return rank, p, Vt[0]


def _matched_wall_normal(openings, idxs, dot_thr=0.9):
    """Return (normal, ok). ok only when the RELIABLE matched normals form ONE dominant
    unoriented wall-normal cluster (else there is no single shared wall to reason about)."""
    ns = [openings[i].normal_line for i in idxs if openings[i].normal_reliable]
    if not ns:
        return np.array([0.0, 0.0, 1.0]), False
    clusters = []                                   # [accum_vector, count]
    for n in ns:
        placed = False
        for cl in clusters:
            ref = cl[0] / (np.linalg.norm(cl[0]) + 1e-12)
            if abs(float(np.dot(n, ref))) >= dot_thr:
                cl[0] += n if np.dot(n, ref) >= 0 else -n
                cl[1] += 1; placed = True; break
        if not placed:
            clusters.append([n.copy(), 1])
    clusters.sort(key=lambda c: -c[1])
    dominant = len(clusters) == 1 or clusters[0][1] > clusters[1][1]
    if not dominant:
        return np.array([0.0, 0.0, 1.0]), False
    v = clusters[0][0]
    return v / (np.linalg.norm(v) + 1e-12), True


def _baseline_roll(s, R, t, p, u):
    """180deg rotation about baseline p+lambda*u (fixes matched centers, flips wall side)."""
    u = u / (np.linalg.norm(u) + 1e-12)
    Q = 2 * np.outer(u, u) - np.eye(3)     # rotation by pi about axis u
    return s, Q @ R, Q @ t + (np.eye(3) - Q) @ p


def _side_dominance(P, wall_p, wall_n, crop_centers, crop_pad):
    """Signed-occupancy dominance D in [-1,1] of a cloud vs a wall, LOCAL to the openings.
    Voxel-balanced + near-wall margin trimmed. Returns (D, coverage, n_voxels)."""
    P = np.asarray(P, np.float64)
    P = P[np.isfinite(P).all(1)]
    if len(P) == 0:
        return 0.0, 0.0, 0
    d = (P - wall_p) @ wall_n
    inplane = (P - wall_p) - np.outer(d, wall_n)
    # crop to a local tangential neighborhood of the matched openings
    keep = np.zeros(len(P), bool)
    for c in crop_centers:
        cin = (c - wall_p) - np.dot(c - wall_p, wall_n) * wall_n
        keep |= np.linalg.norm(inplane - cin, axis=1) <= crop_pad
    P, d = P[keep], d[keep]
    if len(P) < 50:
        return 0.0, 0.0, 0
    H = max(np.percentile(np.abs(d), 90), 1e-3)
    vox = max(0.03 * H, 1e-6)
    keys = np.floor(P / vox).astype(np.int64)
    order = np.lexsort(keys.T)                       # order-independent voxel aggregation
    keys_s, d_s = keys[order], d[order]
    starts = np.concatenate([[0], np.where(np.any(np.diff(keys_s, axis=0) != 0, axis=1))[0] + 1])
    counts = np.diff(np.concatenate([starts, [len(d_s)]]))
    d = np.add.reduceat(d_s, starts) / counts        # mean signed distance per voxel
    delta = 0.05 * H
    pp = float(np.mean(d > delta)); pm = float(np.mean(d < -delta))
    C = pp + pm
    D = (pp - pm) / C if C > 1e-6 else 0.0
    return D, C, len(d)


def _is_collinear(centers, ratio=0.1):
    if len(centers) < 3:
        return True
    S = np.linalg.svd(centers - centers.mean(0), compute_uv=False)
    return S[0] < 1e-9 or (S[1] / S[0]) < ratio


def _refit_scale_t(src, dst, R):
    """Refit ONLY scale + translation with R held fixed (rank-1 centers can't fix rotation)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a = (src - mu_s) @ R.T
    b = dst - mu_d
    s = float(np.sum(a * b) / (np.sum(a * a) + 1e-12))
    return s, mu_d - s * (R @ mu_s)


# --------------------------------------------------------------------------------------
# 4. Consensus scoring (partial bipartite assignment; sign-invariant gates)
# --------------------------------------------------------------------------------------
def _score(A: list[RobustOpening], B: list[RobustOpening], s, R, t,
           T_center, T_angle=np.deg2rad(25), T_logsize=0.45, hard_normal=True):
    nA, nB = len(A), len(B)
    cA = transform_points(np.array([o.center for o in A]), s, R, t)
    BIG = 1e6
    cost = np.full((nA, nB), BIG)
    for i in range(nA):
        for j in range(nB):
            if A[i].category != B[j].category:
                continue
            cres = np.linalg.norm(cA[i] - B[j].center) / T_center
            if cres >= 1.0:                      # HARD gate: center must be close
                continue
            if hard_normal and A[i].normal_reliable and B[j].normal_reliable:
                dot = abs(float(np.dot(R @ A[i].normal_line, B[j].normal_line)))
                if np.arccos(np.clip(dot, 0, 1)) / T_angle >= 1.0:   # HARD gate: normal
                    continue
            # cost = center residual (primary) + tiny tiebreakers (size/normal, NOT gates)
            tie = 0.0
            if A[i].normal_reliable and B[j].normal_reliable:
                dot = abs(float(np.dot(R @ A[i].normal_line, B[j].normal_line)))
                tie += 0.05 * (np.arccos(np.clip(dot, 0, 1)) / T_angle)
            if A[i].size_reliable and B[j].size_reliable:
                la = np.log(np.array(A[i].extents) * s + 1e-9)
                lb = np.log(np.array(B[j].extents) + 1e-9)
                tie += 0.02 * min(float(np.max(np.abs(la - lb))) / T_logsize, 2.0)
            cost[i, j] = cres + tie
    # A gated pair's cost = center residual (<1) + small ties (<~0.25); forbid anything at/above
    # NOMATCH so the solver's acceptance and our extraction use ONE consistent threshold.
    NOMATCH = 1.3
    cost[cost >= NOMATCH] = BIG
    size = nA + nB
    C = np.full((size, size), BIG)              # everything forbidden by default...
    C[:nA, :nB] = cost
    for i in range(nA):
        C[i, nB + i] = NOMATCH                   # ...except each row's own no-match option
    for j in range(nB):
        C[nA + j, j] = NOMATCH                   # ...each col's own no-match option
    C[nA:, nB:] = 0.0                            # dummy<->dummy pairings are free
    row, col = linear_sum_assignment(C)
    matches = []
    for r, c in zip(row, col):
        if r < nA and c < nB and cost[r, c] < BIG:   # any solver-selected gated pair = inlier
            matches.append((r, c, float(cost[r, c])))
    return matches


# --------------------------------------------------------------------------------------
# 5. Main matcher
# --------------------------------------------------------------------------------------
def _triangle_hypotheses(A, B, tau_scene, T_center, log_scale_tol=0.35):
    cA = np.array([o.center for o in A]); cB = np.array([o.center for o in B])
    hyps = []
    for ia in itertools.combinations(range(len(A)), 3):
        if _triangle_bad(cA[list(ia)]):
            continue
        for ib in itertools.permutations(range(len(B)), 3):
            if any(A[ia[k]].category != B[ib[k]].category for k in range(3)):
                continue
            if _triangle_bad(cB[list(ib)]):
                continue
            # log-scale consistency of the 3 edges
            good = True
            ls = []
            for (p, q) in [(0, 1), (0, 2), (1, 2)]:
                da = np.linalg.norm(cA[ia[p]] - cA[ia[q]])
                db = np.linalg.norm(cB[ib[p]] - cB[ib[q]])
                if da < 1e-6 or db < 1e-6:
                    good = False; break
                ls.append(np.log(db / da))
            if not good or (max(ls) - min(ls)) > log_scale_tol:
                continue
            s, R, t = umeyama(cA[list(ia)], cB[list(ib)])
            if s <= 1e-6:
                continue
            hyps.append((s, R, t))
    return hyps


def _two_opening_hypotheses(A, B):
    """2 centers + normal-lines: baseline fixes scale + 2 DoF, roll optimized over normals."""
    cA = np.array([o.center for o in A]); cB = np.array([o.center for o in B])
    hyps = []
    for ia in itertools.combinations(range(len(A)), 2):
        for ib in itertools.permutations(range(len(B)), 2):
            if any(A[ia[k]].category != B[ib[k]].category for k in range(2)):
                continue
            va = cA[ia[1]] - cA[ia[0]]; vb = cB[ib[1]] - cB[ib[0]]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-6 or nb < 1e-6:
                continue
            s = nb / na
            R0 = _rotation_between(va / na, vb / nb)
            axis = vb / nb
            # sample roll; objective = sum over reliable normals of (1 - |R n_a . n_b|)
            best_thetas = []
            vals = []
            thetas = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            for th in thetas:
                R = _axis_rotation(axis, th) @ R0
                v = 0.0; cnt = 0
                for k in range(2):
                    oa, ob = A[ia[k]], B[ib[k]]
                    if oa.normal_reliable and ob.normal_reliable:
                        v += 1 - abs(float(np.dot(R @ oa.normal_line, ob.normal_line))); cnt += 1
                vals.append(v if cnt else 0.0)
            vals = np.array(vals)
            # local minima
            for m in range(len(thetas)):
                if vals[m] <= vals[(m - 1) % len(thetas)] and vals[m] <= vals[(m + 1) % len(thetas)]:
                    R = _axis_rotation(axis, thetas[m]) @ R0
                    t = cB[ib[0]] - s * (R @ cA[ia[0]])
                    hyps.append((s, R, t))
    return hyps


def _triangle_bad(pts, min_height_ratio=0.12):
    e = [np.linalg.norm(pts[i] - pts[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    longest = max(e)
    if longest < 1e-6:
        return True
    # area via cross product -> altitude = 2*area/longest
    area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
    altitude = 2 * area / longest
    return (altitude / longest) < min_height_ratio


def match_openings(clusters_A, clusters_B, tau_scene=0.18, size_floor_frac=0.5,
                   points_A=None, points_B=None, side_policy="off") -> AlignResult:
    """side_policy: "off" | "prefer_opposite" | "require_opposite".
    When the matched openings are on ONE wall (rank-1 roll ambiguity) and side_policy is
    enabled and points_A/points_B are given, disambiguate the 180deg roll by requiring the
    two clouds' bulk to sit on OPPOSITE sides of the shared wall (non-overlap)."""
    if side_policy not in ("off", "prefer_opposite", "require_opposite"):
        side_policy = "off"
    A = [o for o in (build_opening(c) for c in clusters_A) if o is not None]
    B = [o for o in (build_opening(c) for c in clusters_B) if o is not None]
    if len(A) < 2 or len(B) < 2:
        return AlignResult(1.0, np.eye(3), np.zeros(3), [], "NO_SOLUTION",
                           reason=f"too few valid openings (A={len(A)}, B={len(B)})")

    cB = np.array([o.center for o in B])
    L_B = _median_pairwise(cB)
    med_ext_B = float(np.median([np.mean(o.extents) for o in B if o.size_reliable] or [L_B]))
    T_center = max(tau_scene * L_B, size_floor_frac * med_ext_B, 1e-4)

    # Always combine BOTH hypothesis sources, not "2-opening only as a
    # fallback when triangles are empty" - confirmed a real bug on real
    # data (`server`): whenever ANY triangle hypothesis existed at all,
    # even one built from 3 points spanning 2 UNRELATED physical walls
    # (a triangle just needs 3 non-collinear points, not 3 points that
    # actually belong together), `_two_opening_hypotheses` never ran at
    # all - so the correct 2-opening pair for the genuinely-matching wall
    # was silently never even proposed as a candidate, regardless of how
    # good its score would have been. The downstream ranking (most
    # inliers first, then lowest residual) is sound; the bug was that the
    # right candidate could be entirely absent from the pool it ranks.
    hyps = _triangle_hypotheses(A, B, tau_scene, T_center) + _two_opening_hypotheses(A, B)
    if not hyps:
        return AlignResult(1.0, np.eye(3), np.zeros(3), [], "NO_SOLUTION",
                           reason="no valid transform hypothesis")

    scored = []
    for (s, R, t) in hyps:
        m = _score(A, B, s, R, t, T_center)
        res = np.mean([x[2] for x in m]) if m else 1.0
        scored.append((len(m), -res, s, R, t, m))
    # Rank by TOTAL cost (mean cost * inlier count), not raw inlier count -
    # confirmed a real bug on real data (`server`): a 3-inlier hypothesis
    # that padded 2 genuinely excellent pairs (cost ~0.01 each) with 1
    # near-garbage pair (cost 0.99, right at the NOMATCH=1.3 gate) beat a
    # cleaner 2-inlier hypothesis whose BOTH pairs were excellent (cost
    # ~0.006 each) - pure inlier-count ranking rewards padding a good
    # match with a barely-passing one instead of preferring the tighter,
    # more trustworthy correspondence. `z[1]*z[0]` = `-res*n_in` = the
    # negated total cost, so sorting it descending = ascending total cost
    # (lower/better first); `z[0]` (inlier count) is the tie-break for
    # equal-total-cost hypotheses, preferring the one with more support.
    scored.sort(key=lambda z: (z[1] * z[0], z[0]), reverse=True)
    best = scored[0]
    n_in, _, s, R, t, matches = best

    # second-best = best hypothesis with a DIFFERENT correspondence set (duplicate seeds that
    # yield the same match set are NOT competing alternatives).
    best_set = frozenset((i, j) for i, j, _ in matches)
    second_inliers = 0
    for entry in scored[1:]:
        if frozenset((i, j) for i, j, _ in entry[5]) != best_set:
            second_inliers = entry[0]
            break

    def _refit(matches, s, R, t):
        for _ in range(3):
            if len(matches) < 3:
                break
            srcs = np.array([A[i].center for i, _, _ in matches])
            dsts = np.array([B[j].center for _, j, _ in matches])
            if _is_collinear(dsts):
                s, t = _refit_scale_t(srcs, dsts, R)   # keep R: rank-1 can't fix roll
            else:
                s, R, t = umeyama(srcs, dsts)
            matches = _score(A, B, s, R, t, T_center)
        return matches, s, R, t

    matches, s, R, t = _refit(matches, s, R, t)
    n_in = len(matches)

    # Best-effort fallback for under-determined scenes: drop the hard normal gate so we
    # still surface a viewable guess (flagged LOW_CONFIDENCE), instead of NO_SOLUTION.
    if n_in < 3:
        be = []
        for (s2, R2, t2) in hyps:
            m2 = _score(A, B, s2, R2, t2, T_center, hard_normal=False)
            res2 = np.mean([x[2] for x in m2]) if m2 else float("inf")
            be.append((len(m2), -res2, s2, R2, t2, m2))
        # same total-cost-first ranking as the main selection above (not
        # raw count) - this fallback loop is the OTHER place the same bug
        # showed up: it re-ran on real data (`server`) and re-introduced
        # the count-padded-with-garbage winner even after fixing the main
        # selection, because this block picks independently via its own
        # sort.
        be.sort(key=lambda z: (z[1] * z[0], z[0]), reverse=True)
        if be and be[0][0] > n_in:
            _, _, s, R, t, matches = be[0]
            matches, s, R, t = _refit(matches, s, R, t)   # refit proper fit on FINAL matches
            n_in = len(matches)
    mean_res = float(np.mean([x[2] for x in matches])) if matches else float("nan")

    # ---- opposite-side (non-overlap) disambiguation of a rank-1 single-wall roll ----
    side_decided = False
    side_ambiguous = False
    side_score = side_dom_A = side_dom_B = float("nan")
    if side_policy in ("prefer_opposite", "require_opposite") and n_in >= 2:
        pA = np.asarray(points_A, float) if points_A is not None else None
        pB = np.asarray(points_B, float) if points_B is not None else None
        pts_ok = (pA is not None and pB is not None and pA.ndim == 2 and pA.shape[1] == 3
                  and pB.ndim == 2 and pB.shape[1] == 3 and len(pA) > 0 and len(pB) > 0)
        idxB = [j for _, j, _ in matches]
        dsts = np.array([B[j].center for j in idxB])
        wall_n, wall_ok = _matched_wall_normal(B, idxB)
        Sd = np.linalg.svd(dsts - dsts.mean(0), compute_uv=False)
        u = np.linalg.svd(dsts - dsts.mean(0), full_matrices=False)[2][0]
        p_base = dsts.mean(0)
        span_ok = Sd[0] > 0.1 * T_center                       # baseline must actually span
        rank1 = _is_collinear(dsts)
        perp = abs(float(np.dot(u, wall_n))) <= 0.20           # keep post-roll normal within the 25deg gate
        if pts_ok and wall_ok and rank1 and span_ok and perp:
            wall_p = dsts.mean(0)
            pad = 2.5 * (float(np.median([np.mean(B[j].extents) for j in idxB])) or T_center)
            DB, CB, nB = _side_dominance(pB, wall_p, wall_n, dsts, pad)
            DA0, CA0, nA0 = _side_dominance(transform_points(pA, s, R, t), wall_p, wall_n, dsts, pad)
            s2, R2, t2 = _baseline_roll(s, R, t, p_base, u)
            m2 = _score(A, B, s2, R2, t2, T_center)            # rolled candidate must keep consensus
            DA1, CA1, nA1 = _side_dominance(transform_points(pA, s2, R2, t2), wall_p, wall_n, dsts, pad)

            def _reliable(D, C, nv):
                return C >= 0.5 and abs(D) >= 0.6 and nv >= 200

            if _reliable(DB, CB, nB) and _reliable(DA0, CA0, nA0) and _reliable(DA1, CA1, nA1):
                S0, S1 = -DA0 * DB, -DA1 * DB                  # >0 = opposite sides (good)
                if S1 > S0 and len(m2) >= n_in:                # only flip if it stays consistent
                    s, R, t, matches = s2, R2, t2, m2
                    n_in = len(matches)
                    mean_res = float(np.mean([x[2] for x in matches])) if matches else float("nan")
                    DA0, S0 = DA1, S1
                    side_decided = True
                side_score, side_dom_A, side_dom_B = S0, DA0, DB
            else:
                side_ambiguous = True
        else:
            side_ambiguous = True
        # require_opposite: reject unless we have RELIABLE positive opposite-side evidence
        if side_policy == "require_opposite" and not (np.isfinite(side_score) and side_score >= 0.36):
            return AlignResult(s=s, R=R, t=t, matches=matches, status="NO_SOLUTION",
                               reason="require_opposite: no reliable opposite-side evidence",
                               n_inliers=n_in, mean_residual=mean_res,
                               side_score=side_score, side_dominance_A=side_dom_A,
                               side_dominance_B=side_dom_B, side_decided_roll=side_decided)

    # reflection / chirality diagnostic: compare PROPER vs IMPROPER fit on the SAME final matches
    reflection_preferred = False
    chirality_observable = False
    if n_in >= 3:
        srcs = np.array([A[i].center for i, _, _ in matches])
        dsts = np.array([B[j].center for _, j, _ in matches])
        s_r, R_r, t_r = umeyama(srcs, dsts, allow_reflection=True)
        err_proper = np.mean(np.linalg.norm(transform_points(srcs, s, R, t) - dsts, axis=1))
        err_impro = np.mean(np.linalg.norm(transform_points(srcs, s_r, R_r, t_r) - dsts, axis=1))
        centered = srcs - srcs.mean(0)
        rank = np.linalg.matrix_rank(centered, tol=1e-3 * max(np.linalg.norm(centered), 1e-9))
        chirality_observable = rank >= 3 and abs(err_proper - err_impro) > 0.1 * T_center
        # only a genuine reflection if the better fit is actually improper (det < 0)
        reflection_preferred = (chirality_observable and np.linalg.det(R_r) < 0
                                and err_impro < err_proper - 1e-9)

    # wall-diversity of the matched openings (same-wall openings can't fix rotation)
    def _num_wall_dirs(ops, idxs, dot_thr=0.9):
        dirs = []
        for i in idxs:
            if not ops[i].normal_reliable:      # only trust reliable normals as wall evidence
                continue
            n = ops[i].normal_line
            if not any(abs(float(np.dot(n, d))) >= dot_thr for d in dirs):
                dirs.append(n)
        return len(dirs)

    idxA = [i for i, _, _ in matches]
    walls_A = _num_wall_dirs(A, idxA) if idxA else 0
    idxB = [j for _, j, _ in matches]
    walls_B = _num_wall_dirs(B, idxB) if idxB else 0
    single_wall = min(walls_A, walls_B) < 2      # matched openings span only ONE wall
    decisive = (n_in - second_inliers) >= 1

    # Discrete candidate set (only meaningful/needed for the rank-1 single-wall case):
    # distinct-correspondence hypotheses tied at the top inlier tier + baseline-roll mates.
    def _collect_candidates():
        cands, seen = [], []
        top = max((e[0] for e in scored), default=0)
        for e in scored:
            if e[0] < top:
                break
            _, _, sc, Rc, tc, mc = e
            mc2, sc, Rc, tc = _refit(mc, sc, Rc, tc)
            for (ss, RR, tt) in ((sc, Rc, tc),):
                if not any(np.allclose(RR, r2, atol=1e-3) for r2 in seen):
                    seen.append(RR); cands.append((float(ss), RR, tt))
        return cands

    if n_in >= 3 and walls_A >= 2 and walls_B >= 2 and decisive and not reflection_preferred:
        status, reason = "CONFIDENT", "OK"
    elif n_in >= 1 and single_wall:
        # Rank-1: matched openings lie on ONE wall. Roll about that wall -- and with a
        # horizontal baseline the coupled wall-side AND vertical orientation -- are NOT
        # observable from openings alone. Do NOT present (s,R,t) as authoritative.
        status = "AMBIGUOUS"
        provisional = True
        candidates = _collect_candidates()
        reason = ("rank-1: matched openings lie on ONE wall, so roll, wall side and vertical "
                  "orientation are unobservable from openings alone (%d discrete candidate(s)). "
                  "Fix: segment a corresponding opening on a NON-PARALLEL wall in BOTH scenes, "
                  "or supply externally-oriented gravity. Transform is PROVISIONAL (preview only)."
                  % max(len(candidates), 1))
        return AlignResult(s=s, R=R, t=t, matches=matches, status=status, reason=reason,
                           n_inliers=n_in, mean_residual=mean_res,
                           reflection_preferred=reflection_preferred,
                           chirality_observable=chirality_observable,
                           second_best_inliers=second_inliers, provisional=provisional,
                           candidates=candidates,
                           side_decided_roll=side_decided, side_score=side_score,
                           side_dominance_A=side_dom_A, side_dominance_B=side_dom_B)
    elif n_in >= 1:
        status = "LOW_CONFIDENCE"
        bits = []
        if n_in < 3:
            bits.append(f"only {n_in} matched opening(s) (<3)")
        if not decisive:
            bits.append("competing hypotheses near-tied")
        if reflection_preferred:
            bits.append("mirror fits better / chirality unobservable")
        reason = "; ".join(bits) or "weak evidence"
    else:
        status, reason = "NO_SOLUTION", "no consensus inliers"

    return AlignResult(s=s, R=R, t=t, matches=matches, status=status, reason=reason,
                       n_inliers=n_in, mean_residual=mean_res,
                       reflection_preferred=reflection_preferred,
                       chirality_observable=chirality_observable,
                       second_best_inliers=second_inliers,
                       side_decided_roll=side_decided, side_score=side_score,
                       side_dominance_A=side_dom_A, side_dominance_B=side_dom_B)


# --------------------------------------------------------------------------------------
# 6. Loader: OpeningClusters from a CloudCompare manual-segmentation .ply
# --------------------------------------------------------------------------------------
def openings_from_manual_segmentation(path, window_keys=("cua_so",),
                                      door_keys=("cua",)) -> list[OpeningCluster]:
    """Load CloudCompare manual-segmentation openings. Matching is CASE-INSENSITIVE and
    substring-based so it tolerates any naming: 'scalar_cua_ra_vao_1', 'scalar_Cua_ra_vao_2'
    (capital), 'scalar_cua_cuoi_h', etc. A field is a WINDOW if its (lowercased) name
    contains a window key, else a DOOR if it contains a door key. Non-opening scalar
    fields (e.g. 'scalar_Original_cloud_index') are ignored.

    Also treats a field literally named 'scalar_Constant' as a door: CloudCompare's
    "Edit > Scalar Fields > Add constant SF" tool names a new field that way unless the
    person renames it - confirmed on real data (`Q1_outdoor`, `Q2_outdoor`) that a door
    selection was left with this default name, silently dropping the whole opening
    without this case (same fix as `manual_segmentation.UNRENAMED_DEFAULT_FIELD`)."""
    from plyfile import PlyData
    v = PlyData.read(str(Path(path)))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    out = []
    for name in v.dtype.names:
        low = name.lower()
        if low == "scalar_constant":
            cat = "door"
        elif not low.startswith("scalar") or "cua" not in low:
            continue
        elif "cloud" in low or "index" in low or "original" in low:
            continue
        else:
            cat = "window" if any(k in low for k in window_keys) else (
                "door" if any(k in low for k in door_keys) else None)
        if cat is None:
            continue
        mask = ~np.isnan(v[name])
        if mask.sum() == 0:
            continue
        out.append(OpeningCluster(category=cat, points=xyz[mask], source_index=len(out)))
    return out
