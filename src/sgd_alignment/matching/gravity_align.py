"""Gravity-locked, camera-verified alignment of two adjacent indoor spaces.

Phase-1 method (practical breakthrough for the single-shared-wall degeneracy):

The opening-only matcher is rank-1 when the matched openings lie on ONE wall: the
roll about the wall -- and, for a horizontal door baseline, the coupled wall-side and
vertical (up/down) parity -- are unobservable from openings alone. We break the
degeneracy with two RELIABLE external cues carried by the DA3 reconstruction itself:

  * GRAVITY from cameras: each frame's world-up is -R[1,:] (image y points down).
    Averaged over frames this is an extremely stable vertical (empirically cos>0.99),
    far more reliable than plane-based `estimate_up_vector_manhattan`.
  * CAMERA CENTERS: C = -R^T t. Cameras that filmed space A lie INSIDE A, so after a
    correct alignment A's cameras and B's cameras sit on OPPOSITE sides of the shared
    wall. This resolves the wall-side / correspondence-swap ambiguity.

Algorithm: rotate each cloud so its gravity maps to +Z (upright), reducing the unknown
similarity to {scale, azimuth about Z, translation}. Solve that from the opening-center
correspondences (a well-conditioned 2-D similarity + vertical offset). Enumerate the few
correspondences (identity / swap / partial), keep gravity locked, and SELECT the winner by
camera-side opposition + opening-fit residual. Report an honest AMBIGUOUS when cues can't
separate the candidates.

`extrinsics` convention (verified against make_ply.py): world2cam, X_cam = R X_world + t,
R = EX[:3,:3], t = EX[:3,3]  ==>  center = -R^T t, up = -R[1,:], forward = R[2,:].
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .robust_align import (
    build_opening, _matched_wall_normal, transform_points, _median_pairwise,
    _axis_rotation,
)


# --------------------------------------------------------------------------------------
# Camera evidence from a DA3 results.npz
# --------------------------------------------------------------------------------------
@dataclass
class CameraEvidence:
    centers: np.ndarray     # (N,3) camera centers in world/cloud frame
    up: np.ndarray          # (3,) robust mean world-up (gravity, points ceiling-ward)
    up_consistency: float   # median cos(per-frame up, mean up); ~1 = very stable
    forwards: np.ndarray    # (N,3) viewing directions


def camera_evidence(npz_path: str) -> CameraEvidence:
    Z = np.load(npz_path)
    EX = Z["extrinsics"].astype(np.float64)          # (N,3,4) world2cam
    R = EX[:, :3, :3]
    t = EX[:, :3, 3]
    centers = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)
    up_per = -R[:, 1, :]
    up_per /= np.linalg.norm(up_per, axis=1, keepdims=True) + 1e-12
    up = up_per.mean(0)
    up /= np.linalg.norm(up) + 1e-12
    consistency = float(np.median(up_per @ up))
    fwd = R[:, 2, :]
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12
    return CameraEvidence(centers=centers, up=up, up_consistency=consistency, forwards=fwd)


# --------------------------------------------------------------------------------------
# Gravity-locked similarity from opening-center correspondences
# --------------------------------------------------------------------------------------
def _basis_from_up(up: np.ndarray) -> np.ndarray:
    """Orthonormal rows [e_x, e_y, up_hat]; U @ x expresses x with up mapped to +Z."""
    w = up / (np.linalg.norm(up) + 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, w); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(w, e1)
    return np.stack([e1, e2, w], axis=0)   # (3,3), rows orthonormal


def gravity_locked_similarity(srcs: np.ndarray, dsts: np.ndarray,
                              up_src: np.ndarray, up_dst: np.ndarray):
    """Best similarity (s,R,t) mapping srcs->dsts with R @ up_src == up_dst (upright,
    no flip). Rotation reduces to an azimuth about gravity. Needs >=2 correspondences
    with a non-degenerate HORIZONTAL spread. Returns (s, R, t, horiz_span)."""
    srcs = np.asarray(srcs, float); dsts = np.asarray(dsts, float)
    Ua = _basis_from_up(up_src)     # maps src up -> +Z
    Ub = _basis_from_up(up_dst)     # maps dst up -> +Z
    a = srcs @ Ua.T                 # (N,3) in canonical A (up = +Z)
    b = dsts @ Ub.T                 # (N,3) in canonical B (up = +Z)
    axy, az = a[:, :2], a[:, 2]
    bxy, bz = b[:, :2], b[:, 2]
    ca, cb = axy.mean(0), bxy.mean(0)
    A0 = axy - ca; B0 = bxy - cb
    horiz_span = float(np.sqrt((A0 ** 2).sum(1).mean()))
    # 2-D similarity (rotation+scale about gravity) via complex least squares
    za = A0[:, 0] + 1j * A0[:, 1]
    zb = B0[:, 0] + 1j * B0[:, 1]
    denom = float((za.conj() * za).real.sum()) + 1e-12
    w = (za.conj() * zb).sum() / denom          # complex: |w|=scale, arg(w)=azimuth
    s = float(abs(w))
    phi = float(np.angle(w))
    c, sn = np.cos(phi), np.sin(phi)
    Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    # vertical scale is the same uniform s; solve z translation
    tz = float(np.mean(bz - s * az))
    txy = cb - s * (Rz[:2, :2] @ ca)
    # compose back to world: x_dst = Ub^T ( s Rz (Ua x_src) + [txy,tz] )
    R = Ub.T @ Rz @ Ua
    t = Ub.T @ np.array([txy[0], txy[1], tz])
    return s, R, t, horiz_span


def gravity_locked_single(oA, oB, up_src, up_dst):
    """ONE-opening alignment. Gravity fixes vertical; the door NORMAL (horizontal on a
    vertical wall) fixes the azimuth up to its sign; the door SIZE fixes scale; the
    center fixes translation. Returns the TWO azimuth candidates (normal is an unoriented
    line) -- camera-side picks which one is physically correct. List of (s, R, t)."""
    s = float(np.mean(oB.extents) / (np.mean(oA.extents) + 1e-9))
    Ua = _basis_from_up(up_src); Ub = _basis_from_up(up_dst)
    nA = Ua @ oA.normal_line; nB = Ub @ oB.normal_line     # to canonical (up=+Z)
    aA = float(np.arctan2(nA[1], nA[0]))                    # horizontal heading of normal
    aB = float(np.arctan2(nB[1], nB[0]))
    out = []
    for flip in (0.0, np.pi):                              # unoriented normal -> 2 azimuths
        phi = aB - aA + flip
        c, sn = np.cos(phi), np.sin(phi)
        Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        R = Ub.T @ Rz @ Ua
        t = np.asarray(oB.center, float) - s * (R @ np.asarray(oA.center, float))
        out.append((s, R, t))
    return out


# --------------------------------------------------------------------------------------
# Camera-side (non-overlap) score
# --------------------------------------------------------------------------------------
def _signed_side(points, wall_p, wall_n):
    d = (np.asarray(points, float) - wall_p) @ wall_n
    return d


def camera_side_score(cams_A_t, cams_B, wall_p, wall_n):
    """+1 if A-cameras and B-cameras sit on OPPOSITE sides of the wall (good), -1 same.
    Uses robust median side and the fraction of cameras on the dominant side."""
    dA = _signed_side(cams_A_t, wall_p, wall_n)
    dB = _signed_side(cams_B, wall_p, wall_n)
    mA, mB = np.median(dA), np.median(dB)
    # dominance in [0,1]: how one-sided each camera set is
    domA = abs(np.mean(np.sign(dA - 0)))  # ~1 if all on one side
    domB = abs(np.mean(np.sign(dB - 0)))
    opposite = (np.sign(mA) * np.sign(mB) < 0)
    return (1.0 if opposite else -1.0), float(min(domA, domB)), float(mA), float(mB)


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------
def _wall_footprint(P, up):
    """2-D footprint (robust bbox) of a cluster in its own wall-plane basis + plane normal
    and mean depth along that normal. Returns (n, ctr, base_u, base_v, bbox2d)."""
    P = np.asarray(P, float)
    ctr = P.mean(0)
    _, _, Vt = np.linalg.svd(P - ctr, full_matrices=False)
    n = Vt[2] / (np.linalg.norm(Vt[2]) + 1e-12)
    up = up / (np.linalg.norm(up) + 1e-12)
    u = np.cross(up, n); nu = np.linalg.norm(u)
    u = (u / nu) if nu > 1e-6 else Vt[0]
    v = np.cross(n, u)
    Q = P - ctr
    a, b = Q @ u, Q @ v
    box = (np.percentile(a, 2), np.percentile(a, 98), np.percentile(b, 2), np.percentile(b, 98))
    return n, ctr, u, v, box


def _dedupe_openings(clusters, up, iom_thr=0.55, ang_thr_deg=15.0):
    """Merge detections of the SAME physical aperture (e.g. a glass door labelled both
    'door' and 'window'). Decisive test = wall-plane FOOTPRINT overlap (intersection over
    smaller area), plus near-parallel normals and near-equal depth. Keeps the larger, more
    complete footprint. Returns a reduced cluster list (order preserved)."""
    info = [_wall_footprint(c.points, up) for c in clusters]
    keep = [True] * len(clusters)
    for i in range(len(clusters)):
        if not keep[i]:
            continue
        ni, ci, _, _, bi = info[i]
        for j in range(i + 1, len(clusters)):
            if not keep[j]:
                continue
            nj, cj, _, _, bj = info[j]
            if np.degrees(np.arccos(np.clip(abs(float(ni @ nj)), 0, 1))) > ang_thr_deg:
                continue
            nm = ni + (nj if float(ni @ nj) >= 0 else -nj)
            nm /= np.linalg.norm(nm) + 1e-12
            um = np.cross(up / (np.linalg.norm(up) + 1e-12), nm); nu = np.linalg.norm(um)
            um = (um / nu) if nu > 1e-6 else np.array([1.0, 0.0, 0.0])
            vm = np.cross(nm, um)
            def _box(P):
                Q = np.asarray(P, float); a = Q @ um; b = Q @ vm
                return (np.percentile(a, 2), np.percentile(a, 98), np.percentile(b, 2), np.percentile(b, 98))
            Bi, Bj = _box(clusters[i].points), _box(clusters[j].points)
            ix = max(0.0, min(Bi[1], Bj[1]) - max(Bi[0], Bj[0]))
            iy = max(0.0, min(Bi[3], Bj[3]) - max(Bi[2], Bj[2]))
            inter = ix * iy
            ai = (Bi[1] - Bi[0]) * (Bi[3] - Bi[2]); aj = (Bj[1] - Bj[0]) * (Bj[3] - Bj[2])
            iom = inter / (min(ai, aj) + 1e-9)
            depth_gap = abs(float((ci - cj) @ nm))
            if iom >= iom_thr and depth_gap < 0.30 * np.sqrt(min(ai, aj) + 1e-9):
                if ai >= aj:
                    keep[j] = False
                else:
                    keep[i] = False; break
    return [clusters[k] for k in range(len(clusters)) if keep[k]]


def _agg_wall_normal(ops, idxs, up):
    """Aggregate the matched openings' FULL plane normals into one wall normal
    (sign-invariant scatter matrix), qualified by verticality + cross-opening consensus.
    Over-merged auto-seg blobs fit the WALL plane, so this normal is a stable orientation
    cue -- more reliable than a slightly-off camera gravity. Returns (n_unit, reliable)."""
    up = up / (np.linalg.norm(up) + 1e-12)
    ns = []
    for i in idxs:
        o = ops[i]
        if not o.normal_reliable:
            continue
        n = o.normal_line / (np.linalg.norm(o.normal_line) + 1e-12)
        if abs(float(n @ up)) >= 0.34:          # wall must be ~vertical (<~20deg tilt)
            continue
        ns.append(n)
    if not ns:
        return None, False
    Q = np.zeros((3, 3))
    for n in ns:
        Q += np.outer(n, n)                     # sign-invariant (n and -n identical)
    w, V = np.linalg.eigh(Q)
    n_agg = V[:, -1] / (np.linalg.norm(V[:, -1]) + 1e-12)
    resid = [np.degrees(np.arccos(np.clip(abs(float(n @ n_agg)), 0, 1))) for n in ns]
    reliable = len(ns) >= 1 and float(np.median(resid)) < 8.0
    return n_agg, reliable


def _refine_normal_lock(A, B, pi, s, R, t, upA, upB, max_dev_deg=25.0):
    """Tighten the rotation so the shared wall's FULL normals are ANTIPARALLEL
    (R.nA = -nB) -- the physical fact that two spaces meet at ONE plane. The wall normal
    is the PRIMARY axis (trusted over a slightly-tilted camera gravity), gravity is the
    secondary axis (fixes rotation about the normal). Sign chosen to preserve the
    candidate's side (closest to the incoming R); camera-side confirms later. Falls back
    to the incoming R when the wall normal is unreliable or would move R by > max_dev_deg.
    Returns (R, t, applied)."""
    idxA = [i for i, _ in pi]; idxB = [j for _, j in pi]
    nA, okA = _agg_wall_normal(A, idxA, upA)
    nB, okB = _agg_wall_normal(B, idxB, upB)
    if not (okA and okB):
        return R, t, False
    uA = upA / (np.linalg.norm(upA) + 1e-12); uB = upB / (np.linalg.norm(upB) + 1e-12)
    gA = uA - (uA @ nA) * nA                     # gravity component perpendicular to normal
    gA /= np.linalg.norm(gA) + 1e-12
    FA = np.stack([nA, gA, np.cross(nA, gA)], axis=1)     # columns = A basis (normal primary)
    best = None
    for sn in (-1.0, 1.0):
        nBs = sn * nB
        gB = uB - (uB @ nBs) * nBs
        gB /= np.linalg.norm(gB) + 1e-12
        FB = np.stack([nBs, gB, np.cross(nBs, gB)], axis=1)
        Rn = FB @ FA.T
        U, _, Vt = np.linalg.svd(Rn)
        Rn = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
        ang = float(np.arccos(np.clip((np.trace(Rn @ R.T) - 1.0) / 2.0, -1.0, 1.0)))
        if best is None or ang < best[0]:
            best = (ang, Rn)
    ang, Rn = best
    if np.degrees(ang) > max_dev_deg:            # normal disagrees with centres -> distrust
        return R, t, False
    cA = np.array([A[i].center for i, _ in pi]); cB = np.array([B[j].center for _, j in pi])
    t_new = np.median(cB - s * (cA @ Rn.T), axis=0)   # robust translation refit
    return Rn, t_new, True


def _refine_roll_openings(A, B, pi, s, R, t, wall_n, max_deg=12.0, n_samples=97):
    """After the wall-normal lock, optimize the ROLL about the wall normal so the matched
    opening centres line up IN-PLANE. A rotation about the wall normal preserves wall
    coplanarity EXACTLY, so there is no trade-off. It closes the residual door-frame gap
    and removes the REDUCIBLE (azimuthal) part of a floor mismatch. It does NOT change the
    wall-vs-floor invariant angle within a scene, so any floor tilt from the two scenes
    having different wall-floor angles is irreducible here. Only applied when it improves
    the door alignment significantly (else a shallow, noisy minimum would over-rotate).
    Needs >=2 matched openings (one centre can't fix roll once t is free).
    Returns (R, t, roll_deg)."""
    if len(pi) < 2:
        return R, t, 0.0
    n = wall_n / (np.linalg.norm(wall_n) + 1e-12)
    cA = np.array([A[i].center for i, _ in pi], float)
    cB = np.array([B[j].center for _, j in pi], float)

    def _inplane_res(Rn):
        d = s * (cA @ Rn.T) - cB
        d = d - d.mean(0)                       # translation-invariant
        d = d - np.outer(d @ n, n)              # keep the in-wall component
        return float(np.mean(np.linalg.norm(d, axis=1)))

    res0 = _inplane_res(R)                       # door residual with NO roll
    best = (res0, 0.0, R)
    for th in np.linspace(-max_deg, max_deg, n_samples):
        Rn = _axis_rotation(n, np.radians(th)) @ R
        res = _inplane_res(Rn)
        if res < best[0]:
            best = (res, float(th), Rn)
    # Only commit the roll if it improves the door alignment SIGNIFICANTLY. When the doors
    # are already ~aligned at theta=0 (shallow minimum), a big roll would overfit tiny
    # centre noise and needlessly TILT THE FLOOR (rotation about the normal changes the
    # vertical). In that case stay at theta=0.
    if best[0] >= 0.5 * res0:
        Rn, roll = R, 0.0
    else:
        Rn, roll = best[2], best[1]
    tn = np.median(cB - s * (cA @ Rn.T), axis=0)
    return Rn, tn, roll


@dataclass
class GravityAlignResult:
    s: float
    R: np.ndarray
    t: np.ndarray
    matches: list           # [(iA,iB), ...]
    status: str             # CONFIDENT | AMBIGUOUS | NO_SOLUTION
    reason: str = ""
    up_consistency_A: float = float("nan")
    up_consistency_B: float = float("nan")
    grav_dot: float = float("nan")        # R@up_A . up_B  (should be ~+1)
    wall_normal_locked: bool = False      # rotation yaw tightened by antiparallel wall normal
    roll_refined_deg: float = 0.0         # roll about wall normal to align openings (+floors)
    cam_side: float = float("nan")        # +1 opposite (good)
    cam_dominance: float = float("nan")
    opening_residual: float = float("nan")
    candidates: list = field(default_factory=list)


def align_gravity_camera(clusters_A, clusters_B, cam_A: CameraEvidence,
                         cam_B: CameraEvidence, tau_scene: float = 0.18) -> GravityAlignResult:
    # collapse duplicate detections of the same physical aperture (door+window on glass)
    clusters_A = _dedupe_openings(list(clusters_A), cam_A.up)
    clusters_B = _dedupe_openings(list(clusters_B), cam_B.up)
    A, A_pts, B, B_pts = [], [], [], []
    for c in clusters_A:
        o = build_opening(c)
        if o is not None:
            A.append(o); A_pts.append(np.asarray(c.points, float))
    for c in clusters_B:
        o = build_opening(c)
        if o is not None:
            B.append(o); B_pts.append(np.asarray(c.points, float))
    if len(A) < 1 or len(B) < 1:
        return GravityAlignResult(1.0, np.eye(3), np.zeros(3), [], "NO_SOLUTION",
                                  reason=f"no valid openings (A={len(A)}, B={len(B)})")
    cB = np.array([o.center for o in B])
    L_ref = _median_pairwise(cB) if len(B) >= 2 else float(np.mean(B[0].extents))
    T_center = max(tau_scene * L_ref,
                   0.5 * float(np.median([np.mean(o.extents) for o in B])), 1e-4)
    upA, upB = cam_A.up, cam_B.up
    n_min = min(len(A), len(B))

    def _pair_metrics(pi, s, R, t):
        srcs = np.array([A[i].center for i, _ in pi])
        dsts = np.array([B[j].center for _, j in pi])
        res = float(np.mean(np.linalg.norm(transform_points(srcs, s, R, t) - dsts, axis=1)))
        szA = np.array([np.mean(A[i].extents) for i, _ in pi])
        szB = np.array([np.mean(B[j].extents) for _, j in pi])
        size_err = float(np.median(np.abs(s * szA - szB) / (szB + 1e-9)))
        ne = [1.0 - abs(float((R @ A[i].normal_line) @ B[j].normal_line))
              for i, j in pi if A[i].normal_reliable and B[j].normal_reliable]
        norm_err = float(np.mean(ne)) if ne else 0.0
        return res, size_err, norm_err

    cand = []
    def _add_multi(pi):
        if any(A[i].category != B[j].category for i, j in pi):
            return
        srcs = np.array([A[i].center for i, _ in pi])
        dsts = np.array([B[j].center for _, j in pi])
        s, R, t, span = gravity_locked_similarity(srcs, dsts, upA, upB)
        if not np.isfinite(s) or s <= 1e-6 or span < 0.05 * L_ref:
            return
        # tighten yaw using the (stable) shared-wall normal instead of noisy centres
        R, t, locked = _refine_normal_lock(A, B, pi, s, R, t, upA, upB)
        res, size_err, norm_err = _pair_metrics(pi, s, R, t)
        cand.append({"pi": pi, "s": s, "R": R, "t": t, "res": res,
                     "size_err": size_err, "norm_err": norm_err, "wall_locked": locked})

    # ---- multi-opening path (>=2 shared openings): centers give azimuth + scale.
    # Match the SMALLER opening set fully onto a subset of the larger (partial overlap:
    # not every opening in the bigger scene is shared). Handles len(A) </>/== len(B). ----
    if n_min >= 2:
        if len(A) <= len(B):
            for combB in itertools.permutations(range(len(B)), len(A)):
                _add_multi(list(zip(range(len(A)), combB)))
        else:
            for combA in itertools.permutations(range(len(A)), len(B)):
                _add_multi(list(zip(combA, range(len(B)))))
    # ---- single-opening path (exactly one shared opening, or multi degenerate) ----
    if n_min == 1 or not cand:
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i].category != B[j].category:
                    continue
                if not (A[i].normal_reliable and B[j].normal_reliable):
                    continue
                for (s, R, t) in gravity_locked_single(A[i], B[j], upA, upB):
                    if not np.isfinite(s) or s <= 1e-6:
                        continue
                    pi = [(i, j)]
                    res, size_err, norm_err = _pair_metrics(pi, s, R, t)
                    cand.append({"pi": pi, "s": s, "R": R, "t": t, "res": res,
                                 "size_err": size_err, "norm_err": norm_err, "wall_locked": False})
    if not cand:
        return GravityAlignResult(1.0, np.eye(3), np.zeros(3), [], "NO_SOLUTION",
                                  reason="no gravity-locked hypothesis (openings vertically stacked?)")

    # geometric consistency = center residual (normalized) + opening size mismatch.
    # Two centres always fit, so SIZE is what identifies the true shared wall.
    for c in cand:
        idxB = [j for _, j in c["pi"]]
        wall_n, _ = _matched_wall_normal(B, idxB)
        wall_p = np.array([B[j].center for j in idxB]).mean(0)
        camsAt = transform_points(cam_A.centers, c["s"], c["R"], c["t"])
        side, dom, mA, mB = camera_side_score(camsAt, cam_B.centers, wall_p, wall_n)
        c["side"] = side; c["dom"] = dom
        c["grav"] = float((c["R"] @ upA) @ upB)
        c["geom"] = c["res"] / T_center + c["size_err"] + c["norm_err"]
    keep = cand

    SIZE_TOL = 0.35    # max acceptable relative opening-size mismatch
    NORM_TOL = 0.15    # max acceptable normal misalignment (~32 deg)
    valid = [c for c in keep if c["side"] > 0 and c["size_err"] <= SIZE_TOL
             and c["norm_err"] <= NORM_TOL and c["dom"] >= 0.5]
    valid.sort(key=lambda c: c["geom"])

    if valid:
        top = valid[0]
        status, reason = "CONFIDENT", "OK (gravity-locked + size + camera-side)"
        if len(valid) >= 2 and (valid[1]["geom"] - valid[0]["geom"]) < 0.25:
            status = "AMBIGUOUS"
            reason = ("two physically-valid candidates are near-tied (likely identical "
                      "symmetric openings); need portal/see-through evidence to decide")
    else:
        # nothing physically valid: expose the best geometric guess but flag it
        keep.sort(key=lambda c: c["geom"]); top = keep[0]
        status = "AMBIGUOUS"
        if not any(c["side"] > 0 for c in keep):
            reason = "no candidate puts the two camera sets on OPPOSITE sides (check segmentation)"
        elif not any(c["size_err"] <= SIZE_TOL for c in keep):
            reason = "no candidate matches opening SIZES under a common scale (wrong openings?)"
        else:
            reason = "camera side evidence too weak to commit"
    keep.sort(key=lambda c: c["geom"])

    # Roll about the wall normal to align the matched openings IN-PLANE (closes the
    # residual door-frame gap and pulls the two floors parallel). Rotation about the wall
    # normal preserves coplanarity, so this costs nothing; only commit if it keeps the
    # cameras on opposite sides.
    roll_deg = 0.0
    if top.get("wall_locked", False) and len(top["pi"]) >= 2:
        idxB = [j for _, j in top["pi"]]
        nBr, oknr = _agg_wall_normal(B, idxB, upB)
        if oknr:
            R2, t2, roll_deg = _refine_roll_openings(A, B, top["pi"], top["s"],
                                                     top["R"], top["t"], nBr)
            wp = np.array([B[j].center for j in idxB]).mean(0)
            side2, dom2, _, _ = camera_side_score(
                transform_points(cam_A.centers, top["s"], R2, t2), cam_B.centers, wp, nBr)
            if side2 > 0:
                top["R"], top["t"], top["side"], top["dom"] = R2, t2, side2, dom2
                top["grav"] = float((R2 @ upA) @ upB)
            else:
                roll_deg = 0.0

    # Depth (along the wall normal): make the two matched-opening WALL FACES coincide, so
    # the join is tight ("ke sat") without a big gap. We use each door slab's ROBUST MEDIAN
    # depth (the dense wall surface), not blob extremes (percentiles over-separate and can
    # double the wall thickness), and apply the weighted-median offset over matched doors.
    # (The shared wall being captured by both scenes will overlap here; separating the two
    # shells cleanly is a fusion/cropping concern, not a global-transform one.)
    if top.get("wall_locked", False):
        idxB = [j for _, j in top["pi"]]
        nB, oknB = _agg_wall_normal(B, idxB, upB)
        if oknB:
            offs = []
            for i, j in top["pi"]:
                dA = transform_points(np.asarray(A_pts[i], float), top["s"], top["R"], top["t"]) @ nB
                dB = np.asarray(B_pts[j], float) @ nB
                offs.append(float(np.median(dB) - np.median(dA)))
            top["t"] = top["t"] + float(np.median(offs)) * nB

    # gravity-reliability guard: the whole method leans on camera gravity. If either
    # scene's per-frame up is not stable (tilted / 360 capture), don't claim CONFIDENT.
    GRAV_MIN = 0.9
    if status == "CONFIDENT" and min(cam_A.up_consistency, cam_B.up_consistency) < GRAV_MIN:
        status = "AMBIGUOUS"
        reason = ("camera gravity is not stable (up_consistency A=%.2f B=%.2f < %.2f); "
                  "vertical lock unreliable -- supply an oriented gravity/floor cue"
                  % (cam_A.up_consistency, cam_B.up_consistency, GRAV_MIN))

    return GravityAlignResult(
        s=top["s"], R=top["R"], t=top["t"], matches=top["pi"], status=status, reason=reason,
        up_consistency_A=cam_A.up_consistency, up_consistency_B=cam_B.up_consistency,
        grav_dot=top["grav"], wall_normal_locked=bool(top.get("wall_locked", False)),
        roll_refined_deg=float(roll_deg),
        cam_side=top["side"], cam_dominance=top["dom"],
        opening_residual=top["res"],
        candidates=[{"pi": c["pi"], "s": c["s"], "res": c["res"], "size_err": c["size_err"],
                     "norm_err": c["norm_err"], "geom": c["geom"], "side": c["side"],
                     "grav": c["grav"]} for c in keep],
    )
