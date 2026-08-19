"""Regression tests for gravity_align: lock the working cases so future edits can't
silently break them. Run:  python tests/test_gravity_align_regression.py
Needs Python 3.11 env with numpy + plyfile (real workspace data present)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from sgd_alignment.matching.robust_align import openings_from_manual_segmentation, transform_points
from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera, _basis_from_up

ROOT = os.path.join(os.path.dirname(__file__), "..")
FA = os.path.join(ROOT, r"workspace/h_server_room/h_server_room_points - Cloud.remaining.ply")
FB = os.path.join(ROOT, r"workspace/server_room/server_room_points-segment.ply")
NA = os.path.join(ROOT, r"workspace/h_server_room/h_sever_room/results.npz")
NB = os.path.join(ROOT, r"workspace/server_room/results.npz")

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  <-- " + detail))
    if not cond:
        fails.append(name)

cA = openings_from_manual_segmentation(FA)
cB = openings_from_manual_segmentation(FB)
camA, camB = camera_evidence(NA), camera_evidence(NB)

# --- Case 1: two shared doors on one wall (the flagship regression) ---
r = align_gravity_camera(cA, cB, camA, camB)
check("2door.status_confident", r.status == "CONFIDENT", r.status)
check("2door.matches", sorted(r.matches) == [(0, 5), (1, 4)], str(r.matches))
check("2door.scale", abs(r.s - 1.508) < 0.03, f"{r.s:.3f}")
check("2door.upright", r.grav_dot > 0.99, f"{r.grav_dot:.3f}")
check("2door.opposite_side", r.cam_side > 0, f"{r.cam_side}")

# --- Case 2: single shared door, correct correspondence ---
r1 = align_gravity_camera([cA[0]], [cB[4]], camA, camB)
check("1door.status_confident", r1.status == "CONFIDENT", r1.status)
check("1door.matches", sorted(r1.matches) == [(0, 0)], str(r1.matches))
check("1door.scale", abs(r1.s - 1.5) < 0.1, f"{r1.s:.3f}")
check("1door.upright", r1.grav_dot > 0.99, f"{r1.grav_dot:.3f}")
check("1door.opposite_side", r1.cam_side > 0, f"{r1.cam_side}")

# --- Case 3: A has more openings than B (len(A)>len(B), multi-path both ways) ---
#   give A its 2 doors, B a single door -> honest handling, must not crash, upright.
r3 = align_gravity_camera(cA, [cB[5]], camA, camB)
check("AgtB.no_crash_upright", r3.grav_dot > 0.99 and r3.status in ("CONFIDENT", "AMBIGUOUS"),
      f"{r3.status}/{r3.grav_dot:.3f}")

# --- Case 4: indoor/outdoor facade (auto-seg) -> wall-normal lock makes walls coplanar ---
import pickle, os
from sgd_alignment.matching.robust_align import OpeningCluster, build_opening, _matched_wall_normal
from sgd_alignment.matching.gravity_align import _dedupe_openings
QI = os.path.join(ROOT, "outputs/auto_seg/Q815_indoor.pkl")
QO = os.path.join(ROOT, "outputs/auto_seg/Q815_outdoor.pkl")
if os.path.exists(QI) and os.path.exists(QO):
    def _cap(P, n=40000):
        P = np.asarray(P, float)
        return P if len(P) <= n else P[np.random.RandomState(0).choice(len(P), n, replace=False)]
    def _clus(p):
        return [OpeningCluster(o["category"], _cap(o["points"])) for o in pickle.load(open(p, "rb"))]
    ci, co = _clus(QI), _clus(QO)
    cmi = camera_evidence(os.path.join(ROOT, "workspace/Q815_indoor/Q815_indoor/results.npz"))
    cmo = camera_evidence(os.path.join(ROOT, "workspace/Q815_outdoor/Q815_outdoor/results.npz"))
    rq = align_gravity_camera(ci, co, cmi, cmo)
    # measure on the SAME deduped openings the aligner used (matches index into these)
    Ai = [o for o in map(build_opening, _dedupe_openings(ci, cmi.up)) if o]
    Bo = [o for o in map(build_opening, _dedupe_openings(co, cmo.up)) if o]
    nAi, _ = _matched_wall_normal(Ai, [i for i, _ in rq.matches])
    nBo, _ = _matched_wall_normal(Bo, [j for _, j in rq.matches])
    wall_deg = float(np.degrees(np.arccos(np.clip(abs(float((rq.R @ nAi) @ nBo)), 0, 1))))
    # co-axiality: door centres must coincide IN THE WALL PLANE (perpendicular to the wall
    # normal). The depth (wall-normal) component is intentionally non-zero -- the two scenes
    # are separated by the wall thickness so they don't interpenetrate -- so project it out.
    nrm = nBo / (np.linalg.norm(nBo) + 1e-12)
    inplane = []
    for i, j in rq.matches:
        d = transform_points(Ai[i].center[None, :], rq.s, rq.R, rq.t)[0] - Bo[j].center
        inplane.append(float(np.linalg.norm(d - (d @ nrm) * nrm)))
    max_inplane = max(inplane) if inplane else 9.9
    check("Q815.confident", rq.status == "CONFIDENT", rq.status)
    check("Q815.wall_locked", rq.wall_normal_locked, str(rq.wall_normal_locked))
    check("Q815.walls_coplanar", wall_deg < 2.0, f"{wall_deg:.2f} deg (want ~0)")
    check("Q815.doors_inplane_tight", max_inplane < 0.03, f"{max_inplane:.3f} (want <0.03)")
    check("Q815.deduped", len(Ai) <= 3 and len(Bo) <= 2, f"A={len(Ai)} B={len(Bo)}")
    check("Q815.roll_applied", 1.0 <= abs(rq.roll_refined_deg) <= 12.0, f"{rq.roll_refined_deg:.1f}")
    check("Q815.opposite_side", rq.cam_side > 0, f"{rq.cam_side}")
else:
    print("SKIP Q815 case (auto-seg pkls not present)")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
