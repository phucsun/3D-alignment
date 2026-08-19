"""Align two scenes from AUTO-SEGMENTED opening clusters + camera gravity.
Usage: python run_pair_gravity.py <A.pkl> <A.npz> <A.ply> <B.pkl> <B.npz> <B.ply> <out.ply>
"""
import sys, pickle
import numpy as np
sys.path.insert(0, "src")
from plyfile import PlyData, PlyElement
from sgd_alignment.matching.robust_align import OpeningCluster, transform_points
from sgd_alignment.matching.gravity_align import camera_evidence, align_gravity_camera

APKL, ANPZ, APLY, BPKL, BNPZ, BPLY, OUT = sys.argv[1:8]

def _cap(P, n=40000):
    P = np.asarray(P, float)
    return P if len(P) <= n else P[np.random.RandomState(0).choice(len(P), n, replace=False)]
def clusters(pkl):
    with open(pkl, "rb") as f: raw = pickle.load(f)
    return [OpeningCluster(category=o["category"], points=_cap(o["points"])) for o in raw]
def load(p):
    v = PlyData.read(p)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8) if "red" in v.dtype.names else np.full((len(xyz),3),180,np.uint8)
    return xyz, rgb

cA, cB = clusters(APKL), clusters(BPKL)
print("A openings:", [(k,o.category,len(o.points)) for k,o in enumerate(cA)])
print("B openings:", [(k,o.category,len(o.points)) for k,o in enumerate(cB)])
camA, camB = camera_evidence(ANPZ), camera_evidence(BNPZ)
print("gravity A consist=%.3f  B consist=%.3f" % (camA.up_consistency, camB.up_consistency))

res = align_gravity_camera(cA, cB, camA, camB)
print("\n=== KET QUA ===")
print("status =", res.status, "| reason:", res.reason)
print("scale=%.3f | upright(R.upA.upB)=%.3f | cam_side=%+.0f dom=%.2f | open_res=%.4f | matches=%s" % (
    res.s, res.grav_dot, res.cam_side, res.cam_dominance, res.opening_residual, res.matches))
print("--- candidates ---")
for c in res.candidates[:8]:
    print("  pi=%s s=%.2f res=%.4f size_err=%.2f norm_err=%.2f geom=%.3f side=%+.0f" % (
        c["pi"], c["s"], c["res"], c["size_err"], c["norm_err"], c["geom"], c["side"]))

xA, colA = load(APLY); xB, colB = load(BPLY)
xA_t = transform_points(xA, res.s, res.R, res.t).astype(np.float32)
camAt = transform_points(camA.centers, res.s, res.R, res.t)
def markers(C, color):
    P = np.repeat(C, 200, axis=0) + (np.random.RandomState(0).rand(len(C)*200,3)-0.5)*0.04
    return P.astype(np.float32), np.tile(color, (len(P),1)).astype(np.uint8)
mA,mAc = markers(camAt, [255,0,0]); mB,mBc = markers(camB.centers, [0,255,0])
P = np.vstack([xA_t, xB.astype(np.float32), mA, mB]); C = np.vstack([colA, colB, mAc, mBc]).astype(np.uint8)
if len(P) > 5_000_000:
    idx = np.random.RandomState(0).choice(len(P), 5_000_000, replace=False); P,C = P[idx],C[idx]
vx = np.zeros(len(P), dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
vx["x"],vx["y"],vx["z"] = P[:,0],P[:,1],P[:,2]; vx["red"],vx["green"],vx["blue"] = C[:,0],C[:,1],C[:,2]
PlyData([PlyElement.describe(vx,"vertex")], text=False).write(OUT)
print("\nWROTE", OUT, "(cam A=DO, cam B=XANH LA)")
