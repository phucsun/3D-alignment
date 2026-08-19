"""Baseline (original-paper-equivalent) vs RGA (full, all contributions on)
comparison, run automatically across every dataset registered in DATASETS.

"Baseline" = plain two-level Hungarian SGD matching only: no RANSAC
wall-consensus, no ambiguity check, no region constraint, no degenerate
filter, no intrinsic fallback, hand-picked `PairWeights()` (the source paper
never published weight values either, so this is unavoidable on both sides -
not something RGA claims credit for). `estimate_scale`/`normalize_distance`
are kept on for scale-ambiguous (video/DA3) sources in BOTH configs, since
without them matching is not just worse but impossible for that data - they
are a prerequisite for running on that modality at all, not a matching-
robustness contribution being tested here.

"RGA" = the same detections, run through this project's full matching stack
(whichever of RANSAC consensus / ambiguity check / min_dimension / intrinsic
fallback actually apply to that dataset - see each DatasetEntry below).

Add a new dataset by appending one more `DatasetEntry` (see the two
`_load_cloudcompare`/`_load_pickle` helpers) - nothing else in this file
needs to change to pick it up.

Usage:
    python scripts/compare_baseline_vs_rga.py
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgd_alignment.common.types import Detection3D  # noqa: E402
from sgd_alignment.detection.manual_segmentation import load_manual_segmentation  # noqa: E402
from sgd_alignment.detection.multiview_source import DA3Source  # noqa: E402
from sgd_alignment.detection.plane_fitting import estimate_up_vector_from_camera_rotation  # noqa: E402
from sgd_alignment.matching.alignment import align_indoor_outdoor  # noqa: E402

Loader = Callable[[], tuple[list[Detection3D], list[Detection3D]]]


@dataclass
class DatasetEntry:
    name: str
    loader: Loader
    baseline_kwargs: dict = field(default_factory=dict)
    rga_kwargs: dict = field(default_factory=dict)


def _load_cloudcompare(indoor_path: str, outdoor_path: str) -> Loader:
    def _loader():
        indoor = load_manual_segmentation(indoor_path, is_outdoor=False)
        outdoor = load_manual_segmentation(outdoor_path, is_outdoor=True)
        return indoor, outdoor

    return _loader


def _camera_positions(npz_path: str):
    source = DA3Source(npz_path)
    return np.array([-source.pose(v).R.T @ source.pose(v).t for v in source.view_ids])


def _camera_rotations(npz_path: str):
    source = DA3Source(npz_path)
    return np.array([source.pose(v).R for v in source.view_ids])


def _load_cloudcompare_with_camera_pose(
    indoor_path: str, outdoor_path: str, indoor_npz: str, outdoor_npz: str
) -> Loader:
    """Same as `_load_cloudcompare`, but for CloudCompare-labeled data
    sourced from 2 INDEPENDENT DA3/COLMAP reconstructions with recorded
    camera poses (`results.npz` next to each .ply) - uses gravity from
    each frame's own camera ROTATION for the shared up-vector (confirmed
    on real data more reliable than trajectory-position PCA - see
    `estimate_up_vector_from_camera_rotation`'s docstring, includes a
    genuine sign, no cross-scene sign-alignment step needed) and camera
    position for each wall's outward orientation (confirmed on real data
    to fix a density-based near-tie that silently merged indoor/outdoor
    onto the same side of the shared wall - see `orient_walls_outward`'s
    docstring).
    """

    def _loader():
        cam_in = _camera_positions(indoor_npz)
        cam_out = _camera_positions(outdoor_npz)
        rot_in = _camera_rotations(indoor_npz)
        rot_out = _camera_rotations(outdoor_npz)
        up_a, _ = estimate_up_vector_from_camera_rotation(rot_in)
        up_b, _ = estimate_up_vector_from_camera_rotation(rot_out)
        up = up_a + up_b
        up = up / np.linalg.norm(up)
        indoor = load_manual_segmentation(indoor_path, is_outdoor=False, up=up, camera_positions=cam_in)
        outdoor = load_manual_segmentation(outdoor_path, is_outdoor=True, up=up, camera_positions=cam_out)
        return indoor, outdoor

    return _loader


def _load_pickle(indoor_pkl: str, outdoor_pkl: str) -> Loader:
    def _loader():
        with open(indoor_pkl, "rb") as f:
            indoor = pickle.load(f)
        with open(outdoor_pkl, "rb") as f:
            outdoor = pickle.load(f)
        return indoor, outdoor

    return _loader


# ---------------------------------------------------------------------------
# Registered datasets - append new entries here.
#
#   CloudCompare / LiDAR (metric scale already, manual segmentation):
#     DatasetEntry("name", _load_cloudcompare("indoor.ply", "outdoor.ply"),
#                  baseline_kwargs={}, rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5))
#
#   DA3 / monocular video (no shared metric scale, auto 2D->3D detections):
#     DatasetEntry("name", _load_pickle("indoor_detections.pkl", "outdoor_detections.pkl"),
#                  baseline_kwargs=dict(estimate_scale=True, normalize_distance=True),
#                  rga_kwargs=dict(estimate_scale=True, normalize_distance=True,
#                                   use_ransac_consensus=True, ransac_max_residual=0.5))
#
# `home`/`home_phuc` are intentionally NOT registered here - their up-vector
# estimation was never confirmed reliable enough to judge matching
# correctness either way (see CONTRIBUTIONS.md).
# ---------------------------------------------------------------------------

DATASETS: list[DatasetEntry] = [
    DatasetEntry(
        "sc1_room2",
        _load_cloudcompare(
            "data/sc1-room2/scenario1_room2_indoor - Segment.ply",
            "data/sc1-room2/scenario1_room2_outdoor - Segment.ply",
        ),
        baseline_kwargs={},
        rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5),
    ),
    DatasetEntry(
        "sc2_room2",
        _load_cloudcompare(
            "data/sc2-room2/scenario2_room2_indoor - Cloud - segment.ply",
            "data/sc2-room2/scenario2_room2_outdoor - Cloud - segment.ply",
        ),
        baseline_kwargs={},
        rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5),
    ),
    DatasetEntry(
        "sc2_room5",
        _load_cloudcompare(
            "data/sc2-room5/scenario2_room5_indoor - segment.ply",
            "data/sc2-room5/scenario2_room5_outdoor - segment.ply",
        ),
        baseline_kwargs={},
        rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5),
    ),
    DatasetEntry(
        "vkist",
        _load_cloudcompare(
            "data/vkist_lidar_pantry/pantry-10-30 - Cloud - segment.ply",
            "data/vkist_lidar_pantry/hanh-lang-10-34 - Cloud - segment.ply",
        ),
        baseline_kwargs={},
        rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5),
    ),
    DatasetEntry(
        "sc1_room1",
        _load_cloudcompare(
            "data/sc1-room1/scenario1_room1_indoor - Cloud - segment.ply",
            "data/sc1-room1/scenario1_room1_outdoor - Cloud -segment.ply",
        ),
        baseline_kwargs={},
        rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5),
    ),
    DatasetEntry(
        "sc2_room1",
        _load_cloudcompare(
            "data/sc2-room1/scenario2_room1_indoor - Cloud - segment.ply",
            "data/sc2-room1/scenario2_room1_outdoor - Cloud - segment.ply",
        ),
        baseline_kwargs={},
        rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5),
    ),
    DatasetEntry(
        "sc2_room3",
        _load_cloudcompare(
            "data/sc2-room3/scenario2_room3_indoor - Cloud - segment.ply",
            "data/sc2-room3/scenario2_room3_outdoor - Cloud - segment.ply",
        ),
        baseline_kwargs={},
        rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5),
    ),
    DatasetEntry(
        "sc2_room4",
        _load_cloudcompare(
            "data/sc2-room4/scenario2_room4_indoor - Cloud - segment.ply",
            "data/sc2-room4/scenario2_room4_outdoor - Cloud - segment.ply",
        ),
        baseline_kwargs={},
        rga_kwargs=dict(use_ransac_consensus=True, ransac_max_residual=0.5),
    ),
    DatasetEntry(
        # Has camera pose (results.npz) for both sides - camera-trajectory
        # up + camera-position wall orientation confirmed correct (opposite
        # sides of the shared wall) on real data; RANSAC consensus off since
        # only 2 openings on the noisier side (not enough for a meaningful
        # consensus vote) but intrinsic fallback still applies.
        "q1",
        _load_cloudcompare_with_camera_pose(
            "data/Q1/Q1_indoor/Q1_indoor_points - Cloud.ply",
            "data/Q1/Q1_outdoor/Q1_outdoor_points - Cloud.ply",
            "data/Q1/Q1_indoor/results.npz",
            "data/Q1/Q1_outdoor/results.npz",
        ),
        baseline_kwargs=dict(estimate_scale=True, normalize_distance=True),
        rga_kwargs=dict(
            estimate_scale=True, normalize_distance=True,
            use_ransac_consensus=True, ransac_max_residual=0.5, use_intrinsic_fallback=True,
        ),
    ),
    DatasetEntry(
        # Has camera pose (results.npz) for both sides - same camera-pose
        # up/orientation fix as q1, confirmed correct (opposite sides) on
        # real data; only 1 opening per side so RANSAC consensus never
        # triggers, but intrinsic fallback still applies.
        "q2",
        _load_cloudcompare_with_camera_pose(
            "data/Q2/Q2_indoor/Q2_indoor_points - Cloud.ply",
            "data/Q2/Q2_outdoor/Q2_outdoor_points - Cloud.ply",
            "data/Q2/Q2_indoor/results.npz",
            "data/Q2/Q2_outdoor/results.npz",
        ),
        baseline_kwargs=dict(estimate_scale=True, normalize_distance=True),
        rga_kwargs=dict(
            estimate_scale=True, normalize_distance=True,
            use_ransac_consensus=True, ransac_max_residual=0.5, use_intrinsic_fallback=True,
        ),
    ),
    DatasetEntry(
        # Has camera pose (results.npz) for both sides. h_server is a
        # corridor with 2 real opposite-facing walls (one door pair per
        # neighboring room) plus a 3rd end-of-corridor wall - picking the
        # wrong wall for `server` here was traced through 2 confirmed real
        # bugs, not a config issue: (1) `merge_coplanar_planes` wrongly
        # fused the corridor's 2 opposite walls into one nonsensical plane
        # (unconditional `|d1-d2|` offset check, not sign-aware for
        # antiparallel normals - see its docstring), and (2)
        # `ransac_match_with_wall_consensus` broke ties between 2
        # candidate walls with equal inlier-vote counts by combination
        # enumeration order instead of by total residual, silently
        # preferring the higher-cost wall every time. Both fixed; `server`
        # now matches its established-correct wall at cost 0.017 (was
        # 0.12-0.54 depending on config). `aspect_ratio_weight` was never
        # the real fix - see CONTRIBUTIONS.md section 10 for that
        # retraction.
        "server",
        _load_cloudcompare_with_camera_pose(
            "data/server/server/server_room_points-segment.ply",
            "data/server/h_server/h_server_room_points - Cloud - segment.ply",
            "data/server/server/results.npz",
            "data/server/h_server/results.npz",
        ),
        baseline_kwargs=dict(estimate_scale=True, normalize_distance=True),
        rga_kwargs=dict(
            estimate_scale=True, normalize_distance=True,
            use_ransac_consensus=True, ransac_max_residual=0.5, use_intrinsic_fallback=True,
        ),
    ),
    DatasetEntry(
        "video1",
        _load_pickle(
            "outputs/video1_test/indoor_detections.pkl",
            "outputs/video1_test/outdoor_detections.pkl",
        ),
        baseline_kwargs=dict(estimate_scale=True, normalize_distance=True),
        rga_kwargs=dict(estimate_scale=True, normalize_distance=True),
    ),
    DatasetEntry(
        "meeting_room",
        _load_pickle(
            "outputs/door_only/meeting_indoor_detections.pkl",
            "outputs/door_only/meeting_outdoor_detections.pkl",
        ),
        baseline_kwargs=dict(estimate_scale=True, normalize_distance=True),
        rga_kwargs=dict(
            estimate_scale=True, normalize_distance=True,
            use_ransac_consensus=True, ransac_max_residual=0.5,
        ),
    ),
]


def _run(indoor, outdoor, kwargs: dict) -> tuple[str, str]:
    """Returns (detailed_line, short_summary) for one config run."""
    try:
        result = align_indoor_outdoor(indoor, outdoor, **kwargs)
    except Exception as e:  # noqa: BLE001 - report, don't crash the whole sweep
        err = f"ERROR ({type(e).__name__}: {e})"
        return err, f"ERROR: {type(e).__name__}"
    if not result.matches:
        return "0 matched", "0 matched"
    residuals = [round(float(r), 4) for r in result.residuals]
    pairs = ", ".join(f"{i}<->{j}" for i, j, _ in result.matches)
    detailed = f"{len(result.matches)} matched  residuals={residuals}  pairs=[{pairs}]"
    short = f"{len(result.matches)} matched, max_residual={max(residuals):.4f}"
    return detailed, short


def main() -> None:
    rows = []
    for entry in DATASETS:
        print(f"\n=== {entry.name} ===")
        indoor, outdoor = entry.loader()
        print(f"  ({len(indoor)} indoor / {len(outdoor)} outdoor detections)")

        baseline_detailed, baseline_short = _run(indoor, outdoor, entry.baseline_kwargs)
        print(f"  baseline : {baseline_detailed}")

        rga_detailed, rga_short = _run(indoor, outdoor, entry.rga_kwargs)
        print(f"  RGA      : {rga_detailed}")

        flag = "  <-- DIFFERS" if baseline_short != rga_short else ""
        rows.append((entry.name, baseline_short, rga_short, flag))

    print("\n" + "=" * 90)
    print(f"{'dataset':<18}{'baseline':<32}{'RGA':<32}")
    print("=" * 90)
    for name, baseline_short, rga_short, flag in rows:
        print(f"{name:<18}{baseline_short:<32}{rga_short:<32}{flag}")


if __name__ == "__main__":
    main()
