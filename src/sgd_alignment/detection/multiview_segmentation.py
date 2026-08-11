"""Turn 2D door/window masks (from Grounding DINO + SAM2, run on hand-picked
key-frame images) into `Detection3D`s, given a `multiview_source.py` reader
for whichever reconstruction backend produced the poses/depth (COLMAP or
DA3). This is the detection-side counterpart to `manual_segmentation.py`'s
CloudCompare workflow - same output contract (`list[Detection3D]`, straight
into `sgd_alignment.matching`), different input source.

The 2D segmentation itself (Grounding DINO + SAM2 inference, and picking
which images to segment) deliberately stays in the notebook - it needs
interactive visual review and heavy model loads that don't belong in a
library function. Everything downstream of "I have a mask per view" lives
here instead of being copy-pasted across notebooks.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from sgd_alignment.common.types import Detection3D, PointCloud
from sgd_alignment.detection.multiview_source import DEFAULT_DEPTH_RANGE, backproject_mask_to_world
from sgd_alignment.detection.opening_geometry import nearest_wall_normal, orient_walls_outward, points_to_detection
from sgd_alignment.detection.plane_fitting import estimate_up_vector_manhattan, extract_wall_planes


class MultiviewSource(Protocol):
    """Duck-typed interface implemented by `ColmapSource` and `DA3Source`."""

    @property
    def view_ids(self) -> list[int]: ...
    def camera(self, view_id: int): ...
    def pose(self, view_id: int): ...
    def depth(self, view_id: int) -> np.ndarray: ...
    def build_scene_point_cloud(self) -> tuple[np.ndarray, np.ndarray | None]: ...


class ViewInstance:
    """One 2D-detected instance on one view: a category label + binary mask
    at that view's own image resolution (may differ from the depth map's
    resolution - `build_detections_from_instances` resizes as needed).

    `conf`: the 2D detector's own confidence score (Grounding DINO/SAM2),
    carried through so `backproject_instances(min_confidence=...)` can
    drop weak detections before they ever reach the 3D measurement -
    previously computed but silently discarded.
    """

    __slots__ = ("view_id", "label", "mask", "conf")

    def __init__(self, view_id: int, label: str, mask: np.ndarray, conf: float = 1.0):
        self.view_id = view_id
        self.label = label
        self.mask = mask
        self.conf = conf


def merge_instances(
    raw_instances: list[dict], merge_distance: float = 0.6
) -> list[dict]:
    """Greedily group per-view instances of the same physical opening seen
    from multiple views: same category + centroid within `merge_distance`
    of an existing cluster's running centroid -> merged into it.

    `raw_instances`: list of {"label": str, "points": (N,3)}. Returns a
    list of {"label": str, "points": (M,3)} (points from all merged views
    concatenated).
    """
    clusters: list[dict] = []
    for inst in raw_instances:
        centroid = inst["points"].mean(axis=0)
        match = None
        for c in clusters:
            if c["label"] != inst["label"]:
                continue
            c_centroid = np.concatenate(c["points_list"]).mean(axis=0)
            if np.linalg.norm(c_centroid - centroid) < merge_distance:
                match = c
                break
        if match is None:
            clusters.append({"label": inst["label"], "points_list": [inst["points"]]})
        else:
            match["points_list"].append(inst["points"])
    return [{"label": c["label"], "points": np.concatenate(c["points_list"])} for c in clusters]


def backproject_instances(
    source: MultiviewSource,
    instances: list[ViewInstance],
    scale: float = 1.0,
    depth_range: tuple[float, float] = DEFAULT_DEPTH_RANGE,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Backproject every instance's 2D mask to a 3D world point cluster,
    using that instance's own view's depth map + pose. `scale` converts
    from the reconstruction's own (possibly non-metric) unit to whatever
    unit the caller wants Detection3D sizes reported in - pass 1.0 (the
    default) unless you've measured a real known size to calibrate
    against; see `sgd_alignment.matching.alignment.align_indoor_outdoor`'s
    `estimate_scale` for how matching now handles an unknown/inconsistent
    scale automatically instead, which is usually the better fix.

    `min_confidence=0.0` (default) keeps every instance, unchanged from
    before. Raise it to drop weak 2D detections (`inst.conf` below the
    threshold) before they contribute any points to a merged cluster - a
    single low-confidence, poorly-localized mask from one oblique/noisy
    view can otherwise corrupt that opening's measured size regardless of
    how many good views also saw it (see `points_to_detection`'s
    `trim_percentile` for the complementary fix at the point-cluster level).
    """
    raw_instances = []
    for inst in instances:
        if inst.conf < min_confidence:
            continue
        depth = source.depth(inst.view_id)
        camera, pose = source.camera(inst.view_id), source.pose(inst.view_id)
        mask = inst.mask
        if mask.shape != depth.shape:
            mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        points = backproject_mask_to_world(mask, depth, camera, pose, depth_range=depth_range) * scale
        if len(points) == 0:
            continue
        raw_instances.append({"label": inst.label, "points": points})
    return raw_instances


def detections_from_clusters(
    scene_points: np.ndarray,
    clusters: list[dict],
    is_outdoor: bool,
    up: np.ndarray | None = None,
    trim_percentile: float = 0.0,
) -> list[Detection3D]:
    """Measure each merged opening cluster against the scene's own wall
    planes (reusing exactly the same logic `manual_segmentation.py` uses
    for CloudCompare input) to produce `Detection3D`s ready for
    `sgd_alignment.matching`. Split out from `build_detections_from_instances`
    so callers that need the intermediate merged point clusters too (e.g.
    for a colored review-point-cloud export) don't have to redo backproject
    + merge to get them.

    `up`: override the auto-estimated up-vector. `estimate_up_vector_manhattan`
    needs at least 2 independent plane-normal groups (walls vs floor/ceiling)
    to work reliably - a scene with limited coverage (e.g. a narrow corridor
    filmed mostly facing one wall, with little floor/ceiling captured) can
    have its "smallest spatial extent" heuristic pick a wall as "up" instead
    of the other way around. If two scenes are confirmed to share one
    coordinate frame (e.g. two DA3 runs whose up-vectors turn out nearly
    parallel when compared directly - not something to assume, only to
    verify per pair), passing the other scene's already-trustworthy `up`
    here sidesteps a bad estimate on a poorly-covered one.

    `trim_percentile`: forwarded to `points_to_detection` - see there.
    """
    scene_pc = PointCloud(points=scene_points)
    up = up if up is not None else estimate_up_vector_manhattan(scene_pc)
    walls = extract_wall_planes(scene_pc, up=up)
    oriented_normals = orient_walls_outward(walls, scene_pc, is_outdoor)

    detections = []
    for cluster in clusters:
        wall_normal = nearest_wall_normal(cluster["points"].mean(axis=0), walls, oriented_normals)
        detections.append(points_to_detection(cluster["points"], cluster["label"], up, wall_normal,
                                                trim_percentile=trim_percentile))
    return detections


def build_detections_from_instances(
    source: MultiviewSource,
    instances: list[ViewInstance],
    is_outdoor: bool,
    scale: float = 1.0,
    merge_distance: float = 0.6,
    depth_range: tuple[float, float] = DEFAULT_DEPTH_RANGE,
    up: np.ndarray | None = None,
    min_confidence: float = 0.0,
    trim_percentile: float = 0.0,
) -> list[Detection3D]:
    """Full detection-side pipeline: backproject every 2D instance to 3D,
    merge duplicates seen from multiple views, then measure each merged
    opening (`detections_from_clusters`) to produce `Detection3D`s ready
    for `sgd_alignment.matching`. See `detections_from_clusters` for `up`/
    `trim_percentile`, and `backproject_instances` for `min_confidence`.
    """
    raw_instances = backproject_instances(source, instances, scale=scale, depth_range=depth_range,
                                           min_confidence=min_confidence)
    merged = merge_instances(raw_instances, merge_distance=merge_distance)
    scene_points, _ = source.build_scene_point_cloud()
    return detections_from_clusters(scene_points * scale, merged, is_outdoor, up=up, trim_percentile=trim_percentile)
