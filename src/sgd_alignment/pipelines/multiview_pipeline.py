"""Headless end-to-end multi-view detection pipeline: photos + camera poses
(COLMAP or DA3) -> 2D door/window segmentation -> backproject to 3D ->
`Detection3D`s ready for `sgd_alignment.matching`. No notebook required.

Run as a script:

    python -m sgd_alignment.pipelines.multiview_pipeline --config path/to/config.yaml

or via the installed console script (`pyproject.toml`):

    sgd-multiview-pipeline --config path/to/config.yaml

See `configs/multiview_pipeline.example.yaml` for the config schema. Visual
QA replaces a notebook's inline image display: every segmented view's
overlay is saved to `<output_dir>/<output_name>_<view_name>_segmented.jpg` for you to
open afterward.
"""
from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from sgd_alignment.common.types import Detection3D
from sgd_alignment.detection.multiview_segmentation import ViewInstance, build_detections_from_instances
from sgd_alignment.detection.multiview_source import ColmapSource, DA3Source, DEFAULT_DEPTH_RANGE
from sgd_alignment.detection.segment_2d import DEFAULT_TEXT_PROMPT, DoorWindowSegmenter, save_overlay

REVIEW_HIGHLIGHT_COLORS = {"door": (0, 255, 0), "window": (255, 0, 0)}


def _setup_logger(name: str, log_path: Path) -> logging.Logger:
    """Every pipeline run logs to both the console and a file under
    `output_dir`, so a run kicked off non-interactively still leaves a
    readable record of what happened (matches per view, sizes, warnings)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


@dataclass
class SourceConfig:
    backend: str  # "da3" or "colmap"
    npz_path: str | None = None            # da3
    sparse_dir: str | None = None          # colmap
    images_dir: str | None = None          # colmap
    depth_maps_dir: str | None = None      # colmap
    scene_ply: str | None = None           # colmap (required), da3 (optional override)


@dataclass
class SegmenterConfig:
    text_prompt: str = DEFAULT_TEXT_PROMPT
    box_threshold: float = 0.5
    text_threshold: float = 0.25
    nms_iou_thres: float = 0.7
    cross_label_iou_thres: float = 0.7
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    sam_model_id: str = "sam2.1_b.pt"
    device: str | None = None


@dataclass
class MultiviewPipelineConfig:
    source: SourceConfig
    is_outdoor: bool = False
    scale: float = 1.0
    merge_distance: float = 0.6
    depth_range: tuple[float, float] = DEFAULT_DEPTH_RANGE
    selected_views: list[int] = field(default_factory=list)    # da3: view indices, [] = all
    selected_images: list[str] = field(default_factory=list)   # colmap: filenames, [] = all
    output_dir: str = "outputs"
    output_name: str = "indoor"
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    up: list[float] | None = None  # override auto up-vector estimation - see detections_from_clusters
    min_confidence: float = 0.0    # drop 2D detections below this conf before backprojecting - see backproject_instances
    trim_percentile: float = 0.0   # robust width/height measurement - see opening_geometry.points_to_detection


def load_pipeline_config(path: str | Path) -> MultiviewPipelineConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = dict(raw)
    source = SourceConfig(**raw.pop("source"))
    segmenter = SegmenterConfig(**raw.pop("segmenter", {}))
    if "depth_range" in raw:
        raw["depth_range"] = tuple(raw["depth_range"])
    return MultiviewPipelineConfig(source=source, segmenter=segmenter, **raw)


def _build_source(cfg: SourceConfig):
    if cfg.backend == "da3":
        if not cfg.npz_path:
            raise ValueError("source.npz_path is required for backend 'da3'")
        return DA3Source(cfg.npz_path)
    if cfg.backend == "colmap":
        missing = [f for f in ("sparse_dir", "images_dir", "depth_maps_dir") if getattr(cfg, f) is None]
        if missing:
            raise ValueError(f"source.{missing[0]} is required for backend 'colmap'")
        return ColmapSource(cfg.sparse_dir, cfg.images_dir, cfg.depth_maps_dir, cfg.scene_ply)
    raise ValueError(f"unknown source.backend {cfg.backend!r} (expected 'da3' or 'colmap')")


def _resolve_view_ids(source, config: MultiviewPipelineConfig) -> list[int]:
    if config.source.backend == "da3":
        return config.selected_views or source.view_ids
    if config.selected_images:
        return [source.id_by_name(name) for name in config.selected_images]
    return source.view_ids


def _export_review_ply(
    scene_points: np.ndarray,
    scene_colors: np.ndarray | None,
    merged_clusters: list[dict],
    out_path: Path,
) -> None:
    from plyfile import PlyData, PlyElement

    background = scene_colors if scene_colors is not None else np.full((len(scene_points), 3), 160, np.uint8)
    all_points = [scene_points]
    all_colors = [background.astype(np.uint8)]
    for c in merged_clusters:
        color = np.array(REVIEW_HIGHLIGHT_COLORS.get(c["label"], (255, 0, 255)), dtype=np.uint8)
        all_points.append(c["points"])
        all_colors.append(np.tile(color, (len(c["points"]), 1)))

    pts = np.concatenate(all_points)
    cols = np.concatenate(all_colors).astype(np.uint8)
    vertex = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(out_path))


def run_pipeline(config: MultiviewPipelineConfig, segmenter=None) -> list[Detection3D]:
    """`segmenter` defaults to a real `DoorWindowSegmenter` built from
    `config.segmenter` (requires the `segmentation` extra). Overridable so
    tests/tools can inject a stub with a `.segment(image_rgb) ->
    list[DetectedInstance]` method instead of loading real models.
    """
    from sgd_alignment.common.types import PointCloud
    from sgd_alignment.detection.multiview_segmentation import backproject_instances, detections_from_clusters, merge_instances
    from sgd_alignment.detection.plane_fitting import estimate_up_vector_from_camera_rotation, estimate_up_vector_manhattan

    source = _build_source(config.source)
    view_ids = _resolve_view_ids(source, config)
    if not view_ids:
        raise ValueError("no views to segment - check selected_views/selected_images and the source's own view_ids")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logger(f"multiview_pipeline.{config.output_name}", output_dir / f"{config.output_name}_detection.log")

    log.info("=== detection: %s (is_outdoor=%s, backend=%s) ===", config.output_name, config.is_outdoor, config.source.backend)
    log.info("segmenting %d view(s): %s", len(view_ids), view_ids)

    if segmenter is None:
        segmenter = DoorWindowSegmenter(**asdict(config.segmenter))
    instances: list[ViewInstance] = []
    for view_id in view_ids:
        image = source.image(view_id)
        if image is None:
            log.warning("view %s (%s): no image available, skipping", view_id, source.name(view_id))
            continue
        detected = segmenter.segment(image)
        save_overlay(image, detected, str(output_dir / f"{config.output_name}_{source.name(view_id)}_segmented.jpg"))
        log.info("%s: %d object(s)", source.name(view_id), len(detected))
        for inst in detected:
            log.info("    %s: conf=%.2f", inst.label, inst.conf)
            instances.append(ViewInstance(view_id=view_id, label=inst.label, mask=inst.mask, conf=inst.conf))

    if not instances:
        raise ValueError("no door/window detected on any selected view - nothing to backproject")

    raw_instances = backproject_instances(source, instances, scale=config.scale, depth_range=tuple(config.depth_range),
                                           min_confidence=config.min_confidence)
    if config.min_confidence > 0:
        log.info("min_confidence=%.2f: %d/%d instance(s) kept", config.min_confidence, len(raw_instances), len(instances))
    merged = merge_instances(raw_instances, merge_distance=config.merge_distance)
    log.info("%d raw instance(s) -> %d merged opening(s)", len(raw_instances), len(merged))
    scene_points, scene_colors = source.build_scene_point_cloud()
    scene_points = scene_points * config.scale
    log.info("scene point cloud: %d points", len(scene_points))

    camera_positions = np.array([-source.pose(v).R.T @ source.pose(v).t for v in view_ids])
    camera_rotations = np.array([source.pose(v).R for v in view_ids])

    MIN_UP_CONSISTENCY = 0.90

    if config.up is not None:
        up = np.array(config.up)
        log.info("up vector (overridden via config): %s", np.round(up, 4))
    else:
        # gravity from each frame's own camera ROTATION (see
        # estimate_up_vector_from_camera_rotation's docstring) - confirmed
        # on real data more reliable than the trajectory-position-PCA
        # version it replaces (works from any number of frames, not just
        # well-spread ones, and comes with an honest per-capture
        # consistency score). Falls back to the point-cloud heuristic only
        # when that consistency is low (a rolling/tilted rig, or a source
        # with degenerate/duplicated poses) - the rotation estimate itself
        # never "fails" the way trajectory-PCA could, but a low-consistency
        # result should not be trusted blindly either.
        up, consistency = estimate_up_vector_from_camera_rotation(camera_rotations)
        if consistency >= MIN_UP_CONSISTENCY:
            log.info("up vector (from camera rotation, consistency=%.3f): %s", consistency, np.round(up, 4))
        else:
            log.info("camera-rotation up consistency too low (%.3f < %.2f) - falling back to point-cloud heuristic",
                      consistency, MIN_UP_CONSISTENCY)
            up = estimate_up_vector_manhattan(PointCloud(points=scene_points))
            log.info("up vector (auto-estimated): %s", np.round(up, 4))

    detections = detections_from_clusters(scene_points, merged, config.is_outdoor, up=up,
                                           trim_percentile=config.trim_percentile,
                                           camera_positions=camera_positions)

    detections_path = output_dir / f"{config.output_name}_detections.pkl"
    with open(detections_path, "wb") as f:
        pickle.dump(detections, f)
    log.info("saved %d detection(s) -> %s", len(detections), detections_path)
    for d in detections:
        log.info("  %s: center=%s, size=%.2f x %.2f, normal=%s",
                  d.category, np.round(d.center, 3), d.width, d.height, np.round(d.normal, 3))

    review_path = output_dir / f"{config.output_name}_openings.ply"
    _export_review_ply(scene_points, scene_colors, merged, review_path)
    log.info("saved review point cloud -> %s", review_path)

    scene_path = output_dir / f"{config.output_name}_scene.ply"
    _export_review_ply(scene_points, scene_colors, [], scene_path)
    log.info("saved full scene point cloud -> %s", scene_path)

    return detections


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a multiview pipeline YAML config")
    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
