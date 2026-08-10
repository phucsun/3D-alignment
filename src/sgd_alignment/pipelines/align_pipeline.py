"""Connects `multiview_pipeline.py`'s per-side output (`<name>_detections.pkl`
+ `<name>_scene.ply`) to `sgd_alignment.matching`: loads both sides, matches
+ aligns them, then applies the recovered transform to the full colored
scene point clouds and exports one final aligned `.ply`.

Run as a script:

    python -m sgd_alignment.pipelines.align_pipeline --config path/to/config.yaml

or via the installed console script:

    sgd-align-pipeline --config path/to/config.yaml

`estimate_scale`/`normalize_distance` default to True here (unlike
`matching.alignment.align_indoor_outdoor`'s own default of False), because
this pipeline's usual case is indoor/outdoor coming from two independent
DA3/COLMAP reconstructions with no shared metric reference - see
`estimate_rigid_transform` and `sgd.build_sgds` docstrings for why that
needs handling. Pass estimate_scale/normalize_distance: false in the config
if both sides came from one shared reconstruction (already same scale).
"""
from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from sgd_alignment.matching.alignment import AlignmentResult, align_indoor_outdoor, transform_points
from sgd_alignment.matching.sgd import PairWeights


@dataclass
class AlignPipelineConfig:
    indoor_detections: str
    indoor_scene_ply: str
    outdoor_detections: str
    outdoor_scene_ply: str
    output_dir: str = "outputs"
    output_name: str = "aligned"
    estimate_scale: bool = True
    normalize_distance: bool = True
    max_cost: float = 5.0
    weights: dict | None = None  # overrides for PairWeights fields, e.g. {"distance_threshold": 0.5}


def load_align_config(path: str | Path) -> AlignPipelineConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AlignPipelineConfig(**raw)


def _read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    colors = (np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
              if "red" in v.dtype.names else None)
    return points, colors


def _write_ply(points: np.ndarray, colors: np.ndarray | None, path: Path) -> None:
    from plyfile import PlyData, PlyElement

    if colors is None:
        colors = np.full((len(points), 3), 160, np.uint8)
    vertex = np.zeros(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def run_alignment(config: AlignPipelineConfig) -> AlignmentResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("align_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / f"{config.output_name}_matching.log", mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    with open(config.indoor_detections, "rb") as f:
        indoor_detections = pickle.load(f)
    with open(config.outdoor_detections, "rb") as f:
        outdoor_detections = pickle.load(f)
    logger.info("indoor: %d opening(s), outdoor: %d opening(s)", len(indoor_detections), len(outdoor_detections))
    for d in indoor_detections:
        logger.info("  indoor  %s: center=%s, size=%.2f x %.2f", d.category, np.round(d.center, 3), d.width, d.height)
    for d in outdoor_detections:
        logger.info("  outdoor %s: center=%s, size=%.2f x %.2f", d.category, np.round(d.center, 3), d.width, d.height)

    weights = PairWeights(**config.weights) if config.weights else None
    result = align_indoor_outdoor(
        indoor_detections, outdoor_detections,
        weights=weights, max_cost=config.max_cost,
        estimate_scale=config.estimate_scale, normalize_distance=config.normalize_distance,
    )

    logger.info("=== matching: %d pair(s) matched ===", len(result.matches))
    for (i, j, cost), residual in zip(result.matches, result.residuals):
        logger.info("  indoor[%d](%s) <-> outdoor[%d](%s): cost=%.4f residual=%.4f m",
                     i, indoor_detections[i].category, j, outdoor_detections[j].category, cost, residual)
    logger.info("recovered scale (indoor -> outdoor): %.4f", result.scale)
    logger.info("translation t: %s", np.round(result.t, 4))

    indoor_points, indoor_colors = _read_ply(config.indoor_scene_ply)
    outdoor_points, outdoor_colors = _read_ply(config.outdoor_scene_ply)
    aligned_indoor_points = transform_points(indoor_points, result.R, result.t)

    all_points = np.concatenate([aligned_indoor_points, outdoor_points])
    all_colors = np.concatenate([
        indoor_colors if indoor_colors is not None else np.full((len(indoor_points), 3), (255, 120, 120), np.uint8),
        outdoor_colors if outdoor_colors is not None else np.full((len(outdoor_points), 3), (120, 120, 255), np.uint8),
    ])
    out_path = output_dir / f"{config.output_name}.ply"
    _write_ply(all_points, all_colors, out_path)
    logger.info("saved aligned model -> %s", out_path)

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to an align pipeline YAML config")
    args = parser.parse_args(argv)
    config = load_align_config(args.config)
    run_alignment(config)


if __name__ == "__main__":
    main()
