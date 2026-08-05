"""Configuration loading for the detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DetectionConfig:
    """Settings for the window/door detection stage (Section 3.1).

    Populated incrementally as later pipeline stages are implemented;
    only the fields needed so far (data I/O) are required.
    """

    point_clouds: dict[str, str] = field(default_factory=dict)
    output_dir: str = "outputs"

    def resolve_point_cloud(self, name: str, base_dir: Path | None = None) -> Path:
        if name not in self.point_clouds:
            raise KeyError(f"Unknown point cloud '{name}'. Known: {list(self.point_clouds)}")
        path = Path(self.point_clouds[name])
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path


def load_config(path: str | Path) -> DetectionConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return DetectionConfig(**raw)
