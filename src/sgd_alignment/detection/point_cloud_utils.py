"""Point cloud I/O and basic inspection utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData

from sgd_alignment.common.types import PointCloud


def load_point_cloud(path: str | Path) -> PointCloud:
    """Load a .ply point cloud, keeping reflection intensity if present.

    The project's LiDAR exports carry `scalar_intensity` (reflection
    value, used later for door detection in Section 3.1) and
    `scalar_curvature`/normals fields, which are read directly via
    `plyfile` since open3d's PLY reader does not expose custom scalar
    properties.
    """
    path = Path(path)
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names

    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

    normals = None
    if all(n in names for n in ("nx", "ny", "nz")):
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1)
        if not np.any(normals):
            normals = None

    intensity = None
    for candidate in ("scalar_intensity", "intensity", "reflectance"):
        if candidate in names:
            intensity = np.asarray(vertex[candidate], dtype=np.float64)
            break

    return PointCloud(points=points, intensity=intensity, normals=normals)


def project_top_view_png(pc: PointCloud, out_path: str | Path, point_size: float = 0.5) -> Path:
    """Render a top-down (X-Y) scatter plot colored by intensity, for visual review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    color = pc.intensity if pc.intensity is not None else pc.points[:, 2]

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(pc.points[:, 0], pc.points[:, 1], c=color, s=point_size, cmap="viridis")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(out_path.stem)
    fig.colorbar(sc, ax=ax, label="intensity" if pc.intensity is not None else "z")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
