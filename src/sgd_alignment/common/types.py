"""Core data structures shared across the SGD alignment pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PointCloud:
    """A single scanned point cloud with optional per-point attributes.

    Attributes mirror the fields available in the project's LiDAR .ply
    exports (Section 3.1 of the paper: xyz position, normals, reflection
    intensity used later for door/wall discrimination).
    """

    points: np.ndarray  # (N, 3) float64
    intensity: np.ndarray | None = None  # (N,) reflection/intensity value
    normals: np.ndarray | None = None  # (N, 3)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {self.points.shape}")
        if self.intensity is not None:
            self.intensity = np.asarray(self.intensity, dtype=np.float64)
            if self.intensity.shape != (len(self.points),):
                raise ValueError("intensity must have shape (N,) matching points")
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
            if self.normals.shape != self.points.shape:
                raise ValueError("normals must have shape (N, 3) matching points")

    def __len__(self) -> int:
        return len(self.points)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) of the point cloud."""
        return self.points.min(axis=0), self.points.max(axis=0)


@dataclass
class Plane:
    """A fitted planar surface (candidate wall) with its supporting points.

    Plane equation: normal . x + d = 0, with `normal` unit-length.
    """

    normal: np.ndarray  # (3,) unit normal
    d: float
    inlier_indices: np.ndarray  # indices into the source PointCloud

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.inlier_indices = np.asarray(self.inlier_indices, dtype=np.int64)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d


@dataclass
class Detection3D:
    """A detected window/door instance: an oriented rectangular opening on a wall.

    `u_axis`/`v_axis`/`normal` form the wall's local orthonormal frame
    (Figure 3 of the paper): u is horizontal along the wall, v is vertical
    (aligned with true up), normal points away from the wall surface.
    `center` is the opening's centroid on the wall plane.
    """

    category: str  # "window" or "door"
    center: np.ndarray  # (3,)
    u_axis: np.ndarray  # (3,) unit
    v_axis: np.ndarray  # (3,) unit
    normal: np.ndarray  # (3,) unit
    width: float  # extent along u_axis
    height: float  # extent along v_axis
    thickness: float = 0.05  # nominal extent along normal, for a thin 3D box

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.u_axis = self.u_axis / np.linalg.norm(self.u_axis)
        self.v_axis = self.v_axis / np.linalg.norm(self.v_axis)
        self.normal = self.normal / np.linalg.norm(self.normal)

    def corners(self) -> np.ndarray:
        """Return the 8 corners of the thin oriented 3D bounding box."""
        hu, hv, ht = self.width / 2, self.height / 2, self.thickness / 2
        corners = []
        for su in (-1, 1):
            for sv in (-1, 1):
                for sn in (-1, 1):
                    corners.append(
                        self.center
                        + su * hu * self.u_axis
                        + sv * hv * self.v_axis
                        + sn * ht * self.normal
                    )
        return np.array(corners)
