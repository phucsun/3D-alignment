"""Camera pose/depth readers for the multi-view (photo-based) detection
pipeline, plus the 2D-mask -> 3D-world backprojection shared by all of them.

Two concrete sources are provided, `ColmapSource` and `DA3Source`, both
exposing the same small interface so `multiview_segmentation.py` doesn't
need to know which reconstruction backend produced the data:

    source.view_ids                    -> list[int]
    source.name(view_id)                -> str
    source.camera(view_id).intrinsics_matrix() -> (3,3)
    source.pose(view_id).rotation_matrix()     -> (3,3)
    source.pose(view_id).tvec                  -> (3,)
    source.depth(view_id)               -> (H,W) float32, Z-depth in camera space
    source.image(view_id)               -> (H,W,3) uint8 | None
    source.build_scene_point_cloud()    -> (points (N,3), colors (N,3) uint8 | None)

Camera convention (shared by both backends - verified for DA3 against real
capture data via cross-view point-cloud alignment, sub-cm agreement): a
world point X_world projects to camera space as X_cam = R @ X_world + t,
and depth is the Z-depth of a pixel in camera space (not ray length) -
this is what makes the unprojection below a plain pinhole inverse.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COLMAP model_id -> (model_name, num_params), per src/colmap/scene/camera_model.h
_COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}

DEFAULT_DEPTH_RANGE = (0.05, 30.0)


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsics_matrix(self) -> np.ndarray:
        """Only defined for undistorted models (PINHOLE/SIMPLE_PINHOLE) -
        the expected case for COLMAP after `image_undistorter`, and always
        the case for DA3 (its intrinsics are pinhole by construction)."""
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
        elif self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params
            fx = fy = f
        else:
            raise ValueError(
                f"camera {self.id} has model {self.model} (still has lens distortion). "
                "For COLMAP, run `colmap image_undistorter` first and point this reader "
                "at its output (dense/sparse + dense/images)."
            )
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass
class Pose:
    """World->cam pose: X_cam = R @ X_world + t."""

    id: int
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)
    name: str

    def rotation_matrix(self) -> np.ndarray:
        return self.R

    @property
    def tvec(self) -> np.ndarray:
        return self.t


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def backproject_mask_to_world(
    mask: np.ndarray,
    depth: np.ndarray,
    camera: Camera,
    pose: Pose,
    depth_range: tuple[float, float] = DEFAULT_DEPTH_RANGE,
) -> np.ndarray:
    """Unproject every masked pixel with valid depth to a 3D world point.

    `mask` and `depth` must already be the same resolution (resize the
    mask to the depth map's shape first if the 2D detector ran at a
    different resolution than the depth source).
    """
    if mask.shape != depth.shape:
        raise ValueError(f"mask shape {mask.shape} != depth shape {depth.shape}; resize mask to depth's resolution first")

    valid = mask.astype(bool) & np.isfinite(depth) & (depth > depth_range[0]) & (depth < depth_range[1])
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3))

    K = camera.intrinsics_matrix()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = depth[ys, xs]
    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    points_cam = np.stack([x_cam, y_cam, d], axis=1)

    R, t = pose.rotation_matrix(), pose.tvec
    # X_cam = R @ X_world + t  =>  X_world = R.T @ (X_cam - t)
    return (points_cam - t) @ R


def _unproject_depth_map(depth, K, R, t, valid_mask=None):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    mask = np.isfinite(depth) & (depth > DEFAULT_DEPTH_RANGE[0]) & (depth < DEFAULT_DEPTH_RANGE[1])
    if valid_mask is not None:
        mask &= valid_mask
    ys, xs = np.nonzero(mask)
    d = depth[ys, xs]
    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    points_cam = np.stack([x_cam, y_cam, d], axis=1)
    return (points_cam - t) @ R, ys, xs


# --------------------------------------------------------------------------
# COLMAP source
# --------------------------------------------------------------------------


def _read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        cameras[cam_id] = Camera(id=cam_id, model=model, width=width, height=height,
                                  params=np.array([float(p) for p in parts[4:]]))
    return cameras


def _read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            model_name, num_params = _COLMAP_CAMERA_MODELS[model_id]
            params = np.array(struct.unpack("<" + "d" * num_params, f.read(8 * num_params)))
            cameras[cam_id] = Camera(id=cam_id, model=model_name, width=width, height=height, params=params)
    return cameras


def _read_images_text(path: Path) -> dict[int, tuple[Pose, int]]:
    images = {}
    # each image is exactly 2 lines: a pose line, then a POINTS2D[] line -
    # which is legitimately BLANK for an image with zero tracked 2D points,
    # so only comment lines get dropped here; blindly dropping every blank
    # line (as an earlier version of this did) desyncs the pose/points2D
    # pairing for every image after the first blank points2D line.
    lines = [l for l in path.read_text().splitlines() if not l.strip().startswith("#")]
    for i in range(0, len(lines), 2):  # each image = 2 lines (pose line, then 2D-point line)
        parts = lines[i].split()
        img_id = int(parts[0])
        qvec = np.array([float(p) for p in parts[1:5]])
        tvec = np.array([float(p) for p in parts[5:8]])
        cam_id = int(parts[8])
        images[img_id] = (Pose(id=img_id, R=qvec2rotmat(qvec), t=tvec, name=parts[9]), cam_id)
    return images


def _read_images_binary(path: Path) -> dict[int, tuple[Pose, int]]:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2d)  # skip (x, y, point3D_id) triples
            images[img_id] = (Pose(id=img_id, R=qvec2rotmat(qvec), t=tvec, name=name.decode("utf-8")), cam_id)
    return images


class ColmapSource:
    """Reads an undistorted COLMAP dense workspace: `sparse_dir` (cameras +
    images, text or binary), `images_dir` (matching undistorted photos),
    `depth_maps_dir` (`stereo/depth_maps/<name>.geometric.bin` from
    `colmap patch_match_stereo`), and an optional pre-fused scene point
    cloud (`colmap stereo_fusion` output) for wall/up-vector estimation.
    """

    def __init__(self, sparse_dir: str | Path, images_dir: str | Path,
                 depth_maps_dir: str | Path, scene_ply: str | Path | None = None):
        sparse_dir = Path(sparse_dir)
        if (sparse_dir / "cameras.bin").exists():
            self._cameras = _read_cameras_binary(sparse_dir / "cameras.bin")
            images_with_cam_id = _read_images_binary(sparse_dir / "images.bin")
        elif (sparse_dir / "cameras.txt").exists():
            self._cameras = _read_cameras_text(sparse_dir / "cameras.txt")
            images_with_cam_id = _read_images_text(sparse_dir / "images.txt")
        else:
            raise FileNotFoundError(f"no cameras.bin/.txt found under {sparse_dir}")

        self._poses = {img_id: pose for img_id, (pose, _) in images_with_cam_id.items()}
        self._image_camera_id = {img_id: cam_id for img_id, (_, cam_id) in images_with_cam_id.items()}
        self.images_dir = Path(images_dir)
        self.depth_maps_dir = Path(depth_maps_dir)
        self.scene_ply = Path(scene_ply) if scene_ply else None

    @property
    def view_ids(self) -> list[int]:
        return list(self._poses.keys())

    def name(self, view_id: int) -> str:
        return self._poses[view_id].name

    def id_by_name(self, name: str) -> int:
        for vid, pose in self._poses.items():
            if pose.name == name or Path(pose.name).name == Path(name).name:
                return vid
        raise KeyError(f"image {name!r} not found in COLMAP model")

    def camera(self, view_id: int) -> Camera:
        return self._cameras[self._image_camera_id[view_id]]

    def pose(self, view_id: int) -> Pose:
        return self._poses[view_id]

    def depth(self, view_id: int) -> np.ndarray:
        name = self.name(view_id)
        path = self.depth_maps_dir / f"{name}.geometric.bin"
        if not path.exists():
            path = self.depth_maps_dir / f"{name}.photometric.bin"
        return _read_colmap_depth_array(path)

    def image(self, view_id: int) -> np.ndarray | None:
        import cv2
        path = self.images_dir / self.name(view_id)
        if not path.exists():
            return None
        bgr = cv2.imread(str(path))
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def build_scene_point_cloud(self, max_points: int = 1_500_000) -> tuple[np.ndarray, np.ndarray | None]:
        """Reads the pre-fused dense point cloud (`colmap stereo_fusion`
        output) rather than re-unprojecting every depth map here, since
        COLMAP's fusion already does multi-view consistency filtering that
        a naive per-view unprojection wouldn't."""
        if self.scene_ply is None:
            raise ValueError(
                "ColmapSource has no scene_ply configured - pass the path to "
                "`colmap stereo_fusion`'s fused point cloud to build a scene cloud "
                "for wall/up-vector estimation."
            )
        from plyfile import PlyData
        v = PlyData.read(str(self.scene_ply))["vertex"].data
        points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        colors = (np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
                  if "red" in v.dtype.names else None)
        if len(points) > max_points:
            idx = np.random.RandomState(0).choice(len(points), max_points, replace=False)
            points = points[idx]
            colors = colors[idx] if colors is not None else None
        return points, colors


def _read_colmap_depth_array(path: str | Path) -> np.ndarray:
    """Read a COLMAP dense depth/normal map (ascii header
    `width&height&channels&` then column-major float32 data)."""
    with open(path, "rb") as f:
        header = b""
        while True:
            c = f.read(1)
            if c == b"&" and header.count(b"&") == 2:
                break
            header += c
        width, height, channels = map(int, header.split(b"&"))
        data = np.fromfile(f, dtype=np.float32)
    data = data.reshape(channels, width, height).transpose(2, 1, 0)  # -> (H, W, C)
    return data[:, :, 0] if channels == 1 else data


# --------------------------------------------------------------------------
# DA3 (Depth Anything 3) source
# --------------------------------------------------------------------------


class DA3Source:
    """Reads a DA3 `results.npz`: one feed-forward inference giving
    per-view depth + world->cam pose + pinhole intrinsics + confidence +
    the (resized) source image, all natively at the same resolution and
    in one mutually-consistent coordinate frame (no undistort step needed,
    unlike COLMAP - DA3's intrinsics are pinhole by construction).
    """

    def __init__(self, npz_path: str | Path):
        z = np.load(npz_path)
        self._depth = z["depth"].astype(np.float32)             # (N,H,W)
        extr = z["extrinsics"].astype(np.float64)                # (N,3,4) or (N,4,4), world->cam
        self._R = extr[:, :3, :3]
        self._t = extr[:, :3, 3]
        self._K = z["intrinsics"].astype(np.float64)              # (N,3,3)
        self._conf = z["conf"] if "conf" in z.files else None
        self._image = z["image"] if "image" in z.files else None  # (N,H,W,3) uint8

    @property
    def view_ids(self) -> list[int]:
        return list(range(len(self._depth)))

    def name(self, view_id: int) -> str:
        return f"view_{view_id:04d}"

    def camera(self, view_id: int) -> Camera:
        h, w = self._depth[view_id].shape
        return Camera(id=view_id, model="PINHOLE", width=w, height=h,
                      params=np.array([self._K[view_id, 0, 0], self._K[view_id, 1, 1],
                                       self._K[view_id, 0, 2], self._K[view_id, 1, 2]]))

    def pose(self, view_id: int) -> Pose:
        return Pose(id=view_id, R=self._R[view_id], t=self._t[view_id], name=self.name(view_id))

    def depth(self, view_id: int) -> np.ndarray:
        return self._depth[view_id]

    def image(self, view_id: int) -> np.ndarray | None:
        return None if self._image is None else self._image[view_id]

    def build_scene_point_cloud(
        self, conf_percentile: float = 40, max_points: int = 1_500_000,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Unprojects every view's depth map (confidence-filtered) using
        the exact same pose/depth already loaded above, so the scene cloud
        and every backprojected opening are guaranteed to share one frame -
        no risk of loading a mismatched external point cloud."""
        pts_list, col_list = [], []
        for i in self.view_ids:
            d = self._depth[i]
            conf_mask = None
            if self._conf is not None:
                thr = np.percentile(self._conf[i], conf_percentile)
                conf_mask = self._conf[i] >= thr
            pts, ys, xs = _unproject_depth_map(d, self._K[i], self._R[i], self._t[i], conf_mask)
            pts_list.append(pts)
            if self._image is not None:
                col_list.append(self._image[i][ys, xs])
        points = np.concatenate(pts_list).astype(np.float64)
        colors = np.concatenate(col_list).astype(np.uint8) if col_list else None
        if len(points) > max_points:
            idx = np.random.RandomState(0).choice(len(points), max_points, replace=False)
            points = points[idx]
            colors = colors[idx] if colors is not None else None
        return points, colors
