"""Whole-scene door/window segmentation using a pretrained PointNet++ (S3DIS).

Runs the S3DIS-pretrained `pointnet2_sem_seg` checkpoint (vendored under
`third_party/pointnet2/`, from yanx27/Pointnet_Pointnet2_pytorch) directly
on our own point cloud, sidestepping the 2D-projection approach entirely -
useful when the cloud has corridor points overlapping the room (occluding
walls in any 2D projection), since this model reasons in 3D per-point.

The point cloud must already be upright-aligned (world +Z vertical), and
does not need real RGB - S3DIS's blocks are (x,y,z,r,g,b) but the model is
just a PointNet++ backbone, so lacking real color mainly costs accuracy on
color-dependent classes (e.g. board vs wall), not geometry-driven ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sgd_alignment.common.types import PointCloud
from third_party.pointnet2.pointnet2_sem_seg import get_model

S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter",
]
DOOR_LABEL = S3DIS_CLASSES.index("door")
WINDOW_LABEL = S3DIS_CLASSES.index("window")

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "pointnet2_sem_seg_s3dis.pth"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT, device: torch.device | None = None) -> tuple[torch.nn.Module, torch.device]:
    device = device or _pick_device()
    model = get_model(len(S3DIS_CLASSES)).to(device)
    # trusted checkpoint (vendored from the official repo), old pickle
    # format containing numpy scalars alongside tensors
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def _pseudo_rgb(pc: PointCloud) -> np.ndarray:
    """Fabricate an (N, 3) RGB-like array in [0, 255] from intensity.

    S3DIS blocks expect real RGB; we only have grayscale intensity (or
    nothing), so it's replicated across channels as the closest available
    stand-in for per-point texture.
    """
    if pc.intensity is not None:
        v = pc.intensity
        v = 255.0 * (v - v.min()) / (v.max() - v.min() + 1e-9)
    else:
        v = np.full(len(pc), 128.0)
    return np.stack([v, v, v], axis=1)


def _verticality_pseudo_rgb(pc: PointCloud, up: np.ndarray, knn: int = 30) -> np.ndarray:
    """Fabricate an (N, 3) RGB-like array from local surface verticality.

    Used in place of `_pseudo_rgb` when there is no intensity at all (a
    constant gray carries zero information for the model): each point's
    local normal (via per-point PCA over a k-nearest-neighborhood) is
    compared to the up direction. Floor/ceiling (normal parallel to up),
    walls (normal perpendicular to up) and clutter (irregular normals)
    each get a distinct, non-constant signal in the RGB slot instead of a
    flat value the model can't use for anything.
    """
    import open3d as o3d

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc.points)
    o3d_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn))
    normals = np.asarray(o3d_pc.normals)
    verticality = np.abs(normals @ up) * 255.0
    return np.stack([verticality, verticality, verticality], axis=1)


def segment_whole_scene(
    pc: PointCloud,
    model: torch.nn.Module,
    device: torch.device,
    block_size: float = 1.0,
    stride: float = 0.5,
    block_points: int = 4096,
    padding: float = 0.001,
    num_votes: int = 3,
    batch_size: int = 32,
    seed: int = 0,
    up: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Predict a S3DIS class label for every point in `pc`.

    `rgb` (optional, `(N, 3)` in `[0, 255]`): real per-point color, when
    available - S3DIS's own inputs are real RGB, so this is what the
    model actually expects and should give the fairest accuracy read.
    Takes priority over `pc.intensity`/verticality pseudo-color below.

    If neither `rgb` nor `pc.intensity` is given, `up` must be provided
    so a verticality-based pseudo-RGB can be used instead of a flat,
    uninformative gray (see `_verticality_pseudo_rgb`).

    Reimplements the block partitioning from the original repo's
    `ScannetDatasetWholeScene` (1m x 1m columns spanning full room height,
    50% stride, per-block 4096-point resampling), applied directly to a
    single in-memory point cloud instead of a pre-baked .npy dataset.
    """
    rng = np.random.default_rng(seed)
    # S3DIS preprocessing shifts each room so its most-negative point sits
    # at the origin before block normalization (see the original repo's
    # collect_indoor3d_data.py) - skipping this makes coord_max/coord_min
    # inconsistent with what the model was trained on (normalized_xyz can
    # come out negative or off-scale), which is what caused the earlier
    # nonsense predictions (e.g. "floor" for most of the room).
    xyz = pc.points - pc.points.min(axis=0)
    if rgb is not None:
        pass
    elif pc.intensity is not None:
        rgb = _pseudo_rgb(pc)
    else:
        if up is None:
            raise ValueError("no rgb/pc.intensity given; pass `up` to use verticality-based pseudo-RGB")
        rgb = _verticality_pseudo_rgb(pc, up)
    points = np.concatenate([xyz, rgb], axis=1)  # (N, 6)

    coord_min = xyz.min(axis=0)
    coord_max = xyz.max(axis=0)
    grid_x = int(np.ceil((coord_max[0] - coord_min[0] - block_size) / stride)) + 1
    grid_y = int(np.ceil((coord_max[1] - coord_min[1] - block_size) / stride)) + 1

    vote_pool = np.zeros((len(pc), len(S3DIS_CLASSES)), dtype=np.int64)

    for _vote in range(num_votes):
        block_feats = []
        block_point_idx = []

        for iy in range(grid_y):
            for ix in range(grid_x):
                s_x = coord_min[0] + ix * stride
                e_x = min(s_x + block_size, coord_max[0])
                s_x = e_x - block_size
                s_y = coord_min[1] + iy * stride
                e_y = min(s_y + block_size, coord_max[1])
                s_y = e_y - block_size

                mask = (
                    (points[:, 0] >= s_x - padding) & (points[:, 0] <= e_x + padding)
                    & (points[:, 1] >= s_y - padding) & (points[:, 1] <= e_y + padding)
                )
                point_idxs = np.where(mask)[0]
                if point_idxs.size == 0:
                    continue

                num_batch = int(np.ceil(point_idxs.size / block_points))
                point_size = num_batch * block_points
                n_extra = point_size - point_idxs.size
                replace = n_extra > point_idxs.size
                extra = rng.choice(point_idxs, n_extra, replace=replace) if n_extra > 0 else np.array([], dtype=int)
                block_idx = np.concatenate([point_idxs, extra])
                rng.shuffle(block_idx)

                data = points[block_idx].copy()
                normalized_xyz = data[:, :3] / coord_max[:3]
                data[:, 0] -= s_x + block_size / 2.0
                data[:, 1] -= s_y + block_size / 2.0
                data[:, 3:6] /= 255.0
                data9 = np.concatenate([data, normalized_xyz], axis=1)  # (point_size, 9)

                block_feats.append(data9.reshape(-1, block_points, 9))
                block_point_idx.append(block_idx.reshape(-1, block_points))

        all_feats = np.concatenate(block_feats, axis=0)
        all_idx = np.concatenate(block_point_idx, axis=0)

        with torch.no_grad():
            for start in range(0, len(all_feats), batch_size):
                batch = all_feats[start:start + batch_size]
                idx_batch = all_idx[start:start + batch_size]
                tensor = torch.from_numpy(batch).float().to(device).transpose(2, 1)
                pred, _ = model(tensor)
                pred_label = pred.argmax(dim=2).cpu().numpy()
                for b in range(pred_label.shape[0]):
                    np.add.at(vote_pool, (idx_batch[b], pred_label[b]), 1)

    return vote_pool.argmax(axis=1)
