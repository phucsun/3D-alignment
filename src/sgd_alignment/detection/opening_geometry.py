"""Shared geometry helpers for turning a raw 3D point cluster (however it was
obtained - CloudCompare hand selection, or 2D-mask backprojection from
multi-view photos) into a `Detection3D`, given the wall planes of the scene
it sits in.

Used by `manual_segmentation.py` (CloudCompare workflow) and
`multiview_segmentation.py` (COLMAP/DA3 photo workflow), so the wall-normal
orientation logic and per-opening measurement logic only exist once.
"""
from __future__ import annotations

import numpy as np

from sgd_alignment.common.types import Detection3D, PointCloud


def orient_walls_outward(
    walls,
    pc: PointCloud,
    is_outdoor: bool,
    margin: float = 0.10,
    camera_positions: np.ndarray | None = None,
    low_confidence_threshold: float = 0.15,
) -> dict[int, np.ndarray]:
    """Orient each wall's normal outward, computed once per wall (not per
    opening) so every opening on the same physical wall gets the same sign.

    Two heuristics were tried and rejected before this one: PCA on each
    opening's own small point cluster (sign is arbitrary, no consistency
    across openings at all) and "away from the whole scene's centroid" (the
    indoor scene's overall shape and the outdoor scene's overall shape are
    different physical surfaces - interior vs exterior faces - not a
    rotated copy of each other, so their centroids don't relate the way
    this heuristic needs).

    Point density asymmetry across the wall gets the direction right, but
    the RULE FLIPS depending on which side of the wall the scan was taken
    from - this was the actual bug behind a confirmed, systematic ~180
    degree normal mismatch between every indoor/outdoor pair tested:
      - an INDOOR scan sits inside the room, so the room itself (dense,
        richly captured) is on the *interior* side, and the true exterior
        is barely seen at all -> exterior = the side with FEWER points.
      - an OUTDOOR scan sits outside (e.g. in a corridor), so *that*
        space is what gets densely captured, and the room is only
        glimpsed sparsely through the opening -> exterior = the side
        with MORE points, the opposite rule.
    Applying the "fewer points" rule unconditionally to both silently gets
    every outdoor wall's normal backwards.

    Confirmed on real data (`server`) that density asymmetry itself can be
    close to a coin flip for a specific wall even when it's obviously
    correct for every other wall in the same scene: the dominant outdoor
    corridor wall (2.6M of ~3M points) split 0.30/0.23 between its two
    sides - `frac_pos <= frac_neg` still picks *a* side, silently, with no
    signal that the decision was barely better than random, and getting it
    wrong there flips every opening on that wall (server had 2 of its
    matched doors on exactly this wall) to the wrong hemisphere - the
    scenes end up merged onto the same side instead of facing each other,
    even though width/height/scale/residual all still look fine.

    `camera_positions` (optional, `(N, 3)`): when available (a DA3/COLMAP
    source with recorded poses), which side of the wall the CAPTURING
    DEVICE was actually on is a hard geometric fact, not a statistical
    count - the person scanning indoor was inescapably standing inside the
    room, so "outward" is unambiguously the side their camera positions are
    *not* on; the person scanning outdoor was standing in the exterior
    space, so "outward" is the side their camera positions *are* on (same
    indoor/outdoor rule-flip as the density heuristic, just built on a
    signal that can't be a near-tie the way point counts can - the camera
    was on one side or the other, period). Used as the primary signal
    whenever provided; density is still computed and cross-checked against
    it, and a mismatch on a wall the camera signal is confident about is
    logged as a low-confidence resolution in the returned diagnostics
    rather than silently overridden without record.

    Returns a plain `dict[int, np.ndarray]` (existing callers unaffected)
    with one addition: a `.low_confidence` attribute listing wall indices
    whose orientation decision was unreliable (density near-tie with no
    camera signal to break it, or density/camera disagreeing) - callers
    that don't check it see no behavior change.
    """
    oriented: dict[int, np.ndarray] = {}
    low_confidence: list[int] = []
    camera_centroid = camera_positions.mean(axis=0) if camera_positions is not None else None

    for idx, wall in enumerate(walls):
        wall_point = pc.points[wall.inlier_indices].mean(axis=0)
        normal = wall.normal
        signed = pc.points @ normal - np.dot(normal, wall_point)
        frac_pos = float((signed > margin).mean())
        frac_neg = float((signed < -margin).mean())
        density_margin = abs(frac_pos - frac_neg)
        # indoor: outward = fewer points on that side. outdoor: outward =
        # MORE points on that side (see docstring). Either way, decide
        # which side ("+" or "-") is outward.
        density_positive_is_outward = (frac_pos <= frac_neg) if not is_outdoor else (frac_pos > frac_neg)

        if camera_centroid is not None:
            camera_signed = float(np.dot(camera_centroid - wall_point, normal))
            # indoor: outward = away from where the camera was (camera was
            # inside, exterior is the other side). outdoor: outward =
            # toward where the camera was (camera was in the exterior
            # space itself).
            camera_positive_is_outward = (camera_signed <= 0) if not is_outdoor else (camera_signed > 0)
            positive_side_is_outward = camera_positive_is_outward
            if density_margin >= low_confidence_threshold and density_positive_is_outward != camera_positive_is_outward:
                low_confidence.append(idx)
        else:
            positive_side_is_outward = density_positive_is_outward
            if density_margin < low_confidence_threshold:
                low_confidence.append(idx)

        if not positive_side_is_outward:
            normal = -normal
        oriented[idx] = normal

    result = _OrientedNormals(oriented)
    result.low_confidence = low_confidence
    return result


class _OrientedNormals(dict):
    """Plain `dict[int, np.ndarray]` with an extra `.low_confidence`
    attribute (list of wall indices whose orientation decision was
    unreliable) - existing callers that only ever index/iterate the dict
    are completely unaffected; only code that specifically checks
    `.low_confidence` sees the diagnostic."""

    low_confidence: list[int]


def nearest_wall_normal(
    centroid: np.ndarray, walls, oriented_normals: dict[int, np.ndarray], max_distance: float = 0.15
) -> np.ndarray | None:
    """Wall normal of whichever detected wall plane is closest to `centroid`,
    or `None` if even the nearest one is further than `max_distance` away -
    letting the caller (`points_to_detection`) fall back to estimating this
    opening's own normal from its point cluster via PCA instead of silently
    attributing it to a wall it likely isn't actually on.

    Confirmed on real data (`home_phuc`): an under-covered scan left the
    real wall behind 2 of 3 outdoor doors undetected by RANSAC entirely: the
    "nearest" wall found was still 0.39-0.40m away (vs. 0.01-0.05m for every
    correctly-attributed opening across this project's other real datasets),
    and unconditionally using it produced badly distorted width/height
    measurements. `max_distance=0.15` sits comfortably above every verified-
    correct case and below that failure, without needing per-dataset
    tuning.
    """
    if not walls:
        return None
    best_idx = min(range(len(walls)), key=lambda i: abs(walls[i].signed_distance(centroid[None, :])[0]))
    if abs(walls[best_idx].signed_distance(centroid[None, :])[0]) > max_distance:
        return None
    return oriented_normals[best_idx]


def opening_normal_line(points: np.ndarray, up: np.ndarray, verticality_max_dot: float = 0.34) -> tuple[np.ndarray | None, bool]:
    """Up-free, sign-invariant plane normal fit directly to ONE opening's
    own raw point cluster - no dependency on any whole-scene wall-plane
    detection. `n` and `-n` are treated as identical (a plane's normal
    has no inherent sign from a bare PCA fit); a caller that needs a
    correctly-ORIENTED (outward-pointing) normal must resolve the sign
    itself (`aggregate_oriented_wall_normal` below does this via an
    already-trusted reference direction - never assume this function's
    own sign convention means anything).

    Returns `(normal, reliable)`. `reliable=False` (normal still
    returned, just flagged) when the cluster's own normal isn't
    reasonably vertical (`|dot(normal, up)| >= verticality_max_dot`,
    ~20 degrees off a true wall) - a cluster that fit a near-horizontal
    plane isn't sitting flush on a wall, so its normal carries no
    trustworthy wall-orientation information. Also `None`/unreliable for
    a cluster too small to fit a plane meaningfully.
    """
    if len(points) < 10:
        return None, False
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    reliable = abs(float(np.dot(normal, up))) < verticality_max_dot
    return normal, reliable


def aggregate_wall_normal(clusters: list[np.ndarray], up: np.ndarray, max_resid_deg: float = 8.0) -> tuple[np.ndarray | None, bool]:
    """Aggregate several openings' own per-cluster normals
    (`opening_normal_line`) into ONE stable, sign-invariant wall-normal
    estimate, via a sign-invariant scatter matrix (n and -n contribute
    identically, so openings on the exact same wall never cancel each
    other out just because their individual PCA fits happened to land on
    opposite signs).

    More robust for unusual/decorative architecture than reading a normal
    off a whole-scene RANSAC wall-plane list, because it never needs to
    decide WHICH detected wall plane an opening belongs to in the first
    place (confirmed on real data - `chua_thay` - where a window-reveal's
    inner/outer faces fragment whole-scene wall detection into multiple
    close, inconsistent candidate planes; this aggregation sidesteps that
    entirely by working from each opening's own immediate points).

    Returns `(normal, reliable)` - `normal`'s sign is ARBITRARY (see
    `opening_normal_line`); `reliable=False` when no cluster passed
    `opening_normal_line`'s verticality check, or the surviving
    per-cluster normals don't agree with each other well (median angular
    residual to the aggregate >= `max_resid_deg`) - the latter usually
    means the given openings aren't actually all on the same physical
    wall, so aggregating them is not meaningful.
    """
    normals = []
    for points in clusters:
        normal, reliable = opening_normal_line(points, up)
        if normal is not None and reliable:
            normals.append(normal)
    if not normals:
        return None, False
    scatter = sum(np.outer(n, n) for n in normals)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    aggregate = eigenvectors[:, -1]
    residuals_deg = [
        float(np.degrees(np.arccos(np.clip(abs(float(np.dot(n, aggregate))), 0.0, 1.0))))
        for n in normals
    ]
    reliable = float(np.median(residuals_deg)) < max_resid_deg
    return aggregate, reliable


def aggregate_oriented_wall_normal(
    clusters: list[np.ndarray], reference_normals: list[np.ndarray], up: np.ndarray, max_resid_deg: float = 8.0
) -> tuple[np.ndarray | None, bool]:
    """Same as `aggregate_wall_normal`, but SIGN-RESOLVED: the raw
    aggregate is sign-invariant (see that function), so it's flipped to
    agree with the mean of `reference_normals` (e.g. the openings' own
    already-reliably-oriented `Detection3D.normal`, from
    `orient_walls_outward`) whenever they disagree.

    Any caller that cares about ANTIPARALLEL vs PARALLEL (not just "same
    line") - e.g. checking 2 matched walls face directly away from each
    other - must use this, not the bare sign-invariant version: using an
    arbitrarily-signed normal to decide "should I rotate 0 degrees or 180
    degrees" is a real bug this project hit once already (a ~180-degree
    misfire that blew up alignment residuals into the thousands on real
    data) - see CONTRIBUTIONS.md.
    """
    aggregate, reliable = aggregate_wall_normal(clusters, up, max_resid_deg)
    if aggregate is None:
        return None, False
    reference = np.mean(reference_normals, axis=0)
    if np.dot(aggregate, reference) < 0:
        aggregate = -aggregate
    return aggregate, reliable


def points_to_detection(
    points: np.ndarray,
    category: str,
    up: np.ndarray,
    wall_normal: np.ndarray | None,
    trim_percentile: float = 0.0,
    wall_normal_agreement_threshold: float = 0.85,
) -> Detection3D:
    """Measure width/height of a raw 3D point cluster belonging to one
    opening, in a wall-aligned (u, v) frame, using the *actual detected
    wall's* normal (already reliably outward-oriented per-wall, see
    `orient_walls_outward`) rather than re-deriving orientation from this
    small cluster alone - EXCEPT when that wall's normal doesn't actually
    agree with the cluster's own shape (see `wall_normal_agreement_threshold`
    below), since being spatially near a wall doesn't guarantee the opening
    sits flush on it.

    `wall_normal_agreement_threshold=0.85`: `nearest_wall_normal` only
    checks that the closest detected wall plane is *near* the cluster's
    centroid, not that it's actually the *same* surface the opening sits
    on - a real opening set in a recessed or angled nook can be close to a
    nearby larger wall (which wins the whole-scene RANSAC fit on point
    count) while not sharing its plane. Confirmed on real data
    (`Q2_outdoor`): a wall 0.001 m from the door centroid (as close as it
    gets) still had a normal 44 degrees off the door cluster's own PCA
    normal (dot=0.72), producing a badly skewed aspect ratio (0.93 instead
    of the ~0.54 both the cluster's own PCA and the matching indoor
    opening agreed on). When the given `wall_normal` disagrees with the
    cluster's own PCA normal by more than this threshold, the PCA normal
    (sign-aligned to `wall_normal`, so the outward-orientation convention
    from `orient_walls_outward` is preserved) is trusted instead - the
    direct, local measurement over the indirect nearby-wall proxy. Every
    already-verified dataset had wall/PCA normals agreeing far above this
    threshold, so this is a no-op for them (confirmed by regression test,
    not merely assumed).

    `trim_percentile=0.0` (default) measures the exact `min()`/`max()`
    extent - unchanged from before (`np.percentile(x, 0)`/`(x, 100)` equal
    `min()`/`max()` exactly, no interpolation at the boundary, so this is
    a genuine no-op, not an approximation). Set e.g. `2.0` to instead use
    the [2, 98] percentile range: `min()`/`max()` are the least robust
    possible statistic (a *single* outlier point - a mis-segmented mask
    edge pixel from one oblique/noisy view among several merged views -
    directly sets the measured width or height). Only meaningful for the
    multi-view photo pipeline, where multiple independently-noisy views
    get merged into one cluster; CloudCompare hand selections don't need
    it (a human already excluded stray points), so
    `manual_segmentation.py` keeps the default.
    """
    centroid = points.mean(axis=0)

    def _pca_normal() -> np.ndarray:
        _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
        return vt[-1]

    if wall_normal is None:
        # no wall plane was found nearby at all: PCA on the selection
        # itself (orientation may be inconsistent between scenes - see
        # docstring)
        normal = _pca_normal()
    else:
        pca_normal = _pca_normal()
        if abs(float(np.dot(wall_normal, pca_normal))) < wall_normal_agreement_threshold:
            # nearby wall's normal doesn't actually match this cluster's
            # own shape - trust the direct local measurement instead (sign-
            # aligned so the outward convention still roughly holds)
            normal = pca_normal if np.dot(wall_normal, pca_normal) >= 0 else -pca_normal
        else:
            normal = wall_normal

    v_axis = up - np.dot(up, normal) * normal
    v_axis = v_axis / np.linalg.norm(v_axis)
    u_axis = np.cross(v_axis, normal)
    u_axis = u_axis / np.linalg.norm(u_axis)

    # project onto the wall's own plane before measuring extent, so a
    # selection with some depth (frame thickness, or backprojection noise)
    # doesn't inflate width/height
    onto_plane = points - np.outer((points - centroid) @ normal, normal)
    centered = onto_plane - centroid
    u = centered @ u_axis
    v = centered @ v_axis
    u_lo, u_hi = np.percentile(u, [trim_percentile, 100 - trim_percentile])
    v_lo, v_hi = np.percentile(v, [trim_percentile, 100 - trim_percentile])
    width = float(u_hi - u_lo)
    height = float(v_hi - v_lo)
    center = centroid + ((u_hi + u_lo) / 2) * u_axis + ((v_hi + v_lo) / 2) * v_axis

    return Detection3D(
        category=category,
        center=center,
        u_axis=u_axis,
        v_axis=v_axis,
        normal=normal,
        width=width,
        height=height,
    )
