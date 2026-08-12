"""Data-driven calibration of the SGD matching weights (Equation 2's lambdas).

The paper does not publish numeric values for the lambda weights, and this
project's own `PairWeights` defaults were hand-picked constants (see
`sgd.py`'s module docstring) - reasonable starting points, never claimed to
be optimal. This module replaces "chosen by hand" with "fit to margin-
separate verified-correct correspondences from real matched datasets".

Key structural fact this exploits: for a fixed pair of neighbor-groups,
`sgd._pair_feature_cost` is a LINEAR function of the weight vector
(`w.distance_weight * d_e + angle_weights . angle_e + rotation_weights .
rot_e`), and `sgdu_distance` is the cost of the cheapest feasible assignment
between two such neighbor-group sets - i.e. a MIN over finitely many linear
functions of the weights, which makes it concave and piecewise-linear in the
weights. This is the same structure margin/ranking methods like LMNN or
structured-SVM training exploit; it is why a small, derivative-free margin
optimizer (not deep learning) is enough here, and why it can work directly
against the real `sgdu_distance` oracle rather than a linearized
approximation of it.

Honesty note: "ground truth" here means correspondences already verified
correct by prior manual inspection / consistently small alignment residuals
on this project's own real datasets (see the project's dataset table) - not
an independently, externally annotated benchmark. Treat calibrated weights
and their reported cross-validation numbers accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from sgd_alignment.common.types import Detection3D
from sgd_alignment.matching.sgd import PairWeights, build_sgds, sgdu_distance

_WEIGHT_FIELDS = [
    "distance_weight",
    "alpha_weight",
    "beta_weight",
    "theta_weight",
    "rotx_weight",
    "roty_weight",
    "rotz_weight",
]


@dataclass
class CalibrationExample:
    """One scene-pair with already-verified-correct correspondences to
    calibrate/evaluate against."""

    name: str
    indoor: list[Detection3D]
    outdoor: list[Detection3D]
    matches: list[tuple[int, int]]  # verified-correct (indoor_idx, outdoor_idx)
    normalize_distance: bool = False
    use_intrinsic_fallback: bool = False


def _weights_from_log_vector(x: np.ndarray, base: PairWeights) -> PairWeights:
    """Optimizing in log-space keeps every weight positive without needing
    bounded optimization - a weight of 0 or below is meaningless here (it
    would let some geometric error term contribute nothing, or flip sign)."""
    values = np.exp(x)
    kwargs = {name: float(v) for name, v in zip(_WEIGHT_FIELDS, values)}
    return replace(base, **kwargs)


def _log_vector_from_weights(w: PairWeights) -> np.ndarray:
    return np.log(np.array([getattr(w, name) for name in _WEIGHT_FIELDS]))


def margin_loss(
    weights: PairWeights,
    examples: list[CalibrationExample],
    margin_target: float = 0.05,
) -> float:
    """Hinge loss over all verified-correct pairs: the SGDU cost of a known-
    correct pair (i*, j*) should be at least `margin_target` below the cost
    of matching i* to every OTHER object in the same outdoor scene. Averaged
    per verified pair (not summed) so a scene with more openings doesn't
    dominate the objective over one with fewer.

    A verified-correct pair that comes out INFEASIBLE under a candidate
    weight vector (e.g. thresholds too tight for that weighting) is
    penalized with a large fixed cost rather than skipped, so the optimizer
    is never rewarded for "solving" the problem by making the right answer
    unreachable.
    """
    total, count = 0.0, 0
    for ex in examples:
        indoor_sgds = build_sgds(ex.indoor, normalize_distance=ex.normalize_distance)
        outdoor_sgds = build_sgds(ex.outdoor, normalize_distance=ex.normalize_distance)
        for i, j in ex.matches:
            cost_correct, _ = sgdu_distance(indoor_sgds[i], outdoor_sgds[j], weights, ex.use_intrinsic_fallback)
            count += 1
            if not np.isfinite(cost_correct):
                total += 10.0
                continue
            neg_costs = [
                sgdu_distance(indoor_sgds[i], outdoor_sgds[k], weights, ex.use_intrinsic_fallback)[0]
                for k in range(len(ex.outdoor))
                if k != j
            ]
            neg_costs = [c for c in neg_costs if np.isfinite(c)]
            if neg_costs:
                margin = min(neg_costs) - cost_correct
                total += max(0.0, margin_target - margin)
    return total / max(count, 1)


def calibrate_weights(
    examples: list[CalibrationExample],
    base_weights: PairWeights | None = None,
    margin_target: float = 0.05,
    maxiter: int = 3000,
) -> PairWeights:
    """Fit the 7 lambda weights by minimizing `margin_loss` with a
    derivative-free optimizer (Nelder-Mead): `sgdu_distance` is piecewise-
    linear/concave but not smooth (the inner Hungarian assignment can switch
    discretely as weights change), so gradient-based optimizers aren't a
    natural fit; Nelder-Mead needs no gradient and is cheap enough here since
    each objective evaluation is just a handful of `sgdu_distance` calls
    (no plane-fitting/RANSAC involved), not a full pipeline run.
    """
    base = base_weights or PairWeights()
    x0 = _log_vector_from_weights(base)
    result = minimize(
        lambda x: margin_loss(_weights_from_log_vector(x, base), examples, margin_target),
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": maxiter, "maxfev": maxiter},
    )
    return _weights_from_log_vector(result.x, base)


def mean_margin_ratio(weights: PairWeights, examples: list[CalibrationExample]) -> float:
    """Evaluation metric (NOT the training objective): mean of
    cost(correct) / cost(best incorrect alternative) over all verified pairs
    - the same ratio `sgd._ambiguity_ratio` computes at match time. Near 0 =
    the correct match is decisively cheaper than any distractor (robust to
    ambiguity/symmetry); near/above 1 = a wrong candidate scores almost as
    well or better. Pairs with no finite alternative to compare against
    (scene has only 1 same-category object) are skipped - there is nothing
    to be ambiguous with.
    """
    ratios = []
    for ex in examples:
        indoor_sgds = build_sgds(ex.indoor, normalize_distance=ex.normalize_distance)
        outdoor_sgds = build_sgds(ex.outdoor, normalize_distance=ex.normalize_distance)
        for i, j in ex.matches:
            cost_correct, _ = sgdu_distance(indoor_sgds[i], outdoor_sgds[j], weights, ex.use_intrinsic_fallback)
            if not np.isfinite(cost_correct):
                ratios.append(float("inf"))
                continue
            neg_costs = [
                sgdu_distance(indoor_sgds[i], outdoor_sgds[k], weights, ex.use_intrinsic_fallback)[0]
                for k in range(len(ex.outdoor))
                if k != j
            ]
            neg_costs = [c for c in neg_costs if np.isfinite(c)]
            if not neg_costs:
                continue
            next_best = min(neg_costs)
            ratios.append(0.0 if next_best <= 1e-12 else cost_correct / next_best)
    finite = [r for r in ratios if np.isfinite(r)]
    return float(np.mean(finite)) if finite else float("nan")


def leave_one_scene_out_cv(
    examples: list[CalibrationExample],
    base_weights: PairWeights | None = None,
    margin_target: float = 0.05,
) -> list[dict]:
    """For each scene, calibrate weights on all OTHER scenes, then evaluate
    `mean_margin_ratio` on the held-out scene under both the hand-picked
    default weights and the calibrated-on-the-rest weights - so every
    reported number is on a scene the calibration never saw.
    """
    base = base_weights or PairWeights()
    reports = []
    for i, held_out in enumerate(examples):
        train = examples[:i] + examples[i + 1:]
        calibrated = calibrate_weights(train, base_weights=base, margin_target=margin_target)
        reports.append(
            {
                "held_out": held_out.name,
                "default_ratio": mean_margin_ratio(base, [held_out]),
                "calibrated_ratio": mean_margin_ratio(calibrated, [held_out]),
                "calibrated_weights": calibrated,
            }
        )
    return reports
