"""Module 5, Step 2: Distribution Engine.

One reusable sampling function per distribution family, all parameterized
entirely by `FeatureDistributionParams` (Module 1's config) — no statistical
value is ever hardcoded here. `sample()` is the single entry point every
other Module 5 component uses to draw a random behavioural value.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config.schema import FeatureDistributionParams

_MAX_REJECTION_ATTEMPTS = 1000


def sample(rng: np.random.Generator, params: FeatureDistributionParams) -> float:
    """Draw one sample from `params`'s distribution family.

    Non-truncated families have their tails clamped by `clip_min`/`clip_max`
    if given (matching Stage 2's use of clipping for e.g. latency >= 0.5s).
    `truncated_normal` instead *rejection-samples* within
    `[clip_min, clip_max]` (both required — enforced by
    `FeatureDistributionParams`'s own validation), so the returned value's
    distribution is a genuine truncated Gaussian, not a Gaussian with its
    tails piled up at the boundary.
    """

    if params.family == "normal":
        value = float(rng.normal(params.params["mean"], params.params["std"]))
    elif params.family == "gamma":
        value = float(rng.gamma(params.params["shape"], params.params["scale"]))
    elif params.family == "beta":
        value = float(rng.beta(params.params["alpha"], params.params["beta"]))
    elif params.family == "poisson":
        value = float(rng.poisson(params.params["lam"]))
    elif params.family == "truncated_normal":
        return _sample_truncated_normal(
            rng, params.params["mean"], params.params["std"], params.clip_min, params.clip_max
        )
    else:
        raise ValueError(f"unsupported distribution family {params.family!r}")

    if params.clip_min is not None:
        value = max(params.clip_min, value)
    if params.clip_max is not None:
        value = min(params.clip_max, value)
    return value


def _sample_truncated_normal(
    rng: np.random.Generator, mean: float, std: float, lo: float | None, hi: float | None
) -> float:
    """Rejection-sample `Normal(mean, std)` within `[lo, hi]`.

    Falls back to a hard clamp of one final draw if `_MAX_REJECTION_ATTEMPTS`
    is exhausted (only plausible for pathologically tight bounds relative to
    `std`) so sampling always terminates.
    """

    assert lo is not None and hi is not None  # enforced by config validation
    for _ in range(_MAX_REJECTION_ATTEMPTS):
        value = rng.normal(mean, std)
        if lo <= value <= hi:
            return float(value)
    return float(min(max(rng.normal(mean, std), lo), hi))
