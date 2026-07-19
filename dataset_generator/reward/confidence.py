"""Module 10, Step 7: Reward Confidence.

Mirrors `bas.confidence.compute_confidence`'s structure deliberately (same
coverage x agreement formula), but isn't a direct function call: reward
contributions are signed (`[-1, 1]`) rather than BAS's evidence values
(`[0, 1]`), so the variance computation needs its own pass over the
`[-1, 1]` scale. `bas_confidence` — the underlying interaction's own BAS
confidence — is *reused* directly (Module 9 already computed it), not
re-derived, and blended in at `config.confidence_bas_weight`.
"""

from __future__ import annotations

import statistics

from dataset_generator.reward.config import RewardConfig
from dataset_generator.reward.models import RewardContribution


def compute_reward_confidence(
    contributions: list[RewardContribution],
    missing_ratio: float,
    config: RewardConfig,
    bas_confidence: float | None = None,
) -> tuple[float, float, str]:
    """Return `(confidence, uncertainty, reliability_label)` for one interaction."""

    coverage = 1.0 - missing_ratio

    signed_values = [c.signed_evidence for c in contributions]
    variance = statistics.pvariance(signed_values) if len(signed_values) >= 2 else 0.0

    base_confidence = coverage * (1.0 - min(1.0, config.confidence_variance_weight * variance))

    if bas_confidence is not None:
        base_confidence = (
            (1.0 - config.confidence_bas_weight) * base_confidence
            + config.confidence_bas_weight * bas_confidence
        )

    confidence = min(1.0, max(0.0, base_confidence))
    uncertainty = 1.0 - confidence

    if confidence >= 0.75:
        reliability = "high"
    elif confidence >= 0.40:
        reliability = "medium"
    else:
        reliability = "low"

    return confidence, uncertainty, reliability
