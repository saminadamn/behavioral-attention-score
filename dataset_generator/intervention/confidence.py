"""Module 11, Step 9: Decision Confidence.

Blends policy agreement (how much the top candidate outscores the runner-up),
signal coverage (how many observation fields were actually available versus
optional/defaulted), and the observation's own `bas_confidence`/
`reward_confidence` — reused directly, never recomputed, per the module's
"never recompute BAS/Reward" constraint.
"""

from __future__ import annotations

from dataset_generator.intervention.config import InterventionConfig
from dataset_generator.intervention.models import InterventionCandidate, InterventionObservation

ReliabilityLabel = str


def _policy_agreement(candidates: list[InterventionCandidate]) -> float:
    """1.0 when the top candidate clearly dominates, 0.0 when scores are tied."""

    if len(candidates) < 2:
        return 1.0
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    top, runner_up = ranked[0].score, ranked[1].score
    gap = top - runner_up
    scale = max(abs(top), abs(runner_up), 1.0)
    return max(0.0, min(1.0, gap / scale))


def _signal_coverage(observation: InterventionObservation) -> float:
    """Fraction of optional signals that were actually available (not None)."""

    optional_fields = [observation.previous_bas, observation.bas_trend, observation.reward_trend,
                        observation.classifier_confidence]
    available = sum(1 for value in optional_fields if value is not None)
    return available / len(optional_fields)


class InterventionConfidenceEstimator:
    """Computes confidence/uncertainty/reliability for one `InterventionDecision`."""

    def __init__(self, config: InterventionConfig) -> None:
        self._config = config

    def estimate(
        self, observation: InterventionObservation, candidates: list[InterventionCandidate]
    ) -> tuple[float, float, ReliabilityLabel]:
        """Return `(confidence, uncertainty, reliability)`."""

        config = self._config
        agreement = _policy_agreement(candidates)
        coverage = _signal_coverage(observation)
        remaining_weight = max(
            0.0, 1.0 - config.confidence_bas_weight - config.confidence_reward_weight
        )

        confidence = (
            config.confidence_bas_weight * observation.bas_confidence
            + config.confidence_reward_weight * observation.reward_confidence
            + remaining_weight * ((agreement + coverage) / 2.0)
        )
        confidence = max(0.0, min(1.0, confidence))
        uncertainty = 1.0 - confidence

        if confidence >= 0.7:
            reliability = "high"
        elif confidence >= 0.4:
            reliability = "medium"
        else:
            reliability = "low"

        return confidence, uncertainty, reliability
