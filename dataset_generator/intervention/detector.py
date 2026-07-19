"""Module 11, Step 4: Need Detection.

Combines seven independent, config-driven need signals into one
`need_score`, using `config.need_signal_weights` (which sum to 1.0 by
construction). A signal that has "no evidence of decline yet" (e.g. no
`bas_trend` on the first interaction of a session) contributes 0.0 — this
is semantically "nothing wrong detected", not a missing value, so no
renormalization is needed here (unlike Reward's aggregator).
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_generator.intervention.config import InterventionConfig
from dataset_generator.intervention.models import InterventionObservation


def _ratio_above(value: float, threshold: float, reference: float) -> float:
    """How far `value` exceeds `threshold`, scaled by `reference`, clipped to [0, 1]."""

    if value <= threshold:
        return 0.0
    return min(1.0, (value - threshold) / reference)


def _ratio_below(value: float, threshold: float, reference: float) -> float:
    """How far `value` falls below `threshold`, scaled by `reference`, clipped to [0, 1]."""

    if value >= threshold:
        return 0.0
    return min(1.0, (threshold - value) / reference)


@dataclass(frozen=True)
class NeedSignalBreakdown:
    """The seven raw signal strengths (each in [0, 1]) before weighting."""

    low_bas: float
    rapid_decline: float
    persistent_negative_reward: float
    high_fatigue: float
    low_engagement: float
    consecutive_declines: float
    low_confidence: float

    def as_dict(self) -> dict[str, float]:
        return {
            "low_bas": self.low_bas,
            "rapid_decline": self.rapid_decline,
            "persistent_negative_reward": self.persistent_negative_reward,
            "high_fatigue": self.high_fatigue,
            "low_engagement": self.low_engagement,
            "consecutive_declines": self.consecutive_declines,
            "low_confidence": self.low_confidence,
        }


@dataclass(frozen=True)
class NeedDetectionResult:
    """The output of `InterventionDetector.detect`."""

    need_score: float
    trigger_reasons: list[str]
    severity: str
    breakdown: NeedSignalBreakdown


class InterventionDetector:
    """Computes need_score/trigger_reasons/severity from an `InterventionObservation`."""

    def __init__(self, config: InterventionConfig) -> None:
        self._config = config

    def detect(self, observation: InterventionObservation) -> NeedDetectionResult:
        """Combine the seven need signals into one `NeedDetectionResult`."""

        config = self._config

        low_bas = _ratio_below(observation.current_bas, config.min_bas, config.min_bas or 1.0)

        rapid_decline = 0.0
        if observation.bas_trend is not None and observation.bas_trend < 0:
            rapid_decline = min(1.0, abs(observation.bas_trend) / config.bas_decline_reference)

        persistent_negative_reward = 0.0
        if observation.current_reward < config.min_reward:
            persistent_negative_reward = min(
                1.0,
                (config.min_reward - observation.current_reward) / config.reward_decline_reference,
            )

        high_fatigue = _ratio_above(
            observation.fatigue, config.max_fatigue, max(1.0 - config.max_fatigue, 1e-6)
        )

        low_engagement = _ratio_below(
            observation.engagement, config.min_engagement, max(config.min_engagement, 1e-6)
        )

        consecutive_declines = 0.0
        if observation.consecutive_decline_count >= config.consecutive_decline_threshold:
            consecutive_declines = min(
                1.0,
                observation.consecutive_decline_count / (2 * config.consecutive_decline_threshold),
            )

        low_confidence = _ratio_below(
            observation.confidence, config.min_confidence, max(config.min_confidence, 1e-6)
        )

        breakdown = NeedSignalBreakdown(
            low_bas=low_bas,
            rapid_decline=rapid_decline,
            persistent_negative_reward=persistent_negative_reward,
            high_fatigue=high_fatigue,
            low_engagement=low_engagement,
            consecutive_declines=consecutive_declines,
            low_confidence=low_confidence,
        )

        weights = config.need_signal_weights
        need_score = (
            weights.low_bas * low_bas
            + weights.rapid_decline * rapid_decline
            + weights.persistent_negative_reward * persistent_negative_reward
            + weights.high_fatigue * high_fatigue
            + weights.low_engagement * low_engagement
            + weights.consecutive_declines * consecutive_declines
            + weights.low_confidence * low_confidence
        )
        need_score = max(0.0, min(1.0, need_score))

        trigger_reasons = [name for name, value in breakdown.as_dict().items() if value > 0.0]

        if need_score >= config.severity_high_threshold:
            severity = "high"
        elif need_score >= config.severity_medium_threshold:
            severity = "medium"
        else:
            severity = "low"

        return NeedDetectionResult(
            need_score=need_score,
            trigger_reasons=trigger_reasons,
            severity=severity,
            breakdown=breakdown,
        )
