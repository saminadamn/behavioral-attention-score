"""Module 11, Step 2: Intervention Configuration.

Every threshold, weight, and limit the engine uses lives here — nothing is
hardcoded in `detector.py`/`policies.py`/`scorer.py`/`cooldown.py`.
`policy_weights`/`policy_cost_multipliers` reuse the same "weight it to zero
to disable it" convention `RewardConfig.with_category_disabled` established
in Module 10 — an ablation is a one-line config change, not a code change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RankingStrategy = Literal["weighted_score", "gain_only", "cost_adjusted"]


class NeedSignalWeights(BaseModel):
    """Weights combining `detector.py`'s individual need signals into one need score.

    Each signal is independently computed in [0, 1] (how strongly *that*
    signal alone indicates a problem); these weights determine how much
    each one contributes to the overall need score.
    """

    model_config = ConfigDict(frozen=True)

    low_bas: float = Field(ge=0.0, default=0.25)
    rapid_decline: float = Field(ge=0.0, default=0.15)
    persistent_negative_reward: float = Field(ge=0.0, default=0.15)
    high_fatigue: float = Field(ge=0.0, default=0.15)
    low_engagement: float = Field(ge=0.0, default=0.10)
    consecutive_declines: float = Field(ge=0.0, default=0.10)
    low_confidence: float = Field(ge=0.0, default=0.10)

    @model_validator(mode="after")
    def _check_sum(self) -> "NeedSignalWeights":
        total = (
            self.low_bas + self.rapid_decline + self.persistent_negative_reward
            + self.high_fatigue + self.low_engagement + self.consecutive_declines + self.low_confidence
        )
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"NeedSignalWeights must sum to 1.0 (got {total:.4f})")
        return self


class ScoringWeights(BaseModel):
    """Weights combining a policy's gain/cost/confidence/severity into one score."""

    model_config = ConfigDict(frozen=True)

    expected_gain_weight: float = Field(ge=0.0, default=0.45)
    cost_weight: float = Field(ge=0.0, default=0.25)
    confidence_weight: float = Field(ge=0.0, default=0.15)
    severity_weight: float = Field(ge=0.0, default=0.15)

    @model_validator(mode="after")
    def _check_sum(self) -> "ScoringWeights":
        total = self.expected_gain_weight + self.cost_weight + self.confidence_weight + self.severity_weight
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"ScoringWeights must sum to 1.0 (got {total:.4f})")
        return self


class InterventionConfig(BaseModel):
    """Complete, deterministic configuration for one intervention-planning run."""

    model_config = ConfigDict(frozen=True)

    # Need-detection thresholds
    min_bas: float = Field(ge=0.0, le=1.0, default=0.45)
    min_reward: float = Field(ge=-1.0, le=1.0, default=-0.1)
    max_fatigue: float = Field(ge=0.0, le=1.0, default=0.6)
    min_confidence: float = Field(ge=0.0, le=1.0, default=0.4)
    min_engagement: float = Field(ge=0.0, le=1.0, default=0.4)

    # Policy-eligibility thresholds (beyond the five above)
    min_correctness: float = Field(ge=0.0, le=1.0, default=0.5)
    min_semantic_similarity: float = Field(ge=0.0, le=1.0, default=0.4)
    min_difficulty_for_reduction: float = Field(ge=0.0, le=1.0, default=0.6)

    # Reference scales for normalizing raw deltas into [0,1] need signals
    bas_decline_reference: float = Field(gt=0.0, default=0.3)
    reward_decline_reference: float = Field(gt=0.0, default=0.3)
    consecutive_decline_threshold: int = Field(gt=0, default=2)

    need_threshold: float = Field(ge=0.0, le=1.0, default=0.35)
    severity_high_threshold: float = Field(ge=0.0, le=1.0, default=0.66)
    severity_medium_threshold: float = Field(ge=0.0, le=1.0, default=0.33)
    need_signal_weights: NeedSignalWeights = Field(default_factory=NeedSignalWeights)

    # Cooldown / pacing
    cooldown_length: int = Field(ge=0, default=3)
    max_interventions_per_session: int = Field(gt=0, default=5)
    min_interactions_before_intervention: int = Field(ge=1, default=2)
    duplicate_prevention_window: int = Field(ge=0, default=5)

    # Policy scoring
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    policy_weights: dict[str, float] = Field(default_factory=dict)
    policy_cost_multipliers: dict[str, float] = Field(default_factory=dict)
    ranking_strategy: RankingStrategy = "weighted_score"

    # Confidence
    confidence_bas_weight: float = Field(ge=0.0, le=1.0, default=0.3)
    confidence_reward_weight: float = Field(ge=0.0, le=1.0, default=0.3)

    version: str = "1.0.0"

    @model_validator(mode="after")
    def _check_severity_thresholds(self) -> "InterventionConfig":
        if self.severity_medium_threshold >= self.severity_high_threshold:
            raise ValueError(
                "severity_medium_threshold must be < severity_high_threshold "
                f"(got {self.severity_medium_threshold} >= {self.severity_high_threshold})"
            )
        return self

    def policy_weight(self, policy_name: str) -> float:
        """`policy_weights[policy_name]`, defaulting to 1.0 (not configured = full weight)."""

        return self.policy_weights.get(policy_name, 1.0)

    def policy_cost_multiplier(self, policy_name: str) -> float:
        """`policy_cost_multipliers[policy_name]`, defaulting to 1.0."""

        return self.policy_cost_multipliers.get(policy_name, 1.0)

    def with_policy_disabled(self, policy_name: str) -> "InterventionConfig":
        """An ablation helper: a new `InterventionConfig` with `policy_name` zero-weighted."""

        updated_weights = dict(self.policy_weights)
        updated_weights[policy_name] = 0.0
        return InterventionConfig(**{**self.model_dump(), "policy_weights": updated_weights})


def default_intervention_config() -> InterventionConfig:
    """Reference configuration with every threshold at a defensible default."""

    return InterventionConfig()
