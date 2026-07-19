"""Module 10, Step 2: Reward Configuration.

Reuses `bas.config.FeatureNormalizationConfig` directly for signal
normalization — the same primitive Module 9 built, applied to reward's
delta signals instead of raw features (see this package's design-decision
summary: "never duplicate BAS computation" enforced at the code level).
`default_reward_config()` is the one place default weights are decided,
each justified inline.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataset_generator.bas.config import FeatureNormalizationConfig, NormalizationStrategy


class RewardSignalPolarity(str, Enum):
    """How a signal's normalized delta maps to signed reward evidence.

    `PENALTY` differs from `NEGATIVE`: a negative-polarity signal's
    *magnitude* matters (a bigger fatigue increase is a bigger penalty), but
    a penalty signal (`intervention_cost`) is a fixed cost applied whenever
    a boolean condition holds, regardless of magnitude — there's no
    "how much" for "was an intervention used this interaction".
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    PENALTY = "penalty"
    NEUTRAL = "neutral"


class TemporalMode(str, Enum):
    IMMEDIATE = "immediate"
    DISCOUNTED = "discounted"
    MOVING_AVERAGE = "moving_average"


class RewardCategory(str, Enum):
    """Which named sub-total (`RewardRecord.performance_reward` /
    `.behaviour_reward` / `.cost_reward`) a signal contributes to.

    This is what makes `total_reward = performance + behaviour - cost` an
    explicit decomposition rather than an implicit property of a flat
    weighted sum, and what makes ablation a one-line change
    (`RewardConfig.with_category_disabled(...)`) instead of editing several
    individual signal weights by hand. `CONTEXT` signals (weight 0, neutral
    polarity — `session_progress`, `delta_classifier_confidence` by
    default) are tracked but excluded from all three sub-totals.
    """

    PERFORMANCE = "performance"
    BEHAVIOUR = "behaviour"
    COST = "cost"
    CONTEXT = "context"


class RewardSignalConfig(BaseModel):
    """Per-signal configuration: how much it counts, which direction, how it's
    normalized, and which reward category it belongs to.
    """

    model_config = ConfigDict(frozen=True)

    weight: float = Field(ge=0.0)
    polarity: RewardSignalPolarity
    category: RewardCategory
    normalization: FeatureNormalizationConfig


class RewardConfig(BaseModel):
    """Complete, deterministic configuration for one reward computation run."""

    model_config = ConfigDict(frozen=True)

    signal_configs: dict[str, RewardSignalConfig]

    temporal_mode: TemporalMode = TemporalMode.DISCOUNTED
    discount_factor: float = Field(gt=0.0, le=1.0, default=0.9)
    rolling_window: int = Field(gt=0, default=5)

    confidence_variance_weight: float = Field(ge=0.0, default=0.5)
    confidence_bas_weight: float = Field(ge=0.0, le=1.0, default=0.3)

    reward_clip_min: float = Field(default=-1.0)
    reward_clip_max: float = Field(default=1.0)

    trend_tolerance: float = Field(ge=0.0, default=0.02)
    recovery_threshold: float = Field(default=0.3)

    version: str = "1.0.0"

    @model_validator(mode="after")
    def _check(self) -> "RewardConfig":
        if not self.signal_configs:
            raise ValueError("signal_configs must not be empty")
        if self.reward_clip_min >= self.reward_clip_max:
            raise ValueError("reward_clip_min must be < reward_clip_max")
        return self

    def weighted_signals(self) -> list[str]:
        """Signal names with nonzero weight and non-neutral polarity — the ones actually scored."""

        return [
            name
            for name, cfg in self.signal_configs.items()
            if cfg.weight > 0 and cfg.polarity != RewardSignalPolarity.NEUTRAL
        ]

    def weighted_signals_by_category(self, category: RewardCategory) -> list[str]:
        """Scored signal names (per `weighted_signals`) restricted to `category`."""

        return [name for name in self.weighted_signals() if self.signal_configs[name].category == category]

    def with_category_disabled(self, category: RewardCategory) -> "RewardConfig":
        """An ablation helper: a new `RewardConfig` with every signal in `category`
        zero-weighted, everything else unchanged.

        Re-validates through the real constructor (not `model_copy`, which
        skips validators), so the result is guaranteed consistent.
        """

        updated_signals = {
            name: (
                cfg.model_copy(update={"weight": 0.0}) if cfg.category == category else cfg
            )
            for name, cfg in self.signal_configs.items()
        }
        return RewardConfig(**{**self.model_dump(), "signal_configs": updated_signals})


def default_reward_config() -> RewardConfig:
    """Reference reward configuration, decomposed into three named categories
    (per the project's explicit `R = Performance + Behaviour - Cost` framing):

    **Performance** (delta_bas 0.30, delta_correctness 0.15, delta_confidence
    0.10 — 0.55 total): direct evidence of a successful interaction.
    delta_bas is the primary signal — BAS improvement is directly what this
    whole project is trying to encourage. delta_confidence is weighted
    lower — confidence alone isn't reliable (Module 4's Impulsive strategy
    can be falsely confident).

    **Behaviour** (delta_engagement 0.15, delta_latency_deviation 0.10,
    delta_fatigue 0.10 — 0.35 total): how the student's observable
    behaviour is trending, independent of whether they got the answer
    right. delta_latency_deviation rewards shrinking deviation from the
    student's own baseline latency, not raw latency (direction-ambiguous
    otherwise); delta_fatigue rewards fatigue *not* increasing.

    **Cost** (intervention_cost 0.10): applying an intervention has a real
    cost (teacher/system attention is limited), so even a *successful*
    intervention should net out lower than the same improvement happening
    organically.

    Category weight totals (0.55 / 0.35 / 0.10) are exactly what
    `RewardRecord.performance_reward` / `.behaviour_reward` / `.cost_reward`
    report as named sub-totals, and what `RewardConfig.with_category_disabled`
    zeroes out wholesale for an ablation run.

    `delta_classifier_confidence` and `session_progress` are `CONTEXT`
    (tracked, zero-weight) — classifier confidence is opt-in (requires a
    `TrainingArtifact`), and session progress is context, not evidence.
    """

    return RewardConfig(
        signal_configs={
            "delta_bas": RewardSignalConfig(
                weight=0.30, polarity=RewardSignalPolarity.POSITIVE, category=RewardCategory.PERFORMANCE,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=-0.5, max_value=0.5
                ),
            ),
            "delta_correctness": RewardSignalConfig(
                weight=0.15, polarity=RewardSignalPolarity.POSITIVE, category=RewardCategory.PERFORMANCE,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=-0.5, max_value=0.5
                ),
            ),
            "delta_confidence": RewardSignalConfig(
                weight=0.10, polarity=RewardSignalPolarity.POSITIVE, category=RewardCategory.PERFORMANCE,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=-0.5, max_value=0.5
                ),
            ),
            "delta_engagement": RewardSignalConfig(
                weight=0.15, polarity=RewardSignalPolarity.POSITIVE, category=RewardCategory.BEHAVIOUR,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=-0.5, max_value=0.5
                ),
            ),
            "delta_latency_deviation": RewardSignalConfig(
                weight=0.10, polarity=RewardSignalPolarity.NEGATIVE, category=RewardCategory.BEHAVIOUR,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=-2.0, max_value=2.0
                ),
            ),
            "delta_fatigue": RewardSignalConfig(
                weight=0.10, polarity=RewardSignalPolarity.NEGATIVE, category=RewardCategory.BEHAVIOUR,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=-0.5, max_value=0.5
                ),
            ),
            "intervention_cost": RewardSignalConfig(
                weight=0.10, polarity=RewardSignalPolarity.PENALTY, category=RewardCategory.COST,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            # Context / tracked-only:
            "delta_classifier_confidence": RewardSignalConfig(
                weight=0.0, polarity=RewardSignalPolarity.NEUTRAL, category=RewardCategory.CONTEXT,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=-0.5, max_value=0.5
                ),
            ),
            "session_progress": RewardSignalConfig(
                weight=0.0, polarity=RewardSignalPolarity.NEUTRAL, category=RewardCategory.CONTEXT,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
        },
    )
