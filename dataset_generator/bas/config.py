"""Module 9, Step 2: BAS Configuration.

`default_bas_config()` is the one place the default feature set, weights,
and normalization ranges are decided — every number here is justified in
its own comment, per "no magic constants." The default weighted set
deliberately excludes several of Step 3's candidate features to avoid
double-counting (see this module's inline notes and the package's design
summary): `response_engagement_proxy` and `behaviour_rolling_engagement`
are themselves composites of other features already weighted separately.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NormalizationStrategy(str, Enum):
    IDENTITY = "identity"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    CLIP = "clip"


class SmoothingStrategy(str, Enum):
    IDENTITY = "identity"
    EMA = "ema"
    ROLLING_AVERAGE = "rolling_average"


class FeaturePolarity(str, Enum):
    """How a feature's normalized value maps to evidence of attention.

    `NEUTRAL` features are extracted and reported (see `BASObservation`)
    but never enter the weighted score — used for context fields
    (`session_progress`) and fields that would double-count another
    weighted feature (`rolling_engagement`).
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FeatureNormalizationConfig(BaseModel):
    """Normalization parameters for one feature.

    Reference ranges (`min_value`/`max_value`/`mean`/`std`) are fixed,
    configured constants, not fit from whatever dataset is passed to
    `BASEngine.compute()` — a score computed on one dataset must mean the
    same thing on another, which a data-fit range would break.
    """

    model_config = ConfigDict(frozen=True)

    strategy: NormalizationStrategy
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    std: float | None = None
    clip_min: float = 0.0
    clip_max: float = 1.0

    @model_validator(mode="after")
    def _check(self) -> "FeatureNormalizationConfig":
        if self.strategy == NormalizationStrategy.MIN_MAX:
            if self.min_value is None or self.max_value is None:
                raise ValueError("min_max normalization requires min_value and max_value")
            if self.min_value >= self.max_value:
                raise ValueError("min_value must be < max_value")
        if self.strategy == NormalizationStrategy.Z_SCORE:
            if self.mean is None or self.std is None:
                raise ValueError("z_score normalization requires mean and std")
            if self.std <= 0:
                raise ValueError("std must be > 0")
        if self.clip_min >= self.clip_max:
            raise ValueError("clip_min must be < clip_max")
        return self


class FeatureBASConfig(BaseModel):
    """Per-feature configuration: how much it counts, which direction, how it's normalized."""

    model_config = ConfigDict(frozen=True)

    weight: float = Field(ge=0.0)
    polarity: FeaturePolarity
    normalization: FeatureNormalizationConfig


class BASConfig(BaseModel):
    """Complete, deterministic configuration for one BAS computation run."""

    model_config = ConfigDict(frozen=True)

    feature_configs: dict[str, FeatureBASConfig]

    smoothing_strategy: SmoothingStrategy = SmoothingStrategy.EMA
    ema_alpha: float = Field(gt=0.0, le=1.0, default=0.3)
    rolling_window: int = Field(gt=0, default=5)

    confidence_variance_weight: float = Field(ge=0.0, default=0.5)
    confidence_classifier_weight: float = Field(ge=0.0, le=1.0, default=0.3)

    score_clip_min: float = Field(default=0.0)
    score_clip_max: float = Field(default=1.0)

    attention_threshold: float = Field(ge=0.0, le=1.0, default=0.5)

    version: str = "1.0.0"

    @model_validator(mode="after")
    def _check(self) -> "BASConfig":
        if not self.feature_configs:
            raise ValueError("feature_configs must not be empty")
        if self.score_clip_min >= self.score_clip_max:
            raise ValueError("score_clip_min must be < score_clip_max")
        return self

    def weighted_features(self) -> list[str]:
        """Feature names with nonzero weight and non-neutral polarity — the ones actually scored."""

        return [
            name
            for name, cfg in self.feature_configs.items()
            if cfg.weight > 0 and cfg.polarity != FeaturePolarity.NEUTRAL
        ]


def default_bas_config() -> BASConfig:
    """Reference BAS configuration.

    Ten actively-weighted features summing to 1.0:
    - correctness (0.20): the strongest direct signal of on-task success.
    - semantic_similarity (0.15): how on-topic the response is.
    - coherence (0.10): vocabulary richness / non-repetition of the response.
    - lexical_diversity (0.05): overlaps partly with coherence but captures
      vocabulary breadth specifically; kept small to avoid over-weighting
      the same underlying signal twice.
    - confidence (0.05): weak signal — Impulsive responses can be falsely
      confident (Module 4), so this is deliberately low-weight, not a
      strong positive indicator on its own.
    - hesitation (0.10), topic_shift (0.10), repetition_ratio (0.05),
      fatigue (0.10), abs_normalized_latency (0.10): negative-polarity
      signals of disengagement/distraction.

    Five neutral (tracked, unweighted) fields: `response_latency` (direction
    ambiguous — superseded by `abs_normalized_latency`), `rolling_latency`,
    `rolling_engagement` and `response_engagement_proxy`-derived signals
    (would double-count already-weighted features), and `session_progress`
    (context, not evidence). `classifier_confidence` is present but
    zero-weight by default — opt-in, since it requires a `TrainingArtifact`.
    """

    return BASConfig(
        feature_configs={
            "correctness": FeatureBASConfig(
                weight=0.20, polarity=FeaturePolarity.POSITIVE,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "semantic_similarity": FeatureBASConfig(
                weight=0.15, polarity=FeaturePolarity.POSITIVE,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "coherence": FeatureBASConfig(
                weight=0.10, polarity=FeaturePolarity.POSITIVE,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "lexical_diversity": FeatureBASConfig(
                weight=0.05, polarity=FeaturePolarity.POSITIVE,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "confidence": FeatureBASConfig(
                weight=0.05, polarity=FeaturePolarity.POSITIVE,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "hesitation": FeatureBASConfig(
                weight=0.10, polarity=FeaturePolarity.NEGATIVE,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=0.0, max_value=5.0
                ),
            ),
            "topic_shift": FeatureBASConfig(
                weight=0.10, polarity=FeaturePolarity.NEGATIVE,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "repetition_ratio": FeatureBASConfig(
                weight=0.05, polarity=FeaturePolarity.NEGATIVE,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "fatigue": FeatureBASConfig(
                weight=0.10, polarity=FeaturePolarity.NEGATIVE,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "abs_normalized_latency": FeatureBASConfig(
                weight=0.10, polarity=FeaturePolarity.NEGATIVE,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=0.0, max_value=3.0
                ),
            ),
            # Neutral / tracked-only features:
            "response_latency": FeatureBASConfig(
                weight=0.0, polarity=FeaturePolarity.NEUTRAL,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=0.0, max_value=60.0
                ),
            ),
            "rolling_latency": FeatureBASConfig(
                weight=0.0, polarity=FeaturePolarity.NEUTRAL,
                normalization=FeatureNormalizationConfig(
                    strategy=NormalizationStrategy.MIN_MAX, min_value=0.0, max_value=60.0
                ),
            ),
            "rolling_engagement": FeatureBASConfig(
                weight=0.0, polarity=FeaturePolarity.NEUTRAL,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "session_progress": FeatureBASConfig(
                weight=0.0, polarity=FeaturePolarity.NEUTRAL,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
            "classifier_confidence": FeatureBASConfig(
                weight=0.0, polarity=FeaturePolarity.NEUTRAL,
                normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY),
            ),
        },
    )
