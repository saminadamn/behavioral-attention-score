"""Module 9, Step 1: BAS domain models.

`BASArtifact` is the single source of truth this package produces, the same
role `DatasetArtifact` (Module 7) and `TrainingArtifact` (Module 8) play for
theirs. Every model here is plain Pydantic (no non-serializable payloads
like Module 8's wrapped sklearn estimator), so a `BASArtifact` round-trips
through JSON with no special handling — see `serialization.py`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BASObservation(BaseModel):
    """Raw, unnormalized feature values extracted for one interaction.

    `raw_values[feature]` is `None` for a feature that couldn't be observed
    for this interaction (e.g. `classifier_confidence` when no predictor was
    supplied) — missingness is tracked explicitly, never silently defaulted.
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_number: int = Field(gt=0)
    raw_values: dict[str, float | None]


class BASEvidence(BaseModel):
    """Normalized, polarity-adjusted evidence values ready for aggregation.

    Every value in `values` is on the same [0, 1] "more evidence of
    attention" scale regardless of the underlying feature's original
    polarity — `evidence.py` is what performs that adjustment.
    """

    model_config = ConfigDict(frozen=True)

    values: dict[str, float]
    missing_features: list[str]


class BASContribution(BaseModel):
    """One feature's contribution to an aggregated BAS score."""

    model_config = ConfigDict(frozen=True)

    feature: str
    weight: float
    evidence_value: float = Field(ge=0.0, le=1.0)
    contribution: float


class BASRecordMetadata(BaseModel):
    """Provenance for one `BASRecord`."""

    model_config = ConfigDict(frozen=True)

    normalization_strategy: dict[str, str]
    smoothing_strategy: str
    config_fingerprint: str
    config_version: str


class BASRecord(BaseModel):
    """The complete BAS output for one interaction.

    `raw_score` is this interaction's aggregated evidence *before* temporal
    smoothing; `score` (the BAS itself) is after — the field downstream
    consumers should treat as "the" Behavioural Attention Score.
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_number: int = Field(gt=0)

    raw_score: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)

    contributions: list[BASContribution]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    reliability: str
    missing_feature_ratio: float = Field(ge=0.0, le=1.0)

    explanation: str
    top_positive: list[str]
    top_negative: list[str]

    metadata: BASRecordMetadata


class BASSessionSummary(BaseModel):
    """Session-level aggregation of a session's `BASRecord`s (Module 9, Step 10)."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_count: int = Field(gt=0)
    average_bas: float = Field(ge=0.0, le=1.0)
    minimum_bas: float = Field(ge=0.0, le=1.0)
    maximum_bas: float = Field(ge=0.0, le=1.0)
    variance_bas: float = Field(ge=0.0)
    attention_trend: str
    largest_drop: float = Field(ge=0.0, le=1.0)
    largest_recovery: float = Field(ge=0.0, le=1.0)
    time_above_threshold: float = Field(ge=0.0, le=1.0)
    time_below_threshold: float = Field(ge=0.0, le=1.0)


class BASStatistics(BaseModel):
    """Dataset-wide statistics over many `BASRecord`s (analogous to `DatasetStatistics`)."""

    model_config = ConfigDict(frozen=True)

    record_count: int = Field(ge=0)
    average_score: float = Field(ge=0.0, le=1.0)
    score_distribution: dict[str, float]
    average_confidence: float = Field(ge=0.0, le=1.0)
    feature_contribution_summary: dict[str, float]
    missing_value_summary: dict[str, int]


class BASArtifact(BaseModel):
    """The single source of truth for a computed BAS dataset."""

    model_config = ConfigDict(frozen=True)

    records: list[BASRecord]
    session_summaries: list[BASSessionSummary]
    statistics: BASStatistics
    config_fingerprint: str
    schema_version: str
    generation_timestamp: str
