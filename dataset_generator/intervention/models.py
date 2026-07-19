"""Module 11, Step 1: Intervention domain models.

`InterventionArtifact` is the single source of truth this package
produces — the same role `DatasetArtifact`/`TrainingArtifact`/`BASArtifact`/
`RewardArtifact` play for theirs. Pure Pydantic throughout.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InterventionObservation(BaseModel):
    """Everything the planner needs about one interaction, read from
    already-computed `DatasetRecord`/`BASRecord`/`RewardRecord`s — nothing
    here is recomputed or resampled.
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_number: int = Field(gt=0)

    current_bas: float = Field(ge=0.0, le=1.0)
    previous_bas: float | None = Field(ge=0.0, le=1.0, default=None)
    bas_trend: float | None = None

    current_reward: float = Field(ge=-1.0, le=1.0)
    reward_trend: float | None = None

    fatigue: float = Field(ge=0.0, le=1.0)
    engagement: float = Field(ge=0.0, le=1.0)
    latency_deviation: float = Field(ge=0.0)
    correctness: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    prompt_difficulty_score: float = Field(ge=0.0, le=1.0)

    classifier_confidence: float | None = Field(ge=0.0, le=1.0, default=None)
    reward_confidence: float = Field(ge=0.0, le=1.0)
    bas_confidence: float = Field(ge=0.0, le=1.0)

    session_progress: float = Field(ge=0.0, le=1.0)
    previous_interventions_count: int = Field(ge=0)
    consecutive_decline_count: int = Field(ge=0)


class InterventionCandidate(BaseModel):
    """One eligible policy's evaluated candidacy for this interaction."""

    model_config = ConfigDict(frozen=True)

    policy_name: str
    eligible: bool
    estimated_bas_gain: float = Field(ge=0.0)
    estimated_reward_gain: float = Field(ge=0.0)
    estimated_cost: float = Field(ge=0.0)
    score: float
    reason: str


class InterventionDecisionMetadata(BaseModel):
    """Provenance for one `InterventionDecision`."""

    model_config = ConfigDict(frozen=True)

    ranking_strategy: str
    config_fingerprint: str
    config_version: str


class InterventionDecision(BaseModel):
    """The complete intervention decision for one interaction.

    `chosen_policy` is always set (it's `"NoInterventionPolicy"` when
    nothing else is warranted or cooldown/limits block a real
    intervention) — every interaction produces exactly one decision.
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_number: int = Field(gt=0)

    need_score: float = Field(ge=0.0, le=1.0)
    trigger_reasons: list[str]
    severity: str
    intervention_required: bool

    chosen_policy: str
    chosen_reason: str
    candidates: list[InterventionCandidate]

    cooldown_suppressed: bool

    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    reliability: str

    metadata: InterventionDecisionMetadata


class InterventionSessionSummary(BaseModel):
    """Session-level aggregation of a session's `InterventionDecision`s (Module 11, Step 10)."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_count: int = Field(gt=0)
    intervention_count: int = Field(ge=0)
    policy_frequencies: dict[str, int]
    average_confidence: float = Field(ge=0.0, le=1.0)
    average_severity_score: float = Field(ge=0.0, le=1.0)
    average_bas_before_intervention: float | None = Field(ge=0.0, le=1.0, default=None)
    average_bas_after_intervention: float | None = Field(ge=0.0, le=1.0, default=None)
    estimated_cumulative_bas_gain: float = Field(ge=0.0)
    estimated_cumulative_reward_gain: float = Field(ge=0.0)
    cooldown_suppressions: int = Field(ge=0)


class InterventionStatistics(BaseModel):
    """Dataset-wide statistics over many `InterventionDecision`s."""

    model_config = ConfigDict(frozen=True)

    record_count: int = Field(ge=0)
    intervention_rate: float = Field(ge=0.0, le=1.0)
    policy_distribution: dict[str, float]
    average_confidence: float = Field(ge=0.0, le=1.0)
    average_need_score: float = Field(ge=0.0, le=1.0)
    cooldown_suppression_rate: float = Field(ge=0.0, le=1.0)
    missing_value_summary: dict[str, int]


class InterventionArtifact(BaseModel):
    """The single source of truth for a computed intervention plan."""

    model_config = ConfigDict(frozen=True)

    decisions: list[InterventionDecision]
    session_summaries: list[InterventionSessionSummary]
    statistics: InterventionStatistics
    config_fingerprint: str
    schema_version: str
    generation_timestamp: str
