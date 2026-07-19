"""Module 10, Step 1: Reward domain models.

`RewardArtifact` is the single source of truth this package produces — the
same role `DatasetArtifact` (Module 7), `TrainingArtifact` (Module 8), and
`BASArtifact` (Module 9) play for theirs. Pure Pydantic throughout (no
non-serializable payloads), so it round-trips through JSON directly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.reward.config import RewardCategory


class RewardObservation(BaseModel):
    """Raw, unnormalized reward signals extracted for one interaction.

    `raw_signals[signal]` is `None` when the signal doesn't apply to this
    interaction — every session's first interaction has no previous
    interaction to diff against (all deltas `None`), and `intervention_cost`
    is `None` whenever no intervention was applied at this interaction
    (rather than 0.0 — see `aggregator.py` for why that distinction matters).
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_number: int = Field(gt=0)
    raw_signals: dict[str, float | None]


class RewardContribution(BaseModel):
    """One signal's contribution to an aggregated reward."""

    model_config = ConfigDict(frozen=True)

    signal: str
    weight: float
    category: RewardCategory
    evidence_value: float = Field(ge=0.0, le=1.0)
    signed_evidence: float = Field(ge=-1.0, le=1.0)
    contribution: float


class RewardRecordMetadata(BaseModel):
    """Provenance for one `RewardRecord`."""

    model_config = ConfigDict(frozen=True)

    temporal_mode: str
    discount_factor: float
    config_fingerprint: str
    config_version: str


class RewardRecord(BaseModel):
    """The complete reward output for one interaction.

    `raw_reward` is this interaction's aggregated signed evidence *before*
    temporal credit assignment; `reward` is after (immediate passthrough,
    discounted return, or moving average, per `RewardConfig.temporal_mode`).

    `performance_reward` / `behaviour_reward` / `cost_reward` are the
    explicit `R = Performance + Behaviour - Cost` decomposition, each the
    sum of that category's contributions (`cost_reward` sign-flipped so it
    reads as a non-negative magnitude to *subtract*) — computed at the raw
    (pre-temporal-credit-assignment) level, since they describe *what made
    up this interaction's evidence*, not the credit-assigned scalar an
    eventual RL policy would consume. The invariant
    `raw_reward == performance_reward + behaviour_reward - cost_reward`
    holds by construction (see `aggregator.decompose_reward`).
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_number: int = Field(gt=0)

    raw_reward: float = Field(ge=-1.0, le=1.0)
    reward: float = Field(ge=-1.0, le=1.0)

    performance_reward: float = Field(ge=-1.0, le=1.0)
    behaviour_reward: float = Field(ge=-1.0, le=1.0)
    cost_reward: float = Field(ge=0.0, le=1.0)

    contributions: list[RewardContribution]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    reliability: str
    missing_signal_ratio: float = Field(ge=0.0, le=1.0)

    metadata: RewardRecordMetadata


class RewardSessionSummary(BaseModel):
    """Session-level aggregation of a session's `RewardRecord`s (Module 10, Step 8)."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_count: int = Field(gt=0)
    average_reward: float = Field(ge=-1.0, le=1.0)
    maximum_reward: float = Field(ge=-1.0, le=1.0)
    minimum_reward: float = Field(ge=-1.0, le=1.0)
    variance_reward: float = Field(ge=0.0)
    reward_trend: str
    largest_improvement: float = Field(ge=0.0)
    largest_deterioration: float = Field(ge=0.0)
    cumulative_reward: float
    recovery_count: int = Field(ge=0)


class RewardStatistics(BaseModel):
    """Dataset-wide statistics over many `RewardRecord`s."""

    model_config = ConfigDict(frozen=True)

    record_count: int = Field(ge=0)
    average_reward: float
    reward_distribution: dict[str, float]
    average_confidence: float = Field(ge=0.0, le=1.0)
    average_performance_reward: float = Field(ge=-1.0, le=1.0)
    average_behaviour_reward: float = Field(ge=-1.0, le=1.0)
    average_cost_reward: float = Field(ge=0.0, le=1.0)
    contribution_summary: dict[str, float]
    missing_value_summary: dict[str, int]


class RewardArtifact(BaseModel):
    """The single source of truth for a computed reward dataset."""

    model_config = ConfigDict(frozen=True)

    records: list[RewardRecord]
    session_summaries: list[RewardSessionSummary]
    statistics: RewardStatistics
    config_fingerprint: str
    schema_version: str
    generation_timestamp: str
