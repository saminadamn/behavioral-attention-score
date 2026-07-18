"""Dataset-layer models (Module 7).

`DatasetRecord` flattens one `InteractionRecord` (Student + Prompt +
Response + Behaviour + session-level context) into a single row of scalar
fields — no nested objects, so it exports cleanly to CSV/Parquet/JSONL.

Two deliberate de-duplication choices, both explained where they'd
otherwise look like missing data:
- Fields that mean the *same fact* on two source objects (e.g.
  `intervention_applied`, tracked identically on `Response.metadata` and
  `BehaviourRecord`) appear **once** at the top level, not once per source.
- `BehaviourRecord`'s pass-through fields (`response_length`,
  `engagement_score`, `repetition_ratio`, `topic_shift` — themselves read
  from `Response` per Module 5's design) are represented via their
  `response_*` columns only; there is no second `behaviour_response_length`
  column repeating the identical value.
- `Student.transition_modifier` is intentionally omitted — it's profile-
  invariant (identical for every student sharing a profile) and doesn't
  flatten into fixed columns without knowing which entries a given profile
  overrides. Anyone needing it can look it up from `GeneratorConfig` by
  `student_profile`.

Session-level aggregates (`session_total_interactions`,
`session_average_engagement`, etc.) intentionally repeat across every row
belonging to that session — that's standard denormalization, not the kind
of duplication Module 7's brief warns against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DatasetRecord(BaseModel):
    """One flattened interaction row."""

    model_config = ConfigDict(frozen=True)

    # Identifiers
    session_id: str
    student_id: str
    interaction_number: int = Field(gt=0)
    prompt_id: str
    response_id: str

    # Interaction-level facts (each tracked once, not duplicated per source object)
    attention_state: str
    intervention_applied: bool
    session_progress: float = Field(ge=0.0, le=1.0)

    # Student (Module 2)
    student_profile: str
    student_profile_description: str
    student_baseline_latency: float
    student_latency_variance: float
    student_engagement_tendency: float
    student_fatigue_rate: float
    student_intervention_sensitivity: float

    # Prompt (Module 3)
    prompt_subject: str
    prompt_topic: str
    prompt_difficulty: str
    prompt_cognitive_level: str
    prompt_text: str
    prompt_expected_answer_type: str
    prompt_estimated_response_length: int
    prompt_keywords: str  # "|"-joined (flat scalar, not a nested list)
    prompt_learning_objective: str
    prompt_reading_time_seconds: float
    prompt_token_count: int
    prompt_concept_count: int
    prompt_difficulty_score: float
    prompt_cognitive_complexity_score: float
    prompt_readability_grade: float

    # Response (Module 4)
    response_text: str
    response_correctness_score: float = Field(ge=0.0, le=1.0)
    response_length: int
    response_semantic_similarity: float = Field(ge=0.0, le=1.0)
    response_lexical_diversity: float = Field(ge=0.0, le=1.0)
    response_sentiment: float = Field(ge=-1.0, le=1.0)
    response_engagement_proxy: float = Field(ge=0.0, le=1.0)
    response_confidence: float = Field(ge=0.0, le=1.0)
    response_hesitation_markers: str  # "|"-joined
    response_repetition_ratio: float = Field(ge=0.0, le=1.0)
    response_coherence_score: float = Field(ge=0.0, le=1.0)
    response_topic_shift: float = Field(ge=0.0, le=1.0)
    response_strategy_used: str

    # Behaviour (Module 5) — excludes response_length/engagement_score/
    # repetition_ratio/topic_shift, already present above as response_* columns.
    behaviour_response_latency: float
    behaviour_interaction_duration: float
    behaviour_hesitation_duration: float
    behaviour_rolling_latency: float
    behaviour_rolling_engagement: float = Field(ge=0.0, le=1.0)
    behaviour_fatigue_level: float = Field(ge=0.0, le=1.0)
    behaviour_normalized_latency: float
    behaviour_transition_occurred: bool

    # Session-level aggregates (Module 6), denormalized across every row of the session
    session_total_interactions: int
    session_dominant_attention_state: str
    session_intervention_count: int
    session_final_fatigue: float = Field(ge=0.0, le=1.0)
    session_average_engagement: float = Field(ge=0.0, le=1.0)
    session_average_correctness: float = Field(ge=0.0, le=1.0)
    session_average_latency: float


class DatasetMetadata(BaseModel):
    """Descriptive summary of a dataset's content (not versioning — see `DatasetManifest`)."""

    model_config = ConfigDict(frozen=True)

    student_count: int = Field(ge=0)
    session_count: int = Field(ge=0)
    record_count: int = Field(ge=0)
    subjects_covered: list[str]
    profiles_covered: list[str]


class DatasetManifest(BaseModel):
    """Versioning and provenance (Module 7, Step 7)."""

    model_config = ConfigDict(frozen=True)

    dataset_version: str
    schema_version: str
    generator_version: str
    generation_timestamp: str
    seed: int
    config_fingerprint: str
    git_commit_hash: str | None = None

    @staticmethod
    def now_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()


class FeatureDistributionSummary(BaseModel):
    """Summary statistics for one numeric feature column."""

    model_config = ConfigDict(frozen=True)

    mean: float
    std: float
    min: float
    max: float
    missing_count: int = Field(ge=0)


class DatasetStatistics(BaseModel):
    """Aggregate statistics over a dataset's records (Module 7, Step 5)."""

    model_config = ConfigDict(frozen=True)

    record_count: int = Field(ge=0)
    feature_distributions: dict[str, FeatureDistributionSummary]
    class_balance: dict[str, float]
    attention_balance: dict[str, float]
    profile_balance: dict[str, float]
    session_balance: dict[str, int]
    subject_balance: dict[str, float]
    difficulty_balance: dict[str, float]
    correlation_matrix: dict[str, dict[str, float]]
    missing_value_summary: dict[str, int]


class DatasetValidationReport(BaseModel):
    """Structured validation report (Module 7, Step 4)."""

    model_config = ConfigDict(frozen=True)

    record_count: int = Field(ge=0)
    missing_value_issues: dict[str, int]
    duplicate_row_count: int = Field(ge=0)
    duplicate_id_count: int = Field(ge=0)
    invalid_range_issues: dict[str, int]
    impossible_transition_count: int = Field(ge=0)
    invalid_attention_state_count: int = Field(ge=0)
    orphan_session_ids: list[str]
    orphan_student_ids: list[str]
    nan_count: int = Field(ge=0)
    inf_count: int = Field(ge=0)
    schema_consistent: bool

    @property
    def is_valid(self) -> bool:
        return (
            self.duplicate_row_count == 0
            and self.duplicate_id_count == 0
            and self.impossible_transition_count == 0
            and self.invalid_attention_state_count == 0
            and not self.orphan_session_ids
            and not self.orphan_student_ids
            and self.nan_count == 0
            and self.inf_count == 0
            and self.schema_consistent
            and not self.invalid_range_issues
        )


class FeatureCategory(str, Enum):
    """Feature groupings for `FeatureRegistry` (Module 7, Step 3)."""

    STUDENT = "student"
    SESSION = "session"
    PROMPT = "prompt"
    RESPONSE = "response"
    BEHAVIOUR = "behaviour"
    TARGET = "target"
    IDENTIFIER = "identifier"


@dataclass(frozen=True)
class FeatureDefinition:
    """One `DatasetRecord` field's registry entry."""

    name: str
    category: FeatureCategory
    dtype: str  # "str" | "int" | "float" | "bool"
    description: str


class DatasetArtifact(BaseModel):
    """The single source of truth for a generated dataset (Module 7's architectural
    improvement) — records, statistics, validation, metadata, manifest, and
    exported-file locations together, so downstream modules (attention
    classifier, BAS, evaluation, paper figures) consume *this* object
    instead of independently re-reading exported files and re-implementing
    loading/parsing logic.
    """

    model_config = ConfigDict(frozen=True)

    records: list[DatasetRecord]
    statistics: DatasetStatistics
    validation: DatasetValidationReport
    metadata: DatasetMetadata
    manifest: DatasetManifest
    exports: dict[str, str] = Field(default_factory=dict)
