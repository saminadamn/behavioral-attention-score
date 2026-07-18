"""The `BehaviourRecord` domain model produced by the Behaviour Generator (Module 5).

Three models, mirroring `Response`/`ResponseMetadata`/`ResponseFeatures`'s split:
- `BehaviourFeatures`: derived/rolling statistics computed from the sampled
  signals plus session history — self-contained enough to unit test alone.
- `BehaviourMetadata`: generation provenance, for explainability/reporting.
- `BehaviourRecord`: the interaction's behavioural observation itself.

Per Module 5 Step 9 ("do not duplicate logic already implemented in previous
modules"), `response_length`, `repetition_ratio`, `topic_shift`, and
`engagement_score` are **not resampled here** — they're read straight from
the `Response` the Response Generator already computed from real generated
text. Only genuinely new signals (`response_latency`, `hesitation_duration`,
`interaction_duration`, `fatigue_level`, `rolling_latency`) are sampled.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.config.attention_state import AttentionState


class BehaviourFeatures(BaseModel):
    """Derived/rolling behavioural statistics, computed (never randomly assigned)."""

    model_config = ConfigDict(frozen=True)

    normalized_latency: float
    fatigue_progression: float = Field(ge=0.0, le=1.0)
    rolling_latency: float = Field(ge=0.0)
    rolling_engagement: float = Field(ge=0.0, le=1.0)
    transition_occurred: bool


class BehaviourMetadata(BaseModel):
    """Generation provenance: what produced this behavioural record."""

    model_config = ConfigDict(frozen=True)

    prompt_id: str
    subject: str
    topic: str
    student_profile: str
    previous_attention_state: AttentionState | None
    session_progress: float = Field(ge=0.0, le=1.0)
    correctness_score: float = Field(ge=0.0, le=1.0)


class BehaviourRecord(BaseModel):
    """One interaction's behavioural observation.

    Field documentation:
    - `student_id` / `session_id` / `interaction_number`: identifiers for
      joining back to `Student`/`SessionContext` and ordering within a session.
    - `attention_state`: the state this interaction was generated under
      (caller-supplied — see Module 5's module docstring).
    - `response_latency`: seconds from prompt to response, sampled and
      personalized to the student (see `behaviour_scoring.sample_latency`).
    - `interaction_duration`: total seconds spent on the interaction, sampled
      per attention state.
    - `hesitation_duration`: a hesitation-event-equivalent measure sampled
      per attention state (Poisson family by default — a count, reused here
      as a duration-like magnitude rather than literal wall-clock seconds;
      this is a deliberate simplification, not a claim of precise timing).
    - `response_length` / `engagement_score` / `repetition_ratio` /
      `topic_shift`: read from `Response`, not resampled (see module docstring).
    - `rolling_latency` / `rolling_engagement`: exponential-moving-average
      trends, threaded through `SessionContext`.
    - `fatigue_level`: accumulated session fatigue in [0, 1].
    - `intervention_applied`: whether an intervention preceded this interaction.
    - `features` / `metadata`: see `BehaviourFeatures` / `BehaviourMetadata` above.
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    session_id: str
    interaction_number: int = Field(gt=0)
    attention_state: AttentionState

    response_latency: float = Field(gt=0.0)
    interaction_duration: float = Field(gt=0.0)
    hesitation_duration: float = Field(ge=0.0)
    response_length: int = Field(gt=0)
    engagement_score: float = Field(ge=0.0, le=1.0)
    repetition_ratio: float = Field(ge=0.0, le=1.0)
    topic_shift: float = Field(ge=0.0, le=1.0)
    rolling_latency: float = Field(ge=0.0)
    rolling_engagement: float = Field(ge=0.0, le=1.0)
    fatigue_level: float = Field(ge=0.0, le=1.0)
    intervention_applied: bool

    features: BehaviourFeatures
    metadata: BehaviourMetadata
