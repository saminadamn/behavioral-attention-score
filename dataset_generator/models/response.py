"""The `Response` domain model produced by the Response Generator (Module 4).

Three models, each with a distinct job:
- `ResponseFeatures`: pure text-derived statistics (nothing here required
  knowledge of *why* the text looks this way — it's computed the same way
  regardless of student/prompt/strategy).
- `ResponseMetadata`: generation provenance — which prompt/strategy/session
  state produced this response, kept for explainability and reporting.
- `Response`: the interaction record itself, with the headline fields
  Module 4 Step 1 asks for at the top level, plus the two nested models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.config.attention_state import AttentionState
from dataset_generator.config.prompt_generation import CognitiveLevel, Difficulty


class ResponseFeatures(BaseModel):
    """Text-derived statistics, computed from `student_response` itself.

    `token_count` duplicates `Response.response_length` deliberately — it
    lives here too so anything holding only a `ResponseFeatures` (e.g. a
    feature-extraction unit test) can verify internal consistency without
    reaching into the parent `Response`.
    """

    model_config = ConfigDict(frozen=True)

    token_count: int = Field(gt=0)
    repetition_ratio: float = Field(ge=0.0, le=1.0)
    coherence_score: float = Field(ge=0.0, le=1.0)
    topic_shift: float = Field(ge=0.0, le=1.0)


class ResponseMetadata(BaseModel):
    """Generation provenance: what produced this response, for explainability/reporting."""

    model_config = ConfigDict(frozen=True)

    correctness_probability: float = Field(ge=0.0, le=1.0)
    strategy_used: str
    difficulty: Difficulty
    cognitive_level: CognitiveLevel
    subject: str
    topic: str
    attention_state: AttentionState
    student_profile: str
    intervention_applied: bool
    session_progress: float = Field(ge=0.0, le=1.0)


class Response(BaseModel):
    """One simulated student response to one prompt, in one session interaction.

    Field documentation:
    - `response_id`: unique identifier for this interaction (student + turn index).
    - `student_id` / `prompt_id`: foreign keys into the Student and Prompt tables.
    - `response_text`: the generated wording itself.
    - `correctness_score`: continuous [0,1] estimate of answer quality/correctness
      (not a hard true/false label — see `response_scoring.correctness_score`).
    - `response_length`: token count of `response_text`.
    - `semantic_similarity`: token-overlap similarity between the response and the
      prompt's text/topic/keywords — how on-topic the answer is.
    - `lexical_diversity`: type-token ratio of `response_text`.
    - `sentiment`: fixed-lexicon sentiment heuristic in [-1, 1].
    - `engagement_proxy`: composite engagement estimate (Stage 2 formula).
    - `confidence`: how assertive the response *sounds*, independent of whether
      it's actually correct (inferred from attention state + detected hesitation).
    - `hesitation_markers`: which hesitation phrases were actually detected in the text.
    - `features` / `metadata`: see `ResponseFeatures` / `ResponseMetadata` above.
    """

    model_config = ConfigDict(frozen=True)

    response_id: str
    student_id: str
    prompt_id: str
    response_text: str = Field(min_length=1)

    correctness_score: float = Field(ge=0.0, le=1.0)
    response_length: int = Field(gt=0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    lexical_diversity: float = Field(ge=0.0, le=1.0)
    sentiment: float = Field(ge=-1.0, le=1.0)
    engagement_proxy: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    hesitation_markers: list[str] = Field(default_factory=list)

    features: ResponseFeatures
    metadata: ResponseMetadata
