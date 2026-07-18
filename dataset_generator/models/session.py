"""Session models produced by the Temporal Session Simulator (Module 6).

A session is a **path graph** — interaction 1 -> 2 -> 3 -> ... with no
branching — so `SessionRecord.interactions` (an ordered list) already
represents every node faithfully; a dedicated node/edge graph class would
add machinery nothing else here needs. `transition_history` records every
edge (including the "no transition yet" entry before the first interaction);
`intervention_history`, `statistics`, and `summary` are the sibling
collections your diagram attaches to the session as a whole:

    Session
    |-- Interaction 1 -> Interaction 2 -> Interaction 3 -> ...   (interactions)
    |-- Transition History                                       (transition_history)
    |-- Intervention History                                     (intervention_history)
    |-- Rolling Statistics                                        (statistics)
    `-- Session Summary                                           (summary)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.config.attention_state import AttentionState
from dataset_generator.models.behaviour import BehaviourRecord
from dataset_generator.models.prompt import Prompt
from dataset_generator.models.response import Response


class InteractionRecord(BaseModel):
    """One interaction's full bundle: the prompt it used, and what it produced.

    Stores the complete `Prompt` (not just `prompt_id`) so that Module 7's
    `DatasetBuilder` can flatten `prompt_text`/`keywords`/`learning_objective`/
    etc. into a dataset row without ever regenerating a prompt — that
    information exists nowhere else once a session has been simulated.
    """

    model_config = ConfigDict(frozen=True)

    interaction_number: int = Field(gt=0)
    prompt: Prompt
    response: Response
    behaviour: BehaviourRecord

    @property
    def prompt_id(self) -> str:
        """Convenience accessor — equivalent to `prompt.prompt_id`."""

        return self.prompt.prompt_id


class TransitionEvent(BaseModel):
    """One observed (or initial) attention-state transition.

    `from_state=None` marks the session's initial state (there is no prior
    state to transition from); `transitioned` is always `False` in that case.
    """

    model_config = ConfigDict(frozen=True)

    interaction_number: int = Field(gt=0)
    from_state: AttentionState | None
    to_state: AttentionState
    transitioned: bool


class InterventionEvent(BaseModel):
    """One intervention application event."""

    model_config = ConfigDict(frozen=True)

    interaction_number: int = Field(gt=0)
    triggered_by: str
    pre_fatigue: float = Field(ge=0.0, le=1.0)
    post_fatigue: float = Field(ge=0.0, le=1.0)


class SessionStatistics(BaseModel):
    """Rolling/aggregate statistics, updated incrementally during simulation
    (Module 6, Step 7) — never recomputed from scratch after the session ends.
    """

    model_config = ConfigDict(frozen=True)

    interaction_count: int = Field(ge=0)
    rolling_latency: float = Field(ge=0.0)
    rolling_engagement: float = Field(ge=0.0, le=1.0)
    rolling_correctness: float = Field(ge=0.0, le=1.0)
    rolling_similarity: float = Field(ge=0.0, le=1.0)
    transition_counts: dict[str, int]
    state_frequencies: dict[str, int]
    total_duration_seconds: float = Field(ge=0.0)
    intervention_count: int = Field(ge=0)


class SessionSummary(BaseModel):
    """A compact end-of-session summary."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    student_profile: str
    session_id: str
    total_interactions: int = Field(gt=0)
    final_fatigue: float = Field(ge=0.0, le=1.0)
    average_engagement: float = Field(ge=0.0, le=1.0)
    average_correctness: float = Field(ge=0.0, le=1.0)
    average_latency: float = Field(ge=0.0)
    dominant_attention_state: str
    intervention_count: int = Field(ge=0)


class SessionRecord(BaseModel):
    """A complete simulated session (the "SessionGraph")."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    student_id: str
    student_profile: str
    interactions: list[InteractionRecord]
    transition_history: list[TransitionEvent]
    intervention_history: list[InterventionEvent]
    statistics: SessionStatistics
    summary: SessionSummary
