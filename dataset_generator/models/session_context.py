"""`SessionContext`: the session-specific state a `Student` must never hold.

Module 2's design principle was that a `Student` carries only persistent,
cross-session identity. Everything that changes interaction-to-interaction
— how far into the session we are, what happened last turn, whether an
intervention just fired — lives here instead, and is threaded through by
whatever orchestrates a session (the temporal simulator, Step 6 of the
project roadmap).

`session_id` and `rolling_latency` were added for Module 5 (Behaviour
Generator): `session_id` because `BehaviourRecord` needs one to group
interactions, and `rolling_latency` mirrors the existing `rolling_engagement`
field — an incoming running value the caller threads forward, updated via
exponential moving average each interaction. `session_id` defaults to `""`
rather than being required, so existing Module 4 call sites that predate
Module 5 don't need to change.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.config.attention_state import AttentionState


class SessionContext(BaseModel):
    """Everything the Response/Behaviour Generators need to know about "where we are"."""

    model_config = ConfigDict(frozen=True)

    session_id: str = ""
    interaction_number: int = Field(gt=0)
    session_length: int = Field(gt=0)
    previous_response_text: str | None = None
    previous_attention_state: AttentionState | None = None
    rolling_engagement: float = Field(ge=0.0, le=1.0, default=0.5)
    rolling_latency: float | None = Field(ge=0.0, default=None)
    intervention_applied: bool = False

    @property
    def session_progress(self) -> float:
        """Fraction of the session elapsed, in (0, 1]."""

        return min(1.0, self.interaction_number / self.session_length)
