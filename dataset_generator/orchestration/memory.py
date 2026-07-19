"""Module 12, Step 7: Memory.

`WorkflowMemory` is a thin, stateless *view* over an existing
`WorkflowState` — not a new store. It holds nothing itself beyond a
reference to the state it was built from; every query filters/sorts data
that Modules 9-11 and this module's own nodes already computed. Because
there is no independent mutable store, replay is deterministic for free:
the same `WorkflowState` always yields the same query results. No vector
database, no LLM — every method here is plain filtering over Pydantic
records or `TypedDict`s already present in state.
"""

from __future__ import annotations

from dataset_generator.bas.models import BASRecord
from dataset_generator.intervention.models import InterventionDecision
from dataset_generator.reward.models import RewardRecord

from dataset_generator.orchestration.state import SessionOutput, TutorAction, WorkflowState


class WorkflowMemory:
    """A queryable view over one `WorkflowState`'s accumulated history."""

    def __init__(self, state: WorkflowState) -> None:
        self._state = state

    def previous_interventions(
        self,
        session_id: str | None = None,
        student_id: str | None = None,
        before_interaction: int | None = None,
    ) -> list[InterventionDecision]:
        """`InterventionDecision`s already computed by Module 11, optionally filtered."""

        artifact = self._state.get("intervention_artifact")
        if artifact is None:
            return []

        decisions = artifact.decisions
        if session_id is not None:
            decisions = [d for d in decisions if d.session_id == session_id]
        if student_id is not None:
            decisions = [d for d in decisions if d.student_id == student_id]
        if before_interaction is not None:
            decisions = [d for d in decisions if d.interaction_number < before_interaction]
        return sorted(decisions, key=lambda d: (d.session_id, d.interaction_number))

    def previous_tutor_actions(
        self, session_id: str | None = None, student_id: str | None = None
    ) -> list[TutorAction]:
        """`TutorAction`s already produced this run, optionally filtered."""

        actions = self._state.get("tutor_actions", [])
        if session_id is not None:
            actions = [a for a in actions if a["session_id"] == session_id]
        if student_id is not None:
            actions = [a for a in actions if a["student_id"] == student_id]
        return list(actions)

    def session_history(self, student_id: str | None = None) -> list[SessionOutput]:
        """`SessionOutput`s already finalized this run, optionally filtered by student."""

        outputs = self._state.get("session_outputs", [])
        if student_id is not None:
            outputs = [o for o in outputs if o["student_id"] == student_id]
        return list(outputs)

    def reward_history(self, session_id: str | None = None) -> list[RewardRecord]:
        """`RewardRecord`s from the already-computed `RewardArtifact`, optionally filtered."""

        artifact = self._state.get("reward_artifact")
        if artifact is None:
            return []
        records = artifact.records
        if session_id is not None:
            records = [r for r in records if r.session_id == session_id]
        return sorted(records, key=lambda r: (r.session_id, r.interaction_number))

    def bas_history(self, session_id: str | None = None) -> list[BASRecord]:
        """`BASRecord`s from the already-computed `BASArtifact`, optionally filtered."""

        artifact = self._state.get("bas_artifact")
        if artifact is None:
            return []
        records = artifact.records
        if session_id is not None:
            records = [r for r in records if r.session_id == session_id]
        return sorted(records, key=lambda r: (r.session_id, r.interaction_number))
