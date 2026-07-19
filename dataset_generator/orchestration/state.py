"""Module 12, Step 1: Workflow State Model.

`WorkflowState` is a `TypedDict`, not a frozen Pydantic model like every
other artifact in this project. LangGraph nodes return *partial* state
updates that the graph merges into the running state on each step — that
mutation model is fundamentally different from the frozen-Pydantic
convention used everywhere else, so the outer container adapts to
LangGraph's shape while every artifact *inside* it (`DatasetArtifact`,
`BASArtifact`, `RewardArtifact`, `InterventionArtifact`) stays exactly the
existing frozen Pydantic type, untouched. Nothing about BAS/Reward/
Intervention computation changes because of this container.

Two kinds of fields live here:
  - Batch artifacts, set once by Phase 1 (Load/BAS/Reward/Intervention) and
    read-only from then on.
  - Iteration/accumulator fields, mutated once per interaction by Phase 2
    (Tutor/Finalize) as the graph walks each session's interactions.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from dataset_generator.bas.models import BASArtifact
from dataset_generator.intervention.models import InterventionArtifact
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.reward.models import RewardArtifact


def _append(existing: list[Any], update: list[Any]) -> list[Any]:
    """LangGraph reducer: accumulate rather than overwrite for list-valued state fields."""

    return [*existing, *update]


class ExecutionEvent(TypedDict):
    """One entry in `execution_history` — a single node's completion record.

    `interaction_index` is the 0-based walk cursor (`current_interaction_index`
    at the time this node ran) — not the domain-level, 1-based
    `interaction_number` used on `TutorAction`/`InterventionDecision` etc.
    Named differently on purpose so the two are never confused.
    """

    node_name: str
    session_id: str | None
    interaction_index: int | None
    timestamp: str
    detail: str


class TimingEntry(TypedDict):
    """One entry in `timing_stats` — how long one node invocation took."""

    node_name: str
    duration_seconds: float


class WorkflowError(TypedDict):
    """One recorded failure — a node error, not a Python exception object,
    so it survives JSON serialization/checkpointing.

    `interaction_index` is the 0-based walk cursor, not the domain-level
    `interaction_number` — see `ExecutionEvent`'s docstring.
    """

    node_name: str
    session_id: str | None
    interaction_index: int | None
    message: str


class TutorAction(TypedDict):
    """Module 12, Step 6: the tutor-facing translation of one `InterventionDecision`.

    Produced by `TutorAgent` purely by reading `chosen_policy` off an
    already-computed `InterventionDecision` — never re-scored, never
    re-ranked.
    """

    student_id: str
    session_id: str
    interaction_number: int
    action_type: str
    message: str
    source_policy: str
    confidence: float


class SessionOutput(TypedDict):
    """Module 12, Step: the final per-session output `SessionAgent` produces,
    aggregating the interactions this run actually walked.
    """

    student_id: str
    session_id: str
    interactions_processed: int
    interventions_triggered: int
    tutor_actions: list[TutorAction]
    terminated_early: bool
    termination_reason: str | None
    final_bas: float | None
    final_reward: float | None


class WorkflowState(TypedDict, total=False):
    """The single mutable object every orchestration node reads and updates.

    `total=False` because Phase 1 fields are absent until their node runs
    (e.g. `bas_artifact` doesn't exist until `ComputeBASNode` has executed) —
    required-ness is enforced by node preconditions, not the type itself.
    """

    # Phase 1: batch-computed artifacts (each set once, reused everywhere after)
    dataset_artifact: DatasetArtifact
    bas_artifact: BASArtifact
    reward_artifact: RewardArtifact
    intervention_artifact: InterventionArtifact

    # Iteration cursors for Phase 2's per-interaction walk
    session_ids: list[str]
    current_session_index: int
    current_session_id: str | None
    current_student_id: str | None
    current_interaction_index: int
    max_interactions_per_session: int | None

    # Phase 2 accumulators
    tutor_actions: Annotated[list[TutorAction], _append]
    session_outputs: Annotated[list[SessionOutput], _append]

    # Cross-cutting: metadata, diagnostics, timing
    execution_metadata: dict[str, Any]
    errors: Annotated[list[WorkflowError], _append]
    execution_history: Annotated[list[ExecutionEvent], _append]
    timing_stats: Annotated[list[TimingEntry], _append]


def new_workflow_state(
    max_interactions_per_session: int | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> WorkflowState:
    """Construct an empty `WorkflowState` ready for `LoadDatasetNode`."""

    return WorkflowState(
        session_ids=[],
        current_session_index=0,
        current_session_id=None,
        current_student_id=None,
        current_interaction_index=0,
        max_interactions_per_session=max_interactions_per_session,
        tutor_actions=[],
        session_outputs=[],
        execution_metadata=execution_metadata or {},
        errors=[],
        execution_history=[],
        timing_stats=[],
    )
