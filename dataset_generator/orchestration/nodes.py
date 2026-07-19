"""Module 12, Step 3: Node Implementations.

Every node is produced by a factory function (`make_load_dataset_node`,
etc.) so agent instances can be dependency-injected rather than
hardcoded — the factory closes over the agent, and the returned closure is
the actual `(WorkflowState) -> dict` LangGraph invokes. `_traced_node`
wraps every node uniformly so none of the six duplicate timing/history/
error-recording boilerplate.

Nodes only return the *changed* keys, never the whole state — the
`_append`-reducer fields in `state.py` (timing_stats, execution_history,
errors, tutor_actions, session_outputs) are merged by LangGraph itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from dataset_generator.config.schema import GeneratorConfig

from dataset_generator.orchestration.agents import (
    BASAgent,
    InterventionAgent,
    ObserverAgent,
    RewardAgent,
    SessionAgent,
    SessionWalkResult,
    TutorAgent,
)
from dataset_generator.orchestration.memory import WorkflowMemory
from dataset_generator.orchestration.state import (
    ExecutionEvent,
    TimingEntry,
    WorkflowError,
    WorkflowState,
)

NodeFn = Callable[[WorkflowState], dict[str, Any]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _traced_node(node_name: str) -> Callable[[NodeFn], NodeFn]:
    """Wrap a node with timing, history, and error recording.

    On success, returns the wrapped node's own update plus one
    `timing_stats` entry and one `execution_history` entry. On an
    exception, returns one `errors` entry instead — the graph keeps
    running rather than crashing, so a router can inspect `errors` and
    decide whether to halt (Step 5's "error handling").
    """

    def decorator(fn: NodeFn) -> NodeFn:
        def wrapped(state: WorkflowState) -> dict[str, Any]:
            start = time.perf_counter()
            session_id = state.get("current_session_id")
            interaction_index = state.get("current_interaction_index")
            try:
                update = fn(state)
            except Exception as exc:  # noqa: BLE001 - intentionally broad: any node failure is recorded, not fatal
                duration = time.perf_counter() - start
                return {
                    "errors": [
                        WorkflowError(
                            node_name=node_name,
                            session_id=session_id,
                            interaction_index=interaction_index,
                            message=str(exc),
                        )
                    ],
                    "timing_stats": [TimingEntry(node_name=node_name, duration_seconds=duration)],
                    "execution_history": [
                        ExecutionEvent(
                            node_name=node_name,
                            session_id=session_id,
                            interaction_index=interaction_index,
                            timestamp=_now_iso(),
                            detail=f"failed: {exc}",
                        )
                    ],
                }

            duration = time.perf_counter() - start
            merged: dict[str, Any] = dict(update)
            merged["timing_stats"] = [TimingEntry(node_name=node_name, duration_seconds=duration)]
            merged["execution_history"] = [
                ExecutionEvent(
                    node_name=node_name,
                    session_id=update.get("current_session_id", session_id),
                    interaction_index=update.get("current_interaction_index", interaction_index),
                    timestamp=_now_iso(),
                    detail="ok",
                )
            ]
            return merged

        return wrapped

    return decorator


def make_load_dataset_node(
    observer: ObserverAgent | None = None,
    generator_config: GeneratorConfig | None = None,
    student_count: int | None = None,
    sessions_per_student: int = 2,
) -> NodeFn:
    """LoadDatasetNode: produce/inject a `DatasetArtifact` and initialize cursors.

    If `state['dataset_artifact']` is already present (injected by the
    caller), it is passed through unchanged via `ObserverAgent.observe`.
    Otherwise a fresh one is generated via `ObserverAgent.generate`.
    """

    agent = observer or ObserverAgent(config=generator_config)

    @_traced_node("LoadDatasetNode")
    def node(state: WorkflowState) -> dict[str, Any]:
        existing = state.get("dataset_artifact")
        dataset_artifact = (
            agent.observe(existing)
            if existing is not None
            else agent.generate(student_count=student_count, sessions_per_student=sessions_per_student)
        )

        session_ids = sorted({record.session_id for record in dataset_artifact.records})
        student_by_session = {
            record.session_id: record.student_id for record in dataset_artifact.records
        }
        first_session = session_ids[0] if session_ids else None

        return {
            "dataset_artifact": dataset_artifact,
            "session_ids": session_ids,
            "current_session_index": 0,
            "current_session_id": first_session,
            "current_student_id": student_by_session.get(first_session) if first_session else None,
            "current_interaction_index": 0,
        }

    return node


def make_compute_bas_node(agent: BASAgent | None = None) -> NodeFn:
    """ComputeBASNode: wraps `BASAgent.compute` (in turn `BASEngine.compute`)."""

    bas_agent = agent or BASAgent()

    @_traced_node("ComputeBASNode")
    def node(state: WorkflowState) -> dict[str, Any]:
        dataset_artifact = state["dataset_artifact"]
        return {"bas_artifact": bas_agent.compute(dataset_artifact)}

    return node


def make_compute_reward_node(agent: RewardAgent | None = None) -> NodeFn:
    """ComputeRewardNode: wraps `RewardAgent.compute` (in turn `RewardEngine.compute`)."""

    reward_agent = agent or RewardAgent()

    @_traced_node("ComputeRewardNode")
    def node(state: WorkflowState) -> dict[str, Any]:
        dataset_artifact = state["dataset_artifact"]
        bas_artifact = state["bas_artifact"]
        return {"reward_artifact": reward_agent.compute(dataset_artifact, bas_artifact)}

    return node


def make_plan_intervention_node(agent: InterventionAgent | None = None) -> NodeFn:
    """PlanInterventionNode: wraps `InterventionAgent.plan` (in turn `InterventionPlanner.plan`)."""

    intervention_agent = agent or InterventionAgent()

    @_traced_node("PlanInterventionNode")
    def node(state: WorkflowState) -> dict[str, Any]:
        dataset_artifact = state["dataset_artifact"]
        bas_artifact = state["bas_artifact"]
        reward_artifact = state["reward_artifact"]
        return {
            "intervention_artifact": intervention_agent.plan(
                dataset_artifact, bas_artifact, reward_artifact
            )
        }

    return node


def ordered_decisions_for_session(state: WorkflowState, session_id: str) -> list:
    """The current session's `InterventionDecision`s, sorted by interaction number.

    Shared by `GenerateTutorActionNode`, `FinalizeSessionNode`, and the
    router functions in `graph.py` — computed once per call from
    `intervention_artifact.decisions` (already fully computed by Module
    11), never recomputed logic, only re-filtered/sorted.
    """

    intervention_artifact = state["intervention_artifact"]
    return sorted(
        (d for d in intervention_artifact.decisions if d.session_id == session_id),
        key=lambda d: d.interaction_number,
    )


def session_has_error(state: WorkflowState, session_id: str) -> bool:
    """Whether any recorded error belongs to `session_id`.

    Shared by `route_next_step` (graph.py) and `FinalizeSessionNode` — both
    need this exact check (one to decide routing, one to derive
    `termination_reason`), so it lives in one place.
    """

    return any(error["session_id"] == session_id for error in state.get("errors", []))


def max_interactions_reached(state: WorkflowState, walked_count: int) -> bool:
    """Whether `walked_count` interactions have hit `max_interactions_per_session`.

    Shared by `route_next_step` (graph.py) and `FinalizeSessionNode` for the
    same reason as `session_has_error`.
    """

    max_limit = state.get("max_interactions_per_session")
    return max_limit is not None and walked_count >= max_limit


def make_generate_tutor_action_node(agent: TutorAgent | None = None) -> NodeFn:
    """GenerateTutorActionNode: translate the current interaction's already-decided
    `InterventionDecision` into a `TutorAction`, then advance the interaction cursor.

    Always produces exactly one `TutorAction` per call (including an
    explicit "No Intervention" action when `chosen_policy` is
    `NoInterventionPolicy`) — never re-scores, never re-ranks.
    """

    tutor_agent = agent or TutorAgent()

    @_traced_node("GenerateTutorActionNode")
    def node(state: WorkflowState) -> dict[str, Any]:
        session_id = state["current_session_id"]
        assert session_id is not None, "GenerateTutorActionNode requires a current session"
        idx = state["current_interaction_index"]
        ordered = ordered_decisions_for_session(state, session_id)
        if idx >= len(ordered):
            return {}

        decision = ordered[idx]
        action = tutor_agent.generate_action(decision)
        return {
            "tutor_actions": [action],
            "current_interaction_index": idx + 1,
        }

    return node


def make_finalize_session_node(agent: SessionAgent | None = None) -> NodeFn:
    """FinalizeSessionNode: aggregate the current session's walked interactions
    into a `SessionOutput`, then advance to the next session.

    Only aggregates what was *actually walked* (`current_interaction_index`
    interactions) — this is naturally fewer than the full session under
    early termination or a `max_interactions_per_session` limit.
    """

    session_agent = agent or SessionAgent()

    @_traced_node("FinalizeSessionNode")
    def node(state: WorkflowState) -> dict[str, Any]:
        session_id = state["current_session_id"]
        assert session_id is not None, "FinalizeSessionNode requires a current session"
        walked_count = state["current_interaction_index"]

        ordered_decisions = ordered_decisions_for_session(state, session_id)
        walked_decisions = ordered_decisions[:walked_count]

        session_tutor_actions = [
            action for action in state.get("tutor_actions", []) if action["session_id"] == session_id
        ]

        # Reuse WorkflowMemory's session-filtered/sorted BAS/reward queries
        # (Step 7) rather than re-filtering `bas_artifact.records`/
        # `reward_artifact.records` here — only the walked-count slice is new.
        memory = WorkflowMemory(state)
        bas_records = memory.bas_history(session_id)[:walked_count]
        reward_records = memory.reward_history(session_id)[:walked_count]

        student_id: str = (
            ordered_decisions[0].student_id
            if ordered_decisions
            else (state.get("current_student_id") or "")
        )

        terminated_early = walked_count < len(ordered_decisions)
        termination_reason: str | None = None
        if terminated_early:
            if session_has_error(state, session_id):
                termination_reason = "error"
            elif max_interactions_reached(state, walked_count):
                termination_reason = "max_interaction_limit"
            else:
                # Reachable only when a node is invoked directly on
                # hand-crafted state that bypasses `route_next_step` (as
                # some unit tests do) — the compiled graph itself only ever
                # routes here for the "error" or "max_interaction_limit"
                # reasons above.
                termination_reason = "early_termination"

        walk = SessionWalkResult(
            student_id=student_id,
            session_id=session_id,
            decisions=walked_decisions,
            tutor_actions=session_tutor_actions,
            bas_records=bas_records,
            reward_records=reward_records,
            terminated_early=terminated_early,
            termination_reason=termination_reason,
        )
        output = session_agent.finalize(walk)

        session_ids = state["session_ids"]
        next_index = state["current_session_index"] + 1
        next_session_id = session_ids[next_index] if next_index < len(session_ids) else None
        next_student_id = None
        if next_session_id is not None:
            next_student_id = next(
                (d.student_id for d in ordered_decisions_for_session(state, next_session_id)), None
            )

        return {
            "session_outputs": [output],
            "current_session_index": next_index,
            "current_session_id": next_session_id,
            "current_student_id": next_student_id,
            "current_interaction_index": 0,
        }

    return node
