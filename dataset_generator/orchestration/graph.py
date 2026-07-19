"""Module 12, Steps 4-5: Graph Construction and Conditional Routing.

`route_next_step` is the single router reused at three wiring points
(after `plan_intervention`, after `generate_tutor_action`, after
`finalize_session`) — all three ask the same question ("given the current
cursor, what's next?"), so one function answers it rather than three
near-duplicates. Its four possible answers:

  - "end": no sessions at all (empty dataset).
  - "finalize_session": the current session has an error recorded, has
    exhausted its interactions, or has hit `max_interactions_per_session`.
  - "tutor_intervention" / "continue_session": there's another interaction
    to process; both route to `generate_tutor_action` (which produces the
    correct action either way — a real intervention or an explicit
    "No Intervention" action), but are distinct, labeled LangGraph edges
    matching the diagram's fork/merge shape.

Operational note on very large batches: Phase 2's per-interaction walk
takes one LangGraph "step" per interaction, and LangGraph's Pregel
scheduler caps total steps per run via `recursion_limit` (a safety guard
against infinite loops, unrelated to Python's call-stack recursion) --
its default is far smaller than the interaction count of a 100,000+
interaction dataset. For large batches, pass a sufficiently high limit
explicitly: `compiled.invoke(state, config={"recursion_limit": N})` where
`N` comfortably exceeds the total interaction count across all sessions.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dataset_generator.config.schema import GeneratorConfig

from dataset_generator.orchestration.agents import (
    BASAgent,
    InterventionAgent,
    ObserverAgent,
    RewardAgent,
    SessionAgent,
    TutorAgent,
)
from dataset_generator.orchestration.nodes import (
    make_compute_bas_node,
    make_compute_reward_node,
    make_finalize_session_node,
    make_generate_tutor_action_node,
    make_load_dataset_node,
    make_plan_intervention_node,
    max_interactions_reached,
    ordered_decisions_for_session,
    session_has_error,
)
from dataset_generator.orchestration.state import WorkflowState

LOAD_DATASET = "load_dataset"
COMPUTE_BAS = "compute_bas"
COMPUTE_REWARD = "compute_reward"
PLAN_INTERVENTION = "plan_intervention"
GENERATE_TUTOR_ACTION = "generate_tutor_action"
FINALIZE_SESSION = "finalize_session"

NODE_NAMES = (
    LOAD_DATASET, COMPUTE_BAS, COMPUTE_REWARD, PLAN_INTERVENTION,
    GENERATE_TUTOR_ACTION, FINALIZE_SESSION,
)

RouteKey = Literal["tutor_intervention", "continue_session", "finalize_session", "end"]

ROUTE_PATH_MAP = {
    "tutor_intervention": GENERATE_TUTOR_ACTION,
    "continue_session": GENERATE_TUTOR_ACTION,
    "finalize_session": FINALIZE_SESSION,
    "end": END,
}


def route_next_step(state: WorkflowState) -> RouteKey:
    """Decide the next node from the current cursor + intervention artifact.

    Read-only: routers in LangGraph select an edge, they don't mutate state.
    """

    session_ids = state.get("session_ids") or []
    current_session_id = state.get("current_session_id")
    if not session_ids or current_session_id is None:
        return "end"

    ordered = ordered_decisions_for_session(state, current_session_id)
    idx = state.get("current_interaction_index", 0)

    if session_has_error(state, current_session_id):
        return "finalize_session"
    if idx >= len(ordered):
        return "finalize_session"
    if max_interactions_reached(state, idx):
        return "finalize_session"

    decision = ordered[idx]
    return "tutor_intervention" if decision.intervention_required else "continue_session"


def build_graph(
    observer: ObserverAgent | None = None,
    bas_agent: BASAgent | None = None,
    reward_agent: RewardAgent | None = None,
    intervention_agent: InterventionAgent | None = None,
    tutor_agent: TutorAgent | None = None,
    session_agent: SessionAgent | None = None,
    generator_config: GeneratorConfig | None = None,
    student_count: int | None = None,
    sessions_per_student: int = 2,
) -> StateGraph:
    """Build (but do not compile) the Module 12 orchestration graph.

    Every agent is dependency-injectable; omitted ones default to the
    engines' own defaults (`BASEngine()`, `default_config()`, etc.), exactly
    as each agent class itself already defaults.
    """

    graph: StateGraph = StateGraph(WorkflowState)

    # LangGraph's `add_node`/`add_conditional_edges` overloads are generic over
    # very specific Runnable-like protocols that a plain `(WorkflowState) ->
    # dict[str, Any]` closure doesn't structurally match, even though it's
    # exactly what LangGraph expects and calls at runtime (confirmed by the
    # full test suite). The `# type: ignore[call-overload]` below waives
    # mypy's overload resolution at this boundary only, not the rest of the
    # module's typing.
    graph.add_node(
        LOAD_DATASET,
        make_load_dataset_node(
            observer=observer,
            generator_config=generator_config,
            student_count=student_count,
            sessions_per_student=sessions_per_student,
        ),
    )  # type: ignore[call-overload]
    graph.add_node(COMPUTE_BAS, make_compute_bas_node(bas_agent))  # type: ignore[call-overload]
    graph.add_node(COMPUTE_REWARD, make_compute_reward_node(reward_agent))  # type: ignore[call-overload]
    graph.add_node(PLAN_INTERVENTION, make_plan_intervention_node(intervention_agent))  # type: ignore[call-overload]
    graph.add_node(GENERATE_TUTOR_ACTION, make_generate_tutor_action_node(tutor_agent))  # type: ignore[call-overload]
    graph.add_node(FINALIZE_SESSION, make_finalize_session_node(session_agent))  # type: ignore[call-overload]

    graph.add_edge(START, LOAD_DATASET)
    graph.add_edge(LOAD_DATASET, COMPUTE_BAS)
    graph.add_edge(COMPUTE_BAS, COMPUTE_REWARD)
    graph.add_edge(COMPUTE_REWARD, PLAN_INTERVENTION)

    # Same third-party overload friction: `add_conditional_edges` wants a
    # `dict[Hashable, str]` path map; our `dict[str, str]` satisfies that
    # structurally (str is Hashable) but mypy's overload picker doesn't
    # unify it automatically.
    graph.add_conditional_edges(PLAN_INTERVENTION, route_next_step, ROUTE_PATH_MAP)  # type: ignore[arg-type]
    graph.add_conditional_edges(GENERATE_TUTOR_ACTION, route_next_step, ROUTE_PATH_MAP)  # type: ignore[arg-type]
    graph.add_conditional_edges(FINALIZE_SESSION, route_next_step, ROUTE_PATH_MAP)  # type: ignore[arg-type]

    return graph


def compile_graph(
    graph: StateGraph | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    **build_kwargs: Any,
) -> CompiledStateGraph:
    """Compile `graph` (building a default one from `build_kwargs` if omitted).

    `checkpointer` is left as a plain dependency-injection point here — it's
    Step 8's concern (`checkpoint.py`) to decide which checkpointer to pass.
    """

    graph = graph or build_graph(**build_kwargs)
    return graph.compile(checkpointer=checkpointer)
