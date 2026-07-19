"""Module 12: LangGraph Multi-Agent Orchestration.

Coordinates the completed Dataset Generator, BAS Engine, Reward Engine, and
Intervention Engine (Modules 7, 9, 10, 11) as a deterministic LangGraph
`StateGraph` — orchestration only, no internal algorithm from any of those
modules is duplicated or modified here.

Entry points:
  - `build_graph(...)` / `compile_graph(...)`: construct and run the graph.
  - `compile_checkpointed_graph(...)`: the same graph, checkpointed after
    every node, resumable via `checkpoint.py`.
  - `WorkflowMemory(state)`: read-only queries over a run's history.
  - `save_workflow_state`/`load_workflow_state`: portable JSON snapshots.
  - `build_json_report`/`render_markdown_report`: execution reports.
"""

from dataset_generator.orchestration.agents import (
    BASAgent,
    InterventionAgent,
    ObserverAgent,
    RewardAgent,
    SessionAgent,
    SessionWalkResult,
    TutorAgent,
)
from dataset_generator.orchestration.checkpoint import (
    checkpointed_state,
    compile_checkpointed_graph,
    default_checkpointer,
    is_complete,
    recover_failed_session,
    resume_execution,
    run_to_completion,
    thread_config,
)
from dataset_generator.orchestration.graph import (
    NODE_NAMES,
    ROUTE_PATH_MAP,
    build_graph,
    compile_graph,
    route_next_step,
)
from dataset_generator.orchestration.memory import WorkflowMemory
from dataset_generator.orchestration.nodes import (
    make_compute_bas_node,
    make_compute_reward_node,
    make_finalize_session_node,
    make_generate_tutor_action_node,
    make_load_dataset_node,
    make_plan_intervention_node,
    ordered_decisions_for_session,
)
from dataset_generator.orchestration.prompts import POLICY_ACTION_TYPES, action_type_for_policy, format_tutor_message
from dataset_generator.orchestration.report import (
    build_json_report,
    decision_counts,
    failure_summary,
    graph_statistics,
    intervention_frequencies,
    node_timing_summary,
    render_markdown_report,
)
from dataset_generator.orchestration.serialization import (
    ORCHESTRATION_SCHEMA_VERSION,
    config_fingerprints,
    load_workflow_state,
    save_workflow_state,
)
from dataset_generator.orchestration.state import (
    ExecutionEvent,
    SessionOutput,
    TimingEntry,
    TutorAction,
    WorkflowError,
    WorkflowState,
    new_workflow_state,
)

__all__ = [
    "NODE_NAMES",
    "ORCHESTRATION_SCHEMA_VERSION",
    "POLICY_ACTION_TYPES",
    "ROUTE_PATH_MAP",
    "BASAgent",
    "ExecutionEvent",
    "InterventionAgent",
    "ObserverAgent",
    "RewardAgent",
    "SessionAgent",
    "SessionOutput",
    "SessionWalkResult",
    "TimingEntry",
    "TutorAction",
    "TutorAgent",
    "WorkflowError",
    "WorkflowMemory",
    "WorkflowState",
    "action_type_for_policy",
    "build_graph",
    "build_json_report",
    "checkpointed_state",
    "compile_checkpointed_graph",
    "compile_graph",
    "config_fingerprints",
    "decision_counts",
    "default_checkpointer",
    "failure_summary",
    "format_tutor_message",
    "graph_statistics",
    "intervention_frequencies",
    "is_complete",
    "load_workflow_state",
    "make_compute_bas_node",
    "make_compute_reward_node",
    "make_finalize_session_node",
    "make_generate_tutor_action_node",
    "make_load_dataset_node",
    "make_plan_intervention_node",
    "new_workflow_state",
    "node_timing_summary",
    "ordered_decisions_for_session",
    "recover_failed_session",
    "render_markdown_report",
    "resume_execution",
    "route_next_step",
    "run_to_completion",
    "save_workflow_state",
    "thread_config",
]
