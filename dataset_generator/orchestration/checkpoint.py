"""Module 12, Step 8: Checkpointing.

Built directly on LangGraph's own checkpointer mechanism rather than a
parallel implementation: `compile_checkpointed_graph` compiles with
`interrupt_after` set on every node, so LangGraph persists state after each
one and execution actually pauses at every node boundary — "checkpoint
every node" is a consequence of that compile-time setting, not custom code.

Resuming is a direct consequence of `route_next_step` (Steps 4-5) being a
pure function of `WorkflowState` rather than of "which step we're on": once
a checkpointed state is loaded (from LangGraph's checkpointer, or from a
`WorkflowState` restored via `serialization.py`), calling `.invoke()` again
picks the routing back up exactly where it left off. No separate "resume"
code path is needed beyond restoring the state and invoking again.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RunnableConfig

from dataset_generator.orchestration.graph import NODE_NAMES
from dataset_generator.orchestration.state import WorkflowState


def default_checkpointer() -> MemorySaver:
    """An in-process `MemorySaver` that can round-trip this project's own
    Pydantic artifact types (`DatasetArtifact`, `BASArtifact`,
    `RewardArtifact`, `InterventionArtifact`, and their nested models).

    LangGraph's default serializer only allows msgpack-deserializing types
    it doesn't recognize when explicitly told to; without
    `allowed_msgpack_modules=True` here, checkpoint reloads silently drop
    fields it can't deserialize (confirmed: `dataset_artifact`/
    `bas_artifact` vanished mid-graph before this fix), which is far worse
    than the informational log line LangGraph prints once this is set.
    `True` is safe specifically because this state is entirely our own
    trusted, internally-defined types — never externally-supplied data.
    """

    # Workflow checkpoints contain only trusted internal Pydantic artifact
    # models produced by this project. Allowing msgpack serialization for
    # project-defined modules prevents artifact loss during checkpoint
    # restoration while preserving deterministic replay. Do not remove
    # this in favor of the (seemingly stricter) default — the default
    # silently drops these fields on reload instead of erroring, which is
    # the actual bug this line fixes.
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=True))


def compile_checkpointed_graph(
    graph: StateGraph, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Compile `graph` so it checkpoints after every node and can be resumed.

    `checkpointer` defaults to `default_checkpointer()` — swap in any other
    `BaseCheckpointSaver` implementation (e.g. a persistent one) for
    cross-process durability; the pause/resume behavior here doesn't
    depend on which one is used.
    """

    checkpointer = checkpointer or default_checkpointer()
    return graph.compile(checkpointer=checkpointer, interrupt_after=list(NODE_NAMES))


def thread_config(thread_id: str) -> RunnableConfig:
    """The `config` dict LangGraph expects to identify one checkpointed run."""

    return {"configurable": {"thread_id": thread_id}}


# The `.invoke`/`.get_state` calls below interact with LangGraph's own
# heavily overloaded `Pregel` API, whose stub overloads mypy can't always
# resolve against our plain `WorkflowState`/`RunnableConfig` usage even
# though it's exactly the documented call shape (confirmed correct by the
# full test suite, including checkpoint-recovery and stress tests) — the
# `type: ignore` comments below are scoped to those exact boundary calls.


def _drain(compiled: CompiledStateGraph, config: RunnableConfig, result: WorkflowState) -> WorkflowState:
    """Keep resuming `compiled` at `config` until `get_state(config).next` is
    empty, i.e. the graph has reached END. Shared by `run_to_completion` and
    `resume_execution` — the only difference between them is what `result`
    (and the graph's position) starts as.
    """

    while compiled.get_state(config).next:  # type: ignore[arg-type]
        result = compiled.invoke(None, config=config)  # type: ignore[call-overload,assignment]
    return result


def run_to_completion(
    compiled: CompiledStateGraph, initial_state: WorkflowState, thread_id: str
) -> WorkflowState:
    """Run `compiled` to completion, checkpointing after every node."""

    config = thread_config(thread_id)
    result = compiled.invoke(initial_state, config=config)  # type: ignore[call-overload]
    return _drain(compiled, config, result)  # type: ignore[arg-type]


def is_complete(compiled: CompiledStateGraph, thread_id: str) -> bool:
    """Whether the checkpointed run for `thread_id` has reached END."""

    return not compiled.get_state(thread_config(thread_id)).next  # type: ignore[arg-type]


def resume_execution(compiled: CompiledStateGraph, thread_id: str) -> WorkflowState:
    """Continue an interrupted/incomplete checkpointed run to completion."""

    config = thread_config(thread_id)
    result = compiled.get_state(config).values  # type: ignore[arg-type]
    return _drain(compiled, config, result)  # type: ignore[arg-type]


def checkpointed_state(compiled: CompiledStateGraph, thread_id: str) -> WorkflowState:
    """The last checkpointed `WorkflowState` for `thread_id`, without advancing it."""

    return compiled.get_state(thread_config(thread_id)).values  # type: ignore[arg-type,return-value]


def recover_failed_session(state: WorkflowState) -> WorkflowState:
    """Return a new state with the current session's recorded errors cleared.

    A deliberate, explicit operation (not automatic): clearing errors for
    `state['current_session_id']` lets `route_next_step` retry that
    session's current interaction instead of routing straight to
    `finalize_session`. Returns a plain new dict rather than mutating
    `state`, since the caller is expected to feed this into a *fresh*
    thread (a fresh `thread_id` has no prior checkpoint to merge against,
    so the `errors` reducer won't re-append what was just cleared).
    """

    session_id = state.get("current_session_id")
    cleaned = dict(state)
    cleaned["errors"] = [
        error for error in state.get("errors", []) if error.get("session_id") != session_id
    ]
    return cleaned  # type: ignore[return-value]
