"""Tests for Module 12: LangGraph Multi-Agent Orchestration."""

from __future__ import annotations

from dataset_generator.intervention import InterventionPolicyFactory
from dataset_generator.models.dataset import DatasetRecord

from dataset_generator.orchestration import (
    NODE_NAMES,
    POLICY_ACTION_TYPES,
    ObserverAgent,
    SessionAgent,
    SessionWalkResult,
    TutorAgent,
    WorkflowMemory,
    build_graph,
    build_json_report,
    checkpointed_state,
    compile_checkpointed_graph,
    compile_graph,
    config_fingerprints,
    is_complete,
    load_workflow_state,
    new_workflow_state,
    recover_failed_session,
    render_markdown_report,
    resume_execution,
    route_next_step,
    run_to_completion,
    save_workflow_state,
    thread_config,
)
from dataset_generator.orchestration.nodes import (
    make_compute_bas_node,
    make_compute_reward_node,
    make_finalize_session_node,
    make_generate_tutor_action_node,
    make_load_dataset_node,
    make_plan_intervention_node,
    ordered_decisions_for_session,
)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def test_build_graph_has_all_six_nodes() -> None:
    graph = build_graph(student_count=2, sessions_per_student=1)
    compiled = compile_graph(graph)
    node_names = set(compiled.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert node_names == set(NODE_NAMES)


def test_compile_graph_runs_end_to_end() -> None:
    graph = build_graph(student_count=2, sessions_per_student=2)
    compiled = compile_graph(graph)
    result = compiled.invoke(new_workflow_state())
    assert result["current_session_id"] is None
    assert len(result["session_outputs"]) == len(result["session_ids"])


def test_policy_action_type_table_covers_all_registered_policies() -> None:
    assert set(POLICY_ACTION_TYPES.keys()) == set(InterventionPolicyFactory.names())


# ---------------------------------------------------------------------------
# Individual nodes
# ---------------------------------------------------------------------------


REDUCED_FIELDS = {"timing_stats", "execution_history", "errors", "tutor_actions", "session_outputs"}


def _apply(state: dict, update: dict) -> dict:
    for key, value in update.items():
        state[key] = state.get(key, []) + value if key in REDUCED_FIELDS else value
    return state


def _phase1_state(student_count: int = 2, sessions_per_student: int = 2) -> dict:
    state = new_workflow_state()
    _apply(state, make_load_dataset_node(student_count=student_count, sessions_per_student=sessions_per_student)(state))
    _apply(state, make_compute_bas_node()(state))
    _apply(state, make_compute_reward_node()(state))
    _apply(state, make_plan_intervention_node()(state))
    return state


def test_load_dataset_node_sets_cursors() -> None:
    state = new_workflow_state()
    update = make_load_dataset_node(student_count=2, sessions_per_student=2)(state)
    assert update["session_ids"]
    assert update["current_session_id"] == update["session_ids"][0]
    assert update["current_interaction_index"] == 0
    assert update["current_student_id"] is not None


def test_load_dataset_node_passes_through_injected_dataset() -> None:
    observer = ObserverAgent()
    injected = observer.generate(student_count=1, sessions_per_student=1)
    state = new_workflow_state()
    state["dataset_artifact"] = injected
    update = make_load_dataset_node()(state)
    assert update["dataset_artifact"] is injected


def test_compute_bas_node_matches_direct_engine_call() -> None:
    from dataset_generator.bas import BASEngine

    state = _phase1_state()
    direct = BASEngine().compute(state["dataset_artifact"])
    assert state["bas_artifact"].records == direct.records


def test_compute_reward_node_matches_direct_engine_call() -> None:
    from dataset_generator.reward import RewardEngine

    state = _phase1_state()
    direct = RewardEngine().compute(state["dataset_artifact"], state["bas_artifact"])
    assert state["reward_artifact"].records == direct.records


def test_plan_intervention_node_matches_direct_planner_call() -> None:
    from dataset_generator.intervention import InterventionPlanner

    state = _phase1_state()
    direct = InterventionPlanner().plan(
        state["dataset_artifact"], state["bas_artifact"], state["reward_artifact"]
    )
    assert state["intervention_artifact"].decisions == direct.decisions


def test_generate_tutor_action_node_produces_one_action_and_advances_cursor() -> None:
    state = _phase1_state()
    tutor_node = make_generate_tutor_action_node()
    idx_before = state["current_interaction_index"]
    update = tutor_node(state)
    assert len(update["tutor_actions"]) == 1
    assert update["current_interaction_index"] == idx_before + 1


def test_generate_tutor_action_node_no_op_past_session_end() -> None:
    state = _phase1_state(student_count=1, sessions_per_student=1)
    ordered = ordered_decisions_for_session(state, state["current_session_id"])
    state["current_interaction_index"] = len(ordered)
    update = make_generate_tutor_action_node()(state)
    assert "tutor_actions" not in update
    assert "current_interaction_index" not in update


def test_finalize_session_node_advances_to_next_session() -> None:
    state = _phase1_state(student_count=2, sessions_per_student=2)
    session_id = state["current_session_id"]
    ordered = ordered_decisions_for_session(state, session_id)
    tutor_node = make_generate_tutor_action_node()
    while state["current_interaction_index"] < len(ordered):
        _apply(state, tutor_node(state))

    update = make_finalize_session_node()(state)
    assert len(update["session_outputs"]) == 1
    output = update["session_outputs"][0]
    assert output["session_id"] == session_id
    assert output["interactions_processed"] == len(ordered)
    assert output["terminated_early"] is False
    assert update["current_session_index"] == 1
    assert update["current_session_id"] in state["session_ids"]


def test_finalize_session_node_reports_early_termination_on_max_limit() -> None:
    state = _phase1_state(student_count=1, sessions_per_student=1)
    state["max_interactions_per_session"] = 3
    ordered = ordered_decisions_for_session(state, state["current_session_id"])
    tutor_node = make_generate_tutor_action_node()
    for _ in range(min(3, len(ordered))):
        _apply(state, tutor_node(state))

    update = make_finalize_session_node()(state)
    output = update["session_outputs"][0]
    assert output["interactions_processed"] == 3
    assert output["terminated_early"] is True
    assert output["termination_reason"] == "max_interaction_limit"


def test_node_error_is_captured_not_raised() -> None:
    state = new_workflow_state()  # no dataset_artifact/bas_artifact/etc. set
    update = make_compute_bas_node()(state)
    assert update["errors"]
    assert update["errors"][0]["node_name"] == "ComputeBASNode"


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def test_route_next_step_end_when_no_sessions() -> None:
    state = new_workflow_state()
    state["session_ids"] = []
    state["current_session_id"] = None
    assert route_next_step(state) == "end"


def test_route_next_step_finalize_when_interactions_exhausted() -> None:
    state = _phase1_state(student_count=1, sessions_per_student=1)
    ordered = ordered_decisions_for_session(state, state["current_session_id"])
    state["current_interaction_index"] = len(ordered)
    assert route_next_step(state) == "finalize_session"


def test_route_next_step_finalize_on_max_limit() -> None:
    state = _phase1_state(student_count=1, sessions_per_student=1)
    state["max_interactions_per_session"] = 2
    state["current_interaction_index"] = 2
    assert route_next_step(state) == "finalize_session"


def test_route_next_step_finalize_on_session_error() -> None:
    state = _phase1_state(student_count=1, sessions_per_student=1)
    state["errors"] = [{
        "node_name": "GenerateTutorActionNode", "session_id": state["current_session_id"],
        "interaction_index": 0, "message": "boom",
    }]
    assert route_next_step(state) == "finalize_session"


def test_route_next_step_tutor_intervention_or_continue() -> None:
    state = _phase1_state(student_count=3, sessions_per_student=2)
    ordered = ordered_decisions_for_session(state, state["current_session_id"])
    decision = ordered[0]
    key = route_next_step(state)
    if decision.intervention_required:
        assert key == "tutor_intervention"
    else:
        assert key == "continue_session"
    assert key in {"tutor_intervention", "continue_session"}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def test_tutor_agent_never_touches_scoring() -> None:
    state = _phase1_state(student_count=1, sessions_per_student=1)
    decision = state["intervention_artifact"].decisions[0]
    action = TutorAgent().generate_action(decision)
    assert action["source_policy"] == decision.chosen_policy
    assert action["confidence"] == decision.confidence
    assert action["message"] == f"[{POLICY_ACTION_TYPES[decision.chosen_policy]}] {decision.chosen_reason}"


def test_session_agent_finalize_computes_final_bas_and_reward() -> None:
    state = _phase1_state(student_count=1, sessions_per_student=1)
    session_id = state["current_session_id"]
    bas_records = sorted(
        (r for r in state["bas_artifact"].records if r.session_id == session_id),
        key=lambda r: r.interaction_number,
    )[:5]
    reward_records = sorted(
        (r for r in state["reward_artifact"].records if r.session_id == session_id),
        key=lambda r: r.interaction_number,
    )[:5]
    walk = SessionWalkResult(
        student_id="S00001", session_id=session_id, decisions=[], tutor_actions=[],
        bas_records=bas_records, reward_records=reward_records,
    )
    output = SessionAgent().finalize(walk)
    assert output["final_bas"] == bas_records[-1].score
    assert output["final_reward"] == reward_records[-1].reward


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_memory_filters_by_session_and_student() -> None:
    graph = build_graph(student_count=2, sessions_per_student=2)
    result = compile_graph(graph).invoke(new_workflow_state())
    mem = WorkflowMemory(result)

    session_id = result["session_ids"][0]
    student_id = result["session_outputs"][0]["student_id"]

    bas_records = mem.bas_history(session_id)
    assert all(r.session_id == session_id for r in bas_records)
    assert bas_records == sorted(bas_records, key=lambda r: r.interaction_number)

    reward_records = mem.reward_history(session_id)
    assert all(r.session_id == session_id for r in reward_records)

    decisions = mem.previous_interventions(session_id=session_id)
    assert all(d.session_id == session_id for d in decisions)

    actions = mem.previous_tutor_actions(student_id=student_id)
    assert all(a["student_id"] == student_id for a in actions)

    sessions = mem.session_history(student_id=student_id)
    assert all(s["student_id"] == student_id for s in sessions)


def test_memory_empty_state_returns_empty_lists() -> None:
    mem = WorkflowMemory(new_workflow_state())
    assert mem.bas_history() == []
    assert mem.reward_history() == []
    assert mem.previous_interventions() == []
    assert mem.previous_tutor_actions() == []
    assert mem.session_history() == []


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_checkpoint_every_node_pauses_between_calls() -> None:
    graph = build_graph(student_count=1, sessions_per_student=1)
    compiled = compile_checkpointed_graph(graph)
    config = thread_config("t_pause")
    compiled.invoke(new_workflow_state(), config=config)
    assert compiled.get_state(config).next == ("compute_bas",)


def test_run_to_completion_matches_plain_invoke() -> None:
    graph_a = build_graph(student_count=2, sessions_per_student=1)
    plain_result = compile_graph(graph_a).invoke(new_workflow_state())

    graph_b = build_graph(student_count=2, sessions_per_student=1)
    compiled_b = compile_checkpointed_graph(graph_b)
    checkpointed_result = run_to_completion(compiled_b, new_workflow_state(), "t_full")

    assert plain_result["session_outputs"] == checkpointed_result["session_outputs"]
    assert plain_result["tutor_actions"] == checkpointed_result["tutor_actions"]


def test_resume_execution_continues_interrupted_run() -> None:
    graph = build_graph(student_count=2, sessions_per_student=2)
    compiled = compile_checkpointed_graph(graph)
    config = thread_config("t_resume")
    compiled.invoke(new_workflow_state(), config=config)
    assert not is_complete(compiled, "t_resume")

    result = resume_execution(compiled, "t_resume")
    assert is_complete(compiled, "t_resume")
    assert len(result["session_outputs"]) == len(result["session_ids"])


def test_checkpointed_state_does_not_advance() -> None:
    graph = build_graph(student_count=1, sessions_per_student=1)
    compiled = compile_checkpointed_graph(graph)
    config = thread_config("t_peek")
    compiled.invoke(new_workflow_state(), config=config)
    before = checkpointed_state(compiled, "t_peek")
    after = checkpointed_state(compiled, "t_peek")
    assert before["current_interaction_index"] == after["current_interaction_index"]


def test_recover_failed_session_clears_only_current_session_errors() -> None:
    state = _phase1_state(student_count=2, sessions_per_student=1)
    other_session = [sid for sid in state["session_ids"] if sid != state["current_session_id"]][0]
    state["errors"] = [
        {"node_name": "X", "session_id": state["current_session_id"], "interaction_index": 0, "message": "a"},
        {"node_name": "X", "session_id": other_session, "interaction_index": 0, "message": "b"},
    ]
    recovered = recover_failed_session(state)
    assert recovered["errors"] == [
        {"node_name": "X", "session_id": other_session, "interaction_index": 0, "message": "b"}
    ]
    assert state["errors"] != recovered["errors"]  # original state untouched


def test_recover_failed_session_end_to_end() -> None:
    graph = build_graph(student_count=1, sessions_per_student=1)
    compiled = compile_checkpointed_graph(graph)
    config = thread_config("t_fail")
    compiled.invoke(new_workflow_state(), config=config)
    for _ in range(3):
        compiled.invoke(None, config=config)  # through plan_intervention

    mid_state = checkpointed_state(compiled, "t_fail")
    mid_state["errors"] = mid_state.get("errors", []) + [{
        "node_name": "GenerateTutorActionNode", "session_id": mid_state["current_session_id"],
        "interaction_index": mid_state["current_interaction_index"], "message": "simulated crash",
    }]
    recovered = recover_failed_session(mid_state)
    assert recovered["errors"] == []

    result = run_to_completion(compiled, recovered, "t_fail_recovered")
    assert result["errors"] == []
    assert len(result["session_outputs"]) == 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_load_workflow_state_round_trip(tmp_path) -> None:
    graph = build_graph(student_count=2, sessions_per_student=1)
    result = compile_graph(graph).invoke(new_workflow_state())

    save_workflow_state(result, tmp_path)
    loaded = load_workflow_state(tmp_path)

    assert loaded["dataset_artifact"] == result["dataset_artifact"]
    assert loaded["bas_artifact"] == result["bas_artifact"]
    assert loaded["reward_artifact"] == result["reward_artifact"]
    assert loaded["intervention_artifact"] == result["intervention_artifact"]
    assert loaded["tutor_actions"] == result["tutor_actions"]
    assert loaded["session_outputs"] == result["session_outputs"]
    assert loaded["errors"] == result["errors"]


def test_config_fingerprints_present_after_full_run() -> None:
    graph = build_graph(student_count=1, sessions_per_student=1)
    result = compile_graph(graph).invoke(new_workflow_state())
    fingerprints = config_fingerprints(result)
    assert all(fingerprints[key] for key in ("dataset", "bas", "reward", "intervention"))


def test_config_fingerprints_empty_state() -> None:
    fingerprints = config_fingerprints(new_workflow_state())
    assert all(v is None for v in fingerprints.values())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_render_markdown_report_contains_sections() -> None:
    graph = build_graph(student_count=2, sessions_per_student=2)
    result = compile_graph(graph).invoke(new_workflow_state())
    report = render_markdown_report(result)
    assert "# Orchestration Execution Report" in report
    assert "## Execution Summary" in report
    assert "## Decision Counts" in report
    assert "## Intervention Frequencies" in report
    assert "## Node / Agent Timings" in report


def test_build_json_report_structure_and_consistency() -> None:
    graph = build_graph(student_count=2, sessions_per_student=2)
    result = compile_graph(graph).invoke(new_workflow_state())
    report = build_json_report(result)

    stats = report["graph_statistics"]
    total_calls = sum(entry["call_count"] for entry in report["node_timings"].values())
    assert total_calls == stats["nodes_executed"]
    assert stats["tutor_actions_generated"] == report["decision_counts"]["total_actions"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_full_run_deterministic() -> None:
    def run():
        graph = build_graph(student_count=3, sessions_per_student=2)
        return compile_graph(graph).invoke(new_workflow_state())

    r1, r2 = run(), run()
    assert r1["tutor_actions"] == r2["tutor_actions"]
    assert r1["session_outputs"] == r2["session_outputs"]
    assert r1["bas_artifact"].records == r2["bas_artifact"].records


# ---------------------------------------------------------------------------
# Parallel / multiple sessions and batch execution
# ---------------------------------------------------------------------------


def test_multiple_sessions_processed_independently() -> None:
    graph = build_graph(student_count=4, sessions_per_student=3)
    result = compile_graph(graph).invoke(new_workflow_state())
    assert len(result["session_ids"]) == 12
    assert len(result["session_outputs"]) == 12
    session_ids_in_outputs = {o["session_id"] for o in result["session_outputs"]}
    assert session_ids_in_outputs == set(result["session_ids"])


def test_batch_execution_scales_with_student_count() -> None:
    small = compile_graph(build_graph(student_count=1, sessions_per_student=1)).invoke(new_workflow_state())
    large = compile_graph(build_graph(student_count=5, sessions_per_student=2)).invoke(new_workflow_state())
    assert len(large["session_outputs"]) > len(small["session_outputs"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_dataset_produces_no_sessions() -> None:
    class EmptyObserver(ObserverAgent):
        def generate(self, student_count=None, sessions_per_student=2):
            artifact = super().generate(student_count=1, sessions_per_student=1)
            return artifact.model_copy(
                update={"records": [], "metadata": artifact.metadata.model_copy(update={"record_count": 0})}
            )

    graph = build_graph(observer=EmptyObserver())
    result = compile_graph(graph).invoke(new_workflow_state())
    assert result["session_ids"] == []
    assert result["session_outputs"] == []
    assert result["tutor_actions"] == []


def test_single_interaction_session() -> None:
    graph = build_graph(student_count=1, sessions_per_student=1)
    result = compile_graph(graph).invoke(new_workflow_state())
    assert len(result["session_outputs"]) == 1
    assert result["session_outputs"][0]["interactions_processed"] > 0


def test_max_interactions_per_session_enforced_across_all_sessions() -> None:
    graph = build_graph(student_count=3, sessions_per_student=2)
    state = new_workflow_state(max_interactions_per_session=3)
    result = compile_graph(graph).invoke(state)
    assert all(o["interactions_processed"] <= 3 for o in result["session_outputs"])


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------


def test_stress_100000_workflow_executions() -> None:
    """100,000+ workflow executions: as with Modules 7/8/9/10/11's stress
    tests, replicate a real, small simulated batch with re-suffixed IDs
    rather than brute-force re-running full simulation (or the full graph,
    including its LoadDataset/BAS/Reward/Intervention batch phase) 100,000+
    times — a single fresh `compile_graph().invoke()` per run would take
    hours at that scale purely from per-run generation overhead.

    Instead: one large, replicated `DatasetArtifact` is injected once, and
    ONE compiled graph run walks it — Phase 1 (BAS/Reward/Intervention)
    computes over the whole large dataset in one batch call each (already
    proven to scale to 100,000+ records by Modules 9/10/11's own stress
    tests), and Phase 2's per-interaction loop then dispatches
    `GenerateTutorActionNode` once per interaction — genuinely exercising
    the graph's routing/cursor/accumulation control flow 100,000+ times.
    """

    base_dataset_artifact = ObserverAgent().generate(student_count=10, sessions_per_student=2)
    base_records = base_dataset_artifact.records
    target_size = 100_000
    replication_factor = target_size // len(base_records) + 1

    records: list[DatasetRecord] = []
    for batch in range(replication_factor):
        for record in base_records:
            data = record.model_dump(mode="json")
            data["session_id"] = f"{record.session_id}_batch{batch}"
            data["student_id"] = f"{record.student_id}_batch{batch}"
            data["response_id"] = f"{record.response_id}_batch{batch}"
            records.append(DatasetRecord.model_validate(data))

    assert len(records) >= target_size

    large_dataset_artifact = base_dataset_artifact.model_copy(
        update={
            "records": records,
            "metadata": base_dataset_artifact.metadata.model_copy(update={"record_count": len(records)}),
        }
    )

    total_sessions = len({r.session_id for r in records})

    state = new_workflow_state()
    state["dataset_artifact"] = large_dataset_artifact
    graph = build_graph()
    # Phase 2 takes one LangGraph step per interaction (GenerateTutorActionNode)
    # PLUS one step per session (FinalizeSessionNode) — see graph.py's
    # "Operational note on very large batches". A flat +1,000 buffer is not
    # enough once replication produces thousands of sessions (a prior run
    # with `len(records) + 1_000` hit LangGraph's GraphRecursionError after
    # ~2 hours because total_sessions alone exceeded that buffer). Sizing
    # the limit against both terms explicitly, plus the 4 batch-phase nodes
    # and a safety margin, avoids under-provisioning again.
    recursion_limit = len(records) + total_sessions + 1_000
    result = compile_graph(graph).invoke(state, config={"recursion_limit": recursion_limit})

    assert len(result["tutor_actions"]) == len(records)
    assert sum(o["interactions_processed"] for o in result["session_outputs"]) == len(records)
    assert result["errors"] == []
