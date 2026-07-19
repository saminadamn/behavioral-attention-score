"""Module 12, Step 10: Reporting.

Markdown + JSON reports over a `WorkflowState` — no recomputation, no
plots, mirroring `bas/report.py`/`reward/report.py`/`intervention/report.py`.

"Node timings" and "agent timings" are the same aggregation over
`timing_stats` presented once: in this architecture every node wraps
exactly one agent call, so there is no separate "agent-level" duration to
report — computing it twice from the same data would be redundant, not a
second source of truth.

"Intervention frequencies" are counted from this run's own `tutor_actions`
(what was actually walked, respecting early termination/interaction
limits) rather than `intervention_artifact.statistics` (the full
batch-computed dataset, which may include sessions/interactions this run
never walked) — the orchestration-layer report describes what this run did.
"""

from __future__ import annotations

from collections import Counter
from typing import TypedDict

from dataset_generator.orchestration.state import WorkflowState


class FailureSummary(TypedDict):
    """Errors grouped by node and by session, for `render_markdown_report`."""

    total_errors: int
    by_node: dict[str, int]
    by_session: dict[str, int]
    messages: list[str]


def node_timing_summary(state: WorkflowState) -> dict[str, dict[str, float | int]]:
    """Per-node call count, total duration, and average duration from `timing_stats`."""

    totals: dict[str, list[float]] = {}
    for entry in state.get("timing_stats", []):
        totals.setdefault(entry["node_name"], []).append(entry["duration_seconds"])

    return {
        node_name: {
            "call_count": len(durations),
            "total_seconds": sum(durations),
            "average_seconds": sum(durations) / len(durations),
        }
        for node_name, durations in totals.items()
    }


def graph_statistics(state: WorkflowState) -> dict[str, object]:
    """High-level counts describing this run's shape."""

    session_outputs = state.get("session_outputs", [])
    total_interactions = sum(o["interactions_processed"] for o in session_outputs)
    early_terminated = sum(1 for o in session_outputs if o["terminated_early"])

    return {
        "sessions_total": len(state.get("session_ids", [])),
        "sessions_finalized": len(session_outputs),
        "sessions_terminated_early": early_terminated,
        "interactions_processed": total_interactions,
        "average_interactions_per_session": (
            total_interactions / len(session_outputs) if session_outputs else 0.0
        ),
        "nodes_executed": len(state.get("execution_history", [])),
        "tutor_actions_generated": len(state.get("tutor_actions", [])),
    }


def decision_counts(state: WorkflowState) -> dict[str, object]:
    """How many of this run's tutor actions represented a real intervention."""

    tutor_actions = state.get("tutor_actions", [])
    interventions = sum(1 for a in tutor_actions if a["source_policy"] != "NoInterventionPolicy")

    return {
        "total_actions": len(tutor_actions),
        "interventions": interventions,
        "no_interventions": len(tutor_actions) - interventions,
        "intervention_rate": (interventions / len(tutor_actions)) if tutor_actions else 0.0,
    }


def intervention_frequencies(state: WorkflowState) -> dict[str, float]:
    """Share of this run's tutor actions attributable to each policy."""

    tutor_actions = state.get("tutor_actions", [])
    if not tutor_actions:
        return {}
    counts = Counter(a["source_policy"] for a in tutor_actions)
    return {policy: count / len(tutor_actions) for policy, count in counts.items()}


def failure_summary(state: WorkflowState) -> FailureSummary:
    """Errors grouped by node and by session."""

    errors = state.get("errors", [])
    by_node = Counter(e["node_name"] for e in errors)
    by_session = Counter(e["session_id"] for e in errors if e["session_id"] is not None)

    return FailureSummary(
        total_errors=len(errors),
        by_node=dict(by_node),
        by_session=dict(by_session),
        messages=[e["message"] for e in errors],
    )


def build_json_report(state: WorkflowState) -> dict[str, object]:
    """A structured, JSON-friendly report over `state`."""

    timings = node_timing_summary(state)
    return {
        "graph_statistics": graph_statistics(state),
        "decision_counts": decision_counts(state),
        "intervention_frequencies": intervention_frequencies(state),
        "node_timings": timings,
        "agent_timings": timings,
        "failure_summary": failure_summary(state),
    }


def render_markdown_report(state: WorkflowState) -> str:
    """Render a Markdown orchestration report suitable for a research log."""

    stats = graph_statistics(state)
    decisions = decision_counts(state)
    frequencies = intervention_frequencies(state)
    timings = node_timing_summary(state)
    failures = failure_summary(state)

    lines: list[str] = ["# Orchestration Execution Report", ""]

    lines.append("## Execution Summary")
    lines.append(f"- Sessions: {stats['sessions_finalized']} / {stats['sessions_total']}")
    lines.append(f"- Sessions terminated early: {stats['sessions_terminated_early']}")
    lines.append(f"- Interactions processed: {stats['interactions_processed']}")
    lines.append(
        f"- Average interactions per session: {stats['average_interactions_per_session']:.2f}"
    )
    lines.append(f"- Nodes executed: {stats['nodes_executed']}")
    lines.append(f"- Tutor actions generated: {stats['tutor_actions_generated']}")
    lines.append("")

    lines.append("## Decision Counts")
    lines.append(f"- Total actions: {decisions['total_actions']}")
    lines.append(f"- Interventions: {decisions['interventions']}")
    lines.append(f"- No-intervention actions: {decisions['no_interventions']}")
    lines.append(f"- Intervention rate: {decisions['intervention_rate']:.1%}")
    lines.append("")

    lines.append("## Intervention Frequencies")
    lines.append("| Policy | Share |")
    lines.append("|---|---|")
    for policy, share in sorted(frequencies.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {policy} | {share:.1%} |")
    lines.append("")

    lines.append("## Node / Agent Timings")
    lines.append("(Each node wraps exactly one agent call, so these are the same figures.)")
    lines.append("| Node | Calls | Total (s) | Average (s) |")
    lines.append("|---|---|---|---|")
    for node_name, entry in sorted(timings.items()):
        lines.append(
            f"| {node_name} | {entry['call_count']} | {entry['total_seconds']:.4f} | "
            f"{entry['average_seconds']:.6f} |"
        )
    lines.append("")

    if failures["total_errors"]:
        lines.append("## Failure Summary")
        lines.append(f"- Total errors: {failures['total_errors']}")
        lines.append("| Node | Count |")
        lines.append("|---|---|")
        for node_name, count in sorted(failures["by_node"].items()):
            lines.append(f"| {node_name} | {count} |")
        lines.append("")

    return "\n".join(lines)
