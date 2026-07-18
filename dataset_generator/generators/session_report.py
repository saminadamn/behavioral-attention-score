"""Session Report (Module 6, Step 9).

Unlike the Prompt/Response/Behaviour reports (which summarize a *batch* of
independent records), a session report describes **one session's internal
timeline** — everything it needs is already sitting on the `SessionRecord`
itself (`interactions`, `transition_history`, `statistics`), so nothing here
recomputes anything Module 6 already tracked incrementally.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from dataset_generator.models.session import SessionRecord
from dataset_generator.validators.session_validator import validate_session


@dataclass(frozen=True)
class SessionReport:
    session_id: str
    student_id: str
    student_profile: str
    attention_timeline: list[str]
    fatigue_progression: list[float]
    engagement_progression: list[float]
    latency_progression: list[float]
    recovered_transition_matrix: dict[str, dict[str, float]]
    state_frequencies: dict[str, int]
    total_interactions: int
    total_duration_seconds: float
    intervention_count: int
    validation_issues: list[str]


def _recovered_transition_matrix(record: SessionRecord) -> dict[str, dict[str, float]]:
    """Reconstruct an empirical transition matrix from `statistics.transition_counts`.

    "Recovered from observations" — i.e. what this *one* session's actual
    state sequence looked like, not the configured matrix it was sampled
    from (compare the two to sanity-check the simulator against config).
    """

    totals_from: dict[str, int] = defaultdict(int)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for key, count in record.statistics.transition_counts.items():
        from_state, to_state = key.split("->")
        counts[from_state][to_state] += count
        totals_from[from_state] += count

    matrix: dict[str, dict[str, float]] = {}
    for from_state, row_counts in counts.items():
        total = totals_from[from_state]
        matrix[from_state] = {to_state: count / total for to_state, count in row_counts.items()}
    return matrix


def build_session_report(record: SessionRecord) -> SessionReport:
    """Compute a `SessionReport` from `record`'s already-tracked history."""

    interactions = record.interactions
    return SessionReport(
        session_id=record.session_id,
        student_id=record.student_id,
        student_profile=record.student_profile,
        attention_timeline=[i.behaviour.attention_state.value for i in interactions],
        fatigue_progression=[i.behaviour.fatigue_level for i in interactions],
        engagement_progression=[i.response.engagement_proxy for i in interactions],
        latency_progression=[i.behaviour.response_latency for i in interactions],
        recovered_transition_matrix=_recovered_transition_matrix(record),
        state_frequencies=record.statistics.state_frequencies,
        total_interactions=record.summary.total_interactions,
        total_duration_seconds=record.statistics.total_duration_seconds,
        intervention_count=record.summary.intervention_count,
        validation_issues=validate_session(record),
    )


def render_session_report(report: SessionReport) -> str:
    """Render `report` as a plain-text summary."""

    lines = [
        f"Session: {report.session_id} (student {report.student_id}, profile {report.student_profile})",
        f"Total interactions: {report.total_interactions}",
        f"Total duration: {report.total_duration_seconds:.1f}s",
        "",
    ]

    lines.append("Attention timeline:")
    lines.append("  " + " -> ".join(report.attention_timeline))
    lines.append("")

    lines.append("Fatigue progression:")
    lines.append("  " + ", ".join(f"{v:.2f}" for v in report.fatigue_progression))
    lines.append("")

    lines.append("Engagement progression:")
    lines.append("  " + ", ".join(f"{v:.2f}" for v in report.engagement_progression))
    lines.append("")

    lines.append("Latency progression (s):")
    lines.append("  " + ", ".join(f"{v:.1f}" for v in report.latency_progression))
    lines.append("")

    lines.append("Recovered transition matrix (from observations):")
    for from_state in sorted(report.recovered_transition_matrix):
        row = report.recovered_transition_matrix[from_state]
        row_text = ", ".join(f"{to_state}: {p:.0%}" for to_state, p in sorted(row.items()))
        lines.append(f"  {from_state} -> {{{row_text}}}")
    lines.append("")

    lines.append("State frequencies:")
    for state, count in sorted(report.state_frequencies.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {state}: {count}")
    lines.append("")

    lines.append(f"Interventions: {report.intervention_count}")
    lines.append(f"Validation issues: {len(report.validation_issues)}")
    if report.validation_issues:
        for issue in report.validation_issues:
            lines.append(f"  - {issue}")

    return "\n".join(lines)
