"""Module 11, Step 12: Reporting.

Markdown + JSON reports over an already-computed `InterventionArtifact` — no
recomputation, no plots, mirroring `bas/report.py`/`reward/report.py`.
"""

from __future__ import annotations

from dataset_generator.intervention.models import InterventionArtifact


def build_json_report(artifact: InterventionArtifact) -> dict[str, object]:
    """A structured, JSON-friendly report over `artifact`."""

    return {
        "schema_version": artifact.schema_version,
        "config_fingerprint": artifact.config_fingerprint,
        "generation_timestamp": artifact.generation_timestamp,
        "statistics": artifact.statistics.model_dump(mode="json"),
        "session_summaries": [s.model_dump(mode="json") for s in artifact.session_summaries],
    }


def render_markdown_report(artifact: InterventionArtifact) -> str:
    """Render a Markdown intervention report suitable for a research log."""

    stats = artifact.statistics
    lines: list[str] = ["# Adaptive Intervention Engine Report", ""]

    lines.append("## Dataset Summary")
    lines.append(f"- Decisions: {stats.record_count}")
    lines.append(f"- Sessions: {len(artifact.session_summaries)}")
    lines.append(f"- Intervention rate: {stats.intervention_rate:.1%}")
    lines.append(f"- Average confidence: {stats.average_confidence:.3f}")
    lines.append(f"- Average need score: {stats.average_need_score:.3f}")
    lines.append(f"- Cooldown suppression rate: {stats.cooldown_suppression_rate:.1%}")
    lines.append(f"- Schema version: {artifact.schema_version}")
    lines.append(f"- Config fingerprint: `{artifact.config_fingerprint}`")
    lines.append("")

    lines.append("## Policy Distribution")
    lines.append("| Policy | Share |")
    lines.append("|---|---|")
    for policy, share in sorted(stats.policy_distribution.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {policy} | {share:.1%} |")
    lines.append("")

    if stats.missing_value_summary:
        lines.append("## Missing Signals")
        lines.append("| Signal | Missing count |")
        lines.append("|---|---|")
        for signal, count in sorted(stats.missing_value_summary.items()):
            lines.append(f"| {signal} | {count} |")
        lines.append("")

    lines.append("## Session Summaries")
    lines.append(
        "| Session | Student | Interactions | Interventions | Avg confidence | "
        "Avg severity | Cumulative BAS gain | Cumulative reward gain | Suppressions |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for summary in sorted(artifact.session_summaries, key=lambda s: s.session_id):
        lines.append(
            f"| {summary.session_id} | {summary.student_id} | {summary.interaction_count} | "
            f"{summary.intervention_count} | {summary.average_confidence:.3f} | "
            f"{summary.average_severity_score:.3f} | {summary.estimated_cumulative_bas_gain:.3f} | "
            f"{summary.estimated_cumulative_reward_gain:.3f} | {summary.cooldown_suppressions} |"
        )
    lines.append("")

    lines.append("## Decision Table (first 20)")
    lines.append("| Session | Interaction | Need | Severity | Chosen Policy | Confidence |")
    lines.append("|---|---|---|---|---|---|")
    for decision in artifact.decisions[:20]:
        lines.append(
            f"| {decision.session_id} | {decision.interaction_number} | "
            f"{decision.need_score:.3f} | {decision.severity} | {decision.chosen_policy} | "
            f"{decision.confidence:.3f} |"
        )

    return "\n".join(lines)
