"""Module 10, Step 10: Reporting.

Markdown + JSON reports over an already-computed `RewardArtifact` — no
recomputation. "Reward trend metadata" is descriptive data (per-session
reward/confidence sequences), not a rendered plot, per Step 10's explicit
"do not generate plots".
"""

from __future__ import annotations

from dataset_generator.reward.models import RewardArtifact


def reward_trend_metadata(artifact: RewardArtifact) -> list[dict[str, object]]:
    """Descriptive (not rendered) per-session reward trend data."""

    by_session: dict[str, list] = {}
    for record in artifact.records:
        by_session.setdefault(record.session_id, []).append(record)

    trends = []
    for session_id, records in by_session.items():
        ordered = sorted(records, key=lambda r: r.interaction_number)
        trends.append(
            {
                "session_id": session_id,
                "student_id": ordered[0].student_id,
                "x_axis": "interaction_number",
                "y_axis": "reward",
                "series": ["reward", "raw_reward", "confidence"],
                "x_values": [r.interaction_number for r in ordered],
                "series_values": {
                    "reward": [r.reward for r in ordered],
                    "raw_reward": [r.raw_reward for r in ordered],
                    "confidence": [r.confidence for r in ordered],
                },
            }
        )
    return trends


def build_json_report(artifact: RewardArtifact) -> dict[str, object]:
    """A structured, JSON-friendly report over `artifact`."""

    return {
        "schema_version": artifact.schema_version,
        "config_fingerprint": artifact.config_fingerprint,
        "generation_timestamp": artifact.generation_timestamp,
        "statistics": artifact.statistics.model_dump(mode="json"),
        "session_summaries": [s.model_dump(mode="json") for s in artifact.session_summaries],
        "reward_trends": reward_trend_metadata(artifact),
    }


def render_markdown_report(artifact: RewardArtifact) -> str:
    """Render a Markdown reward report suitable for a research log."""

    lines: list[str] = ["# Reward Model Report", ""]

    lines.append("## Dataset Summary")
    lines.append(f"- Records: {artifact.statistics.record_count}")
    lines.append(f"- Sessions: {len(artifact.session_summaries)}")
    lines.append(f"- Average reward: {artifact.statistics.average_reward:+.3f}")
    lines.append(f"- Average confidence: {artifact.statistics.average_confidence:.3f}")
    lines.append(f"- Schema version: {artifact.schema_version}")
    lines.append(f"- Config fingerprint: `{artifact.config_fingerprint}`")
    lines.append("")

    lines.append("## Reward Decomposition (R = Performance + Behaviour - Cost)")
    lines.append(f"- Performance: {artifact.statistics.average_performance_reward:+.3f}")
    lines.append(f"- Behaviour: {artifact.statistics.average_behaviour_reward:+.3f}")
    lines.append(f"- Cost: {artifact.statistics.average_cost_reward:.3f}")
    lines.append(
        f"- Total (Performance + Behaviour - Cost): "
        f"{(artifact.statistics.average_performance_reward + artifact.statistics.average_behaviour_reward - artifact.statistics.average_cost_reward):+.3f}"
    )
    lines.append("")

    lines.append("## Reward Distribution")
    lines.append("| Range | Proportion |")
    lines.append("|---|---|")
    for label, value in sorted(artifact.statistics.reward_distribution.items()):
        lines.append(f"| {label} | {value:.0%} |")
    lines.append("")

    lines.append("## Contribution Summary")
    lines.append("| Signal | Avg. contribution |")
    lines.append("|---|---|")
    for signal, value in sorted(
        artifact.statistics.contribution_summary.items(), key=lambda kv: -abs(kv[1])
    ):
        lines.append(f"| {signal} | {value:+.4f} |")
    lines.append("")

    if artifact.statistics.missing_value_summary:
        lines.append("## Missing Signals")
        lines.append("| Signal | Missing count |")
        lines.append("|---|---|")
        for signal, count in sorted(artifact.statistics.missing_value_summary.items()):
            lines.append(f"| {signal} | {count} |")
        lines.append("")

    lines.append("## Session Summaries")
    lines.append(
        "| Session | Student | Avg reward | Min | Max | Trend | Cumulative | Recoveries |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for summary in sorted(artifact.session_summaries, key=lambda s: s.session_id):
        lines.append(
            f"| {summary.session_id} | {summary.student_id} | {summary.average_reward:+.3f} | "
            f"{summary.minimum_reward:+.3f} | {summary.maximum_reward:+.3f} | {summary.reward_trend} | "
            f"{summary.cumulative_reward:+.3f} | {summary.recovery_count} |"
        )

    return "\n".join(lines)
