"""Module 9, Step 12: Reporting.

Markdown + JSON reports, both built from an already-computed `BASArtifact`
— no recomputation. "Session plots metadata" is exactly that: a description
of what a plot *would* show (axes, series), not a rendered image — Step 12
explicitly says not to generate plots yet.
"""

from __future__ import annotations

from dataset_generator.bas.models import BASArtifact


def session_plot_metadata(artifact: BASArtifact) -> list[dict[str, object]]:
    """Descriptive (not rendered) plot specifications, one per session."""

    by_session: dict[str, list] = {}
    for record in artifact.records:
        by_session.setdefault(record.session_id, []).append(record)

    specs = []
    for session_id, records in by_session.items():
        ordered = sorted(records, key=lambda r: r.interaction_number)
        specs.append(
            {
                "session_id": session_id,
                "student_id": ordered[0].student_id,
                "chart_type": "line",
                "x_axis": "interaction_number",
                "y_axis": "score",
                "series": ["score", "raw_score", "confidence"],
                "x_values": [r.interaction_number for r in ordered],
                "series_values": {
                    "score": [r.score for r in ordered],
                    "raw_score": [r.raw_score for r in ordered],
                    "confidence": [r.confidence for r in ordered],
                },
            }
        )
    return specs


def build_json_report(artifact: BASArtifact) -> dict[str, object]:
    """A structured, JSON-friendly report over `artifact`."""

    return {
        "schema_version": artifact.schema_version,
        "config_fingerprint": artifact.config_fingerprint,
        "generation_timestamp": artifact.generation_timestamp,
        "statistics": artifact.statistics.model_dump(mode="json"),
        "session_summaries": [s.model_dump(mode="json") for s in artifact.session_summaries],
        "session_plots": session_plot_metadata(artifact),
    }


def render_markdown_report(artifact: BASArtifact) -> str:
    """Render a Markdown BAS report suitable for a research log."""

    lines: list[str] = ["# Behavioural Attention Score Report", ""]

    lines.append("## Dataset Summary")
    lines.append(f"- Records: {artifact.statistics.record_count}")
    lines.append(f"- Sessions: {len(artifact.session_summaries)}")
    lines.append(f"- Average BAS: {artifact.statistics.average_score:.3f}")
    lines.append(f"- Average confidence: {artifact.statistics.average_confidence:.3f}")
    lines.append(f"- Schema version: {artifact.schema_version}")
    lines.append(f"- Config fingerprint: `{artifact.config_fingerprint}`")
    lines.append("")

    lines.append("## Score Distribution")
    lines.append("| Range | Proportion |")
    lines.append("|---|---|")
    for label, value in sorted(artifact.statistics.score_distribution.items()):
        lines.append(f"| {label} | {value:.0%} |")
    lines.append("")

    lines.append("## Feature Contribution Summary")
    lines.append("| Feature | Avg. contribution |")
    lines.append("|---|---|")
    for feature, value in sorted(
        artifact.statistics.feature_contribution_summary.items(), key=lambda kv: -abs(kv[1])
    ):
        lines.append(f"| {feature} | {value:+.4f} |")
    lines.append("")

    if artifact.statistics.missing_value_summary:
        lines.append("## Missing Values")
        lines.append("| Feature | Missing count |")
        lines.append("|---|---|")
        for feature, count in sorted(artifact.statistics.missing_value_summary.items()):
            lines.append(f"| {feature} | {count} |")
        lines.append("")

    lines.append("## Session Summaries")
    lines.append("| Session | Student | Avg BAS | Min | Max | Trend | Time above threshold |")
    lines.append("|---|---|---|---|---|---|---|")
    for summary in sorted(artifact.session_summaries, key=lambda s: s.session_id):
        lines.append(
            f"| {summary.session_id} | {summary.student_id} | {summary.average_bas:.3f} | "
            f"{summary.minimum_bas:.3f} | {summary.maximum_bas:.3f} | {summary.attention_trend} | "
            f"{summary.time_above_threshold:.0%} |"
        )

    return "\n".join(lines)
