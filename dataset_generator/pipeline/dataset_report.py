"""Module 7, Step 8: Verification Report.

Renders everything already sitting on a `DatasetArtifact` (statistics,
validation, metadata, manifest, exports) into one plain-text report — no
recomputation, matching every other report module in this project
(`prompt_report.py`, `response_report.py`, `behaviour_report.py`,
`session_report.py`).
"""

from __future__ import annotations

from dataset_generator.models.dataset import DatasetArtifact


def render_dataset_report(artifact: DatasetArtifact) -> str:
    """Render a full verification report for `artifact`."""

    lines: list[str] = []

    # Dataset summary
    lines.append("=== Dataset Summary ===")
    lines.append(f"Records: {artifact.metadata.record_count}")
    lines.append(f"Students: {artifact.metadata.student_count}")
    lines.append(f"Sessions: {artifact.metadata.session_count}")
    lines.append(f"Subjects covered: {', '.join(artifact.metadata.subjects_covered)}")
    lines.append(f"Profiles covered: {', '.join(artifact.metadata.profiles_covered)}")
    lines.append("")

    # Validation summary
    validation = artifact.validation
    lines.append("=== Validation Summary ===")
    lines.append(f"Valid: {validation.is_valid}")
    lines.append(f"Duplicate rows: {validation.duplicate_row_count}")
    lines.append(f"Duplicate IDs: {validation.duplicate_id_count}")
    lines.append(f"Impossible transitions: {validation.impossible_transition_count}")
    lines.append(f"Invalid attention states: {validation.invalid_attention_state_count}")
    lines.append(f"Orphan session IDs: {len(validation.orphan_session_ids)}")
    lines.append(f"Orphan student IDs: {len(validation.orphan_student_ids)}")
    lines.append(f"NaN values: {validation.nan_count}")
    lines.append(f"Inf values: {validation.inf_count}")
    lines.append(f"Schema consistent: {validation.schema_consistent}")
    if validation.missing_value_issues:
        lines.append("Missing values by column:")
        for column, count in sorted(validation.missing_value_issues.items()):
            lines.append(f"  {column}: {count}")
    if validation.invalid_range_issues:
        lines.append("Out-of-range values by column:")
        for column, count in sorted(validation.invalid_range_issues.items()):
            lines.append(f"  {column}: {count}")
    lines.append("")

    # Feature summary
    statistics = artifact.statistics
    lines.append("=== Feature Summary ===")
    lines.append(f"Numeric features tracked: {len(statistics.feature_distributions)}")
    lines.append("Difficulty balance:")
    for key, value in sorted(statistics.difficulty_balance.items()):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("Subject balance:")
    for key, value in sorted(statistics.subject_balance.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    # Session summary
    lines.append("=== Session Summary ===")
    lines.append("Sessions per profile:")
    for key, value in sorted(statistics.session_balance.items()):
        lines.append(f"  {key}: {value}")
    lines.append("")

    # Attention summary
    lines.append("=== Attention Summary ===")
    for key, value in sorted(statistics.attention_balance.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    # Profile summary
    lines.append("=== Profile Summary (row-level) ===")
    for key, value in sorted(statistics.profile_balance.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    # Export summary
    lines.append("=== Export Summary ===")
    if artifact.exports:
        for fmt, path in sorted(artifact.exports.items()):
            lines.append(f"  {fmt}: {path}")
    else:
        lines.append("  (not yet exported)")
    lines.append("")

    # Manifest / versioning
    manifest = artifact.manifest
    lines.append("=== Manifest ===")
    lines.append(f"dataset_version: {manifest.dataset_version}")
    lines.append(f"schema_version: {manifest.schema_version}")
    lines.append(f"generator_version: {manifest.generator_version}")
    lines.append(f"generation_timestamp: {manifest.generation_timestamp}")
    lines.append(f"seed: {manifest.seed}")
    lines.append(f"config_fingerprint: {manifest.config_fingerprint}")
    lines.append(f"git_commit_hash: {manifest.git_commit_hash or '(unavailable)'}")

    return "\n".join(lines)
