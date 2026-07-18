"""Response Generation Report (Module 4, Step 9).

Mirrors `prompt_report.py`'s design: built from an exported list of
`Response`s, independent of any single `ResponseGenerator` instance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from dataset_generator.models.response import Response
from dataset_generator.validators.response_validator import (
    exact_duplicate_rate,
    validate_response_batch,
)

_CORRECTNESS_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.25, "0.00-0.25"),
    (0.25, 0.50, "0.25-0.50"),
    (0.50, 0.75, "0.50-0.75"),
    (0.75, 1.0001, "0.75-1.00"),
)

_SENTIMENT_BINS: tuple[tuple[float, float, str], ...] = (
    (-1.0001, -0.1, "negative"),
    (-0.1, 0.1, "neutral"),
    (0.1, 1.0001, "positive"),
)


def _bucketize(value: float, bins: tuple[tuple[float, float, str], ...]) -> str:
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return bins[-1][2]


def _distribution(labels: list[str]) -> dict[str, float]:
    total = len(labels)
    counts = Counter(labels)
    return {label: count / total for label, count in counts.items()}


@dataclass(frozen=True)
class ResponseValidationReport:
    total_responses: int
    average_response_length: float
    correctness_distribution: dict[str, float]
    semantic_similarity_distribution: dict[str, float]
    sentiment_distribution: dict[str, float]
    profile_distribution: dict[str, float]
    attention_state_distribution: dict[str, float]
    duplicate_rate: float
    validation_failure_count: int


def build_response_report(responses: list[Response]) -> ResponseValidationReport:
    """Compute a `ResponseValidationReport` over `responses`."""

    total = len(responses)
    if total == 0:
        raise ValueError("cannot build a report for an empty response list")

    correctness_labels = [_bucketize(r.correctness_score, _CORRECTNESS_BINS) for r in responses]
    similarity_labels = [_bucketize(r.semantic_similarity, _CORRECTNESS_BINS) for r in responses]
    sentiment_labels = [_bucketize(r.sentiment, _SENTIMENT_BINS) for r in responses]
    profile_labels = [r.metadata.student_profile for r in responses]
    attention_labels = [r.metadata.attention_state.value for r in responses]

    return ResponseValidationReport(
        total_responses=total,
        average_response_length=sum(r.response_length for r in responses) / total,
        correctness_distribution=_distribution(correctness_labels),
        semantic_similarity_distribution=_distribution(similarity_labels),
        sentiment_distribution=_distribution(sentiment_labels),
        profile_distribution=_distribution(profile_labels),
        attention_state_distribution=_distribution(attention_labels),
        duplicate_rate=exact_duplicate_rate(responses),
        validation_failure_count=len(validate_response_batch(responses)),
    )


def render_response_report(report: ResponseValidationReport) -> str:
    """Render `report` as a plain-text summary suitable for logs or a paper figure."""

    lines = [f"Total responses: {report.total_responses}", ""]

    lines.append("Correctness score:")
    for key in ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"):
        if key in report.correctness_distribution:
            lines.append(f"  {key}: {report.correctness_distribution[key]:.0%}")
    lines.append("")

    lines.append("Semantic similarity:")
    for key in ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"):
        if key in report.semantic_similarity_distribution:
            lines.append(f"  {key}: {report.semantic_similarity_distribution[key]:.0%}")
    lines.append("")

    lines.append("Sentiment:")
    for key in ("negative", "neutral", "positive"):
        if key in report.sentiment_distribution:
            lines.append(f"  {key}: {report.sentiment_distribution[key]:.0%}")
    lines.append("")

    lines.append("Student profiles:")
    for key, value in sorted(report.profile_distribution.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    lines.append("Attention states:")
    for key, value in sorted(report.attention_state_distribution.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    lines.append(f"Average response length: {report.average_response_length:.1f} tokens")
    lines.append(f"Duplicate rate: {report.duplicate_rate:.2%}")
    lines.append(f"Validation failures: {report.validation_failure_count}")

    return "\n".join(lines)
