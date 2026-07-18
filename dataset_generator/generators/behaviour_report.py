"""Behaviour Report (Module 5, Step 8).

Mirrors `prompt_report.py`/`response_report.py`'s design: built from an
exported list of `BehaviourRecord`s, independent of any single
`BehaviourGenerator` instance.

"Transition frequencies" is reconstructed empirically from consecutive
records sharing a `(student_id, session_id)` — this assumes `records` is
either already ordered by `interaction_number` within each
(student, session) group, or is sorted before being passed in; the report
sorts internally to avoid relying on caller ordering.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from dataset_generator.models.behaviour import BehaviourRecord
from dataset_generator.validators.behaviour_validator import validate_behaviour_batch

_LATENCY_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 2.0, "0-2s"),
    (2.0, 5.0, "2-5s"),
    (5.0, 10.0, "5-10s"),
    (10.0, 20.0, "10-20s"),
    (20.0, float("inf"), "20s+"),
)

_ENGAGEMENT_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.25, "0.00-0.25"),
    (0.25, 0.50, "0.25-0.50"),
    (0.50, 0.75, "0.50-0.75"),
    (0.75, 1.0001, "0.75-1.00"),
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
class ProfileSummary:
    average_latency: float
    average_engagement: float
    average_fatigue: float
    count: int


@dataclass(frozen=True)
class BehaviourValidationReport:
    total_records: int
    latency_distribution: dict[str, float]
    engagement_distribution: dict[str, float]
    state_frequencies: dict[str, float]
    transition_frequencies: dict[str, float]
    profile_summaries: dict[str, ProfileSummary]
    average_fatigue_by_progress_decile: dict[str, float]
    validation_failure_count: int


def _empirical_transitions(records: list[BehaviourRecord]) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[BehaviourRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.student_id, record.session_id)].append(record)

    transition_counts: Counter[str] = Counter()
    total_transitions = 0
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: r.interaction_number)
        for previous, current in zip(ordered, ordered[1:]):
            key = f"{previous.attention_state.value}->{current.attention_state.value}"
            transition_counts[key] += 1
            total_transitions += 1

    if total_transitions == 0:
        return {}
    return {key: count / total_transitions for key, count in transition_counts.items()}


def _profile_summaries(records: list[BehaviourRecord]) -> dict[str, ProfileSummary]:
    by_profile: dict[str, list[BehaviourRecord]] = defaultdict(list)
    for record in records:
        by_profile[record.metadata.student_profile].append(record)

    summaries: dict[str, ProfileSummary] = {}
    for profile, group in by_profile.items():
        n = len(group)
        summaries[profile] = ProfileSummary(
            average_latency=sum(r.response_latency for r in group) / n,
            average_engagement=sum(r.engagement_score for r in group) / n,
            average_fatigue=sum(r.fatigue_level for r in group) / n,
            count=n,
        )
    return summaries


def _fatigue_by_progress_decile(records: list[BehaviourRecord]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        decile = min(9, int(record.metadata.session_progress * 10))
        label = f"{decile * 10}-{decile * 10 + 10}%"
        buckets[label].append(record.fatigue_level)
    return {label: sum(values) / len(values) for label, values in buckets.items()}


def build_behaviour_report(records: list[BehaviourRecord]) -> BehaviourValidationReport:
    """Compute a `BehaviourValidationReport` over `records`."""

    total = len(records)
    if total == 0:
        raise ValueError("cannot build a report for an empty behaviour-record list")

    latency_labels = [_bucketize(r.response_latency, _LATENCY_BINS) for r in records]
    engagement_labels = [_bucketize(r.engagement_score, _ENGAGEMENT_BINS) for r in records]
    state_labels = [r.attention_state.value for r in records]

    return BehaviourValidationReport(
        total_records=total,
        latency_distribution=_distribution(latency_labels),
        engagement_distribution=_distribution(engagement_labels),
        state_frequencies=_distribution(state_labels),
        transition_frequencies=_empirical_transitions(records),
        profile_summaries=_profile_summaries(records),
        average_fatigue_by_progress_decile=_fatigue_by_progress_decile(records),
        validation_failure_count=len(validate_behaviour_batch(records)),
    )


def render_behaviour_report(report: BehaviourValidationReport) -> str:
    """Render `report` as a plain-text summary suitable for logs or a paper figure."""

    lines = [f"Total behaviour records: {report.total_records}", ""]

    lines.append("Response latency:")
    for _, _, label in _LATENCY_BINS:
        if label in report.latency_distribution:
            lines.append(f"  {label}: {report.latency_distribution[label]:.0%}")
    lines.append("")

    lines.append("Engagement score:")
    for _, _, label in _ENGAGEMENT_BINS:
        if label in report.engagement_distribution:
            lines.append(f"  {label}: {report.engagement_distribution[label]:.0%}")
    lines.append("")

    lines.append("Attention states:")
    for key, value in sorted(report.state_frequencies.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    lines.append("Observed transitions:")
    for key, value in sorted(report.transition_frequencies.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    lines.append("Profile summaries (avg latency / avg engagement / avg fatigue):")
    for profile, summary in sorted(report.profile_summaries.items()):
        lines.append(
            f"  {profile}: {summary.average_latency:.2f}s / "
            f"{summary.average_engagement:.2f} / {summary.average_fatigue:.2f} "
            f"(n={summary.count})"
        )
    lines.append("")

    lines.append("Average fatigue by session progress:")
    for label, value in sorted(report.average_fatigue_by_progress_decile.items(), key=lambda kv: int(kv[0].split("-")[0])):
        lines.append(f"  {label}: {value:.2f}")
    lines.append("")

    lines.append(f"Validation failures: {report.validation_failure_count}")

    return "\n".join(lines)
