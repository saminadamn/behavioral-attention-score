"""Module 9, Step 10: Session Summary.

Consumes already-computed `BASRecord`s (ordered by `interaction_number`) —
no recomputation of anything `BASEngine` already produced.
"""

from __future__ import annotations

import statistics

from dataset_generator.bas.config import BASConfig
from dataset_generator.bas.models import BASRecord, BASSessionSummary

_TREND_TOLERANCE = 0.02


def build_session_summary(records: list[BASRecord], config: BASConfig) -> BASSessionSummary:
    """Build a `BASSessionSummary` from one session's ordered `BASRecord`s."""

    if not records:
        raise ValueError("cannot build a session summary from an empty record list")

    scores = [r.score for r in records]
    n = len(scores)

    average = sum(scores) / n
    minimum = min(scores)
    maximum = max(scores)
    variance = statistics.pvariance(scores) if n >= 2 else 0.0

    half = max(1, n // 2)
    first_half_avg = sum(scores[:half]) / half
    second_half_avg = sum(scores[-half:]) / half
    delta = second_half_avg - first_half_avg
    if delta > _TREND_TOLERANCE:
        trend = "improving"
    elif delta < -_TREND_TOLERANCE:
        trend = "declining"
    else:
        trend = "stable"

    drops = [scores[i] - scores[i + 1] for i in range(n - 1)]
    recoveries = [scores[i + 1] - scores[i] for i in range(n - 1)]
    largest_drop = max((d for d in drops if d > 0), default=0.0)
    largest_recovery = max((r for r in recoveries if r > 0), default=0.0)

    above = sum(1 for s in scores if s >= config.attention_threshold)
    time_above_threshold = above / n
    time_below_threshold = 1.0 - time_above_threshold

    return BASSessionSummary(
        student_id=records[0].student_id,
        session_id=records[0].session_id,
        interaction_count=n,
        average_bas=average,
        minimum_bas=minimum,
        maximum_bas=maximum,
        variance_bas=variance,
        attention_trend=trend,
        largest_drop=largest_drop,
        largest_recovery=largest_recovery,
        time_above_threshold=time_above_threshold,
        time_below_threshold=time_below_threshold,
    )
