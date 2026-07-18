"""Module 7, Step 4: Dataset Validation.

Vectorized (pandas-based) checks across the *whole* dataset at once —
deliberately not a per-row Python loop, since this needs to stay fast at
100,000+ rows (Step 9's stress test). Returns one structured
`DatasetValidationReport` rather than a growing list of ad hoc strings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataset_generator.config.attention_state import AttentionState
from dataset_generator.models.dataset import DatasetRecord, DatasetValidationReport

_RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "response_correctness_score": (0.0, 1.0),
    "response_semantic_similarity": (0.0, 1.0),
    "response_lexical_diversity": (0.0, 1.0),
    "response_sentiment": (-1.0, 1.0),
    "response_engagement_proxy": (0.0, 1.0),
    "response_confidence": (0.0, 1.0),
    "response_repetition_ratio": (0.0, 1.0),
    "response_coherence_score": (0.0, 1.0),
    "response_topic_shift": (0.0, 1.0),
    "behaviour_response_latency": (0.0, float("inf")),
    "behaviour_interaction_duration": (0.0, float("inf")),
    "behaviour_hesitation_duration": (0.0, float("inf")),
    "behaviour_rolling_engagement": (0.0, 1.0),
    "behaviour_fatigue_level": (0.0, 1.0),
    "session_progress": (0.0, 1.0),
}

_VALID_ATTENTION_STATES = {state.value for state in AttentionState}


def records_to_frame(records: list[DatasetRecord]) -> pd.DataFrame:
    """Convert `records` to a `pandas.DataFrame`, columns ordered by field definition."""

    if not records:
        return pd.DataFrame(columns=list(DatasetRecord.model_fields))
    return pd.DataFrame([r.model_dump() for r in records])


def validate_dataset(
    records: list[DatasetRecord],
    known_student_ids: set[str] | None = None,
    known_session_ids: set[str] | None = None,
) -> DatasetValidationReport:
    """Validate `records`, returning a structured `DatasetValidationReport`.

    `known_student_ids`/`known_session_ids`, if given, enable orphan-ID
    detection (a record referencing a student/session that doesn't exist in
    the source generation run) — omitted, those checks report no orphans
    rather than false-flagging everything.
    """

    if not records:
        return DatasetValidationReport(
            record_count=0,
            missing_value_issues={},
            duplicate_row_count=0,
            duplicate_id_count=0,
            invalid_range_issues={},
            impossible_transition_count=0,
            invalid_attention_state_count=0,
            orphan_session_ids=[],
            orphan_student_ids=[],
            nan_count=0,
            inf_count=0,
            schema_consistent=True,
        )

    df = records_to_frame(records)

    expected_columns = set(DatasetRecord.model_fields)
    schema_consistent = set(df.columns) == expected_columns

    missing_value_issues = {
        col: int(count) for col, count in df.isna().sum().items() if count > 0
    }

    duplicate_row_count = int(df.duplicated().sum())
    duplicate_id_count = int(df["response_id"].duplicated().sum())

    invalid_range_issues: dict[str, int] = {}
    for column, (lo, hi) in _RANGE_CHECKS.items():
        if column not in df.columns:
            continue
        out_of_range = ((df[column] < lo) | (df[column] > hi)).sum()
        if out_of_range > 0:
            invalid_range_issues[column] = int(out_of_range)

    invalid_attention_state_count = int((~df["attention_state"].isin(_VALID_ATTENTION_STATES)).sum())

    sorted_df = df.sort_values(["session_id", "interaction_number"])
    previous_state = sorted_df.groupby("session_id")["attention_state"].shift(1)
    has_previous = previous_state.notna()
    actual_transition = has_previous & (previous_state != sorted_df["attention_state"])
    claimed_transition = sorted_df["behaviour_transition_occurred"].astype(bool)
    impossible_transition_count = int((has_previous & (actual_transition != claimed_transition)).sum())

    if known_student_ids is not None:
        orphan_student_ids = sorted(set(df["student_id"]) - known_student_ids)
    else:
        orphan_student_ids = []
    if known_session_ids is not None:
        orphan_session_ids = sorted(set(df["session_id"]) - known_session_ids)
    else:
        orphan_session_ids = []

    numeric_df = df.select_dtypes(include=[np.number])
    nan_count = int(numeric_df.isna().sum().sum())
    inf_count = int(np.isinf(numeric_df.to_numpy(dtype=float, na_value=0.0)).sum())

    return DatasetValidationReport(
        record_count=len(records),
        missing_value_issues=missing_value_issues,
        duplicate_row_count=duplicate_row_count,
        duplicate_id_count=duplicate_id_count,
        invalid_range_issues=invalid_range_issues,
        impossible_transition_count=impossible_transition_count,
        invalid_attention_state_count=invalid_attention_state_count,
        orphan_session_ids=orphan_session_ids,
        orphan_student_ids=orphan_student_ids,
        nan_count=nan_count,
        inf_count=inf_count,
        schema_consistent=schema_consistent,
    )
