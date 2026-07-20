"""Module 7, Step 5: Dataset Statistics.

Also pandas-based (see `validators/dataset_validator.py`'s module docstring
for why) — one vectorized pass over the assembled dataset rather than a
per-record Python loop.

"Class balance" and "attention balance" are the same measurement
(empirical proportions of the `attention_state` column) under two names
Module 7's brief lists separately — computed once, exposed under both
`DatasetStatistics` fields, rather than genuinely duplicated.
"""

from __future__ import annotations

import pandas as pd

from dataset_generator.models.dataset import DatasetRecord, DatasetStatistics, FeatureDistributionSummary
from dataset_generator.pipeline.feature_registry import FeatureRegistry
from dataset_generator.validators.dataset_validator import records_to_frame


def _proportions(series: pd.Series) -> dict[str, float]:
    counts = series.value_counts()
    total = len(series)
    return {str(value): count / total for value, count in counts.items()}


def compute_dataset_statistics(
    records: list[DatasetRecord], registry: FeatureRegistry | None = None
) -> DatasetStatistics:
    """Compute a `DatasetStatistics` snapshot over `records`."""

    registry = registry or FeatureRegistry()
    if not records:
        return DatasetStatistics(
            record_count=0,
            feature_distributions={},
            class_balance={},
            attention_balance={},
            profile_balance={},
            session_balance={},
            subject_balance={},
            difficulty_balance={},
            correlation_matrix={},
            missing_value_summary={},
        )

    df = records_to_frame(records)

    feature_distributions: dict[str, FeatureDistributionSummary] = {}
    numeric_columns = [c for c in registry.numeric_features() if c in df.columns]
    for column in numeric_columns:
        series = df[column]
        feature_distributions[column] = FeatureDistributionSummary(
            mean=float(series.mean()),
            std=float(series.std(ddof=0)) if len(series) > 1 else 0.0,
            min=float(series.min()),
            max=float(series.max()),
            missing_count=int(series.isna().sum()),
        )

    attention_balance = _proportions(df["attention_state"])

    # "profile_balance" is row-level (fraction of interactions per profile);
    # "session_balance" is session-level (how many distinct sessions per
    # profile) — genuinely different denominators, not a duplicate stat.
    profile_balance = _proportions(df["student_profile"])
    session_balance = (
        df.drop_duplicates("session_id").groupby("student_profile")["session_id"].count().to_dict()
    )
    session_balance = {str(k): int(v) for k, v in session_balance.items()}

    subject_balance = _proportions(df["prompt_subject"])
    difficulty_balance = _proportions(df["prompt_difficulty"])

    correlation_frame = df[numeric_columns].corr(numeric_only=True) if numeric_columns else pd.DataFrame()
    correlation_matrix = {
        row: {
            # A column is always perfectly self-correlated by definition,
            # even when pandas reports NaN for a zero-variance column
            # (correlation is technically undefined there) — the diagonal
            # is always 1.0; only genuinely undefined *off*-diagonal pairs
            # (one or both columns constant) fall back to 0.0.
            col: (1.0 if col == row else (0.0 if pd.isna(value) else float(value)))
            for col, value in correlation_frame[row].items()
        }
        for row in correlation_frame.columns
    }

    missing_value_summary = {col: int(count) for col, count in df.isna().sum().items() if count > 0}

    return DatasetStatistics(
        record_count=len(records),
        feature_distributions=feature_distributions,
        class_balance=attention_balance,
        attention_balance=attention_balance,
        profile_balance=profile_balance,
        session_balance=session_balance,
        subject_balance=subject_balance,
        difficulty_balance=difficulty_balance,
        correlation_matrix=correlation_matrix,
        missing_value_summary=missing_value_summary,
    )
