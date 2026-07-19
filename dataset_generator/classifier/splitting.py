"""Module 8, Step 4: Train/Validation Split.

Every split mode is built from scikit-learn's own splitters
(`train_test_split`, `GroupShuffleSplit`, `StratifiedKFold`, `GroupKFold`,
`StratifiedGroupKFold`) — no custom shuffling/grouping logic. The one thing
genuinely specific to this project is `assert_no_student_leakage`, since
student-aware splitting is a correctness requirement the brief calls out
explicitly ("never leak students between train and validation").
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

SplitMode = Literal["random", "stratified", "session_aware", "student_aware"]


def assert_no_student_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Raise `ValueError` if any `student_id` appears in both `train_df` and `val_df`."""

    overlap = set(train_df["student_id"]) & set(val_df["student_id"])
    if overlap:
        raise ValueError(f"student leakage between train/validation: {sorted(overlap)[:10]}...")


def random_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plain i.i.d. row-level split."""

    return train_test_split(df, test_size=test_size, random_state=random_state)


def stratified_split(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-level split preserving `target_column`'s class proportions."""

    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_column]
    )


def session_aware_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by `session_id` — every row of a session stays on one side."""

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df["session_id"]))
    return df.iloc[train_idx], df.iloc[val_idx]


def student_aware_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by `student_id` — every row of a student stays on one side.

    This is the mode that must never leak a student across train/validation;
    `GroupShuffleSplit` groups by `student_id` structurally (a student is
    either entirely in train or entirely in validation), and the result is
    additionally verified with `assert_no_student_leakage` before returning.
    """

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df["student_id"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    assert_no_student_leakage(train_df, val_df)
    return train_df, val_df


def split_dataset(
    df: pd.DataFrame, mode: SplitMode, target_column: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch to the split function matching `mode`."""

    if mode == "random":
        return random_split(df, test_size, random_state)
    if mode == "stratified":
        return stratified_split(df, target_column, test_size, random_state)
    if mode == "session_aware":
        return session_aware_split(df, test_size, random_state)
    if mode == "student_aware":
        return student_aware_split(df, test_size, random_state)
    raise ValueError(f"unknown split mode {mode!r}")


def make_cv_splitter(mode: SplitMode, n_splits: int, random_state: int):
    """Build a scikit-learn cross-validation splitter for `mode`.

    `student_aware` uses `StratifiedGroupKFold` when available (folds are
    both group-disjoint on `student_id` *and* class-balanced); `session_aware`
    uses plain `GroupKFold` (session-disjoint only, no stratification
    guarantee, since a per-session class balance isn't generally meaningful).
    """

    if mode == "random":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if mode == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if mode == "session_aware":
        return GroupKFold(n_splits=n_splits)
    if mode == "student_aware":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    raise ValueError(f"unknown split mode {mode!r}")


def cv_groups(df: pd.DataFrame, mode: SplitMode) -> pd.Series | None:
    """The `groups` array `make_cv_splitter`'s splitter needs (or `None` if group-less)."""

    if mode == "session_aware":
        return df["session_id"]
    if mode == "student_aware":
        return df["student_id"]
    return None
