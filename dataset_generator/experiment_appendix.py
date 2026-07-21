"""Phases 5-9 helper functions for the thesis appendix — statistical
significance, multi-seed aggregation, and confusion-matrix error analysis.

Ablation (Phase 5) and reward-weight sensitivity (Phase 6) reuse Module 13's
existing `AblationRunner`/`SensitivityRunner` directly (see
`run_appendix_analysis.py`) — nothing new was built for those, since the
infrastructure already existed and was already tested. What's new here is
the statistics layer Phase 7 (multi-seed) and Phase 8 (significance
testing) needed, plus the confusion-matrix error-analysis helper for
Phase 9, none of which existed before.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Deliberately excludes the near-perfectly-separating response features
# (semantic_similarity, confidence, correctness) so a classifier trained on
# this list produces genuine errors to analyze — see
# docs/DEEP_LEARNING_COMPARISON.md's separability finding for why the full
# feature set produces a perfect confusion matrix with nothing to analyze.
RESTRICTED_FEATURES = [
    "behaviour_response_latency",
    "behaviour_interaction_duration",
    "behaviour_fatigue_level",
    "session_progress",
    "student_baseline_latency",
    "student_engagement_tendency",
]


@dataclass(frozen=True)
class SeedStatistics:
    values: list[float]
    mean: float
    std: float
    ci_lower: float
    ci_upper: float


def compute_seed_statistics(values: list[float], confidence: float = 0.95) -> SeedStatistics:
    """Mean, sample standard deviation, and a t-distribution confidence
    interval over repeated-seed measurements of one metric. Requires at
    least 2 values (a single seed has no variance to report).
    """

    if len(values) < 2:
        raise ValueError("compute_seed_statistics needs at least 2 seed values to report variance")

    arr = np.array(values)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    standard_error = std / np.sqrt(len(arr))
    ci_lower, ci_upper = stats.t.interval(confidence, len(arr) - 1, loc=mean, scale=standard_error)
    return SeedStatistics(values=list(values), mean=mean, std=std, ci_lower=float(ci_lower), ci_upper=float(ci_upper))


@dataclass(frozen=True)
class PairedSignificance:
    t_statistic: float
    t_test_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    cohens_d: float


def paired_significance(treatment: list[float], baseline: list[float]) -> PairedSignificance:
    """Paired t-test, Wilcoxon signed-rank, and Cohen's d comparing
    `treatment` against `baseline` across the same seeds (paired, not
    independent, samples — the same dataset/seed pair produced both
    values). Requires at least 2 paired observations.
    """

    if len(treatment) != len(baseline):
        raise ValueError("treatment and baseline must have the same length (paired observations)")
    if len(treatment) < 2:
        raise ValueError("paired_significance needs at least 2 paired observations")

    t_statistic, t_p = stats.ttest_rel(treatment, baseline)

    # Wilcoxon requires at least one non-zero difference.
    differences = np.array(treatment) - np.array(baseline)
    if np.all(differences == 0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon_statistic, wilcoxon_p = stats.wilcoxon(treatment, baseline)

    pooled_std = np.sqrt((np.std(treatment, ddof=1) ** 2 + np.std(baseline, ddof=1) ** 2) / 2)
    cohens_d = float((np.mean(treatment) - np.mean(baseline)) / pooled_std) if pooled_std > 0 else 0.0

    return PairedSignificance(
        t_statistic=float(t_statistic),
        t_test_p_value=float(t_p),
        wilcoxon_statistic=float(wilcoxon_statistic),
        wilcoxon_p_value=float(wilcoxon_p),
        cohens_d=cohens_d,
    )


@dataclass(frozen=True)
class ConfusionError:
    true_label: str
    predicted_label: str
    count: int
    share_of_all_errors: float


def analyze_confusion_errors(confusion_matrix: list[list[int]], class_labels: list[str]) -> list[ConfusionError]:
    """Every off-diagonal (true, predicted) cell, ranked by count — the
    concrete basis for an error-analysis writeup instead of eyeballing a
    matrix. `share_of_all_errors` divides by the total off-diagonal count,
    not the total record count, so it answers "of the mistakes made, how
    much does this one pair account for."
    """

    matrix = np.array(confusion_matrix)
    total_errors = int(matrix.sum() - np.trace(matrix))

    errors: list[ConfusionError] = []
    for i, true_label in enumerate(class_labels):
        for j, predicted_label in enumerate(class_labels):
            if i == j:
                continue
            count = int(matrix[i, j])
            if count == 0:
                continue
            share = count / total_errors if total_errors > 0 else 0.0
            errors.append(ConfusionError(true_label, predicted_label, count, share))

    return sorted(errors, key=lambda e: e.count, reverse=True)
