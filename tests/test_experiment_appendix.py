"""Tests for dataset_generator.experiment_appendix — the statistics helpers
Phases 7-9 of the thesis appendix need (multi-seed aggregation,
significance testing, confusion-matrix error ranking).
"""

from __future__ import annotations

import pytest

from dataset_generator.experiment_appendix import (
    RESTRICTED_FEATURES,
    analyze_confusion_errors,
    compute_seed_statistics,
    paired_significance,
)


def test_compute_seed_statistics_basic():
    stats = compute_seed_statistics([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats.mean == pytest.approx(0.3)
    assert stats.std > 0
    assert stats.ci_lower < stats.mean < stats.ci_upper


def test_compute_seed_statistics_requires_at_least_two_values():
    with pytest.raises(ValueError):
        compute_seed_statistics([0.5])


def test_compute_seed_statistics_tighter_values_give_narrower_ci():
    tight = compute_seed_statistics([0.85, 0.86, 0.85, 0.87, 0.86])
    wide = compute_seed_statistics([0.1, 0.9, 0.2, 0.8, 0.5])
    tight_width = tight.ci_upper - tight.ci_lower
    wide_width = wide.ci_upper - wide.ci_lower
    assert tight_width < wide_width


def test_paired_significance_detects_a_real_difference():
    treatment = [0.85, 0.87, 0.86, 0.84, 0.85]
    baseline = [0.05, 0.07, 0.06, 0.04, 0.05]
    result = paired_significance(treatment, baseline)
    assert result.t_test_p_value < 0.05
    assert result.cohens_d > 1.0  # large effect


def test_paired_significance_no_real_difference_gives_high_p_value():
    """Small, non-degenerate noise around a common mean with no
    systematic paired shift — not the exactly-zero-difference case (which
    is a mathematically undefined 0/0 t-statistic, not a high p-value).
    """

    treatment = [0.50, 0.51, 0.49, 0.52, 0.48]
    baseline = [0.51, 0.49, 0.50, 0.50, 0.50]
    result = paired_significance(treatment, baseline)
    assert result.t_test_p_value > 0.3
    assert abs(result.cohens_d) < 0.5


def test_paired_significance_requires_equal_length():
    with pytest.raises(ValueError):
        paired_significance([0.1, 0.2], [0.1, 0.2, 0.3])


def test_paired_significance_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        paired_significance([0.1], [0.2])


def test_analyze_confusion_errors_ranks_by_count():
    matrix = [[134, 33, 0], [19, 276, 4], [0, 5, 80]]
    labels = ["Distracted", "Focused", "Impulsive"]
    errors = analyze_confusion_errors(matrix, labels)

    assert errors[0].true_label == "Distracted"
    assert errors[0].predicted_label == "Focused"
    assert errors[0].count == 33

    total_errors = sum(e.count for e in errors)
    assert total_errors == 33 + 19 + 4 + 5
    assert sum(e.share_of_all_errors for e in errors) == pytest.approx(1.0)


def test_analyze_confusion_errors_empty_for_perfect_diagonal():
    matrix = [[167, 0, 0], [0, 299, 0], [0, 0, 85]]
    labels = ["Distracted", "Focused", "Impulsive"]
    errors = analyze_confusion_errors(matrix, labels)
    assert errors == []


def test_restricted_features_excludes_near_separating_response_features():
    assert "response_semantic_similarity" not in RESTRICTED_FEATURES
    assert "response_confidence" not in RESTRICTED_FEATURES
    assert "response_correctness_score" not in RESTRICTED_FEATURES
    assert len(RESTRICTED_FEATURES) > 0
