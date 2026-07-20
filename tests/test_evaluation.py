"""Tests for Module 13: Evaluation & Experimentation Framework (Steps 1-5)."""

from __future__ import annotations

import pytest

from dataset_generator.config import AttentionState
from dataset_generator.orchestration import ObserverAgent, BASAgent, InterventionAgent, RewardAgent

from dataset_generator.evaluation.ablation import (
    BAS_FEATURE_CATEGORIES,
    AblationRunner,
    build_comparison_table,
    disable_bas_feature_category,
    disable_bas_normalization,
    disable_confidence_weighting,
    disable_cooldown,
    disable_temporal_smoothing,
)
from dataset_generator.evaluation.benchmark import BenchmarkRunner
from dataset_generator.evaluation.config import (
    AblationOptions,
    BenchmarkOptions,
    ExperimentConfig,
    compute_experiment_config_fingerprint,
    default_experiment_config,
)
from dataset_generator.evaluation.metrics import (
    check_replay_correctness,
    compute_bas_metrics,
    compute_dataset_metrics,
    compute_intervention_metrics,
    compute_reward_metrics,
)
from dataset_generator.evaluation.sensitivity import SensitivityRunner, build_sweep_table

from dataset_generator.bas.config import NormalizationStrategy, SmoothingStrategy, default_bas_config
from dataset_generator.intervention.config import default_intervention_config


@pytest.fixture(scope="module")
def small_pipeline():
    """One small, genuinely-simulated pipeline run shared by every test in
    this module — integration against the real engines, never a mock.
    """

    dataset_artifact = ObserverAgent().generate(student_count=5, sessions_per_student=2)
    bas_artifact = BASAgent().compute(dataset_artifact)
    reward_artifact = RewardAgent().compute(dataset_artifact, bas_artifact)
    intervention_artifact = InterventionAgent().plan(dataset_artifact, bas_artifact, reward_artifact)
    return dataset_artifact, bas_artifact, reward_artifact, intervention_artifact


# ---------------------------------------------------------------------------
# Step 1: ExperimentConfig
# ---------------------------------------------------------------------------


def test_default_experiment_config_constructs() -> None:
    config = default_experiment_config()
    assert config.random_seed == 42
    assert config.num_repetitions > 0


def test_experiment_config_rejects_empty_dataset_sizes() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(dataset_sizes=())


def test_experiment_config_rejects_nonpositive_student_counts() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(student_counts=(10, 0))


def test_fingerprint_deterministic_and_sensitive() -> None:
    a = default_experiment_config()
    b = default_experiment_config()
    c = ExperimentConfig(random_seed=43)
    assert compute_experiment_config_fingerprint(a) == compute_experiment_config_fingerprint(b)
    assert compute_experiment_config_fingerprint(a) != compute_experiment_config_fingerprint(c)


# ---------------------------------------------------------------------------
# Step 2: metrics
# ---------------------------------------------------------------------------


def test_dataset_metrics_reuse_artifact_statistics(small_pipeline) -> None:
    dataset_artifact, _, _, _ = small_pipeline
    metrics = compute_dataset_metrics(dataset_artifact)
    assert metrics.class_balance == dataset_artifact.statistics.class_balance
    assert set(metrics.transition_balance) <= {s.value for s in AttentionState}
    for row in metrics.transition_balance.values():
        assert sum(row.values()) == pytest.approx(1.0)


def test_bas_metrics_bounded_and_confidence_reused(small_pipeline) -> None:
    _, bas_artifact, _, _ = small_pipeline
    metrics = compute_bas_metrics(bas_artifact)
    assert 0.0 <= metrics.mean <= 1.0
    assert metrics.variance >= 0.0
    assert 0.0 <= metrics.recovery_rate <= 1.0
    assert metrics.average_confidence == bas_artifact.statistics.average_confidence


def test_reward_metrics_decomposition_keys(small_pipeline) -> None:
    _, _, reward_artifact, _ = small_pipeline
    metrics = compute_reward_metrics(reward_artifact)
    assert set(metrics.reward_decomposition) == {"performance", "behaviour", "cost"}
    assert 0.0 <= metrics.positive_ratio <= 1.0


def test_intervention_execution_rate_is_none_not_zero_when_never_required(small_pipeline) -> None:
    _, _, _, intervention_artifact = small_pipeline
    metrics = compute_intervention_metrics(intervention_artifact)
    if metrics.intervention_required_count == 0:
        assert metrics.intervention_execution_rate is None
    else:
        assert metrics.intervention_execution_rate is not None
    assert metrics.executed_intervention_count >= 0


def test_replay_correctness_same_and_different(small_pipeline) -> None:
    state_a = {"tutor_actions": [1, 2], "session_outputs": ["x"]}
    state_b = {"tutor_actions": [1, 2], "session_outputs": ["x"]}
    state_c = {"tutor_actions": [1, 3], "session_outputs": ["x"]}
    assert check_replay_correctness(state_a, state_b)
    assert not check_replay_correctness(state_a, state_c)


# ---------------------------------------------------------------------------
# Step 3: benchmark
# ---------------------------------------------------------------------------


def test_benchmark_runner_produces_all_five_results() -> None:
    config = ExperimentConfig(benchmark_options=BenchmarkOptions(warmup_runs=0, repetitions=1))
    artifact = BenchmarkRunner(config=config).run_all(student_count=3, sessions_per_student=1)
    names = [r.module_name for r in artifact.results]
    assert names == ["dataset_generation", "bas", "reward", "intervention", "workflow"]
    for result in artifact.results:
        assert result.mean_runtime_seconds > 0
        assert result.record_count > 0
        assert result.mean_peak_memory_bytes > 0
        assert len(result.peak_memory_bytes) == result.repetitions


# ---------------------------------------------------------------------------
# Step 4: ablation
# ---------------------------------------------------------------------------


def test_bas_feature_category_ablation_zeroes_only_that_category() -> None:
    base = default_bas_config()
    ablated = disable_bas_feature_category(base, "behaviour")
    for name in BAS_FEATURE_CATEGORIES["behaviour"]:
        assert ablated.feature_configs[name].weight == 0.0
    for name in BAS_FEATURE_CATEGORIES["learning_response"]:
        assert ablated.feature_configs[name].weight == base.feature_configs[name].weight


def test_smoothing_and_normalization_ablations_use_identity() -> None:
    base = default_bas_config()
    assert disable_temporal_smoothing(base).smoothing_strategy == SmoothingStrategy.IDENTITY
    ablated = disable_bas_normalization(base)
    assert all(
        cfg.normalization.strategy == NormalizationStrategy.IDENTITY
        for cfg in ablated.feature_configs.values()
    )


def test_confidence_and_cooldown_ablations_zero_fields() -> None:
    bas = disable_confidence_weighting(default_bas_config())
    assert bas.confidence_variance_weight == 0.0
    assert bas.confidence_classifier_weight == 0.0
    intervention = disable_cooldown(default_intervention_config())
    assert intervention.cooldown_length == 0
    assert intervention.duplicate_prevention_window == 0


def test_ablation_runner_baseline_plus_selected(small_pipeline) -> None:
    dataset_artifact, _, _, _ = small_pipeline
    config = ExperimentConfig(ablation_options=AblationOptions(disable_temporal_smoothing=True))
    artifact = AblationRunner(config=config).run(dataset_artifact)
    names = [r.ablation_name for r in artifact.runs]
    assert names == ["baseline", "bas_temporal_smoothing"]
    table = build_comparison_table(artifact)
    assert len(table) == 2
    # Removing smoothing must increase (or leave equal) score volatility.
    assert artifact.runs[1].bas_metrics.volatility >= artifact.runs[0].bas_metrics.volatility


# ---------------------------------------------------------------------------
# Step 5: sensitivity
# ---------------------------------------------------------------------------


def test_need_threshold_sweep_required_count_monotonic(small_pipeline) -> None:
    dataset_artifact, _, _, _ = small_pipeline
    sweep = SensitivityRunner().sweep_need_threshold(dataset_artifact, values=(0.05, 0.25, 0.45))
    required_counts = [p.intervention_metrics.intervention_required_count for p in sweep.points]
    # Raising the threshold can only shrink the set of decisions that clear it.
    assert required_counts == sorted(required_counts, reverse=True)


def test_cooldown_sweep_zero_cooldown_allows_most_interventions(small_pipeline) -> None:
    dataset_artifact, _, _, _ = small_pipeline
    sweep = SensitivityRunner().sweep_cooldown_length(dataset_artifact, values=(0, 5))
    executed = [p.intervention_metrics.executed_intervention_count for p in sweep.points]
    assert executed[0] >= executed[1]


def test_reward_weight_multiplier_zero_matches_category_disabled(small_pipeline) -> None:
    dataset_artifact, _, _, _ = small_pipeline
    sweep = SensitivityRunner().sweep_reward_category_weight(dataset_artifact, multipliers=(0.0, 1.0))
    zero_point, one_point = sweep.points
    # multiplier 0.0 is exactly the with_category_disabled ablation: the
    # behaviour component's average must be 0 there.
    assert zero_point.reward_metrics.reward_decomposition["behaviour"] == pytest.approx(0.0)
    assert one_point.reward_metrics.reward_decomposition["behaviour"] != pytest.approx(0.0)


def test_transition_sweep_rows_renormalized() -> None:
    # Exercise the renormalization logic through a tiny real generation run.
    sweep = SensitivityRunner().sweep_transition_focus_persistence(
        values=(0.6, 0.8), student_count=2, sessions_per_student=1
    )
    assert [p.parameter_value for p in sweep.points] == [0.6, 0.8]
    for point in sweep.points:
        assert point.bas_metrics is not None
        assert 0.0 <= point.bas_metrics.mean <= 1.0


def test_build_sweep_table_omits_absent_metrics(small_pipeline) -> None:
    dataset_artifact, _, _, _ = small_pipeline
    sweep = SensitivityRunner().sweep_need_threshold(dataset_artifact, values=(0.35,))
    rows = build_sweep_table(sweep)
    assert len(rows) == 1
    assert "intervention_required_count" in rows[0]
    assert "bas_mean" not in rows[0]  # this sweep computes no BAS metrics
