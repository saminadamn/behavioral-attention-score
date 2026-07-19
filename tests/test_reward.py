"""Tests for Module 10: Reward Model."""

from __future__ import annotations

import pytest

from dataset_generator.bas import BASEngine
from dataset_generator.classifier import AttentionClassifierPredictor, AttentionClassifierTrainer, TrainingConfig
from dataset_generator.config import default_config
from dataset_generator.generators import generate_sessions, generate_students
from dataset_generator.pipeline import build_dataset_artifact
from dataset_generator.reward import (
    RewardAggregator,
    RewardCategory,
    RewardConfig,
    RewardEngine,
    RewardSignalConfig,
    RewardSignalExtractor,
    RewardSignalPolarity,
    TemporalMode,
    apply_temporal_credit_assignment,
    build_json_report,
    build_reward_session_summary,
    compute_reward_confidence,
    decompose_reward,
    default_reward_config,
    load_reward_artifact,
    render_markdown_report,
    save_reward_artifact,
)
from dataset_generator.reward.models import RewardContribution, RewardObservation, RewardRecord
from dataset_generator.bas.config import FeatureNormalizationConfig, NormalizationStrategy
from dataset_generator.models.dataset import DatasetRecord
from dataset_generator.utils import build_rng_streams


def _artifacts(n_students: int = 5, sessions_per_student: int = 2, seed: int | None = None):
    config = default_config()
    seed = seed if seed is not None else config.seed
    streams = build_rng_streams(seed)
    students = generate_students(config, streams)[:n_students]
    sessions = generate_sessions(config, students, sessions_per_student=sessions_per_student, rng_streams=build_rng_streams(seed))
    dataset_artifact = build_dataset_artifact(config, students, sessions)
    bas_artifact = BASEngine().compute(dataset_artifact)
    return dataset_artifact, bas_artifact


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def test_signal_extractor_first_interaction_has_none_deltas() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    extractor = RewardSignalExtractor()
    observations = extractor.extract_batch(dataset_artifact.records, bas_artifact.records)

    first_interactions = [o for o in observations if o.interaction_number == 1]
    assert first_interactions
    for obs in first_interactions:
        assert obs.raw_signals["delta_bas"] is None
        assert obs.raw_signals["delta_engagement"] is None
        assert obs.raw_signals["delta_correctness"] is None


def test_signal_extractor_produces_one_observation_per_record() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    extractor = RewardSignalExtractor()
    observations = extractor.extract_batch(dataset_artifact.records, bas_artifact.records)
    assert len(observations) == len(dataset_artifact.records)


def test_signal_extractor_intervention_cost_only_when_applied() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    extractor = RewardSignalExtractor()
    observations = extractor.extract_batch(dataset_artifact.records, bas_artifact.records)
    by_key = {(o.session_id, o.interaction_number): o for o in observations}
    for record in dataset_artifact.records:
        obs = by_key[(record.session_id, record.interaction_number)]
        if record.intervention_applied:
            assert obs.raw_signals["intervention_cost"] == 1.0
        else:
            assert obs.raw_signals["intervention_cost"] is None


def test_signal_extractor_delta_bas_matches_bas_scores() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    extractor = RewardSignalExtractor()
    observations = extractor.extract_batch(dataset_artifact.records, bas_artifact.records)

    bas_by_key = {(r.session_id, r.interaction_number): r.score for r in bas_artifact.records}
    by_session: dict[str, list] = {}
    for obs in observations:
        by_session.setdefault(obs.session_id, []).append(obs)

    for session_id, obs_list in by_session.items():
        ordered = sorted(obs_list, key=lambda o: o.interaction_number)
        for previous, current in zip(ordered, ordered[1:]):
            expected = bas_by_key[(current.session_id, current.interaction_number)] - bas_by_key[(previous.session_id, previous.interaction_number)]
            assert current.raw_signals["delta_bas"] == pytest.approx(expected)


def test_signal_extractor_with_classifier_confidence() -> None:
    dataset_artifact, bas_artifact = _artifacts(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(dataset_artifact, TrainingConfig(model_name="random_forest", split_mode="random"))
    predictor = AttentionClassifierPredictor(training_artifact)

    extractor = RewardSignalExtractor(predictor=predictor)
    observations = extractor.extract_batch(dataset_artifact.records, bas_artifact.records)
    second_interactions = [o for o in observations if o.interaction_number == 2]
    assert any(o.raw_signals["delta_classifier_confidence"] is not None for o in second_interactions)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregator_weighted_sum_matches_manual_calculation() -> None:
    config = RewardConfig(
        signal_configs={
            "a": RewardSignalConfig(weight=0.6, polarity=RewardSignalPolarity.POSITIVE, category=RewardCategory.PERFORMANCE, normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY)),
            "b": RewardSignalConfig(weight=0.4, polarity=RewardSignalPolarity.POSITIVE, category=RewardCategory.PERFORMANCE, normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY)),
        }
    )
    aggregator = RewardAggregator(config)
    observation = RewardObservation(student_id="S1", session_id="SESS1", interaction_number=2, raw_signals={"a": 1.0, "b": 0.0})
    raw_reward, contributions, missing_ratio = aggregator.aggregate(observation)
    # a: evidence=1.0 -> signed=1.0 -> contribution=0.6; b: evidence=0.0 -> signed=-1.0 -> contribution=-0.4
    assert raw_reward == pytest.approx(0.6 - 0.4)
    assert missing_ratio == 0.0


def test_aggregator_bounded_output() -> None:
    config = default_reward_config()
    aggregator = RewardAggregator(config)
    observation = RewardObservation(
        student_id="S1", session_id="SESS1", interaction_number=2,
        raw_signals={s: 10.0 for s in config.weighted_signals()},
    )
    raw_reward, _, _ = aggregator.aggregate(observation)
    assert config.reward_clip_min <= raw_reward <= config.reward_clip_max


def test_aggregator_penalty_always_negative_when_triggered() -> None:
    config = default_reward_config()
    aggregator = RewardAggregator(config)
    observation = RewardObservation(
        student_id="S1", session_id="SESS1", interaction_number=2,
        raw_signals={"intervention_cost": 1.0},
    )
    raw_reward, contributions, _ = aggregator.aggregate(observation)
    penalty_contribution = next(c for c in contributions if c.signal == "intervention_cost")
    assert penalty_contribution.signed_evidence == -1.0
    assert penalty_contribution.contribution < 0


def test_aggregator_missing_signals_renormalize_not_zero() -> None:
    config = RewardConfig(
        signal_configs={
            "a": RewardSignalConfig(weight=0.5, polarity=RewardSignalPolarity.POSITIVE, category=RewardCategory.PERFORMANCE, normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY)),
            "b": RewardSignalConfig(weight=0.5, polarity=RewardSignalPolarity.POSITIVE, category=RewardCategory.PERFORMANCE, normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY)),
        }
    )
    aggregator = RewardAggregator(config)
    observation = RewardObservation(student_id="S1", session_id="SESS1", interaction_number=2, raw_signals={"a": 1.0})
    raw_reward, _, missing_ratio = aggregator.aggregate(observation)
    assert raw_reward == pytest.approx(1.0)  # not halved by a phantom "b"
    assert missing_ratio == pytest.approx(0.5)


def test_aggregator_all_missing_returns_neutral_zero() -> None:
    config = default_reward_config()
    aggregator = RewardAggregator(config)
    observation = RewardObservation(student_id="S1", session_id="SESS1", interaction_number=2, raw_signals={})
    raw_reward, contributions, missing_ratio = aggregator.aggregate(observation)
    assert raw_reward == 0.0
    assert contributions == []
    assert missing_ratio == 1.0


def test_aggregator_contributions_ranked_descending() -> None:
    config = default_reward_config()
    aggregator = RewardAggregator(config)
    observation = RewardObservation(
        student_id="S1", session_id="SESS1", interaction_number=2,
        raw_signals={"delta_bas": 0.4, "delta_fatigue": 0.4, "delta_correctness": -0.4},
    )
    _, contributions, _ = aggregator.aggregate(observation)
    values = [c.contribution for c in contributions]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Temporal discounting
# ---------------------------------------------------------------------------


def test_temporal_immediate_passthrough() -> None:
    config = default_reward_config().model_copy(update={"temporal_mode": TemporalMode.IMMEDIATE})
    raw = [0.1, -0.2, 0.3]
    assert apply_temporal_credit_assignment(raw, config) == raw


def test_temporal_discounted_bounded_by_segment_range() -> None:
    config = default_reward_config().model_copy(update={"temporal_mode": TemporalMode.DISCOUNTED, "discount_factor": 0.9})
    raw = [0.1, -0.5, 1.0, -1.0]
    result = apply_temporal_credit_assignment(raw, config)
    for t, value in enumerate(result):
        segment = raw[t:]
        assert min(segment) - 1e-9 <= value <= max(segment) + 1e-9


def test_temporal_discounted_last_interaction_equals_raw() -> None:
    config = default_reward_config().model_copy(update={"temporal_mode": TemporalMode.DISCOUNTED, "discount_factor": 0.9})
    raw = [0.2, -0.3, 0.7]
    result = apply_temporal_credit_assignment(raw, config)
    assert result[-1] == pytest.approx(raw[-1])


def test_temporal_moving_average() -> None:
    config = default_reward_config().model_copy(update={"temporal_mode": TemporalMode.MOVING_AVERAGE, "rolling_window": 2})
    raw = [0.2, 0.4, 0.6]
    result = apply_temporal_credit_assignment(raw, config)
    assert result[0] == pytest.approx(0.2)
    assert result[1] == pytest.approx(0.3)  # mean(0.2, 0.4)
    assert result[2] == pytest.approx(0.5)  # mean(0.4, 0.6)


def test_temporal_unknown_mode_raises() -> None:
    config = default_reward_config()
    object.__setattr__(config, "temporal_mode", "not_a_mode")
    with pytest.raises(ValueError):
        apply_temporal_credit_assignment([0.1, 0.2], config)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_full_coverage_agreement_is_high() -> None:
    config = default_reward_config()
    contributions = [
        RewardContribution(signal="delta_bas", weight=0.3, category=RewardCategory.PERFORMANCE, evidence_value=0.9, signed_evidence=0.8, contribution=0.24),
    ]
    confidence, uncertainty, reliability = compute_reward_confidence(contributions, missing_ratio=0.0, config=config)
    assert confidence > 0.7
    assert uncertainty == pytest.approx(1.0 - confidence)


def test_confidence_drops_with_missing_ratio() -> None:
    config = default_reward_config()
    contributions = [
        RewardContribution(signal="delta_bas", weight=0.3, category=RewardCategory.PERFORMANCE, evidence_value=0.9, signed_evidence=0.8, contribution=0.24),
    ]
    full, _, _ = compute_reward_confidence(contributions, missing_ratio=0.0, config=config)
    partial, _, _ = compute_reward_confidence(contributions, missing_ratio=0.7, config=config)
    assert partial < full


def test_confidence_blends_bas_confidence() -> None:
    config = default_reward_config()
    contributions = [
        RewardContribution(signal="delta_bas", weight=0.3, category=RewardCategory.PERFORMANCE, evidence_value=0.5, signed_evidence=0.0, contribution=0.0),
    ]
    low, _, _ = compute_reward_confidence(contributions, missing_ratio=0.0, config=config, bas_confidence=0.0)
    high, _, _ = compute_reward_confidence(contributions, missing_ratio=0.0, config=config, bas_confidence=1.0)
    assert high > low


# ---------------------------------------------------------------------------
# Session summaries
# ---------------------------------------------------------------------------


def _make_reward_record(session_id: str, interaction_number: int, reward: float) -> RewardRecord:
    return RewardRecord(
        student_id="S1", session_id=session_id, interaction_number=interaction_number,
        raw_reward=reward, reward=reward,
        performance_reward=reward, behaviour_reward=0.0, cost_reward=0.0,
        contributions=[], confidence=0.9, uncertainty=0.1,
        reliability="high", missing_signal_ratio=0.0,
        metadata={"temporal_mode": "immediate", "discount_factor": 0.9, "config_fingerprint": "x", "config_version": "1.0.0"},
    )


def _make_observation(session_id: str, interaction_number: int, intervention: bool) -> RewardObservation:
    return RewardObservation(
        student_id="S1", session_id=session_id, interaction_number=interaction_number,
        raw_signals={"intervention_cost": 1.0 if intervention else None},
    )


def test_session_summary_basic_stats() -> None:
    config = default_reward_config()
    records = [_make_reward_record("SESS1", i + 1, r) for i, r in enumerate([0.1, 0.3, -0.2, 0.0])]
    observations = [_make_observation("SESS1", i + 1, False) for i in range(4)]
    summary = build_reward_session_summary(records, observations, config)
    assert summary.average_reward == pytest.approx(0.05)
    assert summary.maximum_reward == 0.3
    assert summary.minimum_reward == -0.2
    assert summary.cumulative_reward == pytest.approx(0.2)


def test_session_summary_recovery_count() -> None:
    config = default_reward_config().model_copy(update={"recovery_threshold": 0.2})
    records = [_make_reward_record("SESS1", i + 1, r) for i, r in enumerate([-0.5, 0.5, -0.1, 0.05])]
    observations = [
        _make_observation("SESS1", 1, intervention=True),
        _make_observation("SESS1", 2, intervention=False),
        _make_observation("SESS1", 3, intervention=True),
        _make_observation("SESS1", 4, intervention=False),
    ]
    summary = build_reward_session_summary(records, observations, config)
    # Intervention at interaction 1 -> reward at interaction 2 is 0.5 >= 0.2 -> recovery.
    # Intervention at interaction 3 -> reward at interaction 4 is 0.05 < 0.2 -> not a recovery.
    assert summary.recovery_count == 1


def test_session_summary_empty_raises() -> None:
    with pytest.raises(ValueError):
        build_reward_session_summary([], [], default_reward_config())


# ---------------------------------------------------------------------------
# End-to-end RewardEngine
# ---------------------------------------------------------------------------


def test_reward_engine_produces_one_record_per_dataset_record() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    assert len(reward_artifact.records) == len(dataset_artifact.records)


def test_reward_engine_bounded() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    for record in reward_artifact.records:
        assert -1.0 <= record.reward <= 1.0
        assert 0.0 <= record.confidence <= 1.0


def test_reward_engine_deterministic() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    a = RewardEngine().compute(dataset_artifact, bas_artifact)
    b = RewardEngine().compute(dataset_artifact, bas_artifact)
    assert [r.model_dump(mode="json") for r in a.records] == [r.model_dump(mode="json") for r in b.records]


def test_reward_engine_session_summaries_count() -> None:
    dataset_artifact, bas_artifact = _artifacts(n_students=5, sessions_per_student=2)
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    distinct_sessions = {r.session_id for r in dataset_artifact.records}
    assert len(reward_artifact.session_summaries) == len(distinct_sessions)


def test_reward_engine_with_classifier_confidence() -> None:
    dataset_artifact, bas_artifact = _artifacts(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(dataset_artifact, TrainingConfig(model_name="random_forest", split_mode="random"))
    predictor = AttentionClassifierPredictor(training_artifact)

    engine = RewardEngine(predictor=predictor)
    reward_artifact = engine.compute(dataset_artifact, bas_artifact)
    assert len(reward_artifact.records) == len(dataset_artifact.records)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_load_reward_artifact_round_trip(tmp_path) -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    save_reward_artifact(reward_artifact, tmp_path)
    loaded = load_reward_artifact(tmp_path)
    assert loaded.model_dump(mode="json") == reward_artifact.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_render_markdown_report_contains_sections() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    text = render_markdown_report(reward_artifact)
    for heading in ("Dataset Summary", "Reward Distribution", "Contribution Summary", "Session Summaries"):
        assert heading in text


def test_build_json_report_structure() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    report = build_json_report(reward_artifact)
    assert report["schema_version"] == reward_artifact.schema_version
    assert "reward_trends" in report
    assert len(report["reward_trends"]) == len(reward_artifact.session_summaries)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_reward_config_rejects_empty_signal_configs() -> None:
    with pytest.raises(ValueError):
        RewardConfig(signal_configs={})


def test_reward_config_rejects_invalid_clip_bounds() -> None:
    with pytest.raises(ValueError):
        RewardConfig(signal_configs=default_reward_config().signal_configs, reward_clip_min=1.0, reward_clip_max=-1.0)


def test_weighted_signals_excludes_zero_weight_and_neutral() -> None:
    config = default_reward_config()
    weighted = config.weighted_signals()
    assert "session_progress" not in weighted
    assert "delta_classifier_confidence" not in weighted
    assert "delta_bas" in weighted


# ---------------------------------------------------------------------------
# Reward decomposition (R = Performance + Behaviour - Cost)
# ---------------------------------------------------------------------------


def test_default_config_categorizes_signals_as_expected() -> None:
    config = default_reward_config()
    assert config.signal_configs["delta_bas"].category == RewardCategory.PERFORMANCE
    assert config.signal_configs["delta_correctness"].category == RewardCategory.PERFORMANCE
    assert config.signal_configs["delta_confidence"].category == RewardCategory.PERFORMANCE
    assert config.signal_configs["delta_engagement"].category == RewardCategory.BEHAVIOUR
    assert config.signal_configs["delta_latency_deviation"].category == RewardCategory.BEHAVIOUR
    assert config.signal_configs["delta_fatigue"].category == RewardCategory.BEHAVIOUR
    assert config.signal_configs["intervention_cost"].category == RewardCategory.COST
    assert config.signal_configs["session_progress"].category == RewardCategory.CONTEXT


def test_decompose_reward_invariant_holds() -> None:
    contributions = [
        RewardContribution(signal="delta_bas", weight=0.3, category=RewardCategory.PERFORMANCE, evidence_value=0.9, signed_evidence=0.8, contribution=0.24),
        RewardContribution(signal="delta_fatigue", weight=0.1, category=RewardCategory.BEHAVIOUR, evidence_value=0.3, signed_evidence=-0.4, contribution=-0.04),
        RewardContribution(signal="intervention_cost", weight=0.1, category=RewardCategory.COST, evidence_value=0.0, signed_evidence=-1.0, contribution=-0.10),
    ]
    performance, behaviour, cost = decompose_reward(contributions)
    assert performance == pytest.approx(0.24)
    assert behaviour == pytest.approx(-0.04)
    assert cost == pytest.approx(0.10)  # sign-flipped to a positive magnitude
    total = sum(c.contribution for c in contributions)
    assert total == pytest.approx(performance + behaviour - cost)


def test_reward_engine_decomposition_invariant_holds_end_to_end() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    for record in reward_artifact.records:
        assert record.raw_reward == pytest.approx(
            record.performance_reward + record.behaviour_reward - record.cost_reward, abs=1e-9
        )


def test_cost_reward_always_nonnegative() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    for record in reward_artifact.records:
        assert record.cost_reward >= 0.0


def test_with_category_disabled_zeroes_only_that_category() -> None:
    config = default_reward_config()
    ablated = config.with_category_disabled(RewardCategory.BEHAVIOUR)

    for name in ("delta_engagement", "delta_latency_deviation", "delta_fatigue"):
        assert ablated.signal_configs[name].weight == 0.0
    # Performance/cost signals untouched.
    assert ablated.signal_configs["delta_bas"].weight == config.signal_configs["delta_bas"].weight
    assert ablated.signal_configs["intervention_cost"].weight == config.signal_configs["intervention_cost"].weight


def test_with_category_disabled_ablation_changes_reward() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    full_config = default_reward_config()
    ablated_config = full_config.with_category_disabled(RewardCategory.BEHAVIOUR)

    full_artifact = RewardEngine(full_config).compute(dataset_artifact, bas_artifact)
    ablated_artifact = RewardEngine(ablated_config).compute(dataset_artifact, bas_artifact)

    for record in ablated_artifact.records:
        assert record.behaviour_reward == 0.0

    # At least some rewards should differ once behaviour signals stop contributing.
    full_rewards = [r.raw_reward for r in full_artifact.records]
    ablated_rewards = [r.raw_reward for r in ablated_artifact.records]
    assert full_rewards != ablated_rewards


def test_statistics_expose_category_averages() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    stats = reward_artifact.statistics
    assert -1.0 <= stats.average_performance_reward <= 1.0
    assert -1.0 <= stats.average_behaviour_reward <= 1.0
    assert 0.0 <= stats.average_cost_reward <= 1.0


def test_markdown_report_shows_decomposition() -> None:
    dataset_artifact, bas_artifact = _artifacts()
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    text = render_markdown_report(reward_artifact)
    assert "R = Performance + Behaviour - Cost" in text
    assert "Performance:" in text
    assert "Behaviour:" in text
    assert "Cost:" in text


# ---------------------------------------------------------------------------
# Stress test: 100,000+ reward computations
# ---------------------------------------------------------------------------


def test_stress_100000_reward_computations() -> None:
    """As with Modules 7/8/9's stress tests: replicate a real, small
    simulated batch with re-suffixed IDs rather than re-running full
    simulation 100,000+ times.
    """

    base_dataset_artifact, base_bas_artifact = _artifacts(n_students=10, sessions_per_student=2)
    base_records = base_dataset_artifact.records
    base_bas_records = base_bas_artifact.records
    target_size = 100_000
    replication_factor = target_size // len(base_records) + 1

    dataset_records: list[DatasetRecord] = []
    bas_records = []
    for batch in range(replication_factor):
        for record in base_records:
            data = record.model_dump(mode="json")
            data["session_id"] = f"{record.session_id}_batch{batch}"
            data["student_id"] = f"{record.student_id}_batch{batch}"
            data["response_id"] = f"{record.response_id}_batch{batch}"
            dataset_records.append(DatasetRecord.model_validate(data))
        for bas_record in base_bas_records:
            bas_data = bas_record.model_dump(mode="json")
            bas_data["session_id"] = f"{bas_record.session_id}_batch{batch}"
            bas_data["student_id"] = f"{bas_record.student_id}_batch{batch}"
            bas_records.append(type(bas_record).model_validate(bas_data))

    assert len(dataset_records) >= target_size

    large_dataset_artifact = base_dataset_artifact.model_copy(
        update={
            "records": dataset_records,
            "metadata": base_dataset_artifact.metadata.model_copy(update={"record_count": len(dataset_records)}),
        }
    )
    large_bas_artifact = base_bas_artifact.model_copy(update={"records": bas_records})

    reward_artifact = RewardEngine().compute(large_dataset_artifact, large_bas_artifact)
    assert len(reward_artifact.records) == len(dataset_records)
    assert reward_artifact.statistics.record_count == len(dataset_records)
    for record in reward_artifact.records[:1000]:
        assert -1.0 <= record.reward <= 1.0
