"""Tests for Module 9: Behavioural Attention Score (BAS)."""

from __future__ import annotations

import pytest

from dataset_generator.bas import (
    BASConfig,
    BASEngine,
    BASFeatureExtractor,
    BehaviouralAttentionScorer,
    FeatureBASConfig,
    FeatureNormalizationConfig,
    FeaturePolarity,
    NormalizationStrategy,
    Normalizer,
    SmoothingStrategy,
    build_json_report,
    build_session_summary,
    compute_confidence,
    default_bas_config,
    generate_explanation,
    load_bas_artifact,
    map_to_evidence,
    normalize_value,
    render_markdown_report,
    save_bas_artifact,
    smooth,
    top_contributors,
)
from dataset_generator.bas.models import BASContribution, BASEvidence, BASObservation, BASRecord
from dataset_generator.classifier import AttentionClassifierPredictor, AttentionClassifierTrainer, TrainingConfig
from dataset_generator.config import default_config
from dataset_generator.generators import generate_sessions, generate_students
from dataset_generator.pipeline import build_dataset_artifact
from dataset_generator.utils import build_rng_streams


def _dataset_artifact(n_students: int = 5, sessions_per_student: int = 2, seed: int | None = None):
    config = default_config()
    seed = seed if seed is not None else config.seed
    streams = build_rng_streams(seed)
    students = generate_students(config, streams)[:n_students]
    sessions = generate_sessions(config, students, sessions_per_student=sessions_per_student, rng_streams=build_rng_streams(seed))
    return build_dataset_artifact(config, students, sessions)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_feature_extractor_extracts_all_expected_features() -> None:
    artifact = _dataset_artifact()
    extractor = BASFeatureExtractor()
    observation = extractor.extract(artifact.records[0])
    expected = {
        "response_latency", "abs_normalized_latency", "prompt_relative_latency",
        "rolling_latency", "rolling_engagement",
        "fatigue", "correctness", "confidence", "semantic_similarity", "coherence",
        "lexical_diversity", "hesitation", "topic_shift", "repetition_ratio", "session_progress",
        "classifier_confidence",
    }
    assert set(observation.raw_values.keys()) == expected


def test_feature_extractor_classifier_confidence_missing_without_predictor() -> None:
    artifact = _dataset_artifact()
    extractor = BASFeatureExtractor(predictor=None)
    observation = extractor.extract(artifact.records[0])
    assert observation.raw_values["classifier_confidence"] is None


def test_feature_extractor_classifier_confidence_present_with_predictor() -> None:
    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="random"))
    predictor = AttentionClassifierPredictor(training_artifact)

    extractor = BASFeatureExtractor(predictor=predictor)
    observation = extractor.extract(artifact.records[0])
    assert observation.raw_values["classifier_confidence"] is not None
    assert 0.0 <= observation.raw_values["classifier_confidence"] <= 1.0


def test_extract_batch_matches_extract_for_each_record() -> None:
    artifact = _dataset_artifact()
    extractor = BASFeatureExtractor()
    records = artifact.records[:5]
    batch = extractor.extract_batch(records)
    singles = [extractor.extract(r) for r in records]
    assert [b.raw_values for b in batch] == [s.raw_values for s in singles]


def test_abs_normalized_latency_is_nonnegative() -> None:
    artifact = _dataset_artifact()
    extractor = BASFeatureExtractor()
    for record in artifact.records[:50]:
        observation = extractor.extract(record)
        assert observation.raw_values["abs_normalized_latency"] >= 0.0


# ---------------------------------------------------------------------------
# Robustness audit regression: prompt-complexity-relative latency
#
# `abs_normalized_latency` alone only compares a response's latency to that
# same student's own baseline, so it cannot distinguish "answered quickly
# because fluent" from "answered implausibly fast for the question asked."
# `prompt_relative_latency` (docs/ROBUSTNESS_AUDIT.md) closes that gap.
# These tests fail on a revert of the fix: `prompt_relative_latency` would
# either be absent from `raw_values` (feature_extractor.py) or absent from
# `feature_configs` (config.py), and the end-to-end case would collapse back
# to the two scores being indistinguishable.
# ---------------------------------------------------------------------------


def _record(**overrides):
    from dataset_generator.models.dataset import DatasetRecord

    kwargs = dict(
        session_id="s1", student_id="st1", interaction_number=1,
        prompt_id="p1", response_id="r1",
        attention_state="focused", intervention_applied=False, session_progress=0.5,
        student_profile="consistently_focused", student_profile_description="x",
        student_baseline_latency=0.5, student_latency_variance=0.1,
        student_engagement_tendency=0.7, student_fatigue_rate=0.1,
        student_intervention_sensitivity=0.5,
        prompt_subject="math", prompt_topic="algebra", prompt_difficulty="medium",
        prompt_cognitive_level="apply", prompt_text="x", prompt_expected_answer_type="numeric",
        prompt_estimated_response_length=20, prompt_keywords="x", prompt_learning_objective="x",
        prompt_reading_time_seconds=6.0, prompt_token_count=14, prompt_concept_count=2,
        prompt_difficulty_score=0.5, prompt_cognitive_complexity_score=0.5, prompt_readability_grade=6.0,
        response_text="x", response_correctness_score=1.0, response_length=5,
        response_semantic_similarity=0.9, response_lexical_diversity=0.8, response_sentiment=0.0,
        response_engagement_proxy=0.8, response_confidence=0.9, response_hesitation_markers="",
        response_repetition_ratio=0.0, response_coherence_score=0.9, response_topic_shift=0.05,
        response_strategy_used="direct",
        behaviour_response_latency=0.4, behaviour_interaction_duration=0.4,
        behaviour_hesitation_duration=0.0, behaviour_rolling_latency=0.4,
        behaviour_rolling_engagement=0.8, behaviour_fatigue_level=0.1,
        behaviour_normalized_latency=-0.9, behaviour_transition_occurred=False,
        session_total_interactions=1, session_dominant_attention_state="focused",
        session_intervention_count=0, session_final_fatigue=0.1,
        session_average_engagement=0.8, session_average_correctness=0.9,
        session_average_latency=0.5,
    )
    kwargs.update(overrides)
    return DatasetRecord(**kwargs)


def test_prompt_relative_latency_low_for_premature_answer() -> None:
    # Answered in 0.3s to a question that takes 14s just to read.
    record = _record(prompt_reading_time_seconds=14.0, behaviour_response_latency=0.3)
    observation = BASFeatureExtractor().extract(record)
    assert observation.raw_values["prompt_relative_latency"] == pytest.approx(0.3 / 14.0)
    assert observation.raw_values["prompt_relative_latency"] < 0.1


def test_prompt_relative_latency_guards_against_zero_reading_time() -> None:
    record = _record(prompt_reading_time_seconds=0.0, behaviour_response_latency=1.0)
    observation = BASFeatureExtractor().extract(record)
    assert observation.raw_values["prompt_relative_latency"] == pytest.approx(10.0)


def test_premature_answer_scores_lower_than_fluent_fast_answer() -> None:
    """The actual blind spot: before the fix, both of these scored ~equal."""

    engine = BASEngine()

    fluent_fast = _record(
        student_baseline_latency=0.5, student_latency_variance=0.1,
        prompt_reading_time_seconds=0.6, behaviour_response_latency=0.4,
        behaviour_normalized_latency=-0.9,
    )
    premature_for_complex_question = _record(
        student_baseline_latency=0.5, student_latency_variance=0.1,
        prompt_reading_time_seconds=14.0, behaviour_response_latency=0.3,
        behaviour_normalized_latency=-1.0,
    )

    fluent_score = engine.compute(_single_record_artifact(fluent_fast)).records[0].score
    premature_score = engine.compute(_single_record_artifact(premature_for_complex_question)).records[0].score

    assert premature_score < fluent_score - 0.05


def _single_record_artifact(record) -> "DatasetArtifact":
    from dataset_generator.models.dataset import (
        DatasetArtifact, DatasetManifest, DatasetMetadata, DatasetStatistics, DatasetValidationReport,
    )

    return DatasetArtifact(
        records=[record],
        statistics=DatasetStatistics(
            record_count=1, feature_distributions={}, class_balance={}, attention_balance={},
            profile_balance={}, session_balance={}, subject_balance={}, difficulty_balance={},
            correlation_matrix={}, missing_value_summary={},
        ),
        validation=DatasetValidationReport(
            record_count=1, missing_value_issues={}, duplicate_row_count=0, duplicate_id_count=0,
            invalid_range_issues={}, impossible_transition_count=0, invalid_attention_state_count=0,
            orphan_session_ids=[], orphan_student_ids=[], nan_count=0, inf_count=0, schema_consistent=True,
        ),
        metadata=DatasetMetadata(
            student_count=1, session_count=1, record_count=1,
            subjects_covered=["math"], profiles_covered=["consistently_focused"],
        ),
        manifest=DatasetManifest(
            dataset_version="1.0", schema_version="1.0", generator_version="1.0",
            generation_timestamp=DatasetManifest.now_timestamp(), seed=0, config_fingerprint="x",
        ),
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_identity_passthrough_with_clip() -> None:
    cfg = FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY, clip_min=0.0, clip_max=1.0)
    assert normalize_value(0.5, cfg) == 0.5
    assert normalize_value(1.5, cfg) == 1.0
    assert normalize_value(-0.5, cfg) == 0.0


def test_normalize_min_max() -> None:
    cfg = FeatureNormalizationConfig(strategy=NormalizationStrategy.MIN_MAX, min_value=0.0, max_value=10.0)
    assert normalize_value(5.0, cfg) == pytest.approx(0.5)
    assert normalize_value(0.0, cfg) == pytest.approx(0.0)
    assert normalize_value(10.0, cfg) == pytest.approx(1.0)
    assert normalize_value(20.0, cfg) == pytest.approx(1.0)  # clipped


def test_normalize_z_score() -> None:
    cfg = FeatureNormalizationConfig(strategy=NormalizationStrategy.Z_SCORE, mean=0.0, std=1.0)
    assert normalize_value(0.0, cfg) == pytest.approx(0.5)  # at the mean -> midpoint


def test_min_max_requires_bounds() -> None:
    with pytest.raises(ValueError):
        FeatureNormalizationConfig(strategy=NormalizationStrategy.MIN_MAX, min_value=0.0)


def test_z_score_requires_mean_and_std() -> None:
    with pytest.raises(ValueError):
        FeatureNormalizationConfig(strategy=NormalizationStrategy.Z_SCORE, mean=0.0)
    with pytest.raises(ValueError):
        FeatureNormalizationConfig(strategy=NormalizationStrategy.Z_SCORE, mean=0.0, std=-1.0)


def test_normalizer_each_feature_independent() -> None:
    config = default_bas_config()
    normalizer = Normalizer(config)
    observation = BASObservation(
        student_id="S1", session_id="SESS1", interaction_number=1,
        raw_values={name: 0.5 for name in config.feature_configs},
    )
    normalized = normalizer.normalize(observation)
    assert set(normalized.keys()) == set(config.feature_configs.keys())


def test_normalizer_preserves_missing() -> None:
    config = default_bas_config()
    normalizer = Normalizer(config)
    values = {name: 0.5 for name in config.feature_configs}
    values["correctness"] = None
    observation = BASObservation(student_id="S1", session_id="SESS1", interaction_number=1, raw_values=values)
    normalized = normalizer.normalize(observation)
    assert normalized["correctness"] is None


# ---------------------------------------------------------------------------
# Evidence mapping / weighting
# ---------------------------------------------------------------------------


def test_evidence_positive_polarity_passthrough() -> None:
    config = default_bas_config()
    normalized = {name: 0.8 for name in config.feature_configs}
    evidence = map_to_evidence(normalized, config)
    assert evidence.values["correctness"] == 0.8  # positive polarity


def test_evidence_negative_polarity_flips() -> None:
    config = default_bas_config()
    normalized = {name: 0.8 for name in config.feature_configs}
    evidence = map_to_evidence(normalized, config)
    assert evidence.values["fatigue"] == pytest.approx(0.2)  # negative polarity: 1 - 0.8


def test_evidence_neutral_excluded() -> None:
    config = default_bas_config()
    normalized = {name: 0.8 for name in config.feature_configs}
    evidence = map_to_evidence(normalized, config)
    assert "session_progress" not in evidence.values  # neutral


def test_evidence_tracks_missing() -> None:
    config = default_bas_config()
    normalized = {name: 0.8 for name in config.feature_configs}
    normalized["correctness"] = None
    evidence = map_to_evidence(normalized, config)
    assert "correctness" in evidence.missing_features
    assert "correctness" not in evidence.values


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------


def test_scorer_weighted_sum_matches_manual_calculation() -> None:
    config = BASConfig(
        feature_configs={
            "a": FeatureBASConfig(weight=0.6, polarity=FeaturePolarity.POSITIVE, normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY)),
            "b": FeatureBASConfig(weight=0.4, polarity=FeaturePolarity.POSITIVE, normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY)),
        }
    )
    scorer = BehaviouralAttentionScorer(config)
    evidence = BASEvidence(values={"a": 1.0, "b": 0.0}, missing_features=[])
    raw_score, contributions = scorer.score(evidence)
    assert raw_score == pytest.approx(0.6)
    assert len(contributions) == 2


def test_scorer_bounded_output() -> None:
    config = default_bas_config()
    scorer = BehaviouralAttentionScorer(config)
    evidence = BASEvidence(values={f: 1.0 for f in config.weighted_features()}, missing_features=[])
    raw_score, _ = scorer.score(evidence)
    assert 0.0 <= raw_score <= 1.0


def test_scorer_missing_values_renormalize_not_zero() -> None:
    """A missing feature should not silently count as zero evidence."""

    config = BASConfig(
        feature_configs={
            "a": FeatureBASConfig(weight=0.5, polarity=FeaturePolarity.POSITIVE, normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY)),
            "b": FeatureBASConfig(weight=0.5, polarity=FeaturePolarity.POSITIVE, normalization=FeatureNormalizationConfig(strategy=NormalizationStrategy.IDENTITY)),
        }
    )
    scorer = BehaviouralAttentionScorer(config)
    # Only "a" observed, at a perfect 1.0 -- score should reflect that, not be halved by a phantom zero for "b".
    evidence = BASEvidence(values={"a": 1.0}, missing_features=["b"])
    raw_score, _ = scorer.score(evidence)
    assert raw_score == pytest.approx(1.0)


def test_scorer_contributions_ranked_descending() -> None:
    config = default_bas_config()
    scorer = BehaviouralAttentionScorer(config)
    evidence = BASEvidence(
        values={"correctness": 0.9, "fatigue": 0.9, "hesitation": 0.1}, missing_features=[]
    )
    _, contributions = scorer.score(evidence)
    values = [c.contribution for c in contributions]
    assert values == sorted(values, reverse=True)


def test_scorer_all_missing_returns_neutral_default() -> None:
    config = default_bas_config()
    scorer = BehaviouralAttentionScorer(config)
    evidence = BASEvidence(values={}, missing_features=list(config.weighted_features()))
    raw_score, contributions = scorer.score(evidence)
    assert raw_score == 0.5
    assert contributions == []


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def test_smooth_identity_passthrough() -> None:
    config = default_bas_config().model_copy(update={"smoothing_strategy": SmoothingStrategy.IDENTITY})
    assert smooth(0.7, 0.3, config) == 0.7


def test_smooth_ema_first_interaction_passthrough() -> None:
    config = default_bas_config()  # EMA by default
    assert smooth(0.7, None, config) == 0.7


def test_smooth_ema_blends_with_previous() -> None:
    config = default_bas_config().model_copy(update={"ema_alpha": 0.3})
    result = smooth(1.0, 0.0, config)
    assert result == pytest.approx(0.3)


def test_smooth_rolling_average() -> None:
    config = default_bas_config().model_copy(
        update={"smoothing_strategy": SmoothingStrategy.ROLLING_AVERAGE, "rolling_window": 3}
    )
    history = [0.2, 0.4, 0.6]
    result = smooth(0.6, 0.4, config, history=history)
    assert result == pytest.approx(0.4)


def test_smooth_unknown_strategy_raises() -> None:
    config = default_bas_config().model_copy(update={"smoothing_strategy": "identity"})
    object.__setattr__(config, "smoothing_strategy", "not_a_strategy")
    with pytest.raises(ValueError):
        smooth(0.5, 0.5, config)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_full_coverage_low_variance_is_high() -> None:
    config = default_bas_config()
    evidence = BASEvidence(values={f: 0.9 for f in config.weighted_features()}, missing_features=[])
    confidence, uncertainty, reliability = compute_confidence(evidence, config)
    assert confidence > 0.8
    assert reliability == "high"
    assert uncertainty == pytest.approx(1.0 - confidence)


def test_confidence_drops_with_missing_features() -> None:
    config = default_bas_config()
    features = config.weighted_features()
    full = BASEvidence(values={f: 0.9 for f in features}, missing_features=[])
    half_missing = BASEvidence(values={f: 0.9 for f in features[: len(features) // 2]}, missing_features=features[len(features) // 2 :])
    full_conf, _, _ = compute_confidence(full, config)
    partial_conf, _, _ = compute_confidence(half_missing, config)
    assert partial_conf < full_conf


def test_confidence_drops_with_conflicting_evidence() -> None:
    config = default_bas_config()
    features = config.weighted_features()
    agreeing = BASEvidence(values={f: 0.9 for f in features}, missing_features=[])
    conflicting_values = {f: (0.9 if i % 2 == 0 else 0.1) for i, f in enumerate(features)}
    conflicting = BASEvidence(values=conflicting_values, missing_features=[])
    agree_conf, _, _ = compute_confidence(agreeing, config)
    conflict_conf, _, _ = compute_confidence(conflicting, config)
    assert conflict_conf < agree_conf


def test_confidence_blends_classifier_confidence() -> None:
    config = default_bas_config()
    evidence = BASEvidence(values={f: 0.5 for f in config.weighted_features()}, missing_features=[])
    low_classifier, _, _ = compute_confidence(evidence, config, classifier_confidence=0.0)
    high_classifier, _, _ = compute_confidence(evidence, config, classifier_confidence=1.0)
    assert high_classifier > low_classifier


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


def test_top_contributors_positive_and_negative() -> None:
    contributions = [
        BASContribution(feature="a", weight=0.5, evidence_value=0.9, contribution=0.45),
        BASContribution(feature="b", weight=0.5, evidence_value=0.1, contribution=-0.2),
    ]
    positives = top_contributors(contributions, 5, positive=True)
    negatives = top_contributors(contributions, 5, positive=False)
    assert [c.feature for c in positives] == ["a"]
    assert [c.feature for c in negatives] == ["b"]


def test_generate_explanation_snapshot_mode() -> None:
    contributions = [
        BASContribution(feature="correctness", weight=0.2, evidence_value=0.9, contribution=0.18),
        BASContribution(feature="fatigue", weight=0.1, evidence_value=0.1, contribution=-0.09),
    ]
    text = generate_explanation(contributions, previous_contributions=None)
    assert "correctness" in text
    assert "fatigue" in text


def test_generate_explanation_trend_mode() -> None:
    previous = [BASContribution(feature="correctness", weight=0.2, evidence_value=0.5, contribution=0.10)]
    current = [BASContribution(feature="correctness", weight=0.2, evidence_value=0.9, contribution=0.18)]
    text = generate_explanation(current, previous_contributions=previous)
    assert "increased" in text or "decreased" in text
    assert "correctness" in text


# ---------------------------------------------------------------------------
# Session summaries
# ---------------------------------------------------------------------------


def _make_record(session_id: str, interaction_number: int, score: float) -> BASRecord:
    return BASRecord(
        student_id="S1", session_id=session_id, interaction_number=interaction_number,
        raw_score=score, score=score, contributions=[], confidence=0.9, uncertainty=0.1,
        reliability="high", missing_feature_ratio=0.0, explanation="test",
        top_positive=[], top_negative=[],
        metadata={"normalization_strategy": {}, "smoothing_strategy": "ema", "config_fingerprint": "x", "config_version": "1.0.0"},
    )


def test_session_summary_basic_stats() -> None:
    config = default_bas_config()
    records = [_make_record("SESS1", i + 1, s) for i, s in enumerate([0.5, 0.6, 0.7, 0.4])]
    summary = build_session_summary(records, config)
    assert summary.average_bas == pytest.approx(0.55)
    assert summary.minimum_bas == 0.4
    assert summary.maximum_bas == 0.7
    assert summary.largest_drop == pytest.approx(0.3)  # 0.7 -> 0.4
    assert summary.largest_recovery == pytest.approx(0.1)  # 0.5 -> 0.6


def test_session_summary_trend_declining() -> None:
    config = default_bas_config()
    records = [_make_record("SESS1", i + 1, s) for i, s in enumerate([0.9, 0.8, 0.3, 0.2])]
    summary = build_session_summary(records, config)
    assert summary.attention_trend == "declining"


def test_session_summary_trend_improving() -> None:
    config = default_bas_config()
    records = [_make_record("SESS1", i + 1, s) for i, s in enumerate([0.2, 0.3, 0.8, 0.9])]
    summary = build_session_summary(records, config)
    assert summary.attention_trend == "improving"


def test_session_summary_time_above_below_threshold() -> None:
    config = default_bas_config().model_copy(update={"attention_threshold": 0.5})
    records = [_make_record("SESS1", i + 1, s) for i, s in enumerate([0.9, 0.1, 0.9, 0.1])]
    summary = build_session_summary(records, config)
    assert summary.time_above_threshold == pytest.approx(0.5)
    assert summary.time_below_threshold == pytest.approx(0.5)


def test_session_summary_empty_raises() -> None:
    with pytest.raises(ValueError):
        build_session_summary([], default_bas_config())


# ---------------------------------------------------------------------------
# End-to-end BASEngine
# ---------------------------------------------------------------------------


def test_bas_engine_produces_one_record_per_dataset_record() -> None:
    artifact = _dataset_artifact()
    bas_artifact = BASEngine().compute(artifact)
    assert len(bas_artifact.records) == len(artifact.records)


def test_bas_engine_scores_bounded() -> None:
    artifact = _dataset_artifact()
    bas_artifact = BASEngine().compute(artifact)
    for record in bas_artifact.records:
        assert 0.0 <= record.score <= 1.0
        assert 0.0 <= record.confidence <= 1.0


def test_bas_engine_deterministic() -> None:
    artifact = _dataset_artifact()
    a = BASEngine().compute(artifact)
    b = BASEngine().compute(artifact)
    assert [r.model_dump(mode="json") for r in a.records] == [r.model_dump(mode="json") for r in b.records]


def test_bas_engine_session_summaries_count_matches_sessions() -> None:
    artifact = _dataset_artifact(n_students=5, sessions_per_student=2)
    bas_artifact = BASEngine().compute(artifact)
    distinct_sessions = {r.session_id for r in artifact.records}
    assert len(bas_artifact.session_summaries) == len(distinct_sessions)


def test_bas_engine_first_interaction_of_session_unsmoothed() -> None:
    artifact = _dataset_artifact()
    bas_artifact = BASEngine().compute(artifact)
    first_interactions = [r for r in bas_artifact.records if r.interaction_number == 1]
    for record in first_interactions:
        assert record.score == pytest.approx(record.raw_score)


def test_bas_engine_with_classifier_confidence() -> None:
    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="random"))
    predictor = AttentionClassifierPredictor(training_artifact)

    engine = BASEngine(predictor=predictor)
    bas_artifact = engine.compute(artifact)
    assert len(bas_artifact.records) == len(artifact.records)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_load_bas_artifact_round_trip(tmp_path) -> None:
    artifact = _dataset_artifact()
    bas_artifact = BASEngine().compute(artifact)
    save_bas_artifact(bas_artifact, tmp_path)
    loaded = load_bas_artifact(tmp_path)
    assert loaded.model_dump(mode="json") == bas_artifact.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_render_markdown_report_contains_sections() -> None:
    artifact = _dataset_artifact()
    bas_artifact = BASEngine().compute(artifact)
    text = render_markdown_report(bas_artifact)
    for heading in ("Dataset Summary", "Score Distribution", "Feature Contribution Summary", "Session Summaries"):
        assert heading in text


def test_build_json_report_structure() -> None:
    artifact = _dataset_artifact()
    bas_artifact = BASEngine().compute(artifact)
    report = build_json_report(bas_artifact)
    assert report["schema_version"] == bas_artifact.schema_version
    assert "session_plots" in report
    assert len(report["session_plots"]) == len(bas_artifact.session_summaries)
    for spec in report["session_plots"]:
        assert "x_values" in spec and "series_values" in spec


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_bas_config_rejects_empty_feature_configs() -> None:
    with pytest.raises(ValueError):
        BASConfig(feature_configs={})


def test_bas_config_rejects_invalid_clip_bounds() -> None:
    # model_copy() doesn't re-run validators in Pydantic v2, so this must go
    # through the real constructor to actually exercise the validator.
    with pytest.raises(ValueError):
        BASConfig(feature_configs=default_bas_config().feature_configs, score_clip_min=1.0, score_clip_max=0.0)


def test_weighted_features_excludes_zero_weight_and_neutral() -> None:
    config = default_bas_config()
    weighted = config.weighted_features()
    assert "session_progress" not in weighted
    assert "classifier_confidence" not in weighted
    assert "correctness" in weighted


# ---------------------------------------------------------------------------
# Stress test: 100,000+ BAS computations
# ---------------------------------------------------------------------------


def test_stress_100000_bas_computations() -> None:
    """As with Modules 7/8's stress tests: replicate a real, small simulated
    batch with re-suffixed session/student IDs rather than re-running full
    simulation 100,000+ times.
    """

    base_artifact = _dataset_artifact(n_students=10, sessions_per_student=2)
    base_records = base_artifact.records
    target_size = 100_000
    replication_factor = target_size // len(base_records) + 1

    records = []
    for batch in range(replication_factor):
        for record in base_records:
            data = record.model_dump(mode="json")
            data["session_id"] = f"{record.session_id}_batch{batch}"
            data["student_id"] = f"{record.student_id}_batch{batch}"
            data["response_id"] = f"{record.response_id}_batch{batch}"
            records.append(type(record).model_validate(data))

    assert len(records) >= target_size

    large_artifact = base_artifact.model_copy(
        update={"records": records, "metadata": base_artifact.metadata.model_copy(update={"record_count": len(records)})}
    )

    bas_artifact = BASEngine().compute(large_artifact)
    assert len(bas_artifact.records) == len(records)
    assert bas_artifact.statistics.record_count == len(records)
    for record in bas_artifact.records[:1000]:
        assert 0.0 <= record.score <= 1.0
