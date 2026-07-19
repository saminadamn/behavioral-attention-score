"""Tests for Module 11: Adaptive Intervention Engine."""

from __future__ import annotations

import pytest

from dataset_generator.bas import BASEngine
from dataset_generator.config import default_config
from dataset_generator.generators import generate_sessions, generate_students
from dataset_generator.models.dataset import DatasetRecord
from dataset_generator.pipeline import build_dataset_artifact
from dataset_generator.reward import RewardEngine
from dataset_generator.utils import build_rng_streams

from dataset_generator.intervention import (
    NO_INTERVENTION_POLICY_NAME,
    BreakRecommendationPolicy,
    ConceptReviewPolicy,
    CooldownManager,
    DifficultyReductionPolicy,
    EncouragementPolicy,
    HintPolicy,
    InterventionCandidate,
    InterventionConfidenceEstimator,
    InterventionConfig,
    InterventionDetector,
    InterventionObservation,
    InterventionObservationExtractor,
    InterventionPlanner,
    InterventionPolicyFactory,
    MotivationalPromptPolicy,
    NeedSignalWeights,
    NoInterventionPolicy,
    PolicyScorer,
    QuestionReframingPolicy,
    ScoringWeights,
    build_intervention_session_summary,
    build_json_report,
    default_intervention_config,
    load_intervention_artifact,
    render_markdown_report,
    save_intervention_artifact,
)


def _artifacts(n_students: int = 5, sessions_per_student: int = 2, seed: int | None = None):
    config = default_config()
    seed = seed if seed is not None else config.seed
    streams = build_rng_streams(seed)
    students = generate_students(config, streams)[:n_students]
    sessions = generate_sessions(
        config, students, sessions_per_student=sessions_per_student, rng_streams=build_rng_streams(seed)
    )
    dataset_artifact = build_dataset_artifact(config, students, sessions)
    bas_artifact = BASEngine().compute(dataset_artifact)
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
    return dataset_artifact, bas_artifact, reward_artifact


def _observation(**overrides) -> InterventionObservation:
    defaults = dict(
        student_id="s1",
        session_id="sess1",
        interaction_number=3,
        current_bas=0.5,
        previous_bas=0.5,
        bas_trend=0.0,
        current_reward=0.0,
        reward_trend=0.0,
        fatigue=0.2,
        engagement=0.7,
        latency_deviation=0.1,
        correctness=0.7,
        confidence=0.7,
        semantic_similarity=0.8,
        prompt_difficulty_score=0.5,
        classifier_confidence=0.7,
        reward_confidence=0.7,
        bas_confidence=0.7,
        session_progress=0.5,
        previous_interventions_count=0,
        consecutive_decline_count=0,
    )
    defaults.update(overrides)
    return InterventionObservation(**defaults)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_default_config_constructs() -> None:
    config = default_intervention_config()
    assert isinstance(config, InterventionConfig)


def test_need_signal_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        NeedSignalWeights(low_bas=0.9)


def test_scoring_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        ScoringWeights(expected_gain_weight=0.9)


def test_severity_medium_must_be_below_high() -> None:
    with pytest.raises(ValueError):
        InterventionConfig(severity_medium_threshold=0.8, severity_high_threshold=0.5)


def test_policy_weight_defaults_to_one() -> None:
    config = default_intervention_config()
    assert config.policy_weight("SomeUnconfiguredPolicy") == 1.0


def test_with_policy_disabled_zeroes_only_that_policy() -> None:
    config = default_intervention_config()
    disabled = config.with_policy_disabled("HintPolicy")
    assert disabled.policy_weight("HintPolicy") == 0.0
    assert disabled.policy_weight("EncouragementPolicy") == 1.0


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------


def test_observation_extractor_one_observation_per_record() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    extractor = InterventionObservationExtractor()
    observations = extractor.extract_batch(dataset_artifact, bas_artifact, reward_artifact)
    assert len(observations) == len(dataset_artifact.records)


def test_observation_extractor_first_interaction_has_none_trends() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    extractor = InterventionObservationExtractor()
    observations = extractor.extract_batch(dataset_artifact, bas_artifact, reward_artifact)
    first_interactions = [o for o in observations if o.interaction_number == 1]
    assert first_interactions
    for obs in first_interactions:
        assert obs.previous_bas is None
        assert obs.bas_trend is None
        assert obs.reward_trend is None
        assert obs.previous_interventions_count == 0


def test_observation_extractor_matches_bas_and_reward_scores() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    extractor = InterventionObservationExtractor()
    observations = extractor.extract_batch(dataset_artifact, bas_artifact, reward_artifact)

    bas_by_key = {(r.session_id, r.interaction_number): r.score for r in bas_artifact.records}
    reward_by_key = {(r.session_id, r.interaction_number): r.reward for r in reward_artifact.records}
    for obs in observations:
        key = (obs.session_id, obs.interaction_number)
        assert obs.current_bas == pytest.approx(bas_by_key[key])
        assert obs.current_reward == pytest.approx(reward_by_key[key])


def test_observation_extractor_consecutive_decline_count_resets() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    extractor = InterventionObservationExtractor()
    observations = extractor.extract_batch(dataset_artifact, bas_artifact, reward_artifact)

    by_session: dict[str, list[InterventionObservation]] = {}
    for obs in observations:
        by_session.setdefault(obs.session_id, []).append(obs)

    for session_observations in by_session.values():
        ordered = sorted(session_observations, key=lambda o: o.interaction_number)
        for prev, curr in zip(ordered, ordered[1:]):
            if curr.bas_trend is not None and curr.bas_trend < 0:
                assert curr.consecutive_decline_count == prev.consecutive_decline_count + 1
            else:
                assert curr.consecutive_decline_count == 0


# ---------------------------------------------------------------------------
# Need detection
# ---------------------------------------------------------------------------


def test_detector_healthy_observation_scores_zero_need() -> None:
    config = default_intervention_config()
    detector = InterventionDetector(config)
    obs = _observation(
        current_bas=0.95, bas_trend=0.05, current_reward=0.6, reward_trend=0.1,
        fatigue=0.05, engagement=0.95, correctness=0.95, confidence=0.95,
        consecutive_decline_count=0,
    )
    result = detector.detect(obs)
    assert result.need_score == pytest.approx(0.0)
    assert result.trigger_reasons == []
    assert result.severity == "low"


def test_detector_severe_observation_scores_high_need() -> None:
    config = default_intervention_config()
    detector = InterventionDetector(config)
    obs = _observation(
        current_bas=0.1, bas_trend=-0.4, current_reward=-0.8, reward_trend=-0.4,
        fatigue=0.95, engagement=0.05, correctness=0.1, confidence=0.05,
        consecutive_decline_count=5,
    )
    result = detector.detect(obs)
    assert result.need_score > 0.5
    assert set(result.trigger_reasons) == {
        "low_bas", "rapid_decline", "persistent_negative_reward", "high_fatigue",
        "low_engagement", "consecutive_declines", "low_confidence",
    }
    assert result.severity == "high"


def test_detector_need_score_bounded() -> None:
    config = default_intervention_config()
    detector = InterventionDetector(config)
    obs = _observation(
        current_bas=0.0, bas_trend=-1.0, current_reward=-1.0, reward_trend=-1.0,
        fatigue=1.0, engagement=0.0, correctness=0.0, confidence=0.0,
        consecutive_decline_count=100,
    )
    result = detector.detect(obs)
    assert 0.0 <= result.need_score <= 1.0


def test_detector_severity_thresholds_are_respected() -> None:
    config = InterventionConfig(severity_medium_threshold=0.3, severity_high_threshold=0.6)
    detector = InterventionDetector(config)

    low_obs = _observation(current_bas=0.9, correctness=0.9, confidence=0.9, engagement=0.9, fatigue=0.1)
    assert detector.detect(low_obs).severity == "low"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_policy_factory_registers_all_eight_policies() -> None:
    names = set(InterventionPolicyFactory.names())
    assert names == {
        "NoInterventionPolicy", "HintPolicy", "ConceptReviewPolicy", "DifficultyReductionPolicy",
        "MotivationalPromptPolicy", "BreakRecommendationPolicy", "EncouragementPolicy",
        "QuestionReframingPolicy",
    }


def test_no_intervention_policy_always_eligible() -> None:
    config = default_intervention_config()
    policy = NoInterventionPolicy(config)
    assert policy.eligible(_observation()) is True
    assert policy.estimate_bas_gain(_observation()) == 0.0
    assert policy.estimated_cost(_observation()) == 0.0


def test_hint_policy_eligible_when_struggling_but_engaged() -> None:
    config = default_intervention_config()
    policy = HintPolicy(config)
    obs = _observation(correctness=0.2, engagement=0.8)
    assert policy.eligible(obs) is True
    obs_not_engaged = _observation(correctness=0.2, engagement=0.1)
    assert policy.eligible(obs_not_engaged) is False


def test_concept_review_requires_persistent_decline_and_low_correctness() -> None:
    config = default_intervention_config()
    policy = ConceptReviewPolicy(config)
    obs = _observation(consecutive_decline_count=3, correctness=0.2)
    assert policy.eligible(obs) is True
    obs_ok = _observation(consecutive_decline_count=0, correctness=0.2)
    assert policy.eligible(obs_ok) is False


def test_difficulty_reduction_requires_hard_prompt_and_low_correctness() -> None:
    config = default_intervention_config()
    policy = DifficultyReductionPolicy(config)
    obs = _observation(correctness=0.2, prompt_difficulty_score=0.9)
    assert policy.eligible(obs) is True
    obs_easy = _observation(correctness=0.2, prompt_difficulty_score=0.1)
    assert policy.eligible(obs_easy) is False


def test_motivational_prompt_requires_disengagement_with_capability() -> None:
    config = default_intervention_config()
    policy = MotivationalPromptPolicy(config)
    obs = _observation(engagement=0.1, correctness=0.8)
    assert policy.eligible(obs) is True


def test_break_recommendation_requires_high_fatigue() -> None:
    config = default_intervention_config()
    policy = BreakRecommendationPolicy(config)
    assert policy.eligible(_observation(fatigue=0.9)) is True
    assert policy.eligible(_observation(fatigue=0.1)) is False


def test_encouragement_requires_low_confidence_with_capability() -> None:
    config = default_intervention_config()
    policy = EncouragementPolicy(config)
    obs = _observation(confidence=0.1, correctness=0.8)
    assert policy.eligible(obs) is True


def test_question_reframing_requires_low_semantic_similarity() -> None:
    config = default_intervention_config()
    policy = QuestionReframingPolicy(config)
    assert policy.eligible(_observation(semantic_similarity=0.1)) is True
    assert policy.eligible(_observation(semantic_similarity=0.9)) is False


def test_policy_score_respects_policy_weight_zero() -> None:
    from dataset_generator.intervention.detector import NeedDetectionResult, NeedSignalBreakdown

    config = default_intervention_config().with_policy_disabled("HintPolicy")
    policy = HintPolicy(config)
    obs = _observation(correctness=0.2, engagement=0.8)
    need_result = NeedDetectionResult(
        need_score=0.5, trigger_reasons=["low_bas"], severity="medium",
        breakdown=NeedSignalBreakdown(0, 0, 0, 0, 0, 0, 0),
    )
    assert policy.score(obs, need_result) == 0.0


# ---------------------------------------------------------------------------
# Policy scoring
# ---------------------------------------------------------------------------


def test_scorer_only_returns_eligible_candidates() -> None:
    config = default_intervention_config()
    detector = InterventionDetector(config)
    scorer = PolicyScorer()
    policies = InterventionPolicyFactory.create_all(config)

    obs = _observation(correctness=0.9, engagement=0.9, confidence=0.9, fatigue=0.1, semantic_similarity=0.9)
    need_result = detector.detect(obs)
    candidates = scorer.evaluate_all(obs, policies, need_result)
    assert all(c.eligible for c in candidates)
    assert any(c.policy_name == NO_INTERVENTION_POLICY_NAME for c in candidates)


def test_scorer_ranks_candidates_by_score_descending() -> None:
    config = default_intervention_config()
    detector = InterventionDetector(config)
    scorer = PolicyScorer()
    policies = InterventionPolicyFactory.create_all(config)

    obs = _observation(correctness=0.1, engagement=0.1, confidence=0.1, fatigue=0.9, semantic_similarity=0.1)
    need_result = detector.detect(obs)
    candidates = scorer.evaluate_all(obs, policies, need_result)
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


def test_cooldown_blocks_before_min_interactions() -> None:
    config = InterventionConfig(min_interactions_before_intervention=3)
    manager = CooldownManager(config)
    assert manager.can_intervene(1) is False
    assert manager.can_intervene(3) is True


def test_cooldown_enforces_spacing() -> None:
    config = InterventionConfig(cooldown_length=3, min_interactions_before_intervention=1)
    manager = CooldownManager(config)
    manager.record_intervention(5, "HintPolicy")
    assert manager.can_intervene(6) is False
    assert manager.can_intervene(8) is False
    assert manager.can_intervene(9) is True


def test_cooldown_enforces_max_interventions_per_session() -> None:
    config = InterventionConfig(max_interventions_per_session=1, cooldown_length=0, min_interactions_before_intervention=1)
    manager = CooldownManager(config)
    manager.record_intervention(1, "HintPolicy")
    assert manager.can_intervene(2) is False


def test_cooldown_filter_candidates_suppresses_when_blocked() -> None:
    config = InterventionConfig(min_interactions_before_intervention=5)
    manager = CooldownManager(config)
    candidates = [
        InterventionCandidate(
            policy_name="HintPolicy", eligible=True, estimated_bas_gain=0.5,
            estimated_reward_gain=0.5, estimated_cost=0.1, score=0.9, reason="test",
        ),
        InterventionCandidate(
            policy_name=NO_INTERVENTION_POLICY_NAME, eligible=True, estimated_bas_gain=0.0,
            estimated_reward_gain=0.0, estimated_cost=0.0, score=0.1, reason="baseline",
        ),
    ]
    allowed, suppressed = manager.filter_candidates(candidates, 1)
    assert suppressed is True
    assert all(c.policy_name == NO_INTERVENTION_POLICY_NAME for c in allowed)


def test_cooldown_duplicate_prevention_window() -> None:
    config = InterventionConfig(
        cooldown_length=0, min_interactions_before_intervention=1, duplicate_prevention_window=5,
    )
    manager = CooldownManager(config)
    manager.record_intervention(1, "HintPolicy")
    candidates = [
        InterventionCandidate(
            policy_name="HintPolicy", eligible=True, estimated_bas_gain=0.5,
            estimated_reward_gain=0.5, estimated_cost=0.1, score=0.9, reason="test",
        ),
        InterventionCandidate(
            policy_name=NO_INTERVENTION_POLICY_NAME, eligible=True, estimated_bas_gain=0.0,
            estimated_reward_gain=0.0, estimated_cost=0.0, score=0.1, reason="baseline",
        ),
    ]
    allowed, suppressed = manager.filter_candidates(candidates, 2)
    assert suppressed is True
    assert all(c.policy_name != "HintPolicy" for c in allowed)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_bounded() -> None:
    config = default_intervention_config()
    estimator = InterventionConfidenceEstimator(config)
    obs = _observation()
    candidates = [
        InterventionCandidate(
            policy_name="HintPolicy", eligible=True, estimated_bas_gain=0.5,
            estimated_reward_gain=0.5, estimated_cost=0.1, score=0.9, reason="test",
        ),
        InterventionCandidate(
            policy_name=NO_INTERVENTION_POLICY_NAME, eligible=True, estimated_bas_gain=0.0,
            estimated_reward_gain=0.0, estimated_cost=0.0, score=0.1, reason="baseline",
        ),
    ]
    confidence, uncertainty, reliability = estimator.estimate(obs, candidates)
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= uncertainty <= 1.0
    assert confidence + uncertainty == pytest.approx(1.0)
    assert reliability in {"low", "medium", "high"}


def test_confidence_high_when_signals_fully_available_and_agreement_strong() -> None:
    config = default_intervention_config()
    estimator = InterventionConfidenceEstimator(config)
    obs = _observation(bas_confidence=0.95, reward_confidence=0.95)
    candidates = [
        InterventionCandidate(
            policy_name="HintPolicy", eligible=True, estimated_bas_gain=0.9,
            estimated_reward_gain=0.9, estimated_cost=0.1, score=1.0, reason="test",
        ),
        InterventionCandidate(
            policy_name=NO_INTERVENTION_POLICY_NAME, eligible=True, estimated_bas_gain=0.0,
            estimated_reward_gain=0.0, estimated_cost=0.0, score=0.0, reason="baseline",
        ),
    ]
    confidence, _, reliability = estimator.estimate(obs, candidates)
    assert confidence > 0.7
    assert reliability == "high"


# ---------------------------------------------------------------------------
# Planner (end-to-end)
# ---------------------------------------------------------------------------


def test_planner_produces_one_decision_per_dataset_record() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    assert len(artifact.decisions) == len(dataset_artifact.records)


def test_planner_every_decision_has_a_chosen_policy() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    for decision in artifact.decisions:
        assert decision.chosen_policy
        assert any(c.policy_name == decision.chosen_policy for c in decision.candidates)


def test_planner_respects_max_interventions_per_session() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    config = InterventionConfig(max_interventions_per_session=1, cooldown_length=0, min_interactions_before_intervention=1)
    artifact = InterventionPlanner(config).plan(dataset_artifact, bas_artifact, reward_artifact)

    by_session: dict[str, int] = {}
    for decision in artifact.decisions:
        if decision.chosen_policy != NO_INTERVENTION_POLICY_NAME:
            by_session[decision.session_id] = by_session.get(decision.session_id, 0) + 1
    assert all(count <= 1 for count in by_session.values())


def test_planner_deterministic() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    a1 = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    a2 = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    assert a1.decisions == a2.decisions
    assert a1.session_summaries == a2.session_summaries
    assert a1.statistics == a2.statistics
    assert a1.config_fingerprint == a2.config_fingerprint


def test_planner_session_summaries_count_matches_sessions() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    session_ids = {r.session_id for r in dataset_artifact.records}
    assert len(artifact.session_summaries) == len(session_ids)


def test_planner_with_classifier_confidence() -> None:
    from dataset_generator.classifier import AttentionClassifierPredictor, AttentionClassifierTrainer, TrainingConfig

    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(
        dataset_artifact, TrainingConfig(model_name="random_forest", split_mode="random")
    )
    predictor = AttentionClassifierPredictor(training_artifact)

    artifact = InterventionPlanner(predictor=predictor).plan(dataset_artifact, bas_artifact, reward_artifact)
    assert artifact.statistics.missing_value_summary["classifier_confidence"] == 0


def test_ablation_disabling_policy_changes_distribution() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts(n_students=8, sessions_per_student=3)
    base_config = default_intervention_config()
    ablated_config = base_config.with_policy_disabled("HintPolicy")

    base_artifact = InterventionPlanner(base_config).plan(dataset_artifact, bas_artifact, reward_artifact)
    if "HintPolicy" not in base_artifact.statistics.policy_distribution:
        pytest.skip("HintPolicy never chosen in this sample; ablation has nothing to change")

    ablated_artifact = InterventionPlanner(ablated_config).plan(dataset_artifact, bas_artifact, reward_artifact)
    assert "HintPolicy" not in ablated_artifact.statistics.policy_distribution


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


def test_session_summary_basic_stats() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    for summary in artifact.session_summaries:
        assert summary.interaction_count > 0
        assert 0 <= summary.intervention_count <= summary.interaction_count
        assert 0.0 <= summary.average_confidence <= 1.0
        assert summary.estimated_cumulative_bas_gain >= 0.0
        assert summary.estimated_cumulative_reward_gain >= 0.0


def test_session_summary_policy_frequencies_sum_to_interaction_count() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    for summary in artifact.session_summaries:
        assert sum(summary.policy_frequencies.values()) == summary.interaction_count


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_load_intervention_artifact_round_trip(tmp_path) -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    save_intervention_artifact(artifact, tmp_path)
    loaded = load_intervention_artifact(tmp_path)
    assert loaded == artifact


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_render_markdown_report_contains_sections() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    report = render_markdown_report(artifact)
    assert "# Adaptive Intervention Engine Report" in report
    assert "## Policy Distribution" in report
    assert "## Session Summaries" in report
    assert "## Decision Table" in report


def test_build_json_report_structure() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts()
    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    report = build_json_report(artifact)
    assert report["schema_version"] == artifact.schema_version
    assert report["config_fingerprint"] == artifact.config_fingerprint
    assert "statistics" in report
    assert "session_summaries" in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_interaction_session_has_no_trends() -> None:
    config = default_config()
    seed = config.seed
    streams = build_rng_streams(seed)
    students = generate_students(config, streams)[:2]
    sessions = generate_sessions(config, students, sessions_per_student=1, rng_streams=build_rng_streams(seed))
    dataset_artifact = build_dataset_artifact(config, students, sessions)
    bas_artifact = BASEngine().compute(dataset_artifact)
    reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)

    artifact = InterventionPlanner().plan(dataset_artifact, bas_artifact, reward_artifact)
    assert len(artifact.decisions) == len(dataset_artifact.records)


def test_empty_dataset_produces_empty_artifact() -> None:
    dataset_artifact, bas_artifact, reward_artifact = _artifacts(n_students=1, sessions_per_student=1)
    empty_dataset = dataset_artifact.model_copy(
        update={
            "records": [],
            "metadata": dataset_artifact.metadata.model_copy(update={"record_count": 0}),
        }
    )
    empty_bas = bas_artifact.model_copy(update={"records": []})
    empty_reward = reward_artifact.model_copy(update={"records": []})

    artifact = InterventionPlanner().plan(empty_dataset, empty_bas, empty_reward)
    assert artifact.decisions == []
    assert artifact.session_summaries == []
    assert artifact.statistics.record_count == 0


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------


def test_stress_100000_intervention_decisions() -> None:
    """As with Modules 7/8/9/10's stress tests: replicate a real, small
    simulated batch with re-suffixed IDs rather than re-running full
    simulation 100,000+ times.
    """

    base_dataset_artifact, base_bas_artifact, base_reward_artifact = _artifacts(
        n_students=10, sessions_per_student=2
    )
    base_records = base_dataset_artifact.records
    base_bas_records = base_bas_artifact.records
    base_reward_records = base_reward_artifact.records
    target_size = 100_000
    replication_factor = target_size // len(base_records) + 1

    dataset_records: list[DatasetRecord] = []
    bas_records = []
    reward_records = []
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
        for reward_record in base_reward_records:
            reward_data = reward_record.model_dump(mode="json")
            reward_data["session_id"] = f"{reward_record.session_id}_batch{batch}"
            reward_data["student_id"] = f"{reward_record.student_id}_batch{batch}"
            reward_records.append(type(reward_record).model_validate(reward_data))

    assert len(dataset_records) >= target_size

    large_dataset_artifact = base_dataset_artifact.model_copy(
        update={
            "records": dataset_records,
            "metadata": base_dataset_artifact.metadata.model_copy(update={"record_count": len(dataset_records)}),
        }
    )
    large_bas_artifact = base_bas_artifact.model_copy(update={"records": bas_records})
    large_reward_artifact = base_reward_artifact.model_copy(update={"records": reward_records})

    artifact = InterventionPlanner().plan(large_dataset_artifact, large_bas_artifact, large_reward_artifact)
    assert len(artifact.decisions) == len(dataset_records)
    assert artifact.statistics.record_count == len(dataset_records)
    for decision in artifact.decisions[:1000]:
        assert 0.0 <= decision.need_score <= 1.0
        assert 0.0 <= decision.confidence <= 1.0
