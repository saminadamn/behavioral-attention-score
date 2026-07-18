"""Tests for Module 5: Behaviour Generator."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from dataset_generator.config import AttentionState, GeneratorConfig, default_config
from dataset_generator.config.schema import FeatureDistributionParams
from dataset_generator.distributions.sampler import sample as sample_distribution
from dataset_generator.generators.behaviour_generator import BehaviourGenerator
from dataset_generator.generators.behaviour_report import build_behaviour_report, render_behaviour_report
from dataset_generator.generators.behaviour_scoring import (
    fatigue_level,
    normalized_latency,
    sample_response_latency,
    update_rolling_value,
)
from dataset_generator.generators.profiles import ProfileFactory
from dataset_generator.generators.prompt_generator import PromptGenerator
from dataset_generator.generators.response_generator import ResponseGenerator
from dataset_generator.models.behaviour import BehaviourRecord
from dataset_generator.models.session_context import SessionContext
from dataset_generator.utils import build_rng_streams
from dataset_generator.validators.behaviour_validator import (
    validate_behaviour_batch,
    validate_behaviour_record,
)


def _setup(config=None, seed: int | None = None):
    config = config or default_config()
    streams = build_rng_streams(seed if seed is not None else config.seed)
    prompt_gen = PromptGenerator(config, streams.prompt_rng)
    response_gen = ResponseGenerator(config, streams.response_rng)
    behaviour_gen = BehaviourGenerator(config, streams.noise_rng)
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    return config, prompt_gen, response_gen, behaviour_gen, student


def _ctx(**overrides) -> SessionContext:
    defaults = dict(session_id="S1", interaction_number=1, session_length=20)
    defaults.update(overrides)
    return SessionContext(**defaults)


# ---------------------------------------------------------------------------
# Distribution sampling (Step 2)
# ---------------------------------------------------------------------------


def test_sample_normal_respects_clip_bounds() -> None:
    rng = np.random.default_rng(1)
    params = FeatureDistributionParams(family="normal", params={"mean": 5.0, "std": 10.0}, clip_min=0.0, clip_max=1.0)
    values = [sample_distribution(rng, params) for _ in range(500)]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_sample_gamma_is_nonnegative() -> None:
    rng = np.random.default_rng(2)
    params = FeatureDistributionParams(family="gamma", params={"shape": 2.0, "scale": 3.0})
    values = [sample_distribution(rng, params) for _ in range(200)]
    assert all(v >= 0.0 for v in values)


def test_sample_beta_within_unit_interval() -> None:
    rng = np.random.default_rng(3)
    params = FeatureDistributionParams(family="beta", params={"alpha": 2.0, "beta": 5.0})
    values = [sample_distribution(rng, params) for _ in range(200)]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_sample_poisson_returns_nonnegative_integral_floats() -> None:
    rng = np.random.default_rng(4)
    params = FeatureDistributionParams(family="poisson", params={"lam": 1.5})
    values = [sample_distribution(rng, params) for _ in range(200)]
    assert all(v >= 0.0 and v == int(v) for v in values)


def test_sample_truncated_normal_stays_within_bounds() -> None:
    rng = np.random.default_rng(5)
    params = FeatureDistributionParams(
        family="truncated_normal", params={"mean": 50.0, "std": 30.0}, clip_min=10.0, clip_max=60.0
    )
    values = [sample_distribution(rng, params) for _ in range(1000)]
    assert all(10.0 <= v <= 60.0 for v in values)
    # Should exercise both tails given the wide std relative to the range.
    assert min(values) < 20.0
    assert max(values) > 50.0


def test_truncated_normal_requires_both_clip_bounds() -> None:
    with pytest.raises(ValidationError):
        FeatureDistributionParams(family="truncated_normal", params={"mean": 5.0, "std": 1.0}, clip_min=0.0)


def test_unsupported_family_raises() -> None:
    rng = np.random.default_rng(6)
    # Construct via model_construct to bypass Literal validation for this
    # deliberately-invalid-at-the-sampler-level test.
    params = FeatureDistributionParams.model_construct(family="unknown", params={}, clip_min=None, clip_max=None)
    with pytest.raises(ValueError):
        sample_distribution(rng, params)


# ---------------------------------------------------------------------------
# Attention-state adaptation (Step 3)
# ---------------------------------------------------------------------------


def test_latency_ordering_by_attention_state() -> None:
    config = default_config()
    rng = np.random.default_rng(config.seed)
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    ctx = _ctx()
    fatigue = fatigue_level(student, ctx, config.behaviour_generation)

    def avg_latency(state: AttentionState) -> float:
        return sum(
            sample_response_latency(rng, student, state, fatigue, config.behaviour_generation)
            for _ in range(200)
        ) / 200

    focused = avg_latency(AttentionState.FOCUSED)
    distracted = avg_latency(AttentionState.DISTRACTED)
    impulsive = avg_latency(AttentionState.IMPULSIVE)
    assert impulsive < focused < distracted


# ---------------------------------------------------------------------------
# Student profile adaptation (Step 4)
# ---------------------------------------------------------------------------


def test_highly_distractible_has_higher_latency_variance_than_focused() -> None:
    config = default_config()
    rng = np.random.default_rng(config.seed)
    focused_student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    distractible_student = ProfileFactory.create("Highly_Distractible").generate_student(0, config)
    ctx = _ctx()

    def latencies(student):
        fatigue = fatigue_level(student, ctx, config.behaviour_generation)
        return [
            sample_response_latency(rng, student, AttentionState.FOCUSED, fatigue, config.behaviour_generation)
            for _ in range(300)
        ]

    focused_std = float(np.std(latencies(focused_student)))
    distractible_std = float(np.std(latencies(distractible_student)))
    assert distractible_std > focused_std


def test_highly_impulsive_has_shorter_interaction_duration_distribution() -> None:
    config = default_config()
    focused_dist = config.distributions.for_state(AttentionState.FOCUSED).interaction_duration
    impulsive_dist = config.distributions.for_state(AttentionState.IMPULSIVE).interaction_duration
    assert impulsive_dist.params["mean"] < focused_dist.params["mean"]


def test_gradually_fatigued_latency_increases_across_session() -> None:
    config = default_config()
    rng = np.random.default_rng(config.seed)
    student = ProfileFactory.create("Gradually_Fatigued").generate_student(0, config)

    early_ctx = _ctx(interaction_number=1, session_length=30)
    late_ctx = _ctx(interaction_number=30, session_length=30)

    early_fatigue = fatigue_level(student, early_ctx, config.behaviour_generation)
    late_fatigue = fatigue_level(student, late_ctx, config.behaviour_generation)

    early_avg = sum(
        sample_response_latency(rng, student, AttentionState.FOCUSED, early_fatigue, config.behaviour_generation)
        for _ in range(300)
    ) / 300
    late_avg = sum(
        sample_response_latency(rng, student, AttentionState.FOCUSED, late_fatigue, config.behaviour_generation)
        for _ in range(300)
    ) / 300
    assert late_avg > early_avg


def test_recovering_learner_fatigue_reduced_by_intervention() -> None:
    config = default_config()
    student = ProfileFactory.create("Recovering_Learner").generate_student(0, config)
    without = fatigue_level(student, _ctx(interaction_number=15, session_length=30), config.behaviour_generation)
    with_intervention = fatigue_level(
        student, _ctx(interaction_number=15, session_length=30, intervention_applied=True), config.behaviour_generation
    )
    assert with_intervention < without


# ---------------------------------------------------------------------------
# Fatigue progression
# ---------------------------------------------------------------------------


def test_fatigue_level_bounded_and_monotonic_in_progress() -> None:
    config = default_config()
    student = ProfileFactory.create("Gradually_Fatigued").generate_student(0, config)
    values = [
        fatigue_level(student, _ctx(interaction_number=i, session_length=30), config.behaviour_generation)
        for i in range(1, 31)
    ]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values[-1] >= values[0]


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def test_update_rolling_value_first_observation_returns_current() -> None:
    assert update_rolling_value(None, 7.0, window=5) == 7.0


def test_update_rolling_value_moves_toward_current() -> None:
    updated = update_rolling_value(previous=10.0, current=0.0, window=5)
    assert 0.0 < updated < 10.0
    # alpha = 1/5 = 0.2 -> 0.2*0 + 0.8*10 = 8.0
    assert updated == pytest.approx(8.0)


def test_normalized_latency_zero_at_baseline() -> None:
    config = default_config()
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    assert normalized_latency(student.baseline_latency, student) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Transition correctness
# ---------------------------------------------------------------------------


def test_transition_occurred_flag_matches_state_change() -> None:
    _, prompt_gen, response_gen, behaviour_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Algebra")

    ctx1 = _ctx(interaction_number=1)
    resp1 = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=ctx1
    )
    b1 = behaviour_gen.generate_behaviour(
        student=student, prompt=prompt, response=resp1, attention_state=AttentionState.FOCUSED, session_context=ctx1
    )
    assert b1.features.transition_occurred is False

    ctx2 = _ctx(interaction_number=2, previous_attention_state=AttentionState.FOCUSED)
    resp2 = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.DISTRACTED, session_context=ctx2
    )
    b2 = behaviour_gen.generate_behaviour(
        student=student, prompt=prompt, response=resp2, attention_state=AttentionState.DISTRACTED, session_context=ctx2
    )
    assert b2.features.transition_occurred is True


# ---------------------------------------------------------------------------
# No duplicated logic (Step 9): response-derived fields pass through unchanged
# ---------------------------------------------------------------------------


def test_response_derived_fields_pass_through_unchanged() -> None:
    _, prompt_gen, response_gen, behaviour_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Science", topic="Biology")
    ctx = _ctx()
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=ctx
    )
    behaviour = behaviour_gen.generate_behaviour(
        student=student, prompt=prompt, response=response, attention_state=AttentionState.FOCUSED, session_context=ctx
    )
    assert behaviour.response_length == response.response_length
    assert behaviour.engagement_score == response.engagement_proxy
    assert behaviour.repetition_ratio == response.features.repetition_ratio
    assert behaviour.topic_shift == response.features.topic_shift


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_behaviour_batch_flags_nothing_for_generator_output() -> None:
    _, prompt_gen, response_gen, behaviour_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Science", topic="Physics")
    records = []
    prev_state = None
    for i in range(1, 21):
        ctx = _ctx(interaction_number=i, previous_attention_state=prev_state)
        response = response_gen.generate_response(
            prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=ctx
        )
        record = behaviour_gen.generate_behaviour(
            student=student, prompt=prompt, response=response,
            attention_state=AttentionState.FOCUSED, session_context=ctx,
        )
        records.append(record)
        prev_state = AttentionState.FOCUSED
    assert validate_behaviour_batch(records) == {}


def test_validate_behaviour_record_detects_hesitation_exceeding_duration() -> None:
    _, prompt_gen, response_gen, behaviour_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Science", topic="Physics")
    ctx = _ctx()
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=ctx
    )
    record = behaviour_gen.generate_behaviour(
        student=student, prompt=prompt, response=response, attention_state=AttentionState.FOCUSED, session_context=ctx
    )
    data = record.model_dump(mode="json")
    data["hesitation_duration"] = data["interaction_duration"] + 100.0
    tampered = BehaviourRecord.model_validate(data)
    assert "hesitation_duration exceeds interaction_duration" in validate_behaviour_record(tampered)


def test_validate_behaviour_record_detects_transition_flag_inconsistency() -> None:
    _, prompt_gen, response_gen, behaviour_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Science", topic="Physics")
    ctx = _ctx(previous_attention_state=AttentionState.DISTRACTED)
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=ctx
    )
    record = behaviour_gen.generate_behaviour(
        student=student, prompt=prompt, response=response, attention_state=AttentionState.FOCUSED, session_context=ctx
    )
    data = record.model_dump(mode="json")
    data["features"]["transition_occurred"] = False  # should be True (Distracted -> Focused)
    tampered = BehaviourRecord.model_validate(data)
    assert "transition_occurred" in " ".join(validate_behaviour_record(tampered))


def test_behaviour_generation_config_rejects_missing_state_multiplier() -> None:
    data = default_config().model_dump(mode="json")
    del data["behaviour_generation"]["attention_state_latency_multiplier"]["Impulsive"]
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_behaviour_record() -> None:
    config = default_config()

    def run():
        _, prompt_gen, response_gen, behaviour_gen, student = _setup(config, seed=config.seed)
        prompt = prompt_gen.generate_prompt(subject="Science", topic="Chemistry")
        ctx = _ctx()
        response = response_gen.generate_response(
            prompt=prompt, student=student, attention_state=AttentionState.DISTRACTED, session_context=ctx
        )
        return behaviour_gen.generate_behaviour(
            student=student, prompt=prompt, response=response,
            attention_state=AttentionState.DISTRACTED, session_context=ctx,
        )

    a, b = run(), run()
    assert a.model_dump(mode="json") == b.model_dump(mode="json")


def test_different_seeds_produce_different_latency() -> None:
    config = default_config()
    _, prompt_gen_a, response_gen_a, behaviour_gen_a, student_a = _setup(config, seed=config.seed)
    prompt_a = prompt_gen_a.generate_prompt(subject="Science", topic="Chemistry")
    ctx = _ctx()
    resp_a = response_gen_a.generate_response(
        prompt=prompt_a, student=student_a, attention_state=AttentionState.DISTRACTED, session_context=ctx
    )
    behaviour_a = behaviour_gen_a.generate_behaviour(
        student=student_a, prompt=prompt_a, response=resp_a, attention_state=AttentionState.DISTRACTED, session_context=ctx
    )

    _, prompt_gen_b, response_gen_b, behaviour_gen_b, student_b = _setup(config, seed=config.seed + 1)
    prompt_b = prompt_gen_b.generate_prompt(subject="Science", topic="Chemistry")
    resp_b = response_gen_b.generate_response(
        prompt=prompt_b, student=student_b, attention_state=AttentionState.DISTRACTED, session_context=ctx
    )
    behaviour_b = behaviour_gen_b.generate_behaviour(
        student=student_b, prompt=prompt_b, response=resp_b, attention_state=AttentionState.DISTRACTED, session_context=ctx
    )
    assert behaviour_a.response_latency != behaviour_b.response_latency


def test_behaviour_rng_independent_of_other_streams() -> None:
    """Changing behaviour-generation randomness must not perturb other streams."""

    config = default_config()
    streams_a = build_rng_streams(config.seed)
    streams_b = build_rng_streams(config.seed)

    prompt_gen_a = PromptGenerator(config, streams_a.prompt_rng)
    prompt_gen_b = PromptGenerator(config, streams_b.prompt_rng)

    prompts_a = prompt_gen_a.generate_batch(5)
    streams_a.noise_rng.random(1000)  # exhaust behaviour's stream on side "a" only
    prompts_a += prompt_gen_a.generate_batch(5)
    prompts_b = prompt_gen_b.generate_batch(10)

    assert [p.model_dump(mode="json") for p in prompts_a] == [p.model_dump(mode="json") for p in prompts_b]


# ---------------------------------------------------------------------------
# Stress test with 100,000 interactions
# ---------------------------------------------------------------------------


def test_generate_100000_behaviour_records_and_report() -> None:
    config = default_config()
    streams = build_rng_streams(config.seed)
    prompt_gen = PromptGenerator(config, streams.prompt_rng)
    response_gen = ResponseGenerator(config, streams.response_rng)
    behaviour_gen = BehaviourGenerator(config, streams.noise_rng)
    students = [
        ProfileFactory.create(key).generate_student(i, config)
        for i, key in enumerate(ProfileFactory.available_profiles())
    ]

    records = []
    states = list(AttentionState)
    session_length = 50
    for i in range(100_000):
        student = students[i % len(students)]
        interaction_number = (i % session_length) + 1
        prev_state = states[(i - 1) % 3] if interaction_number > 1 else None
        prompt = prompt_gen.generate_prompt()
        state = states[i % 3]
        ctx = SessionContext(
            session_id=f"session-{i // session_length}",
            interaction_number=interaction_number,
            session_length=session_length,
            previous_attention_state=prev_state,
            intervention_applied=(i % 20 == 0),
        )
        response = response_gen.generate_response(
            prompt=prompt, student=student, attention_state=state, session_context=ctx
        )
        record = behaviour_gen.generate_behaviour(
            student=student, prompt=prompt, response=response, attention_state=state, session_context=ctx
        )
        records.append(record)

    assert len(records) == 100_000
    assert validate_behaviour_batch(records) == {}

    report = build_behaviour_report(records)
    assert report.total_records == 100_000
    assert abs(sum(report.state_frequencies.values()) - 1.0) < 1e-6

    text = render_behaviour_report(report)
    assert "Total behaviour records: 100000" in text
    assert "Validation failures: 0" in text
