"""Tests for Module 4: Response Generator (strategy-based rebuild)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataset_generator.config import AttentionState, Difficulty, GeneratorConfig, default_config
from dataset_generator.generators.profiles import ProfileFactory
from dataset_generator.generators.prompt_analyzer import PromptAnalyzer
from dataset_generator.generators.prompt_generator import PromptGenerator
from dataset_generator.generators.response_generator import ResponseGenerator
from dataset_generator.generators.response_report import build_response_report, render_response_report
from dataset_generator.generators.response_scoring import (
    coherence_score,
    confidence_score,
    engagement_proxy,
    expected_correctness,
)
from dataset_generator.generators.response_strategies import (
    DistractedStrategy,
    FocusedStrategy,
    ImpulsiveStrategy,
    ResponseStrategyFactory,
)
from dataset_generator.models.response import Response
from dataset_generator.models.session_context import SessionContext
from dataset_generator.utils import build_rng_streams
from dataset_generator.validators.response_validator import (
    duplicate_response_ids,
    exact_duplicate_rate,
    is_immediate_repeat,
    validate_response,
    validate_response_batch,
)


def _setup(config=None, seed: int | None = None):
    config = config or default_config()
    streams = build_rng_streams(seed if seed is not None else config.seed)
    prompt_generator = PromptGenerator(config, streams.prompt_rng)
    response_generator = ResponseGenerator(config, streams.response_rng)
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    return config, prompt_generator, response_generator, student


def _ctx(**overrides) -> SessionContext:
    defaults = dict(interaction_number=1, session_length=20)
    defaults.update(overrides)
    return SessionContext(**defaults)


# ---------------------------------------------------------------------------
# Prompt Analyzer (Step 2)
# ---------------------------------------------------------------------------


def test_prompt_analyzer_extracts_all_fields() -> None:
    config = default_config()
    streams = build_rng_streams(config.seed)
    prompt_gen = PromptGenerator(config, streams.prompt_rng)
    prompt = prompt_gen.generate_prompt(subject="Science", topic="Chemistry", difficulty=Difficulty.MEDIUM)

    analysis = PromptAnalyzer().analyze(prompt)
    assert analysis.prompt_id == prompt.prompt_id
    assert analysis.subject == "Science"
    assert analysis.topic == "Chemistry"
    assert analysis.topic_display == "Chemistry"
    assert analysis.difficulty == Difficulty.MEDIUM
    assert analysis.expected_response_length == prompt.estimated_response_length
    assert analysis.keywords == prompt.keywords


def test_prompt_analyzer_underscore_topic_becomes_display_name() -> None:
    config = default_config()
    streams = build_rng_streams(config.seed)
    prompt_gen = PromptGenerator(config, streams.prompt_rng)
    prompt = prompt_gen.generate_prompt(subject="Reading", topic="Literary_Analysis")
    analysis = PromptAnalyzer().analyze(prompt)
    assert analysis.topic_display == "Literary Analysis"


# ---------------------------------------------------------------------------
# Strategy correctness (Step 3)
# ---------------------------------------------------------------------------


def test_strategy_factory_returns_correct_class_for_each_state() -> None:
    assert isinstance(ResponseStrategyFactory.for_state(AttentionState.FOCUSED), FocusedStrategy)
    assert isinstance(ResponseStrategyFactory.for_state(AttentionState.DISTRACTED), DistractedStrategy)
    assert isinstance(ResponseStrategyFactory.for_state(AttentionState.IMPULSIVE), ImpulsiveStrategy)


def test_strategy_traits_ordered_focused_most_complete() -> None:
    focused = ResponseStrategyFactory.for_state(AttentionState.FOCUSED)
    distracted = ResponseStrategyFactory.for_state(AttentionState.DISTRACTED)
    impulsive = ResponseStrategyFactory.for_state(AttentionState.IMPULSIVE)

    assert focused.completeness > distracted.completeness
    assert focused.reasoning_depth > impulsive.reasoning_depth
    assert focused.length_multiplier > distracted.length_multiplier > impulsive.length_multiplier
    assert impulsive.error_tendency > focused.error_tendency
    assert distracted.error_tendency > focused.error_tendency


def test_strategy_subclass_missing_traits_raises_at_definition() -> None:
    from dataset_generator.generators.response_strategies import ResponseStrategy

    with pytest.raises(TypeError):

        class IncompleteStrategy(ResponseStrategy):
            attention_state = AttentionState.FOCUSED
            completeness = 0.5
            # remaining traits deliberately omitted


def test_target_length_scales_by_multiplier() -> None:
    focused = FocusedStrategy()
    impulsive = ImpulsiveStrategy()
    assert focused.target_length(20) > impulsive.target_length(20)


# ---------------------------------------------------------------------------
# Response model validation (Step 1)
# ---------------------------------------------------------------------------


def test_response_model_rejects_empty_text() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Algebra")
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=_ctx()
    )
    data = response.model_dump(mode="json")
    data["response_text"] = ""
    with pytest.raises(ValidationError):
        Response.model_validate(data)


def test_response_model_rejects_out_of_range_confidence() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Algebra")
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=_ctx()
    )
    data = response.model_dump(mode="json")
    data["confidence"] = 5.0
    with pytest.raises(ValidationError):
        Response.model_validate(data)


def test_response_features_token_count_matches_response_length() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Algebra")
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=_ctx()
    )
    assert response.features.token_count == response.response_length


# ---------------------------------------------------------------------------
# Profile adaptation (Step 4)
# ---------------------------------------------------------------------------


def test_recovering_learner_benefits_more_from_intervention_than_focused() -> None:
    config = default_config()
    focused_student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    recovering_student = ProfileFactory.create("Recovering_Learner").generate_student(0, config)
    strategy = DistractedStrategy()
    ctx = _ctx(intervention_applied=True)

    focused_value = expected_correctness(Difficulty.MEDIUM, strategy, focused_student, ctx, config.response_generation)
    recovering_value = expected_correctness(Difficulty.MEDIUM, strategy, recovering_student, ctx, config.response_generation)
    assert recovering_value > focused_value


def test_gradually_fatigued_correctness_drops_over_session() -> None:
    config = default_config()
    student = ProfileFactory.create("Gradually_Fatigued").generate_student(0, config)
    strategy = FocusedStrategy()

    early = expected_correctness(
        Difficulty.MEDIUM, strategy, student, _ctx(interaction_number=1, session_length=20), config.response_generation
    )
    late = expected_correctness(
        Difficulty.MEDIUM, strategy, student, _ctx(interaction_number=20, session_length=20), config.response_generation
    )
    assert late < early


# ---------------------------------------------------------------------------
# Difficulty adaptation
# ---------------------------------------------------------------------------


def test_expected_correctness_decreases_with_difficulty() -> None:
    config = default_config()
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    strategy = FocusedStrategy()
    ctx = _ctx()
    values = {
        difficulty: expected_correctness(difficulty, strategy, student, ctx, config.response_generation)
        for difficulty in Difficulty
    }
    assert values[Difficulty.EASY] > values[Difficulty.MEDIUM] > values[Difficulty.HARD]


# ---------------------------------------------------------------------------
# Attention-state adaptation
# ---------------------------------------------------------------------------


def test_attention_states_produce_structurally_different_responses() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Science", topic="Biology", difficulty=Difficulty.EASY)

    focused = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=_ctx()
    )
    impulsive = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.IMPULSIVE, session_context=_ctx()
    )
    assert impulsive.response_length < focused.response_length


def test_impulsive_confident_even_when_incorrect() -> None:
    config = default_config()
    strategy = ImpulsiveStrategy()
    confident = confidence_score(strategy, hesitation_markers=[], config=config.response_generation)
    assert confident == pytest.approx(config.response_generation.confidence_by_attention_state[AttentionState.IMPULSIVE])


def test_hesitation_reduces_confidence() -> None:
    config = default_config()
    strategy = DistractedStrategy()
    no_hesitation = confidence_score(strategy, [], config.response_generation)
    with_hesitation = confidence_score(strategy, ["um", "maybe"], config.response_generation)
    assert with_hesitation < no_hesitation


# ---------------------------------------------------------------------------
# Session-memory influence (Step 5)
# ---------------------------------------------------------------------------


def test_rolling_engagement_shifts_expected_correctness() -> None:
    config = default_config()
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    strategy = FocusedStrategy()
    high = expected_correctness(
        Difficulty.MEDIUM, strategy, student, _ctx(rolling_engagement=0.9), config.response_generation
    )
    low = expected_correctness(
        Difficulty.MEDIUM, strategy, student, _ctx(rolling_engagement=0.1), config.response_generation
    )
    assert high > low


def test_momentum_bonus_when_repeating_attention_state() -> None:
    config = default_config()
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    strategy = FocusedStrategy()
    with_momentum = expected_correctness(
        Difficulty.MEDIUM, strategy, student,
        _ctx(previous_attention_state=AttentionState.FOCUSED), config.response_generation,
    )
    without_momentum = expected_correctness(
        Difficulty.MEDIUM, strategy, student,
        _ctx(previous_attention_state=AttentionState.DISTRACTED), config.response_generation,
    )
    assert with_momentum > without_momentum


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_response_sequence() -> None:
    config = default_config()

    def run():
        _, prompt_gen, response_gen, student = _setup(config, seed=config.seed)
        prompt = prompt_gen.generate_prompt(subject="Science", topic="Physics", difficulty=Difficulty.HARD)
        return response_gen.generate_response(
            prompt=prompt, student=student, attention_state=AttentionState.DISTRACTED, session_context=_ctx()
        )

    a, b = run(), run()
    assert a.model_dump(mode="json") == b.model_dump(mode="json")


def test_different_seeds_produce_different_responses() -> None:
    config = default_config()
    _, prompt_gen_a, response_gen_a, student_a = _setup(config, seed=config.seed)
    prompt_a = prompt_gen_a.generate_prompt(subject="Science", topic="Physics", difficulty=Difficulty.HARD)
    response_a = response_gen_a.generate_response(
        prompt=prompt_a, student=student_a, attention_state=AttentionState.DISTRACTED, session_context=_ctx()
    )

    _, prompt_gen_b, response_gen_b, student_b = _setup(config, seed=config.seed + 1)
    prompt_b = prompt_gen_b.generate_prompt(subject="Science", topic="Physics", difficulty=Difficulty.HARD)
    response_b = response_gen_b.generate_response(
        prompt=prompt_b, student=student_b, attention_state=AttentionState.DISTRACTED, session_context=_ctx()
    )
    assert response_a.response_text != response_b.response_text


def test_response_rng_independent_of_prompt_and_student_rng() -> None:
    """Step 7: changing response randomness must not affect prompt/student sampling."""

    config = default_config()
    streams_a = build_rng_streams(config.seed)
    streams_b = build_rng_streams(config.seed)

    prompt_gen_a = PromptGenerator(config, streams_a.prompt_rng)
    prompt_gen_b = PromptGenerator(config, streams_b.prompt_rng)
    prompt_a = prompt_gen_a.generate_batch(5)
    # Exhaust the response stream on side "a" only, between prompt draws.
    ResponseGenerator(config, streams_a.response_rng)._rng.random(1000)
    prompt_a += prompt_gen_a.generate_batch(5)
    prompt_b = prompt_gen_b.generate_batch(10)

    assert [p.model_dump(mode="json") for p in prompt_a] == [p.model_dump(mode="json") for p in prompt_b]


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_is_immediate_repeat() -> None:
    assert is_immediate_repeat("I don't know.", "I don't know.") is True
    assert is_immediate_repeat("I don't know.", "Something else.") is False
    assert is_immediate_repeat("I don't know.", None) is False


def test_generator_avoids_immediate_verbatim_repeat_when_possible() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Algebra", difficulty=Difficulty.EASY)
    first = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.DISTRACTED, session_context=_ctx()
    )
    second = response_gen.generate_response(
        prompt=prompt,
        student=student,
        attention_state=AttentionState.DISTRACTED,
        session_context=_ctx(interaction_number=2, previous_response_text=first.response_text),
    )
    assert second.response_text != first.response_text


def test_duplicate_response_ids_and_rate_are_consistent() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(
        subject="Mathematics", topic="Algebra", difficulty=Difficulty.EASY, cognitive_level=None
    )
    responses = [
        response_gen.generate_response(
            prompt=prompt,
            student=student,
            attention_state=AttentionState.IMPULSIVE,
            session_context=_ctx(interaction_number=i + 1),
        )
        for i in range(40)
    ]
    dupes = duplicate_response_ids(responses)
    rate = exact_duplicate_rate(responses)
    assert rate == pytest.approx(len(dupes) / len(responses))


# ---------------------------------------------------------------------------
# Feature consistency (Step 6)
# ---------------------------------------------------------------------------


def test_coherence_score_penalizes_repetition() -> None:
    high = coherence_score(lexical_diversity=0.9, repetition_ratio=0.0)
    low = coherence_score(lexical_diversity=0.9, repetition_ratio=0.9)
    assert high > low


def test_engagement_proxy_rewards_similarity_and_diversity() -> None:
    weights = (0.3, 0.3, 0.2, 0.2)
    high = engagement_proxy(10, 10, semantic_similarity=0.9, lexical_diversity=0.9, repetition_ratio=0.0, weights=weights)
    low = engagement_proxy(10, 10, semantic_similarity=0.1, lexical_diversity=0.1, repetition_ratio=0.9, weights=weights)
    assert high > low


def test_features_derived_not_random_for_fixed_text() -> None:
    """Same generated text should always yield the same derived features."""

    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Science", topic="Biology", difficulty=Difficulty.EASY)
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=_ctx()
    )
    from dataset_generator.utils import word_tokenize

    tokens = word_tokenize(response.response_text)
    recomputed_diversity = len({t.lower() for t in tokens}) / len(tokens)
    assert response.lexical_diversity == pytest.approx(recomputed_diversity)


# ---------------------------------------------------------------------------
# Quality validation (Step 8)
# ---------------------------------------------------------------------------


def test_validate_response_batch_flags_nothing_for_generator_output() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Geometry", difficulty=Difficulty.MEDIUM)
    responses = [
        response_gen.generate_response(
            prompt=prompt,
            student=student,
            attention_state=AttentionState.FOCUSED,
            session_context=_ctx(interaction_number=i + 1),
        )
        for i in range(30)
    ]
    assert validate_response_batch(responses) == {}


def test_validate_response_rejects_unfilled_placeholder() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Geometry")
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=_ctx()
    )
    data = response.model_dump(mode="json")
    data["response_text"] = "The answer is {keyword}."
    tampered = Response.model_validate(data)
    assert "unfilled template placeholder" in validate_response(tampered)


def test_validate_response_detects_prompt_metadata_inconsistency() -> None:
    _, prompt_gen, response_gen, student = _setup()
    prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Geometry", difficulty=Difficulty.EASY)
    response = response_gen.generate_response(
        prompt=prompt, student=student, attention_state=AttentionState.FOCUSED, session_context=_ctx()
    )
    other_prompt = prompt_gen.generate_prompt(subject="Mathematics", topic="Geometry", difficulty=Difficulty.HARD)
    issues = validate_response(response, prompt=other_prompt)
    assert any("difficulty" in issue for issue in issues)


def test_response_generation_config_rejects_bad_confidence_map() -> None:
    data = default_config().model_dump(mode="json")
    del data["response_generation"]["confidence_by_attention_state"]["Impulsive"]
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Stress test with 10,000 generated responses
# ---------------------------------------------------------------------------


def test_generate_10000_responses_and_report() -> None:
    config = default_config()
    streams = build_rng_streams(config.seed)
    prompt_gen = PromptGenerator(config, streams.prompt_rng)
    response_gen = ResponseGenerator(config, streams.response_rng)
    student_pool = [
        ProfileFactory.create(key).generate_student(i, config)
        for i, key in enumerate(ProfileFactory.available_profiles())
    ]

    responses = []
    previous_text = None
    previous_state = None
    states = list(AttentionState)
    for i in range(10_000):
        prompt = prompt_gen.generate_prompt()
        student = student_pool[i % len(student_pool)]
        state = states[i % 3]
        ctx = SessionContext(
            interaction_number=(i % 30) + 1,
            session_length=30,
            previous_response_text=previous_text,
            previous_attention_state=previous_state,
            intervention_applied=(i % 10 == 0),
        )
        response = response_gen.generate_response(
            prompt=prompt, student=student, attention_state=state, session_context=ctx
        )
        responses.append(response)
        previous_text = response.response_text
        previous_state = state

    assert len(responses) == 10_000
    assert validate_response_batch(responses) == {}

    report = build_response_report(responses)
    assert report.total_responses == 10_000
    assert abs(sum(report.attention_state_distribution.values()) - 1.0) < 1e-6
    assert abs(sum(report.profile_distribution.values()) - 1.0) < 1e-6

    text = render_response_report(report)
    assert "Total responses: 10000" in text
    assert "Duplicate rate" in text
