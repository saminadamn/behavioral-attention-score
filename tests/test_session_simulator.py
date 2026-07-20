"""Tests for Module 6: Temporal Session Simulator."""

from __future__ import annotations

import numpy as np

from dataset_generator.config import AttentionState, default_config
from dataset_generator.config.attention_state import combine_transition_matrix
from dataset_generator.generators.profiles import ProfileFactory
from dataset_generator.generators.session_batch import build_session_simulator, generate_sessions
from dataset_generator.generators.session_report import build_session_report, render_session_report
from dataset_generator.generators.student_profile_generator import generate_students
from dataset_generator.generators.transition_engine import TransitionEngine
from dataset_generator.utils import build_rng_streams
from dataset_generator.validators.session_validator import validate_session, validate_session_batch


def _simulator(config=None, seed: int | None = None):
    config = config or default_config()
    streams = build_rng_streams(seed if seed is not None else config.seed)
    return build_session_simulator(config, streams)


# ---------------------------------------------------------------------------
# Transition matrix sampling
# ---------------------------------------------------------------------------


def test_effective_matrix_matches_shared_combine_function() -> None:
    config = default_config()
    rng = np.random.default_rng(config.seed)
    engine = TransitionEngine(config, rng)

    for profile_key, profile in config.profiles.items():
        expected = combine_transition_matrix(config.transition_matrix.matrix, profile.transition_modifiers)
        assert engine.effective_matrix(profile_key) == expected


def test_sample_next_state_respects_configured_probabilities_approximately() -> None:
    config = default_config()
    rng = np.random.default_rng(config.seed)
    engine = TransitionEngine(config, rng)

    profile_key = "Consistently_Focused"
    expected_row = engine.effective_matrix(profile_key)[AttentionState.FOCUSED]

    n = 20_000
    counts = {state: 0 for state in AttentionState}
    for _ in range(n):
        counts[engine.sample_next_state(AttentionState.FOCUSED, profile_key)] += 1

    for state, expected_p in expected_row.items():
        observed_p = counts[state] / n
        assert abs(observed_p - expected_p) < 0.02


def test_sample_initial_state_respects_class_balance_approximately() -> None:
    config = default_config()
    rng = np.random.default_rng(config.seed)
    engine = TransitionEngine(config, rng)

    n = 20_000
    counts = {state: 0 for state in AttentionState}
    for _ in range(n):
        counts[engine.sample_initial_state()] += 1

    for state, expected_p in config.class_balance.items():
        observed_p = counts[state] / n
        assert abs(observed_p - expected_p) < 0.02


def test_intervention_boost_shifts_mass_toward_focused() -> None:
    config = default_config()
    rng = np.random.default_rng(config.seed)
    engine = TransitionEngine(config, rng)

    n = 5000
    profile_key = "Recovering_Learner"
    student = ProfileFactory.create(profile_key).generate_student(0, config)

    without = sum(
        engine.sample_next_state(AttentionState.DISTRACTED, profile_key) == AttentionState.FOCUSED
        for _ in range(n)
    ) / n
    with_intervention = sum(
        engine.sample_next_state(
            AttentionState.DISTRACTED, profile_key,
            intervention_applied=True, intervention_sensitivity=student.intervention_sensitivity,
        ) == AttentionState.FOCUSED
        for _ in range(n)
    ) / n
    assert with_intervention > without


def test_intervention_never_overwrites_state_directly() -> None:
    """The boosted row must still be a valid probability distribution over all states."""

    config = default_config()
    rng = np.random.default_rng(config.seed)
    engine = TransitionEngine(config, rng)
    row = engine._apply_intervention_boost(
        engine.effective_matrix("Recovering_Learner")[AttentionState.DISTRACTED], intervention_sensitivity=2.0
    )
    assert abs(sum(row.values()) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in row.values())


# ---------------------------------------------------------------------------
# Session reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_session() -> None:
    config = default_config()

    def run():
        streams = build_rng_streams(config.seed)
        students = generate_students(config, streams)
        simulator = build_session_simulator(config, build_rng_streams(config.seed))
        return simulator.simulate_session(students[0], "S00001_SESS01")

    a, b = run(), run()
    assert a.model_dump(mode="json") == b.model_dump(mode="json")


def test_different_seeds_produce_different_sessions() -> None:
    config = default_config()
    streams_a = build_rng_streams(config.seed)
    students_a = generate_students(config, streams_a)
    session_a = build_session_simulator(config, build_rng_streams(config.seed)).simulate_session(
        students_a[0], "S00001_SESS01"
    )

    streams_b = build_rng_streams(config.seed + 1)
    students_b = generate_students(config, streams_b)
    session_b = build_session_simulator(config, build_rng_streams(config.seed + 1)).simulate_session(
        students_b[0], "S00001_SESS01"
    )

    assert [i.behaviour.attention_state for i in session_a.interactions] != [
        i.behaviour.attention_state for i in session_b.interactions
    ] or session_a.summary.average_latency != session_b.summary.average_latency


# ---------------------------------------------------------------------------
# Profile effects
# ---------------------------------------------------------------------------


def test_highly_distractible_transitions_more_than_consistently_focused() -> None:
    config = default_config()

    def transition_rate(profile_key: str, seed: int) -> float:
        streams = build_rng_streams(seed)
        simulator = build_session_simulator(config, streams)
        student = ProfileFactory.create(profile_key).generate_student(0, config)
        session = simulator.simulate_session(student, f"{profile_key}_SESS01")
        transitions = sum(1 for t in session.transition_history if t.transitioned)
        return transitions / len(session.interactions)

    focused_rates = [transition_rate("Consistently_Focused", seed) for seed in range(5)]
    distractible_rates = [transition_rate("Highly_Distractible", seed) for seed in range(5)]
    assert sum(distractible_rates) > sum(focused_rates)


# ---------------------------------------------------------------------------
# Fatigue progression
# ---------------------------------------------------------------------------


def test_fatigue_progression_is_monotonic_absent_intervention() -> None:
    simulator = _simulator()
    config = default_config()
    student = ProfileFactory.create("Gradually_Fatigued").generate_student(0, config)
    session = simulator.simulate_session(student, "GF_SESS01")

    for previous, current in zip(session.interactions, session.interactions[1:]):
        if current.behaviour.intervention_applied:
            continue
        assert current.behaviour.fatigue_level >= previous.behaviour.fatigue_level - 1e-9


# ---------------------------------------------------------------------------
# Intervention effects
# ---------------------------------------------------------------------------


def test_intervention_history_matches_flagged_interactions() -> None:
    simulator = _simulator()
    config = default_config()
    student = ProfileFactory.create("Recovering_Learner").generate_student(0, config)
    session = simulator.simulate_session(student, "RL_SESS01")

    flagged_numbers = {i.interaction_number for i in session.interactions if i.behaviour.intervention_applied}
    recorded_numbers = {e.interaction_number for e in session.intervention_history}
    assert flagged_numbers == recorded_numbers


# ---------------------------------------------------------------------------
# History consistency / validation
# ---------------------------------------------------------------------------


def test_generated_session_passes_validation() -> None:
    simulator = _simulator()
    config = default_config()
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    session = simulator.simulate_session(student, "CF_SESS01")
    assert validate_session(session) == []


def test_validate_session_detects_broken_transition_consistency() -> None:
    simulator = _simulator()
    config = default_config()
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    session = simulator.simulate_session(student, "CF_SESS02")

    data = session.model_dump(mode="json")
    data["transition_history"][0]["to_state"] = (
        "Distracted" if data["transition_history"][0]["to_state"] != "Distracted" else "Impulsive"
    )
    from dataset_generator.models.session import SessionRecord

    tampered = SessionRecord.model_validate(data)
    assert validate_session(tampered) != []


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def test_rolling_statistics_are_bounded_and_present() -> None:
    simulator = _simulator()
    config = default_config()
    student = ProfileFactory.create("Consistently_Focused").generate_student(0, config)
    session = simulator.simulate_session(student, "CF_SESS03")

    stats = session.statistics
    assert 0.0 <= stats.rolling_engagement <= 1.0
    assert 0.0 <= stats.rolling_correctness <= 1.0
    assert 0.0 <= stats.rolling_similarity <= 1.0
    assert stats.rolling_latency > 0.0
    assert stats.interaction_count == len(session.interactions)
    assert sum(stats.state_frequencies.values()) == len(session.interactions)


# ---------------------------------------------------------------------------
# Session report
# ---------------------------------------------------------------------------


def test_session_report_renders_without_error() -> None:
    simulator = _simulator()
    config = default_config()
    student = ProfileFactory.create("Highly_Distractible").generate_student(0, config)
    session = simulator.simulate_session(student, "HD_SESS01")
    report = build_session_report(session)
    assert report.validation_issues == []
    text = render_session_report(report)
    assert "Attention timeline" in text
    assert "Recovered transition matrix" in text


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def test_generate_sessions_multiple_students_and_sessions() -> None:
    config = default_config()
    streams = build_rng_streams(config.seed)
    students = generate_students(config, streams)[:5]
    sessions = generate_sessions(config, students, sessions_per_student=3, rng_streams=build_rng_streams(config.seed))

    assert len(sessions) == 15
    session_ids = [s.session_id for s in sessions]
    assert len(session_ids) == len(set(session_ids))
    assert validate_session_batch(sessions) == {}


# ---------------------------------------------------------------------------
# 1000-session stress test
# ---------------------------------------------------------------------------


def test_generate_1000_sessions_without_error() -> None:
    config = default_config()
    streams = build_rng_streams(config.seed)
    students = generate_students(config, streams)  # 100 students by default config

    session_streams = build_rng_streams(config.seed)
    # Reuse 100 students across 10 sessions each = 1000 sessions.
    sessions = generate_sessions(config, students, sessions_per_student=10, rng_streams=session_streams)

    assert len(sessions) == 1000
    failures = validate_session_batch(sessions)
    assert failures == {}

    total_interactions = sum(len(s.interactions) for s in sessions)
    assert total_interactions > 0
