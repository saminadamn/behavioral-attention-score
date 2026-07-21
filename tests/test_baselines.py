"""Tests for dataset_generator.rl_experimental.baselines — the live,
causal baseline-policy comparison (Phase 2). Verifies each policy actually
controls generation the way its name claims, not just that it runs.
"""

from __future__ import annotations

from dataset_generator.config import default_config
from dataset_generator.generators import generate_sessions, generate_students
from dataset_generator.rl_experimental.baselines import make_random_policy, no_intervention_policy, rule_based_policy
from dataset_generator.utils import build_rng_streams


def _students(n: int = 4):
    config = default_config()
    return config, generate_students(config, build_rng_streams(config.seed))[:n]


def test_rule_based_policy_is_none_sentinel():
    assert rule_based_policy is None


def test_no_intervention_policy_yields_zero_interventions_end_to_end():
    config, students = _students()
    sessions = generate_sessions(
        config, students, sessions_per_student=2, rng_streams=build_rng_streams(config.seed),
        intervention_policy=no_intervention_policy,
    )
    total_interventions = sum(len(s.intervention_history) for s in sessions)
    assert total_interventions == 0


def test_random_policy_intervention_rate_is_near_configured_probability():
    config, students = _students(n=20)
    sessions = generate_sessions(
        config, students, sessions_per_student=3, rng_streams=build_rng_streams(config.seed),
        intervention_policy=make_random_policy(config, seed=123),
    )
    total_interactions = sum(len(s.interactions) for s in sessions)
    total_interventions = sum(len(s.intervention_history) for s in sessions)
    observed_rate = total_interventions / total_interactions
    # Loose tolerance — this is a stochastic check on a real generative run,
    # not an exact-match assertion.
    assert abs(observed_rate - config.intervention_probability) < 0.15


def test_random_policy_is_deterministic_given_a_seed():
    config, students = _students()
    policy_a = make_random_policy(config, seed=7)
    policy_b = make_random_policy(config, seed=7)

    sessions_a = generate_sessions(
        config, students, sessions_per_student=1, rng_streams=build_rng_streams(config.seed),
        intervention_policy=policy_a,
    )
    sessions_b = generate_sessions(
        config, students, sessions_per_student=1, rng_streams=build_rng_streams(config.seed),
        intervention_policy=policy_b,
    )
    flags_a = [e.interaction_number for s in sessions_a for e in s.intervention_history]
    flags_b = [e.interaction_number for s in sessions_b for e in s.intervention_history]
    assert flags_a == flags_b


def test_different_policies_produce_different_intervention_counts():
    config, students = _students(n=10)
    rng_streams_kwargs = dict(sessions_per_student=2)

    sessions_none = generate_sessions(
        config, students, rng_streams=build_rng_streams(config.seed), **rng_streams_kwargs
    )
    sessions_no_intervention = generate_sessions(
        config, students, rng_streams=build_rng_streams(config.seed),
        intervention_policy=no_intervention_policy, **rng_streams_kwargs,
    )

    count_none = sum(len(s.intervention_history) for s in sessions_none)
    count_no_intervention = sum(len(s.intervention_history) for s in sessions_no_intervention)
    assert count_no_intervention == 0
    assert count_none > count_no_intervention
