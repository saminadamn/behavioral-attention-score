"""Tests for Module 2: Student Profile Generator."""

from __future__ import annotations

import pytest

from dataset_generator.config import AttentionState, GeneratorConfig, default_config
from dataset_generator.generators.profiles import (
    BaseProfile,
    DistractibleProfile,
    FatiguedProfile,
    FocusedProfile,
    ImpulsiveProfile,
    ProfileFactory,
    RecoveringProfile,
)
from dataset_generator.generators.student_profile_generator import generate_students
from dataset_generator.utils import build_rng_streams

EXPECTED_PROFILE_CLASSES = {
    "Consistently_Focused": FocusedProfile,
    "Gradually_Fatigued": FatiguedProfile,
    "Highly_Distractible": DistractibleProfile,
    "Highly_Impulsive": ImpulsiveProfile,
    "Recovering_Learner": RecoveringProfile,
}


# ---------------------------------------------------------------------------
# ProfileFactory
# ---------------------------------------------------------------------------


def test_profile_factory_returns_correct_class_for_each_key() -> None:
    for key, expected_cls in EXPECTED_PROFILE_CLASSES.items():
        profile = ProfileFactory.create(key)
        assert isinstance(profile, expected_cls)


def test_profile_factory_available_profiles_matches_defaults() -> None:
    config = default_config()
    assert set(ProfileFactory.available_profiles()) == set(config.profiles.keys())


def test_profile_factory_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        ProfileFactory.create("Nonexistent_Profile")


def test_profile_subclass_without_metadata_raises_at_definition() -> None:
    with pytest.raises(TypeError):

        class IncompleteProfile(BaseProfile):
            profile_key = "Some_Profile"
            # display_name / description deliberately omitted


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_students() -> None:
    config = default_config()

    streams_a = build_rng_streams(config.seed)
    students_a = generate_students(config, streams_a)

    streams_b = build_rng_streams(config.seed)
    students_b = generate_students(config, streams_b)

    assert students_a == students_b


def test_different_seeds_produce_different_students() -> None:
    config = default_config()
    data = config.model_dump(mode="json")
    data["seed"] = config.seed + 1
    other_config = GeneratorConfig.model_validate(data)

    students_a = generate_students(config, build_rng_streams(config.seed))
    students_b = generate_students(other_config, build_rng_streams(other_config.seed))

    baseline_latencies_a = [s.baseline_latency for s in students_a]
    baseline_latencies_b = [s.baseline_latency for s in students_b]
    assert baseline_latencies_a != baseline_latencies_b


def test_single_student_reproducible_independent_of_population_size() -> None:
    config = default_config()

    small = config.model_dump(mode="json")
    small["students"] = 5
    small_config = GeneratorConfig.model_validate(small)

    large = config.model_dump(mode="json")
    large["students"] = 50
    large_config = GeneratorConfig.model_validate(large)

    small_students = generate_students(small_config, build_rng_streams(small_config.seed))
    large_students = generate_students(large_config, build_rng_streams(large_config.seed))

    # Same profile_seed for index 0 regardless of total population size...
    assert small_students[0].profile_seed == large_students[0].profile_seed
    # ...and if they landed on the same archetype, identical sampled parameters.
    if small_students[0].profile_name == large_students[0].profile_name:
        assert small_students[0].baseline_latency == large_students[0].baseline_latency


# ---------------------------------------------------------------------------
# Student object contents / uniqueness
# ---------------------------------------------------------------------------


def test_generated_students_have_unique_ids() -> None:
    config = default_config()
    students = generate_students(config, build_rng_streams(config.seed))
    ids = [s.student_id for s in students]
    assert len(ids) == len(set(ids)) == config.students


def test_students_of_same_profile_are_not_identical() -> None:
    config = default_config()
    students = generate_students(config, build_rng_streams(config.seed))

    by_profile: dict[str, list] = {}
    for student in students:
        by_profile.setdefault(student.profile_name, []).append(student)

    for profile_key, group in by_profile.items():
        if len(group) < 2:
            continue
        latencies = {s.baseline_latency for s in group}
        assert len(latencies) > 1, f"all {profile_key} students share one baseline_latency"


def test_student_never_carries_session_state_fields() -> None:
    config = default_config()
    students = generate_students(config, build_rng_streams(config.seed))
    forbidden = {"bas", "attention_state", "current_latency", "rolling_engagement"}
    field_names = set(type(students[0]).model_fields.keys())
    assert forbidden.isdisjoint({f.lower() for f in field_names})


# ---------------------------------------------------------------------------
# Parameter ranges / validation
# ---------------------------------------------------------------------------


def test_generated_parameters_within_configured_bounds() -> None:
    config = default_config()
    students = generate_students(config, build_rng_streams(config.seed))

    for student in students:
        assert student.baseline_latency > 0
        assert student.latency_variance > 0
        assert 0.0 <= student.engagement_tendency <= 1.0
        assert 0.0 <= student.fatigue_rate <= 1.0
        assert 0.0 <= student.intervention_sensitivity <= 2.0


def test_every_profile_generates_1000_valid_students() -> None:
    config = default_config()
    for profile_key in ProfileFactory.available_profiles():
        profile = ProfileFactory.create(profile_key)
        for index in range(1000):
            student = profile.generate_student(index, config)
            assert student.baseline_latency > 0
            assert 0.0 <= student.fatigue_rate <= 1.0
            assert 0.0 <= student.intervention_sensitivity <= 2.0


# ---------------------------------------------------------------------------
# Transition modifiers
# ---------------------------------------------------------------------------


def test_transition_modifier_combines_with_base_to_valid_distribution() -> None:
    config = default_config()
    students = generate_students(config, build_rng_streams(config.seed))

    for student in students:
        effective = {
            state: dict(config.transition_matrix.matrix[state]) for state in AttentionState
        }
        for from_state, deltas in student.transition_modifier.items():
            for to_state, delta in deltas.items():
                effective[from_state][to_state] += delta

        for from_state, row in effective.items():
            clipped = {state: max(0.0, value) for state, value in row.items()}
            total = sum(clipped.values())
            assert total > 0, f"{student.student_id}: degenerate row from {from_state}"
            normalized = {state: value / total for state, value in clipped.items()}
            assert abs(sum(normalized.values()) - 1.0) < 1e-9
            for probability in normalized.values():
                assert 0.0 <= probability <= 1.0


def test_transition_modifier_is_sparse_not_a_full_matrix() -> None:
    config = default_config()
    focused_profile = ProfileFactory.create("Consistently_Focused")
    student = focused_profile.generate_student(0, config)
    # Consistently_Focused only overrides the Focused row (Stage 2 defaults).
    assert set(student.transition_modifier.keys()) == {AttentionState.FOCUSED}


# ---------------------------------------------------------------------------
# Descriptor export
# ---------------------------------------------------------------------------


def test_student_descriptor_export() -> None:
    config = default_config()
    students = generate_students(config, build_rng_streams(config.seed))
    descriptor = students[0].descriptor()
    assert set(descriptor.keys()) == {"student_id", "profile", "description"}
    assert descriptor["student_id"] == students[0].student_id
    assert descriptor["profile"] == students[0].profile_name
