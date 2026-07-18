"""Tests for the configuration system (Step 1)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dataset_generator.config import (
    AttentionState,
    FeatureDistributionParams,
    GeneratorConfig,
    compute_fingerprint,
    default_config,
    load_config,
    reachability_violations,
    resolve_profile_parameters,
    save_config,
)
from dataset_generator.utils import build_rng_streams


def test_default_config_is_valid() -> None:
    config = default_config()
    assert config.seed == 42
    assert config.students == 100
    assert set(config.profiles.keys()) == set(config.profile_distribution.keys())
    assert set(config.class_balance.keys()) == set(AttentionState)


def test_default_config_probabilities_sum_to_one() -> None:
    config = default_config()
    assert abs(sum(config.class_balance.values()) - 1.0) < 1e-6
    assert abs(sum(config.profile_distribution.values()) - 1.0) < 1e-6
    for row in config.transition_matrix.matrix.values():
        assert abs(sum(row.values()) - 1.0) < 1e-6


def test_class_balance_must_sum_to_one() -> None:
    data = default_config().model_dump(mode="json")
    data["class_balance"] = {"Focused": 0.9, "Distracted": 0.3, "Impulsive": 0.2}
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)


def test_profile_distribution_keys_must_match_profiles() -> None:
    data = default_config().model_dump(mode="json")
    data["profile_distribution"]["Unknown_Profile"] = 0.1
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)


def test_transition_matrix_row_must_sum_to_one() -> None:
    data = default_config().model_dump(mode="json")
    data["transition_matrix"]["matrix"]["Focused"]["Focused"] = 0.99
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)


def test_interactions_per_session_range_validated() -> None:
    data = default_config().model_dump(mode="json")
    data["interactions_per_session"] = [40, 20]
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)


def test_feature_distribution_requires_correct_params() -> None:
    with pytest.raises(ValidationError):
        FeatureDistributionParams(family="normal", params={"mean": 1.0})  # missing std
    with pytest.raises(ValidationError):
        FeatureDistributionParams(family="beta", params={"alpha": 1.0, "beta": -1.0})


def test_load_config_without_path_returns_default() -> None:
    assert load_config(None) == default_config()


def test_save_and_load_yaml_round_trip(tmp_path) -> None:
    config = default_config()
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.model_dump(mode="json") == config.model_dump(mode="json")


def test_save_and_load_json_round_trip(tmp_path) -> None:
    config = default_config()
    path = tmp_path / "config.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.model_dump(mode="json") == config.model_dump(mode="json")
    with path.open(encoding="utf-8") as f:
        json.load(f)  # confirm valid JSON


def test_load_config_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_unsupported_extension_raises(tmp_path) -> None:
    path = tmp_path / "config.txt"
    path.write_text("seed: 1", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


# ---------------------------------------------------------------------------
# Multiplier-derived profile parameters
# ---------------------------------------------------------------------------


def test_resolved_profile_params_scale_with_multipliers() -> None:
    config = default_config()
    focused_latency_mean = config.distributions.Focused.response_latency.params["mean"]

    impulsive = resolve_profile_parameters(config, "Highly_Impulsive")
    focused = resolve_profile_parameters(config, "Consistently_Focused")

    # Highly_Impulsive has latency_multiplier=0.45 -> center should sit near
    # 0.45 * focused_latency_mean, well below Consistently_Focused's center.
    impulsive_center = sum(impulsive.baseline_latency_mean_range) / 2
    focused_center = sum(focused.baseline_latency_mean_range) / 2
    assert impulsive_center < focused_center
    assert abs(impulsive_center - 0.45 * focused_latency_mean) < 1e-6

    assert 0.0 <= impulsive.engagement_tendency <= 1.0
    assert 0.0 <= focused.engagement_tendency <= 1.0


def test_resolve_profile_parameters_retunes_with_focused_distribution() -> None:
    data = default_config().model_dump(mode="json")
    data["distributions"]["Focused"]["response_latency"]["params"]["mean"] = 13.0
    retuned = GeneratorConfig.model_validate(data)

    resolved = resolve_profile_parameters(retuned, "Highly_Impulsive")
    center = sum(resolved.baseline_latency_mean_range) / 2
    assert abs(center - 0.45 * 13.0) < 1e-6


def test_resolve_profile_parameters_unknown_profile_raises() -> None:
    config = default_config()
    with pytest.raises(KeyError):
        resolve_profile_parameters(config, "Nonexistent_Profile")


# ---------------------------------------------------------------------------
# Transition reachability validator
# ---------------------------------------------------------------------------


def test_reachability_violations_empty_for_default_matrix() -> None:
    config = default_config()
    assert reachability_violations(config.transition_matrix.matrix) == []


def test_base_transition_matrix_rejects_unreachable_state() -> None:
    data = default_config().model_dump(mode="json")
    data["transition_matrix"]["matrix"] = {
        "Focused": {"Focused": 1.0, "Distracted": 0.0, "Impulsive": 0.0},
        "Distracted": {"Focused": 0.0, "Distracted": 1.0, "Impulsive": 0.0},
        "Impulsive": {"Focused": 1.0, "Distracted": 0.0, "Impulsive": 0.0},
    }
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)


def test_profile_transition_modifiers_cannot_isolate_a_state() -> None:
    data = default_config().model_dump(mode="json")
    # Cancel out every route into/out of Distracted for one profile.
    data["profiles"]["Consistently_Focused"]["transition_modifiers"] = {
        "Focused": {"Focused": 0.15, "Distracted": -0.15, "Impulsive": 0.0},
        "Distracted": {"Focused": 0.30, "Distracted": 0.0, "Impulsive": -0.30},
        "Impulsive": {"Focused": 0.25, "Distracted": -0.35, "Impulsive": 0.10},
    }
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Configuration fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic() -> None:
    config = default_config()
    assert compute_fingerprint(config) == compute_fingerprint(default_config())


def test_fingerprint_changes_with_generation_parameters() -> None:
    config = default_config()
    data = config.model_dump(mode="json")
    data["seed"] = 999
    other = GeneratorConfig.model_validate(data)
    assert compute_fingerprint(config) != compute_fingerprint(other)


def test_fingerprint_ignores_experiment_metadata() -> None:
    config = default_config()
    data = config.model_dump(mode="json")
    data["experiment"] = {
        "experiment_name": "pilot_run",
        "experiment_description": "testing",
        "author": "someone",
        "date_created": "2026-07-18T00:00:00Z",
        "git_commit": "abc123",
        "notes": "irrelevant to output",
    }
    annotated = GeneratorConfig.model_validate(data)
    assert compute_fingerprint(config) == compute_fingerprint(annotated)


# ---------------------------------------------------------------------------
# RNG stream separation
# ---------------------------------------------------------------------------


def test_rng_streams_reproducible_for_same_seed() -> None:
    a = build_rng_streams(42)
    b = build_rng_streams(42)
    assert list(a.student_rng.random(5)) == list(b.student_rng.random(5))
    assert list(a.noise_rng.integers(0, 1000, 5)) == list(b.noise_rng.integers(0, 1000, 5))


def test_rng_streams_are_mutually_independent() -> None:
    streams = build_rng_streams(42)
    draws = {
        name: list(getattr(streams, f"{name}_rng").random(10))
        for name in ("student", "session", "prompt", "response", "noise")
    }
    values = list(draws.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            assert values[i] != values[j]


def test_rng_stream_draws_do_not_affect_other_streams() -> None:
    streams_a = build_rng_streams(7)
    streams_b = build_rng_streams(7)

    # Exhaust one stream on `streams_a` only.
    streams_a.response_rng.random(1000)

    # Every other stream must still match `streams_b` exactly.
    assert list(streams_a.student_rng.random(5)) == list(streams_b.student_rng.random(5))
    assert list(streams_a.session_rng.random(5)) == list(streams_b.session_rng.random(5))
