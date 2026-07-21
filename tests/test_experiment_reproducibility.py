"""Tests for dataset_generator.experiment_reproducibility (Phase 1)."""

from __future__ import annotations

import random

import numpy as np

from dataset_generator.experiment_reproducibility import (
    build_reproducibility_report,
    collect_environment_info,
    set_global_seed,
)


def test_set_global_seed_makes_random_deterministic():
    set_global_seed(42)
    a = random.random()
    set_global_seed(42)
    b = random.random()
    assert a == b


def test_set_global_seed_makes_numpy_global_rng_deterministic():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_collect_environment_info_has_expected_keys():
    info = collect_environment_info()
    assert "timestamp_utc" in info
    assert "python_version" in info
    assert "platform" in info
    assert "package_versions" in info
    assert "numpy" in info["package_versions"]
    assert "scikit-learn" in info["package_versions"]


def test_build_reproducibility_report_includes_seed_and_config():
    report = build_reproducibility_report(seed=42, config_summary={"students": 10})
    assert report["seed"] == 42
    assert report["config"] == {"students": 10}
    assert "environment" in report
