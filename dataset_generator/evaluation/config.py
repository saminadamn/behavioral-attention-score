"""Module 13, Step 1: Experiment Configuration.

`ExperimentConfig` says *how many times, at what scale, and with which
ablations* to run the already-complete Modules 1-12 pipeline — it never
says *how* BAS, reward, or interventions are computed. None of its fields
duplicate anything in `GeneratorConfig`/`BASConfig`/`RewardConfig`/
`InterventionConfig`; evaluation code only ever calls into those configs
and their engines, never reimplements them.

Composed of sub-models (`BenchmarkOptions`, `AblationOptions`,
`MetricsSelection`, `ExportOptions`) rather than one flat field bag, the
same way `GeneratorConfig` composes `PromptGenerationConfig` etc.
Fingerprinting reuses the exact convention every prior config in this
project uses: a SHA-256 hash of the config's own JSON, computed by a
free function, not a method the config computes about itself.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetricName = Literal[
    "dataset", "bas", "reward", "intervention", "workflow",
]

DatasetMetricName = Literal[
    "class_balance", "profile_balance", "transition_balance",
    "missing_values", "feature_distributions",
]

BASMetricName = Literal["mean", "variance", "session_trend", "recovery_rate", "volatility"]

RewardMetricName = Literal[
    "average_reward", "positive_ratio", "reward_decomposition", "temporal_consistency",
]

InterventionMetricName = Literal[
    "intervention_execution_rate", "policy_frequencies", "cooldown_activations",
    "average_intervention_spacing", "policy_diversity",
]

WorkflowMetricName = Literal[
    "execution_time", "node_timings", "memory_usage",
    "checkpoint_latency", "serialization_latency", "replay_correctness",
]

ExportFormat = Literal["markdown", "json", "csv", "latex"]


class BenchmarkOptions(BaseModel):
    """Configuration for Module 13, Step 3's benchmark runner."""

    model_config = ConfigDict(frozen=True)

    measure_runtime: bool = True
    measure_peak_memory: bool = True
    measure_cpu_time: bool = True
    warmup_runs: int = Field(ge=0, default=1)
    repetitions: int = Field(gt=0, default=3)


class AblationOptions(BaseModel):
    """Which ablations Module 13, Step 4's ablation framework should run.

    Each flag toggles a category of ablation on/off; the actual disabling
    reuses each engine's own existing ablation helper
    (`RewardConfig.with_category_disabled`,
    `InterventionConfig.with_policy_disabled`, `BASConfig`'s per-category
    weighting) rather than a new mechanism.
    """

    model_config = ConfigDict(frozen=True)

    disable_bas_feature_categories: bool = False
    disable_reward_categories: bool = False
    disable_intervention_policies: bool = False
    disable_temporal_smoothing: bool = False
    disable_cooldown: bool = False
    disable_confidence_weighting: bool = False
    disable_normalization: bool = False


class MetricsSelection(BaseModel):
    """Which metric groups (Module 13, Step 2) an experiment computes."""

    model_config = ConfigDict(frozen=True)

    dataset_metrics: tuple[DatasetMetricName, ...] = (
        "class_balance", "profile_balance", "transition_balance",
        "missing_values", "feature_distributions",
    )
    bas_metrics: tuple[BASMetricName, ...] = (
        "mean", "variance", "session_trend", "recovery_rate", "volatility",
    )
    reward_metrics: tuple[RewardMetricName, ...] = (
        "average_reward", "positive_ratio", "reward_decomposition", "temporal_consistency",
    )
    intervention_metrics: tuple[InterventionMetricName, ...] = (
        "intervention_execution_rate", "policy_frequencies", "cooldown_activations",
        "average_intervention_spacing", "policy_diversity",
    )
    workflow_metrics: tuple[WorkflowMetricName, ...] = (
        "execution_time", "node_timings", "memory_usage",
        "checkpoint_latency", "serialization_latency", "replay_correctness",
    )


class ExportOptions(BaseModel):
    """Which report formats Module 13, Step 9 should emit, and where."""

    model_config = ConfigDict(frozen=True)

    formats: tuple[ExportFormat, ...] = ("markdown", "json")
    include_plots: bool = True
    plot_formats: tuple[Literal["png", "svg", "pdf"], ...] = ("png",)


class ExperimentConfig(BaseModel):
    """Complete, deterministic configuration for one evaluation run."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = Field(min_length=1, default="default_experiment")
    random_seed: int = Field(ge=0, default=42)

    dataset_sizes: tuple[int, ...] = Field(default=(1_000, 5_000, 10_000))
    student_counts: tuple[int, ...] = Field(default=(10, 50, 100))
    sessions_per_student: int = Field(gt=0, default=2)
    num_repetitions: int = Field(gt=0, default=3)
    parallel_workers: int = Field(ge=1, default=1)

    output_directory: str = Field(min_length=1, default="evaluation_output")

    benchmark_options: BenchmarkOptions = Field(default_factory=BenchmarkOptions)
    ablation_options: AblationOptions = Field(default_factory=AblationOptions)
    metrics_to_compute: MetricsSelection = Field(default_factory=MetricsSelection)
    export_options: ExportOptions = Field(default_factory=ExportOptions)

    version: str = "1.0.0"

    @model_validator(mode="after")
    def _check_scale_lists_nonempty_and_positive(self) -> "ExperimentConfig":
        if not self.dataset_sizes:
            raise ValueError("dataset_sizes must not be empty")
        if any(size <= 0 for size in self.dataset_sizes):
            raise ValueError("dataset_sizes must all be positive")
        if not self.student_counts:
            raise ValueError("student_counts must not be empty")
        if any(count <= 0 for count in self.student_counts):
            raise ValueError("student_counts must all be positive")
        return self


def compute_experiment_config_fingerprint(config: ExperimentConfig) -> str:
    """A deterministic SHA-256 fingerprint of `config`, matching every other
    module's `compute_*_config_fingerprint` convention.
    """

    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


def default_experiment_config() -> ExperimentConfig:
    """A ready-to-use `ExperimentConfig` with sensible defaults."""

    return ExperimentConfig()
