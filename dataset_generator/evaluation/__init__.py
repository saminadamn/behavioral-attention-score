"""Module 13: Evaluation & Experimentation Framework.

Publication-oriented experimental infrastructure over the completed
Modules 1-12 pipeline: experiment configuration, descriptive metrics,
per-module benchmarks, one-at-a-time ablations, and parameter-sensitivity
sweeps. Introduces no new inference logic — every number this package
produces is computed from artifacts the existing engines already emit,
via the same Module 12 agents used everywhere else.

Entry points:
  - `ExperimentConfig` / `default_experiment_config()`: what to run, at
    what scale, with which ablations.
  - `compute_*_metrics(...)`: dataset/BAS/reward/intervention/workflow
    descriptive metrics.
  - `BenchmarkRunner(config).run_all(...)`: per-module runtime/memory/
    throughput benchmarks.
  - `AblationRunner(config).run(dataset_artifact)`: baseline-vs-ablated
    comparison tables.
  - `SensitivityRunner(config)`: one-at-a-time parameter sweeps.
"""

from dataset_generator.evaluation.ablation import (
    BAS_FEATURE_CATEGORIES,
    AblationArtifact,
    AblationRun,
    AblationRunner,
    build_comparison_table,
    disable_bas_feature_category,
    disable_bas_normalization,
    disable_confidence_weighting,
    disable_cooldown,
    disable_temporal_smoothing,
)
from dataset_generator.evaluation.benchmark import (
    BenchmarkArtifact,
    BenchmarkResult,
    BenchmarkRunner,
    ThroughputMetrics,
)
from dataset_generator.evaluation.config import (
    AblationOptions,
    BenchmarkOptions,
    ExperimentConfig,
    ExportOptions,
    MetricsSelection,
    compute_experiment_config_fingerprint,
    default_experiment_config,
)
from dataset_generator.evaluation.metrics import (
    BASMetrics,
    DatasetMetrics,
    InterventionMetrics,
    RewardMetrics,
    WorkflowMetrics,
    check_replay_correctness,
    compute_bas_metrics,
    compute_dataset_metrics,
    compute_intervention_metrics,
    compute_reward_metrics,
    compute_workflow_metrics,
)
from dataset_generator.evaluation.sensitivity import (
    SensitivityArtifact,
    SensitivityPoint,
    SensitivityRunner,
    SensitivitySweep,
    build_sweep_table,
)

__all__ = [
    "BAS_FEATURE_CATEGORIES",
    "AblationArtifact",
    "AblationOptions",
    "AblationRun",
    "AblationRunner",
    "BASMetrics",
    "BenchmarkArtifact",
    "BenchmarkOptions",
    "BenchmarkResult",
    "BenchmarkRunner",
    "DatasetMetrics",
    "ExperimentConfig",
    "ExportOptions",
    "InterventionMetrics",
    "MetricsSelection",
    "RewardMetrics",
    "SensitivityArtifact",
    "SensitivityPoint",
    "SensitivityRunner",
    "SensitivitySweep",
    "ThroughputMetrics",
    "WorkflowMetrics",
    "build_comparison_table",
    "build_sweep_table",
    "check_replay_correctness",
    "compute_bas_metrics",
    "compute_dataset_metrics",
    "compute_experiment_config_fingerprint",
    "compute_intervention_metrics",
    "compute_reward_metrics",
    "compute_workflow_metrics",
    "default_experiment_config",
    "disable_bas_feature_category",
    "disable_bas_normalization",
    "disable_confidence_weighting",
    "disable_cooldown",
    "disable_temporal_smoothing",
]
