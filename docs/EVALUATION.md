# Evaluation & Experimentation (Module 13)

`dataset_generator/evaluation/` provides the experimental infrastructure for
answering research questions about the pipeline — which features matter,
which policies matter, how sensitive results are to configuration, and how
the system scales. It introduces **no new inference logic**: every number it
produces is computed from artifacts the existing engines already emit,
invoked through the same Module 12 agents used everywhere else.

## Experiment configuration (`config.py`)

`ExperimentConfig` says *how many times, at what scale, and with which
ablations* to run the existing pipeline — never *how* BAS, reward, or
interventions are computed. It composes `BenchmarkOptions`,
`AblationOptions`, `MetricsSelection`, and `ExportOptions` sub-models, and
carries the same SHA-256 fingerprint convention every other config in this
project uses.

## Metrics (`metrics.py`)

Descriptive metrics over already-produced artifacts, reusing each
artifact's own `.statistics` wherever a value already exists:

| Group | Metrics |
|---|---|
| Dataset | class balance, profile balance, empirical transition balance, missing values, feature distributions |
| BAS | mean, variance, per-session trend, recovery rate, volatility, average confidence |
| Reward | average, positive ratio, performance/behaviour/cost decomposition, temporal consistency |
| Intervention | execution rate, policy frequencies, cooldown activations, mean spacing, policy diversity (normalized entropy) |
| Workflow | total/node timings (reused from Module 12's own report), memory/latency figures received from `benchmark.py`, replay correctness |

**A deliberate naming decision:** the headline intervention metric is
**Intervention Execution Rate** (`executed / required`), not "need detection
precision". This repository contains no externally-labeled
"should intervene" ground truth, so no precision/recall/F1/AUROC is
reported for intervention need detection — those would be classification
metrics against labels that do not exist here. Execution rate describes
planner behavior operationally. It is reported as `None` (never a
fabricated `0.0`) when the `intervention_required` denominator is zero,
which under the default `need_threshold=0.35` is the common case — a real
threshold-calibration property, surfaced rather than hidden (see the
sensitivity sweep below). If labeled data becomes available, conventional
classification metrics can be added without changing this framework.

## Benchmarks (`benchmark.py`)

`BenchmarkRunner` times each of Modules 7/9/10/11/12 independently
(wall-clock via `perf_counter`, CPU via `process_time`, peak memory via
`tracemalloc`), with configurable warmup and repetition counts. Raw
per-repetition measurements are kept alongside means so downstream
statistics can compute variance/confidence intervals. Throughput is
reported in the unit natural to each module (rows/sec, interactions/sec,
workflows/sec).

## Ablations (`ablation.py`)

`AblationRunner` produces a baseline-vs-ablated comparison table where each
row disables exactly one thing, reusing each engine's own existing ablation
mechanism:

- Reward categories → `RewardConfig.with_category_disabled` (Module 10)
- Intervention policies → `InterventionConfig.with_policy_disabled` (Module 11)
- BAS feature categories, temporal smoothing, normalization, confidence
  weighting → rebuilt `BASConfig`s using Module 9's own
  `SmoothingStrategy.IDENTITY` / `NormalizationStrategy.IDENTITY` values
  and weight fields
- Cooldown → zeroed `cooldown_length` / `duplicate_prevention_window`

## Sensitivity analysis (`sensitivity.py`)

One-at-a-time (OAT) sweeps: each varies exactly one parameter while holding
everything else — including the input dataset, for downstream parameters —
fixed. OAT cannot detect interaction effects between parameters (a
documented limitation), but every result row has an unambiguous single
cause.

Downstream sweeps (fixed dataset): `need_threshold`, `cooldown_length`,
`min_correctness`, reward-category weight multipliers. Upstream sweeps
(fixed seed, dataset regenerated per value, because the parameter's effect
*is* on generation): Module 4's overlap-noise std and the Focused→Focused
transition persistence (with row renormalization so the matrix stays
valid).

An empirical example from the `need_threshold` sweep on a small run
(5 students × 2 sessions): decisions clearing the threshold drop
45 → 4 → 0 across thresholds 0.05 / 0.15 / 0.25 — quantifying the
calibration cliff that makes the default execution rate `None`.

## Running it

```python
from dataset_generator.orchestration import ObserverAgent
from dataset_generator.evaluation import (
    AblationOptions, ExperimentConfig,
    AblationRunner, BenchmarkRunner, SensitivityRunner,
    build_comparison_table, build_sweep_table,
)

dataset = ObserverAgent().generate(student_count=8, sessions_per_student=2)

benchmarks = BenchmarkRunner().run_all(student_count=8, sessions_per_student=2)

config = ExperimentConfig(ablation_options=AblationOptions(
    disable_reward_categories=True, disable_intervention_policies=True,
))
ablations = AblationRunner(config=config).run(dataset)
for row in build_comparison_table(ablations):
    print(row)

sweeps = SensitivityRunner().run_all(dataset)
for row in build_sweep_table(sweeps.sweeps[0]):
    print(row)
```

Tests: `pytest tests/test_evaluation.py -q`.
