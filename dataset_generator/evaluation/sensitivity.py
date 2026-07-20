"""Module 13, Step 5: Sensitivity Analysis.

One-at-a-time (OAT) parameter sweeps: each sweep varies exactly one
parameter across a configured list of values while holding every other
parameter — and the input `DatasetArtifact` — fixed, then records the
same metric set the ablation framework uses. OAT is the honest baseline
for sensitivity analysis: it cannot detect interaction effects between
parameters (a documented limitation, not an oversight), but every result
row has an unambiguous single cause.

Like `ablation.py`, every varied configuration is built through the real
config constructor (never `model_copy`, which skips validators), and the
engines are invoked through the same Module 12 agents used everywhere
else — no new inference logic.

Two kinds of parameters are swept:

- Downstream parameters (reward weights, need/policy thresholds, cooldown)
  re-score the SAME fixed dataset, so differences are attributable purely
  to the parameter.
- Upstream generation parameters (`noise` and transition-matrix diagonal
  scaling on `GeneratorConfig`) necessarily regenerate the dataset per
  value — the parameter's effect IS on generation. These sweeps therefore
  hold the seed fixed and vary only the one generation parameter.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from dataset_generator.config import AttentionState, GeneratorConfig, default_config
from dataset_generator.intervention.config import InterventionConfig, default_intervention_config
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.orchestration.agents import BASAgent, InterventionAgent, ObserverAgent, RewardAgent
from dataset_generator.reward.config import RewardCategory, RewardConfig, default_reward_config

from dataset_generator.evaluation.config import (
    ExperimentConfig,
    compute_experiment_config_fingerprint,
    default_experiment_config,
)
from dataset_generator.evaluation.metrics import (
    BASMetrics,
    InterventionMetrics,
    RewardMetrics,
    compute_bas_metrics,
    compute_intervention_metrics,
    compute_reward_metrics,
)

SCHEMA_VERSION = "1.0"


class SensitivityPoint(BaseModel):
    """One (parameter, value) evaluation — a single point on a sweep curve."""

    model_config = ConfigDict(frozen=True)

    parameter_name: str
    parameter_value: float
    bas_metrics: BASMetrics | None = None
    reward_metrics: RewardMetrics | None = None
    intervention_metrics: InterventionMetrics | None = None


class SensitivitySweep(BaseModel):
    """All points for one parameter, in ascending parameter-value order."""

    model_config = ConfigDict(frozen=True)

    parameter_name: str
    description: str
    points: list[SensitivityPoint]


class SensitivityArtifact(BaseModel):
    """The single source of truth for one sensitivity analysis."""

    model_config = ConfigDict(frozen=True)

    sweeps: list[SensitivitySweep]
    config_fingerprint: str
    schema_version: str
    generation_timestamp: str


class SensitivityRunner:
    """Runs one-at-a-time parameter sweeps against a fixed dataset (for
    downstream parameters) or a fixed seed (for generation parameters).
    """

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config or default_experiment_config()

    # -- downstream sweeps (fixed dataset) --------------------------------

    def sweep_need_threshold(
        self, dataset_artifact: DatasetArtifact, values: tuple[float, ...] = (0.05, 0.15, 0.25, 0.35, 0.45),
    ) -> SensitivitySweep:
        """Vary `InterventionConfig.need_threshold` — directly probes the
        threshold-calibration question surfaced by `metrics.py`'s
        `intervention_execution_rate` (whose denominator is exactly the
        decisions this threshold gates).
        """

        bas_artifact = BASAgent().compute(dataset_artifact)
        reward_artifact = RewardAgent().compute(dataset_artifact, bas_artifact)

        points = []
        for value in values:
            cfg = InterventionConfig(**{**default_intervention_config().model_dump(), "need_threshold": value})
            intervention_artifact = InterventionAgent(config=cfg).plan(dataset_artifact, bas_artifact, reward_artifact)
            points.append(SensitivityPoint(
                parameter_name="need_threshold", parameter_value=value,
                intervention_metrics=compute_intervention_metrics(intervention_artifact),
            ))
        return SensitivitySweep(
            parameter_name="need_threshold",
            description="InterventionConfig.need_threshold swept; fixed dataset/BAS/reward.",
            points=points,
        )

    def sweep_cooldown_length(
        self, dataset_artifact: DatasetArtifact, values: tuple[int, ...] = (0, 1, 3, 5, 8),
    ) -> SensitivitySweep:
        """Vary `InterventionConfig.cooldown_length`."""

        bas_artifact = BASAgent().compute(dataset_artifact)
        reward_artifact = RewardAgent().compute(dataset_artifact, bas_artifact)

        points = []
        for value in values:
            cfg = InterventionConfig(**{**default_intervention_config().model_dump(), "cooldown_length": value})
            intervention_artifact = InterventionAgent(config=cfg).plan(dataset_artifact, bas_artifact, reward_artifact)
            points.append(SensitivityPoint(
                parameter_name="cooldown_length", parameter_value=float(value),
                intervention_metrics=compute_intervention_metrics(intervention_artifact),
            ))
        return SensitivitySweep(
            parameter_name="cooldown_length",
            description="InterventionConfig.cooldown_length swept; fixed dataset/BAS/reward.",
            points=points,
        )

    def sweep_policy_threshold_min_correctness(
        self, dataset_artifact: DatasetArtifact, values: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    ) -> SensitivitySweep:
        """Vary `InterventionConfig.min_correctness` — the eligibility gate
        shared by Hint/ConceptReview/DifficultyReduction (low side) and
        MotivationalPrompt/Encouragement (high side).
        """

        bas_artifact = BASAgent().compute(dataset_artifact)
        reward_artifact = RewardAgent().compute(dataset_artifact, bas_artifact)

        points = []
        for value in values:
            cfg = InterventionConfig(**{**default_intervention_config().model_dump(), "min_correctness": value})
            intervention_artifact = InterventionAgent(config=cfg).plan(dataset_artifact, bas_artifact, reward_artifact)
            points.append(SensitivityPoint(
                parameter_name="min_correctness", parameter_value=value,
                intervention_metrics=compute_intervention_metrics(intervention_artifact),
            ))
        return SensitivitySweep(
            parameter_name="min_correctness",
            description="InterventionConfig.min_correctness (policy eligibility threshold) swept.",
            points=points,
        )

    def sweep_reward_category_weight(
        self,
        dataset_artifact: DatasetArtifact,
        category: RewardCategory = RewardCategory.BEHAVIOUR,
        multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
    ) -> SensitivitySweep:
        """Scale every signal weight in one reward category by each multiplier.

        Generalizes `RewardConfig.with_category_disabled` (multiplier 0.0 is
        exactly that ablation) to a graded sweep, using the same
        real-constructor rebuild.
        """

        bas_artifact = BASAgent().compute(dataset_artifact)
        base = default_reward_config()

        points = []
        for multiplier in multipliers:
            updated_signals = {
                name: (
                    cfg.model_copy(update={"weight": cfg.weight * multiplier})
                    if cfg.category == category else cfg
                )
                for name, cfg in base.signal_configs.items()
            }
            cfg = RewardConfig(**{**base.model_dump(), "signal_configs": updated_signals})
            reward_artifact = RewardAgent(config=cfg).compute(dataset_artifact, bas_artifact)
            points.append(SensitivityPoint(
                parameter_name=f"reward_weight_multiplier:{category.value}", parameter_value=multiplier,
                reward_metrics=compute_reward_metrics(reward_artifact),
            ))
        return SensitivitySweep(
            parameter_name=f"reward_weight_multiplier:{category.value}",
            description=f"All '{category.value}'-category reward signal weights scaled by each multiplier.",
            points=points,
        )

    # -- upstream sweeps (fixed seed, dataset regenerated per value) -------

    def sweep_generation_noise(
        self, values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2), student_count: int = 8, sessions_per_student: int = 2,
    ) -> SensitivitySweep:
        """Vary the Module 4 overlap-noise standard deviation
        (`ResponseGenerationConfig.semantic_similarity_noise_std`, applied
        uniformly to all three attention states). This is the noise
        parameter that genuinely shapes generation — it controls how much
        the per-attention-state semantic-similarity distributions overlap,
        the exact knob introduced to stop a downstream classifier from
        learning a trivial constant threshold. The dataset is regenerated
        per value (the parameter's effect IS on generation), seed held fixed.
        """

        points = []
        for value in values:
            base = default_config()
            updated_response_generation = base.response_generation.model_copy(
                update={"semantic_similarity_noise_std": {state: value for state in AttentionState}}
            )
            generator_config = GeneratorConfig(**{
                **base.model_dump(),
                "response_generation": updated_response_generation.model_dump(),
            })
            dataset_artifact = ObserverAgent(config=generator_config).generate(
                student_count=student_count, sessions_per_student=sessions_per_student
            )
            bas_artifact = BASAgent().compute(dataset_artifact)
            points.append(SensitivityPoint(
                parameter_name="semantic_similarity_noise_std", parameter_value=value,
                bas_metrics=compute_bas_metrics(bas_artifact),
            ))
        return SensitivitySweep(
            parameter_name="semantic_similarity_noise_std",
            description="Module 4 overlap-noise std swept uniformly across attention states; dataset regenerated per value with fixed seed.",
            points=points,
        )

    def sweep_transition_focus_persistence(
        self,
        values: tuple[float, ...] = (0.55, 0.65, 0.75, 0.85),
        student_count: int = 8,
        sessions_per_student: int = 2,
    ) -> SensitivitySweep:
        """Vary the Focused->Focused self-transition probability, spreading
        the removed/added mass proportionally over Focused's other two
        transitions so the row still sums to 1 (a requirement
        `TransitionMatrixConfig`'s own validator enforces).
        """

        points = []
        for value in values:
            base = default_config()
            matrix = {
                from_state: dict(row) for from_state, row in base.transition_matrix.matrix.items()
            }
            focused_row = matrix[AttentionState.FOCUSED]
            old_self = focused_row[AttentionState.FOCUSED]
            other_total = 1.0 - old_self
            new_other_total = 1.0 - value
            for to_state in focused_row:
                if to_state == AttentionState.FOCUSED:
                    focused_row[to_state] = value
                else:
                    focused_row[to_state] = (
                        focused_row[to_state] / other_total * new_other_total if other_total > 0 else 0.0
                    )
            generator_config = GeneratorConfig(**{
                **base.model_dump(),
                "transition_matrix": {**base.transition_matrix.model_dump(), "matrix": matrix},
            })
            dataset_artifact = ObserverAgent(config=generator_config).generate(
                student_count=student_count, sessions_per_student=sessions_per_student
            )
            bas_artifact = BASAgent().compute(dataset_artifact)
            points.append(SensitivityPoint(
                parameter_name="transition_focused_persistence", parameter_value=value,
                bas_metrics=compute_bas_metrics(bas_artifact),
            ))
        return SensitivitySweep(
            parameter_name="transition_focused_persistence",
            description="Focused->Focused base transition probability swept (row renormalized); dataset regenerated per value.",
            points=points,
        )

    # -- entry point --------------------------------------------------------

    def run_all(self, dataset_artifact: DatasetArtifact) -> SensitivityArtifact:
        """Run every downstream sweep against `dataset_artifact` plus both
        upstream generation sweeps, and bundle the results.
        """

        sweeps = [
            self.sweep_need_threshold(dataset_artifact),
            self.sweep_cooldown_length(dataset_artifact),
            self.sweep_policy_threshold_min_correctness(dataset_artifact),
            self.sweep_reward_category_weight(dataset_artifact),
            self.sweep_generation_noise(),
            self.sweep_transition_focus_persistence(),
        ]
        return SensitivityArtifact(
            sweeps=sweeps,
            config_fingerprint=compute_experiment_config_fingerprint(self._config),
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )


def build_sweep_table(sweep: SensitivitySweep) -> list[dict[str, object]]:
    """Flatten one sweep into rows for a Markdown/CSV table — absent metrics
    are simply omitted from a row, never rendered as fabricated zeros.
    """

    rows: list[dict[str, object]] = []
    for point in sweep.points:
        row: dict[str, object] = {"parameter": point.parameter_name, "value": point.parameter_value}
        if point.bas_metrics is not None:
            row["bas_mean"] = point.bas_metrics.mean
            row["bas_volatility"] = point.bas_metrics.volatility
        if point.reward_metrics is not None:
            row["reward_average"] = point.reward_metrics.average_reward
            row["reward_positive_ratio"] = point.reward_metrics.positive_ratio
        if point.intervention_metrics is not None:
            row["intervention_required_count"] = point.intervention_metrics.intervention_required_count
            row["executed_intervention_count"] = point.intervention_metrics.executed_intervention_count
            row["intervention_execution_rate"] = point.intervention_metrics.intervention_execution_rate
            row["cooldown_activations"] = point.intervention_metrics.cooldown_activations
        rows.append(row)
    return rows
