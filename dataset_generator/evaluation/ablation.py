"""Module 13, Step 4: Ablation Framework.

Every ablation here is produced by reusing an existing engine's own
config-copy mechanism — never a new mechanism, and never new BAS/reward/
intervention logic:

- Reward category ablation reuses `RewardConfig.with_category_disabled`
  (Module 10) verbatim.
- Intervention policy ablation reuses `InterventionConfig.with_policy_disabled`
  (Module 11) verbatim.
- BAS feature-category ablation, temporal-smoothing ablation, and
  normalization ablation reuse `BASConfig`'s *existing* fields and enum
  values (`SmoothingStrategy.IDENTITY`, `NormalizationStrategy.IDENTITY`
  already mean "no smoothing"/"no transformation" in Module 9's own
  design) — this module only decides *which* already-legal `BASConfig`
  to construct, via the real constructor, exactly as `with_category_disabled`
  does for `RewardConfig`. Modules 1-12 are not modified: the BAS
  feature-category taxonomy is Module 13's own concern (deciding what an
  ablation study groups together), not Module 9's.
- Cooldown ablation sets `InterventionConfig.cooldown_length` and
  `duplicate_prevention_window` to 0 — both already-existing, already
  independently-configurable fields.
- Confidence-weighting ablation sets `BASConfig.confidence_variance_weight`/
  `confidence_classifier_weight` to 0 — same pattern.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dataset_generator.bas.config import BASConfig, NormalizationStrategy, SmoothingStrategy, default_bas_config
from dataset_generator.bas.scorer import BASEngine, compute_bas_config_fingerprint
from dataset_generator.intervention.config import InterventionConfig, default_intervention_config
from dataset_generator.intervention.planner import compute_intervention_config_fingerprint
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.orchestration.agents import BASAgent, InterventionAgent, RewardAgent
from dataset_generator.reward.aggregator import compute_reward_config_fingerprint
from dataset_generator.reward.config import RewardCategory, RewardConfig, default_reward_config

from dataset_generator.evaluation.config import (
    AblationOptions,
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

BASFeatureCategoryName = Literal["learning_response", "behaviour"]

BAS_FEATURE_CATEGORIES: dict[BASFeatureCategoryName, tuple[str, ...]] = {
    "learning_response": ("correctness", "semantic_similarity", "coherence", "lexical_diversity", "confidence"),
    "behaviour": ("hesitation", "topic_shift", "repetition_ratio", "fatigue", "abs_normalized_latency"),
}


class AblationRun(BaseModel):
    """One configuration variant's metrics — the baseline uses
    `ablation_name="baseline"`; every other row zero-weights exactly one
    category/policy/mechanism relative to the baseline, everything else
    held fixed.
    """

    model_config = ConfigDict(frozen=True)

    ablation_name: str
    description: str
    config_fingerprint: str
    bas_metrics: BASMetrics | None = None
    reward_metrics: RewardMetrics | None = None
    intervention_metrics: InterventionMetrics | None = None


class AblationArtifact(BaseModel):
    """The single source of truth for one ablation study's comparison table."""

    model_config = ConfigDict(frozen=True)

    runs: list[AblationRun]
    config_fingerprint: str
    schema_version: str
    generation_timestamp: str


# ---------------------------------------------------------------------------
# BAS config ablations
# ---------------------------------------------------------------------------


def disable_bas_feature_category(config: BASConfig, category: BASFeatureCategoryName) -> BASConfig:
    """A new `BASConfig` with every feature in `category` zero-weighted."""

    feature_names = BAS_FEATURE_CATEGORIES[category]
    updated_features = dict(config.feature_configs)
    for name in feature_names:
        if name in updated_features:
            updated_features[name] = updated_features[name].model_copy(update={"weight": 0.0})
    return BASConfig(**{**config.model_dump(), "feature_configs": updated_features})


def disable_temporal_smoothing(config: BASConfig) -> BASConfig:
    """A new `BASConfig` with smoothing set to `SmoothingStrategy.IDENTITY`
    (Module 9's own "no smoothing" value)."""

    return BASConfig(**{**config.model_dump(), "smoothing_strategy": SmoothingStrategy.IDENTITY})


def disable_bas_normalization(config: BASConfig) -> BASConfig:
    """A new `BASConfig` with every feature's normalization set to
    `NormalizationStrategy.IDENTITY` (Module 9's own "no transformation" value)."""

    updated_features = {
        name: cfg.model_copy(update={
            "normalization": cfg.normalization.model_copy(update={"strategy": NormalizationStrategy.IDENTITY})
        })
        for name, cfg in config.feature_configs.items()
    }
    return BASConfig(**{**config.model_dump(), "feature_configs": updated_features})


def disable_confidence_weighting(config: BASConfig) -> BASConfig:
    """A new `BASConfig` with both confidence-related weights zeroed."""

    return BASConfig(**{
        **config.model_dump(),
        "confidence_variance_weight": 0.0,
        "confidence_classifier_weight": 0.0,
    })


# ---------------------------------------------------------------------------
# Intervention config ablations
# ---------------------------------------------------------------------------


def disable_cooldown(config: InterventionConfig) -> InterventionConfig:
    """A new `InterventionConfig` with cooldown spacing and duplicate-prevention disabled."""

    return InterventionConfig(**{
        **config.model_dump(),
        "cooldown_length": 0,
        "duplicate_prevention_window": 0,
    })


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


class AblationRunner:
    """Runs a baseline plus every ablation requested by `AblationOptions`,
    rebuilding only the affected engine (BAS/reward/intervention) each time
    and holding everything else — including the input `DatasetArtifact` —
    fixed, so every row in the resulting table differs from the baseline
    in exactly one respect.
    """

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config or default_experiment_config()

    def _options(self) -> AblationOptions:
        return self._config.ablation_options

    def run(
        self,
        dataset_artifact: DatasetArtifact,
        base_bas_config: BASConfig | None = None,
        base_reward_config: RewardConfig | None = None,
        base_intervention_config: InterventionConfig | None = None,
    ) -> AblationArtifact:
        base_bas_config = base_bas_config or default_bas_config()
        base_reward_config = base_reward_config or default_reward_config()
        base_intervention_config = base_intervention_config or default_intervention_config()
        options = self._options()

        runs: list[AblationRun] = []

        baseline_bas_artifact = BASAgent(engine=BASEngine(config=base_bas_config)).compute(dataset_artifact)
        baseline_reward_artifact = RewardAgent(config=base_reward_config).compute(
            dataset_artifact, baseline_bas_artifact
        )
        baseline_intervention_artifact = InterventionAgent(config=base_intervention_config).plan(
            dataset_artifact, baseline_bas_artifact, baseline_reward_artifact
        )
        runs.append(AblationRun(
            ablation_name="baseline",
            description="Unmodified default configuration for BAS, reward, and intervention.",
            config_fingerprint=compute_bas_config_fingerprint(base_bas_config),
            bas_metrics=compute_bas_metrics(baseline_bas_artifact),
            reward_metrics=compute_reward_metrics(baseline_reward_artifact),
            intervention_metrics=compute_intervention_metrics(baseline_intervention_artifact),
        ))

        if options.disable_bas_feature_categories:
            for category in BAS_FEATURE_CATEGORIES:
                runs.append(self._run_bas_ablation(
                    dataset_artifact, disable_bas_feature_category(base_bas_config, category),
                    base_reward_config, base_intervention_config,
                    ablation_name=f"bas_feature_category:{category}",
                    description=f"BAS features in category '{category}' zero-weighted.",
                ))

        if options.disable_temporal_smoothing:
            runs.append(self._run_bas_ablation(
                dataset_artifact, disable_temporal_smoothing(base_bas_config),
                base_reward_config, base_intervention_config,
                ablation_name="bas_temporal_smoothing",
                description="BAS temporal smoothing disabled (SmoothingStrategy.IDENTITY).",
            ))

        if options.disable_normalization:
            runs.append(self._run_bas_ablation(
                dataset_artifact, disable_bas_normalization(base_bas_config),
                base_reward_config, base_intervention_config,
                ablation_name="bas_normalization",
                description="BAS feature normalization disabled (NormalizationStrategy.IDENTITY for all features).",
            ))

        if options.disable_confidence_weighting:
            runs.append(self._run_bas_ablation(
                dataset_artifact, disable_confidence_weighting(base_bas_config),
                base_reward_config, base_intervention_config,
                ablation_name="bas_confidence_weighting",
                description="BAS confidence-derived weighting (variance + classifier) zeroed.",
            ))

        if options.disable_reward_categories:
            for reward_category in RewardCategory:
                ablated_reward_config = base_reward_config.with_category_disabled(reward_category)
                runs.append(self._run_reward_ablation(
                    dataset_artifact, baseline_bas_artifact, ablated_reward_config, base_intervention_config,
                    ablation_name=f"reward_category:{reward_category.value}",
                    description=f"Reward category '{reward_category.value}' zero-weighted.",
                ))

        if options.disable_intervention_policies:
            for policy_name in self._registered_policy_names():
                if policy_name == "NoInterventionPolicy":
                    continue
                ablated_intervention_config = base_intervention_config.with_policy_disabled(policy_name)
                runs.append(self._run_intervention_ablation(
                    dataset_artifact, baseline_bas_artifact, baseline_reward_artifact, ablated_intervention_config,
                    ablation_name=f"intervention_policy:{policy_name}",
                    description=f"Intervention policy '{policy_name}' zero-weighted.",
                ))

        if options.disable_cooldown:
            runs.append(self._run_intervention_ablation(
                dataset_artifact, baseline_bas_artifact, baseline_reward_artifact,
                disable_cooldown(base_intervention_config),
                ablation_name="intervention_cooldown",
                description="Intervention cooldown spacing and duplicate-prevention disabled.",
            ))

        return AblationArtifact(
            runs=runs,
            config_fingerprint=compute_experiment_config_fingerprint(self._config),
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _run_bas_ablation(
        self, dataset_artifact, bas_config, reward_config, intervention_config, *, ablation_name, description
    ) -> AblationRun:
        bas_artifact = BASAgent(engine=BASEngine(config=bas_config)).compute(dataset_artifact)
        reward_artifact = RewardAgent(config=reward_config).compute(dataset_artifact, bas_artifact)
        intervention_artifact = InterventionAgent(config=intervention_config).plan(
            dataset_artifact, bas_artifact, reward_artifact
        )
        return AblationRun(
            ablation_name=ablation_name,
            description=description,
            config_fingerprint=compute_bas_config_fingerprint(bas_config),
            bas_metrics=compute_bas_metrics(bas_artifact),
            reward_metrics=compute_reward_metrics(reward_artifact),
            intervention_metrics=compute_intervention_metrics(intervention_artifact),
        )

    def _run_reward_ablation(
        self, dataset_artifact, bas_artifact, reward_config, intervention_config, *, ablation_name, description
    ) -> AblationRun:
        reward_artifact = RewardAgent(config=reward_config).compute(dataset_artifact, bas_artifact)
        intervention_artifact = InterventionAgent(config=intervention_config).plan(
            dataset_artifact, bas_artifact, reward_artifact
        )
        return AblationRun(
            ablation_name=ablation_name,
            description=description,
            config_fingerprint=compute_reward_config_fingerprint(reward_config),
            reward_metrics=compute_reward_metrics(reward_artifact),
            intervention_metrics=compute_intervention_metrics(intervention_artifact),
        )

    def _run_intervention_ablation(
        self, dataset_artifact, bas_artifact, reward_artifact, intervention_config, *, ablation_name, description
    ) -> AblationRun:
        intervention_artifact = InterventionAgent(config=intervention_config).plan(
            dataset_artifact, bas_artifact, reward_artifact
        )
        return AblationRun(
            ablation_name=ablation_name,
            description=description,
            config_fingerprint=compute_intervention_config_fingerprint(intervention_config),
            intervention_metrics=compute_intervention_metrics(intervention_artifact),
        )

    @staticmethod
    def _registered_policy_names() -> list[str]:
        from dataset_generator.intervention import InterventionPolicyFactory

        return InterventionPolicyFactory.names()


def build_comparison_table(artifact: AblationArtifact) -> list[dict[str, object]]:
    """Flatten `artifact.runs` into rows suitable for a Markdown/CSV table:
    one row per ablation, with `None` metrics simply absent from that row
    rather than rendered as a fabricated zero.
    """

    rows: list[dict[str, object]] = []
    for run in artifact.runs:
        row: dict[str, object] = {"ablation": run.ablation_name, "description": run.description}
        if run.bas_metrics is not None:
            row["bas_mean"] = run.bas_metrics.mean
            row["bas_volatility"] = run.bas_metrics.volatility
            row["bas_average_confidence"] = run.bas_metrics.average_confidence
        if run.reward_metrics is not None:
            row["reward_average"] = run.reward_metrics.average_reward
            row["reward_positive_ratio"] = run.reward_metrics.positive_ratio
        if run.intervention_metrics is not None:
            row["intervention_execution_rate"] = run.intervention_metrics.intervention_execution_rate
            row["cooldown_activations"] = run.intervention_metrics.cooldown_activations
        rows.append(row)
    return rows
