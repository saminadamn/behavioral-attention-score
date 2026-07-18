"""Default configuration matching the Stage 2 dataset specification.

`default_config()` returns a fully populated, validated `GeneratorConfig`.
It is the single reference implementation of "what Stage 2 specified" —
every value here traces back to a specific Stage 2 section, noted in
comments. Treat this module as data, not logic: no other module should
duplicate these numbers.

Student profiles are expressed as *multipliers* relative to the Focused
distribution and `BaseRates` (see `StudentProfileConfig`/`derive.py`), not
as hardcoded absolute ranges — retuning the Focused distribution keeps every
profile proportionally consistent automatically.
"""

from __future__ import annotations

from dataset_generator import __version__ as _generator_version
from dataset_generator.config.schema import (
    AttentionState,
    BaseRates,
    DistributionConfig,
    ExperimentMetadata,
    FeatureDistributionParams,
    GeneratorConfig,
    OutputConfig,
    ProfileMultipliers,
    StateDistributionConfig,
    StudentProfileConfig,
    TransitionMatrixConfig,
    VersionMetadata,
)

# ---------------------------------------------------------------------------
# Per-state feature distributions (Stage 2, Section 5)
# ---------------------------------------------------------------------------


def _focused_distribution() -> StateDistributionConfig:
    return StateDistributionConfig(
        response_latency=FeatureDistributionParams(
            family="normal", params={"mean": 6.5, "std": 1.5}, clip_min=0.5
        ),
        response_length=FeatureDistributionParams(
            family="gamma", params={"shape": 6.0, "scale": 3.0}
        ),
        topic_similarity=FeatureDistributionParams(
            family="beta", params={"alpha": 8.0, "beta": 2.0}
        ),
        sentiment=FeatureDistributionParams(
            family="normal", params={"mean": 0.1, "std": 0.2}, clip_min=-1.0, clip_max=1.0
        ),
        engagement=FeatureDistributionParams(family="beta", params={"alpha": 7.0, "beta": 2.0}),
        lexical_diversity=FeatureDistributionParams(
            family="beta", params={"alpha": 6.0, "beta": 3.0}
        ),
        topic_shift=FeatureDistributionParams(family="beta", params={"alpha": 2.0, "beta": 6.0}),
        hesitation=FeatureDistributionParams(family="poisson", params={"lam": 0.5}),
        repetition_ratio=FeatureDistributionParams(
            family="beta", params={"alpha": 2.0, "beta": 8.0}
        ),
        interaction_duration=FeatureDistributionParams(
            family="truncated_normal", params={"mean": 45.0, "std": 10.0}, clip_min=10.0, clip_max=120.0
        ),
    )


def _distracted_distribution() -> StateDistributionConfig:
    return StateDistributionConfig(
        response_latency=FeatureDistributionParams(
            family="normal", params={"mean": 11.0, "std": 3.5}, clip_min=0.5
        ),
        response_length=FeatureDistributionParams(
            family="gamma", params={"shape": 2.0, "scale": 2.5}
        ),
        topic_similarity=FeatureDistributionParams(
            family="beta", params={"alpha": 3.0, "beta": 4.0}
        ),
        sentiment=FeatureDistributionParams(
            family="normal", params={"mean": -0.05, "std": 0.2}, clip_min=-1.0, clip_max=1.0
        ),
        engagement=FeatureDistributionParams(family="beta", params={"alpha": 2.0, "beta": 5.0}),
        lexical_diversity=FeatureDistributionParams(
            family="beta", params={"alpha": 3.0, "beta": 5.0}
        ),
        topic_shift=FeatureDistributionParams(family="beta", params={"alpha": 4.0, "beta": 4.0}),
        hesitation=FeatureDistributionParams(family="poisson", params={"lam": 1.8}),
        repetition_ratio=FeatureDistributionParams(
            family="beta", params={"alpha": 3.0, "beta": 6.0}
        ),
        interaction_duration=FeatureDistributionParams(
            family="truncated_normal", params={"mean": 60.0, "std": 20.0}, clip_min=10.0, clip_max=150.0
        ),
    )


def _impulsive_distribution() -> StateDistributionConfig:
    return StateDistributionConfig(
        response_latency=FeatureDistributionParams(
            family="normal", params={"mean": 2.5, "std": 1.0}, clip_min=0.5
        ),
        response_length=FeatureDistributionParams(
            family="gamma", params={"shape": 2.0, "scale": 2.0}
        ),
        topic_similarity=FeatureDistributionParams(
            family="beta", params={"alpha": 3.0, "beta": 5.0}
        ),
        sentiment=FeatureDistributionParams(
            family="normal", params={"mean": 0.0, "std": 0.4}, clip_min=-1.0, clip_max=1.0
        ),
        engagement=FeatureDistributionParams(family="beta", params={"alpha": 3.0, "beta": 4.0}),
        lexical_diversity=FeatureDistributionParams(
            family="beta", params={"alpha": 3.0, "beta": 6.0}
        ),
        topic_shift=FeatureDistributionParams(family="beta", params={"alpha": 4.0, "beta": 3.0}),
        hesitation=FeatureDistributionParams(family="poisson", params={"lam": 0.2}),
        repetition_ratio=FeatureDistributionParams(
            family="beta", params={"alpha": 4.0, "beta": 5.0}
        ),
        interaction_duration=FeatureDistributionParams(
            family="truncated_normal", params={"mean": 20.0, "std": 8.0}, clip_min=5.0, clip_max=60.0
        ),
    )


def _default_distributions() -> DistributionConfig:
    return DistributionConfig(
        Focused=_focused_distribution(),
        Distracted=_distracted_distribution(),
        Impulsive=_impulsive_distribution(),
    )


# ---------------------------------------------------------------------------
# Base transition matrix (Stage 2, Section 6.1)
# ---------------------------------------------------------------------------


def _default_transition_matrix() -> TransitionMatrixConfig:
    return TransitionMatrixConfig(
        matrix={
            AttentionState.FOCUSED: {
                AttentionState.FOCUSED: 0.75,
                AttentionState.DISTRACTED: 0.15,
                AttentionState.IMPULSIVE: 0.10,
            },
            AttentionState.DISTRACTED: {
                AttentionState.FOCUSED: 0.30,
                AttentionState.DISTRACTED: 0.55,
                AttentionState.IMPULSIVE: 0.15,
            },
            AttentionState.IMPULSIVE: {
                AttentionState.FOCUSED: 0.25,
                AttentionState.DISTRACTED: 0.35,
                AttentionState.IMPULSIVE: 0.40,
            },
        }
    )


# ---------------------------------------------------------------------------
# Base rates that profile multipliers scale (Stage 2, Section 3)
# ---------------------------------------------------------------------------


def _default_base_rates() -> BaseRates:
    return BaseRates(base_fatigue_rate=0.08, base_intervention_sensitivity=0.16)


# ---------------------------------------------------------------------------
# Student profiles (Stage 2, Section 3) — multiplier-derived
# ---------------------------------------------------------------------------


def _default_profiles() -> dict[str, StudentProfileConfig]:
    return {
        "Consistently_Focused": StudentProfileConfig(
            name="Consistently Focused",
            description="Highest stability: near-Focused-baseline latency with low variance.",
            multipliers=ProfileMultipliers(
                latency_multiplier=1.00,
                latency_variance_multiplier=0.70,
                engagement_multiplier=1.05,
                fatigue_multiplier=0.30,
                intervention_multiplier=0.70,
            ),
            param_spread=0.10,
            transition_modifiers={
                AttentionState.FOCUSED: {
                    AttentionState.FOCUSED: 0.10,
                    AttentionState.DISTRACTED: -0.05,
                    AttentionState.IMPULSIVE: -0.05,
                }
            },
        ),
        "Gradually_Fatigued": StudentProfileConfig(
            name="Gradually Fatigued",
            description="Starts near Focused baseline; fatigue rate dominates over session duration.",
            multipliers=ProfileMultipliers(
                latency_multiplier=1.05,
                latency_variance_multiplier=1.10,
                engagement_multiplier=0.85,
                fatigue_multiplier=2.50,
                intervention_multiplier=1.00,
            ),
            param_spread=0.15,
            transition_modifiers={
                AttentionState.FOCUSED: {
                    AttentionState.FOCUSED: -0.05,
                    AttentionState.DISTRACTED: 0.05,
                    AttentionState.IMPULSIVE: 0.0,
                }
            },
        ),
        "Highly_Distractible": StudentProfileConfig(
            name="Highly Distractible",
            description="Higher latency variance; frequent transitions into Distracted.",
            multipliers=ProfileMultipliers(
                latency_multiplier=1.40,
                latency_variance_multiplier=1.80,
                engagement_multiplier=0.60,
                fatigue_multiplier=1.30,
                intervention_multiplier=0.80,
            ),
            param_spread=0.20,
            transition_modifiers={
                AttentionState.FOCUSED: {
                    AttentionState.FOCUSED: -0.15,
                    AttentionState.DISTRACTED: 0.15,
                    AttentionState.IMPULSIVE: 0.0,
                },
                AttentionState.DISTRACTED: {
                    AttentionState.FOCUSED: -0.10,
                    AttentionState.DISTRACTED: 0.10,
                    AttentionState.IMPULSIVE: 0.0,
                },
            },
        ),
        "Highly_Impulsive": StudentProfileConfig(
            name="Highly Impulsive",
            description="Lowest latency; elevated impulsive transitions.",
            multipliers=ProfileMultipliers(
                latency_multiplier=0.45,
                latency_variance_multiplier=0.90,
                engagement_multiplier=0.75,
                fatigue_multiplier=1.20,
                intervention_multiplier=1.15,
            ),
            param_spread=0.15,
            transition_modifiers={
                AttentionState.FOCUSED: {
                    AttentionState.FOCUSED: -0.10,
                    AttentionState.DISTRACTED: 0.0,
                    AttentionState.IMPULSIVE: 0.10,
                },
                AttentionState.DISTRACTED: {
                    AttentionState.FOCUSED: 0.0,
                    AttentionState.DISTRACTED: -0.10,
                    AttentionState.IMPULSIVE: 0.10,
                },
                AttentionState.IMPULSIVE: {
                    AttentionState.FOCUSED: -0.05,
                    AttentionState.DISTRACTED: -0.05,
                    AttentionState.IMPULSIVE: 0.10,
                },
            },
        ),
        "Recovering_Learner": StudentProfileConfig(
            name="Recovering Learner",
            description="Adaptive after intervention: high intervention-sensitivity multiplier.",
            multipliers=ProfileMultipliers(
                latency_multiplier=1.10,
                latency_variance_multiplier=1.10,
                engagement_multiplier=0.90,
                fatigue_multiplier=1.20,
                intervention_multiplier=2.50,
            ),
            param_spread=0.15,
            transition_modifiers={
                AttentionState.DISTRACTED: {
                    AttentionState.FOCUSED: 0.05,
                    AttentionState.DISTRACTED: -0.05,
                    AttentionState.IMPULSIVE: 0.0,
                }
            },
        ),
    }


def _default_profile_distribution() -> dict[str, float]:
    return {
        "Consistently_Focused": 0.25,
        "Gradually_Fatigued": 0.20,
        "Highly_Distractible": 0.20,
        "Highly_Impulsive": 0.15,
        "Recovering_Learner": 0.20,
    }


# ---------------------------------------------------------------------------
# Top-level default configuration (Stage 2, Section 9)
# ---------------------------------------------------------------------------


def default_config() -> GeneratorConfig:
    """Build the reference `GeneratorConfig` specified in Stage 2, Section 9."""

    return GeneratorConfig(
        seed=42,
        students=100,
        sessions_per_student=5,
        interactions_per_session=(20, 40),
        noise=0.15,
        fatigue_enabled=True,
        intervention_probability=0.25,
        rolling_window=5,
        class_balance={
            AttentionState.FOCUSED: 0.50,
            AttentionState.DISTRACTED: 0.30,
            AttentionState.IMPULSIVE: 0.20,
        },
        profile_distribution=_default_profile_distribution(),
        transition_matrix=_default_transition_matrix(),
        distributions=_default_distributions(),
        profiles=_default_profiles(),
        base_rates=_default_base_rates(),
        output=OutputConfig(),
        version_metadata=VersionMetadata(
            dataset_version="1.0.0",
            schema_version="1.0",
            paper_version="stage2",
            generator_version=_generator_version,
        ),
        experiment=ExperimentMetadata(),
        version="0.1.0",
    )
