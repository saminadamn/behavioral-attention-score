"""Typed configuration schema for the BAS synthetic dataset generator.

Every model here is a Pydantic `BaseModel`, so a config loaded from YAML/JSON
is validated at load time (bad probabilities, missing states, malformed
ranges, unreachable Markov states, etc. fail fast with a readable error)
instead of surfacing as silent generation bugs three modules downstream.

Design reference: Stage 1 (research design) and Stage 2 (dataset
specification) of the BAS project. Section numbers in docstrings refer to
Stage 2.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataset_generator.config._validation import validate_probability_mapping
from dataset_generator.config.attention_state import (
    BEHAVIOURAL_FEATURES,
    AttentionState,
    combine_transition_matrix,
    reachability_violations,
)
from dataset_generator.config.behaviour_generation import (
    BehaviourGenerationConfig,
    default_behaviour_generation_config,
)
from dataset_generator.config.curriculum import CurriculumConfig, default_curriculum
from dataset_generator.config.prompt_generation import (
    PromptGenerationConfig,
    default_prompt_generation_config,
)
from dataset_generator.config.response_generation import (
    ResponseGenerationConfig,
    default_response_generation_config,
)
from dataset_generator.config.session_simulation import (
    SessionSimulationConfig,
    default_session_simulation_config,
)

_DISTRIBUTION_REQUIRED_PARAMS: dict[str, set[str]] = {
    "normal": {"mean", "std"},
    "gamma": {"shape", "scale"},
    "beta": {"alpha", "beta"},
    "poisson": {"lam"},
    "truncated_normal": {"mean", "std"},
}


# ---------------------------------------------------------------------------
# Distribution parameters (Stage 2, Section 5)
# ---------------------------------------------------------------------------


class FeatureDistributionParams(BaseModel):
    """Parameters of the probability distribution for one feature in one state.

    `family` selects the distribution shape; `params` must contain exactly
    the keys that family requires (Stage 2, Section 5 rationale: Gaussian for
    latency, Gamma for response length, Beta for [0,1]-bounded features,
    Poisson for hesitation counts). `clip_min`/`clip_max` implement the
    truncation used e.g. for latency (>= 0.5s); for `family="truncated_normal"`
    (Module 5's Distribution Engine) both bounds are required, since they
    define the truncation range rather than merely clamping tail values.
    """

    model_config = ConfigDict(frozen=True)

    family: Literal["normal", "gamma", "beta", "poisson", "truncated_normal"]
    params: dict[str, float]
    clip_min: float | None = None
    clip_max: float | None = None

    @model_validator(mode="after")
    def _check_params(self) -> "FeatureDistributionParams":
        required = _DISTRIBUTION_REQUIRED_PARAMS[self.family]
        given = set(self.params.keys())
        if given != required:
            raise ValueError(
                f"distribution family {self.family!r} requires params "
                f"{sorted(required)}, got {sorted(given)}"
            )
        if self.family == "beta" and (self.params["alpha"] <= 0 or self.params["beta"] <= 0):
            raise ValueError("beta distribution alpha/beta must be > 0")
        if self.family == "gamma" and (self.params["shape"] <= 0 or self.params["scale"] <= 0):
            raise ValueError("gamma distribution shape/scale must be > 0")
        if self.family in ("normal", "truncated_normal") and self.params["std"] <= 0:
            raise ValueError(f"{self.family} distribution std must be > 0")
        if self.family == "poisson" and self.params["lam"] <= 0:
            raise ValueError("poisson distribution lam must be > 0")
        if self.family == "truncated_normal" and (self.clip_min is None or self.clip_max is None):
            raise ValueError("truncated_normal requires both clip_min and clip_max")
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise ValueError("clip_min must be <= clip_max")
        return self


class StateDistributionConfig(BaseModel):
    """Per-feature distribution parameters for a single attention state."""

    model_config = ConfigDict(frozen=True)

    response_latency: FeatureDistributionParams
    response_length: FeatureDistributionParams
    topic_similarity: FeatureDistributionParams
    sentiment: FeatureDistributionParams
    engagement: FeatureDistributionParams
    lexical_diversity: FeatureDistributionParams
    topic_shift: FeatureDistributionParams
    hesitation: FeatureDistributionParams
    repetition_ratio: FeatureDistributionParams
    interaction_duration: FeatureDistributionParams

    def get(self, feature: str) -> FeatureDistributionParams:
        if feature not in BEHAVIOURAL_FEATURES:
            raise KeyError(f"unknown behavioural feature {feature!r}")
        return getattr(self, feature)


class DistributionConfig(BaseModel):
    """Distribution parameters for every attention state (Stage 2, Section 5)."""

    model_config = ConfigDict(frozen=True)

    Focused: StateDistributionConfig
    Distracted: StateDistributionConfig
    Impulsive: StateDistributionConfig

    def for_state(self, state: AttentionState) -> StateDistributionConfig:
        return getattr(self, state.value)


# ---------------------------------------------------------------------------
# Transition matrix (Stage 2, Section 6 / 8)
# ---------------------------------------------------------------------------


class TransitionMatrixConfig(BaseModel):
    """Base Markov transition matrix over attention states.

    `matrix[from_state][to_state]` is a probability; each row must sum to 1
    and every state must be reachable (see `reachability_violations`).
    Per-profile modifiers (`StudentProfileConfig.transition_modifiers`) are
    applied additively on top of this base matrix at generation time, and
    are separately checked for reachability at the `GeneratorConfig` level.
    """

    model_config = ConfigDict(frozen=True)

    matrix: dict[AttentionState, dict[AttentionState, float]]

    @model_validator(mode="after")
    def _check_matrix(self) -> "TransitionMatrixConfig":
        states = set(AttentionState)
        if set(self.matrix.keys()) != states:
            raise ValueError(f"transition matrix must define a row for every state: {states}")
        for from_state, row in self.matrix.items():
            if set(row.keys()) != states:
                raise ValueError(
                    f"transition row for {from_state} must define a probability to every state"
                )
            validate_probability_mapping(row, label=f"transition_matrix.matrix[{from_state}]")

        violations = reachability_violations(self.matrix)
        if violations:
            raise ValueError(
                "transition matrix fails reachability: " + "; ".join(violations)
            )
        return self


# ---------------------------------------------------------------------------
# Student profiles (Stage 2, Section 3) — multiplier-derived, not hardcoded
# ---------------------------------------------------------------------------


class ProfileMultipliers(BaseModel):
    """Relative multipliers that derive a profile's parameters from base rates.

    Rather than hardcoding e.g. `baseline_latency_mean_range = (1.5, 3.5)`,
    a profile states *how it differs* from the Focused-state distribution and
    the configured base rates (e.g. "45% of the Focused latency mean"). If
    the underlying distributions are retuned later, every profile stays
    proportionally consistent automatically — see `config.derive`.
    """

    model_config = ConfigDict(frozen=True)

    latency_multiplier: float = Field(gt=0.0)
    latency_variance_multiplier: float = Field(gt=0.0, default=1.0)
    engagement_multiplier: float = Field(gt=0.0)
    fatigue_multiplier: float = Field(ge=0.0)
    intervention_multiplier: float = Field(ge=0.0)


class StudentProfileConfig(BaseModel):
    """One student archetype (Stage 2, Section 3).

    `multipliers` + `param_spread` are resolved into concrete sampling
    ranges by `config.derive.resolve_profile_parameters`, relative to the
    Focused-state distribution and `GeneratorConfig.base_rates`.
    `transition_modifiers` are additive deltas applied to the base
    transition matrix for students of this profile, then renormalized.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    multipliers: ProfileMultipliers
    param_spread: float = Field(gt=0.0, lt=1.0, default=0.15)
    transition_modifiers: dict[AttentionState, dict[AttentionState, float]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _check_transition_modifier_states(self) -> "StudentProfileConfig":
        states = set(AttentionState)
        for from_state, deltas in self.transition_modifiers.items():
            if from_state not in states:
                raise ValueError(f"unknown state {from_state!r} in transition_modifiers")
            for to_state in deltas:
                if to_state not in states:
                    raise ValueError(f"unknown target state {to_state!r} in transition_modifiers")
        return self


class BaseRates(BaseModel):
    """Population-level base rates that profile multipliers scale (Stage 2, Section 3)."""

    model_config = ConfigDict(frozen=True)

    base_fatigue_rate: float = Field(ge=0.0, default=0.08)
    base_intervention_sensitivity: float = Field(ge=0.0, default=0.16)


# ---------------------------------------------------------------------------
# Output configuration (Stage 2, Section 9 / spec Step 9)
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    """Where and how generated data is written to disk."""

    model_config = ConfigDict(frozen=True)

    directory: str = "output"
    formats: list[Literal["csv", "json"]] = Field(default_factory=lambda: ["csv", "json"])
    export_metadata: bool = True
    filename_prefix: str = "bas_synthetic"

    @model_validator(mode="after")
    def _check_formats(self) -> "OutputConfig":
        if not self.formats:
            raise ValueError("output.formats must list at least one format")
        return self


# ---------------------------------------------------------------------------
# Version / experiment metadata — recorded with every generated dataset
# ---------------------------------------------------------------------------


class VersionMetadata(BaseModel):
    """Versioning recorded into every dataset's metadata.json."""

    model_config = ConfigDict(frozen=True)

    dataset_version: str = "1.0.0"
    schema_version: str = "1.0"
    paper_version: str = "stage2"
    generator_version: str = "1.0.0"


class ExperimentMetadata(BaseModel):
    """Free-form, descriptive run metadata.

    Deliberately excluded from `config.fingerprint.compute_fingerprint` —
    changing `author` or `notes` must not change the fingerprint of an
    otherwise identical, byte-for-byte-reproducible generation run.
    `date_created` and `git_commit` are populated by the exporter (Step 9)
    at actual generation time, not by config construction, so config
    equality/fingerprinting stays deterministic.
    """

    model_config = ConfigDict(frozen=True)

    experiment_name: str = "unnamed_experiment"
    experiment_description: str = ""
    author: str = ""
    date_created: str | None = None
    git_commit: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Top-level generator configuration
# ---------------------------------------------------------------------------


class GeneratorConfig(BaseModel):
    """Single source of truth for a generation run (Stage 2, Section 9).

    Every generator/model/distribution module is constructed from (a slice
    of) this object; no module hardcodes a generation parameter.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = Field(ge=0)
    students: int = Field(gt=0)
    sessions_per_student: int = Field(gt=0)
    interactions_per_session: tuple[int, int]
    noise: float = Field(ge=0.0, le=1.0)
    fatigue_enabled: bool = True
    intervention_probability: float = Field(ge=0.0, le=1.0)
    rolling_window: int = Field(gt=0, default=5)

    class_balance: dict[AttentionState, float]
    profile_distribution: dict[str, float]
    transition_matrix: TransitionMatrixConfig
    distributions: DistributionConfig
    profiles: dict[str, StudentProfileConfig]
    base_rates: BaseRates = Field(default_factory=BaseRates)
    curriculum: CurriculumConfig = Field(default_factory=default_curriculum)
    prompt_generation: PromptGenerationConfig = Field(default_factory=default_prompt_generation_config)
    response_generation: ResponseGenerationConfig = Field(
        default_factory=default_response_generation_config
    )
    behaviour_generation: BehaviourGenerationConfig = Field(
        default_factory=default_behaviour_generation_config
    )
    session_simulation: SessionSimulationConfig = Field(
        default_factory=default_session_simulation_config
    )
    output: OutputConfig = Field(default_factory=OutputConfig)
    version_metadata: VersionMetadata = Field(default_factory=VersionMetadata)
    experiment: ExperimentMetadata = Field(default_factory=ExperimentMetadata)
    version: str = "0.1.0"

    @model_validator(mode="after")
    def _check_cross_references(self) -> "GeneratorConfig":
        lo, hi = self.interactions_per_session
        if lo > hi:
            raise ValueError("interactions_per_session must have (min, max) with min <= max")
        if lo < 1:
            raise ValueError("interactions_per_session minimum must be >= 1")

        states = set(AttentionState)
        if set(self.class_balance.keys()) != states:
            raise ValueError(f"class_balance must define a proportion for every state: {states}")
        validate_probability_mapping(self.class_balance, label="class_balance")

        if set(self.profile_distribution.keys()) != set(self.profiles.keys()):
            raise ValueError(
                "profile_distribution keys must exactly match profiles keys: "
                f"{set(self.profile_distribution.keys())} != {set(self.profiles.keys())}"
            )
        validate_probability_mapping(self.profile_distribution, label="profile_distribution")

        if not self.profiles:
            raise ValueError("at least one student profile must be defined")

        for profile_key, profile in self.profiles.items():
            self._check_profile_effective_matrix_reachable(profile_key, profile)

        return self

    def _check_profile_effective_matrix_reachable(
        self, profile_key: str, profile: StudentProfileConfig
    ) -> None:
        """Verify the profile's effective transition matrix (base + modifiers,
        clipped and renormalized by `combine_transition_matrix`) still leaves
        every state reachable — a profile should never be able to make a
        state permanently unreachable.
        """

        try:
            effective = combine_transition_matrix(
                self.transition_matrix.matrix, profile.transition_modifiers
            )
        except ValueError as exc:
            raise ValueError(f"profile {profile_key!r}: {exc}") from exc

        violations = reachability_violations(effective)
        if violations:
            raise ValueError(
                f"profile {profile_key!r} effective transition matrix fails reachability: "
                + "; ".join(violations)
            )
