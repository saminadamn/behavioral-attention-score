"""Derive concrete per-profile sampling ranges from multipliers + base rates.

This is the resolver referenced in `StudentProfileConfig`'s docstring: it
turns a profile's *relative* multipliers into the absolute ranges the
Student Profile Generator (Step 2) samples from, computed against the
Focused-state distribution and `GeneratorConfig.base_rates`. Retuning the
Focused distribution or a base rate automatically keeps every profile
proportionally consistent — nothing here hardcodes an absolute number.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_generator.config.schema import GeneratorConfig


@dataclass(frozen=True)
class ResolvedProfileParams:
    """Concrete per-student sampling ranges resolved for one profile."""

    baseline_latency_mean_range: tuple[float, float]
    baseline_latency_std_range: tuple[float, float]
    fatigue_rate_range: tuple[float, float]
    intervention_sensitivity_range: tuple[float, float]
    engagement_tendency: float


def _spread_range(center: float, spread: float, lower_bound: float = 0.0) -> tuple[float, float]:
    """Build a `(lo, hi)` range of half-width `spread` (fractional) around `center`."""

    lo = max(lower_bound, center * (1.0 - spread))
    hi = max(lower_bound, center * (1.0 + spread))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def resolve_profile_parameters(config: GeneratorConfig, profile_key: str) -> ResolvedProfileParams:
    """Resolve `profile_key`'s multipliers into concrete sampling ranges.

    Raises `KeyError` if `profile_key` is not defined in `config.profiles`.
    """

    if profile_key not in config.profiles:
        raise KeyError(f"unknown profile {profile_key!r}; known: {sorted(config.profiles)}")

    profile = config.profiles[profile_key]
    mult = profile.multipliers
    spread = profile.param_spread

    focused = config.distributions.Focused
    focused_latency_mean = focused.response_latency.params["mean"]
    focused_latency_std = focused.response_latency.params["std"]

    engagement_params = focused.engagement.params
    focused_engagement_mean = engagement_params["alpha"] / (
        engagement_params["alpha"] + engagement_params["beta"]
    )

    latency_mean_center = focused_latency_mean * mult.latency_multiplier
    latency_std_center = focused_latency_std * mult.latency_variance_multiplier
    fatigue_center = config.base_rates.base_fatigue_rate * mult.fatigue_multiplier
    intervention_center = config.base_rates.base_intervention_sensitivity * mult.intervention_multiplier
    engagement_tendency = min(1.0, max(0.0, focused_engagement_mean * mult.engagement_multiplier))

    return ResolvedProfileParams(
        baseline_latency_mean_range=_spread_range(latency_mean_center, spread, lower_bound=0.5),
        baseline_latency_std_range=_spread_range(latency_std_center, spread, lower_bound=0.1),
        fatigue_rate_range=_spread_range(fatigue_center, spread, lower_bound=0.0),
        intervention_sensitivity_range=_spread_range(intervention_center, spread, lower_bound=0.0),
        engagement_tendency=engagement_tendency,
    )
