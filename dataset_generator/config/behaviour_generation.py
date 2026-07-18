"""Behaviour-generation settings (Module 5).

`response_latency` is personalized (Module 2's `Student.baseline_latency`/
`latency_variance`, not the population-level Stage 2 distribution directly)
via multipliers held here — analogous to how `StudentProfileConfig` derives
profile parameters from multipliers rather than absolute numbers (Step 1's
config design principle applied again).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataset_generator.config.attention_state import AttentionState


class BehaviourGenerationConfig(BaseModel):
    """Tunable parameters for behavioural-signal generation (Module 5, Steps 3/4/5)."""

    model_config = ConfigDict(frozen=True)

    attention_state_latency_multiplier: dict[AttentionState, float]
    attention_state_variance_multiplier: dict[AttentionState, float]

    fatigue_latency_weight: float = Field(ge=0.0, default=0.40)
    fatigue_accumulation_weight: float = Field(ge=0.0, default=1.0)
    intervention_recovery_weight: float = Field(ge=0.0, le=1.0, default=0.60)

    latency_clip_min: float = Field(gt=0.0, default=0.3)
    latency_clip_max: float = Field(gt=0.0, default=90.0)

    @model_validator(mode="after")
    def _check(self) -> "BehaviourGenerationConfig":
        if set(self.attention_state_latency_multiplier) != set(AttentionState):
            raise ValueError("attention_state_latency_multiplier must define every attention state")
        for state, value in self.attention_state_latency_multiplier.items():
            if value <= 0:
                raise ValueError(f"attention_state_latency_multiplier[{state}] must be > 0")

        if set(self.attention_state_variance_multiplier) != set(AttentionState):
            raise ValueError("attention_state_variance_multiplier must define every attention state")
        for state, value in self.attention_state_variance_multiplier.items():
            if value <= 0:
                raise ValueError(f"attention_state_variance_multiplier[{state}] must be > 0")

        if self.latency_clip_min >= self.latency_clip_max:
            raise ValueError("latency_clip_min must be < latency_clip_max")

        return self


def default_behaviour_generation_config() -> BehaviourGenerationConfig:
    """Reference settings: multipliers derived from Stage 2's per-state latency
    means/stds, expressed relative to the Focused state (mean=6.5s, std=1.5s)
    so they compose with `Student.baseline_latency`/`latency_variance`.
    """

    return BehaviourGenerationConfig(
        attention_state_latency_multiplier={
            AttentionState.FOCUSED: 1.00,
            AttentionState.DISTRACTED: 11.0 / 6.5,
            AttentionState.IMPULSIVE: 2.5 / 6.5,
        },
        attention_state_variance_multiplier={
            AttentionState.FOCUSED: 1.00,
            AttentionState.DISTRACTED: 3.5 / 1.5,
            AttentionState.IMPULSIVE: 1.0 / 1.5,
        },
    )
