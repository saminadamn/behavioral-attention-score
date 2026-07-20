"""Response-generation settings (Module 4).

Every constant the Response Generator uses to turn (difficulty, attention
state, strategy, student, session context) into a correctness score,
confidence, or an engagement-proxy weighting lives here — none of it is
hardcoded in `generators/response_scoring.py`, `generators/response_strategies.py`,
or `generators/response_generator.py`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataset_generator.config.attention_state import AttentionState
from dataset_generator.config.prompt_generation import Difficulty

_DEFAULT_HESITATION_PHRASES: tuple[str, ...] = (
    "um", "uh", "umm", "i think", "maybe", "not sure", "i guess",
)


class ResponseGenerationConfig(BaseModel):
    """Tunable parameters for response generation (Module 4, Steps 3/4/5/6/7)."""

    model_config = ConfigDict(frozen=True)

    base_correctness_by_difficulty: dict[Difficulty, float]
    attention_state_correctness_modifier: dict[AttentionState, float]

    engagement_tendency_weight: float = Field(ge=0.0, default=0.30)
    fatigue_penalty_weight: float = Field(ge=0.0, default=0.50)
    intervention_bonus_weight: float = Field(ge=0.0, default=0.50)
    momentum_bonus: float = Field(ge=0.0, default=0.05)
    rolling_engagement_weight: float = Field(ge=0.0, default=0.20)
    strategy_error_weight: float = Field(ge=0.0, default=0.20)
    correctness_noise_std: float = Field(ge=0.0, default=0.05)

    correctness_prob_min: float = Field(ge=0.0, le=1.0, default=0.05)
    correctness_prob_max: float = Field(ge=0.0, le=1.0, default=0.95)

    engagement_proxy_weights: tuple[float, float, float, float] = (0.30, 0.30, 0.20, 0.20)
    hesitation_phrases: list[str] = Field(default_factory=lambda: list(_DEFAULT_HESITATION_PHRASES))

    confidence_by_attention_state: dict[AttentionState, float] = Field(
        default_factory=lambda: {
            AttentionState.FOCUSED: 0.75,
            AttentionState.DISTRACTED: 0.35,
            AttentionState.IMPULSIVE: 0.90,
        }
    )
    hesitation_confidence_penalty: float = Field(ge=0.0, default=0.15)

    # Per-state Gaussian noise added to otherwise near-constant evidence
    # (semantic_similarity, confidence) so attention states have realistic,
    # overlapping distributions rather than exactly-constant per-state
    # values (e.g. Focused semantic_similarity was exactly 1.0 for every
    # response, Impulsive exactly 0.5) — a downstream classifier trained on
    # exact constants learns a trivial threshold rather than a genuine
    # pattern. Values are clipped back into their valid range after noise.
    semantic_similarity_noise_std: dict[AttentionState, float] = Field(
        default_factory=lambda: {
            AttentionState.FOCUSED: 0.05,
            AttentionState.DISTRACTED: 0.05,
            AttentionState.IMPULSIVE: 0.08,
        }
    )
    confidence_noise_std: dict[AttentionState, float] = Field(
        default_factory=lambda: {
            AttentionState.FOCUSED: 0.05,
            AttentionState.DISTRACTED: 0.05,
            AttentionState.IMPULSIVE: 0.05,
        }
    )

    duplicate_retry_limit: int = Field(gt=0, default=10)
    max_response_words: int = Field(gt=0, default=200)

    @model_validator(mode="after")
    def _check(self) -> "ResponseGenerationConfig":
        if set(self.base_correctness_by_difficulty) != set(Difficulty):
            raise ValueError("base_correctness_by_difficulty must define every difficulty")
        for difficulty, value in self.base_correctness_by_difficulty.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"base_correctness_by_difficulty[{difficulty}] must be in [0,1]")

        if set(self.attention_state_correctness_modifier) != set(AttentionState):
            raise ValueError("attention_state_correctness_modifier must define every attention state")

        if set(self.confidence_by_attention_state) != set(AttentionState):
            raise ValueError("confidence_by_attention_state must define every attention state")
        for state, value in self.confidence_by_attention_state.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"confidence_by_attention_state[{state}] must be in [0,1]")

        if set(self.semantic_similarity_noise_std) != set(AttentionState):
            raise ValueError("semantic_similarity_noise_std must define every attention state")
        if set(self.confidence_noise_std) != set(AttentionState):
            raise ValueError("confidence_noise_std must define every attention state")
        for mapping_name in ("semantic_similarity_noise_std", "confidence_noise_std"):
            for state, value in getattr(self, mapping_name).items():
                if value < 0.0:
                    raise ValueError(f"{mapping_name}[{state}] must be >= 0")

        if self.correctness_prob_min > self.correctness_prob_max:
            raise ValueError("correctness_prob_min must be <= correctness_prob_max")

        weight_sum = sum(self.engagement_proxy_weights)
        if abs(weight_sum - 1.0) > 1e-3:
            raise ValueError(f"engagement_proxy_weights must sum to 1.0 (got {weight_sum:.4f})")

        if not self.hesitation_phrases:
            raise ValueError("hesitation_phrases must not be empty")

        return self


def default_response_generation_config() -> ResponseGenerationConfig:
    """Reference settings matching Module 4's difficulty/attention-state design."""

    return ResponseGenerationConfig(
        base_correctness_by_difficulty={
            Difficulty.EASY: 0.85,
            Difficulty.MEDIUM: 0.65,
            Difficulty.HARD: 0.45,
        },
        attention_state_correctness_modifier={
            AttentionState.FOCUSED: 0.10,
            AttentionState.DISTRACTED: -0.30,
            AttentionState.IMPULSIVE: -0.15,
        },
    )
