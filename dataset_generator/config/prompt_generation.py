"""Prompt-generation enums and settings (Module 3).

`Difficulty` and `CognitiveLevel` live here (not in `models/prompt.py`) for
the same reason `AttentionState` lives in `schema.py`: config and templates
both need to reference them, and `models/` depends on `config/`, not the
other way around.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataset_generator.config._validation import validate_probability_mapping


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class CognitiveLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels, in ascending order of complexity."""

    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"


# Ascending complexity order — index / (len - 1) gives a 0..1 complexity score.
BLOOM_ORDER: tuple[CognitiveLevel, ...] = (
    CognitiveLevel.REMEMBER,
    CognitiveLevel.UNDERSTAND,
    CognitiveLevel.APPLY,
    CognitiveLevel.ANALYZE,
    CognitiveLevel.EVALUATE,
    CognitiveLevel.CREATE,
)


class PromptGenerationConfig(BaseModel):
    """Tunable parameters for prompt sampling (Module 3, Steps 3/4/8/9)."""

    model_config = ConfigDict(frozen=True)

    difficulty_distribution: dict[Difficulty, float]
    cognitive_level_distribution: dict[CognitiveLevel, float]
    min_prompt_words: int = Field(gt=0, default=3)
    duplicate_retry_limit: int = Field(gt=0, default=20)
    keywords_per_prompt: int = Field(gt=0, default=2)

    @model_validator(mode="after")
    def _check_distributions(self) -> "PromptGenerationConfig":
        if set(self.difficulty_distribution.keys()) != set(Difficulty):
            raise ValueError(f"difficulty_distribution must define every difficulty: {set(Difficulty)}")
        validate_probability_mapping(self.difficulty_distribution, label="difficulty_distribution")

        if set(self.cognitive_level_distribution.keys()) != set(CognitiveLevel):
            raise ValueError(
                f"cognitive_level_distribution must define every level: {set(CognitiveLevel)}"
            )
        validate_probability_mapping(
            self.cognitive_level_distribution, label="cognitive_level_distribution"
        )
        return self


def default_prompt_generation_config() -> PromptGenerationConfig:
    """Reference settings: near-uniform difficulty/cognitive-level sampling."""

    return PromptGenerationConfig(
        difficulty_distribution={
            Difficulty.EASY: 0.34,
            Difficulty.MEDIUM: 0.33,
            Difficulty.HARD: 0.33,
        },
        cognitive_level_distribution={level: 1.0 / 6.0 for level in CognitiveLevel},
        min_prompt_words=3,
        duplicate_retry_limit=20,
        keywords_per_prompt=2,
    )
