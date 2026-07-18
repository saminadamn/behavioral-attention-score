"""The `Prompt` domain model produced by the Prompt Generator (Module 3).

Deliberately knows nothing about BAS, attention states, or reinforcement
learning — its only responsibility is representing one educational prompt
plus the metadata later stages (Response Generator, Feature Extraction)
need.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.config.prompt_generation import CognitiveLevel, Difficulty


class PromptMetadata(BaseModel):
    """Computed, inspectable analytics about one prompt's text."""

    model_config = ConfigDict(frozen=True)

    estimated_reading_time_seconds: float = Field(ge=0.0)
    token_count: int = Field(gt=0)
    concept_count: int = Field(gt=0)
    difficulty_score: float = Field(ge=0.0, le=1.0)
    cognitive_complexity_score: float = Field(ge=0.0, le=1.0)
    readability_grade: float
    subject_id: str
    topic_id: str


class Prompt(BaseModel):
    """One teacher prompt, ready for a Response Generator to consume."""

    model_config = ConfigDict(frozen=True)

    prompt_id: str
    subject: str
    topic: str
    difficulty: Difficulty
    cognitive_level: CognitiveLevel
    prompt_text: str = Field(min_length=1)
    expected_answer_type: str
    estimated_response_length: int = Field(gt=0)
    keywords: list[str] = Field(min_length=1)
    learning_objective: str = Field(min_length=1)
    metadata: PromptMetadata
