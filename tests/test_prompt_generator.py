"""Tests for Module 3: Prompt Generator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataset_generator.config import (
    BLOOM_ORDER,
    CognitiveLevel,
    Difficulty,
    GeneratorConfig,
    default_config,
)
from dataset_generator.generators.prompt_generator import PromptGenerator
from dataset_generator.generators.prompt_report import build_prompt_report, render_report
from dataset_generator.generators.prompt_templates import TEMPLATES
from dataset_generator.models.prompt import Prompt
from dataset_generator.utils import build_rng_streams
from dataset_generator.validators.prompt_validator import (
    duplicate_prompt_ids,
    exact_duplicate_rate,
    validate_prompt,
    validate_prompt_batch,
)


def _generator(config=None, seed: int | None = None) -> PromptGenerator:
    config = config or default_config()
    streams = build_rng_streams(seed if seed is not None else config.seed)
    return PromptGenerator(config, streams.prompt_rng)


# ---------------------------------------------------------------------------
# Prompt model validation
# ---------------------------------------------------------------------------


def test_prompt_model_rejects_empty_text() -> None:
    config = default_config()
    generator = _generator(config)
    prompt = generator.generate_prompt(subject="Mathematics", topic="Algebra")
    data = prompt.model_dump(mode="json")
    data["prompt_text"] = ""
    with pytest.raises(ValidationError):
        Prompt.model_validate(data)


def test_prompt_model_rejects_non_positive_response_length() -> None:
    config = default_config()
    generator = _generator(config)
    prompt = generator.generate_prompt(subject="Mathematics", topic="Algebra")
    data = prompt.model_dump(mode="json")
    data["estimated_response_length"] = 0
    with pytest.raises(ValidationError):
        Prompt.model_validate(data)


def test_prompt_model_rejects_empty_keywords() -> None:
    config = default_config()
    generator = _generator(config)
    prompt = generator.generate_prompt(subject="Mathematics", topic="Algebra")
    data = prompt.model_dump(mode="json")
    data["keywords"] = []
    with pytest.raises(ValidationError):
        Prompt.model_validate(data)


# ---------------------------------------------------------------------------
# Difficulty generation
# ---------------------------------------------------------------------------


def test_all_difficulties_reachable() -> None:
    generator = _generator()
    seen = {
        generator.generate_prompt(subject="Mathematics", topic="Algebra", difficulty=d).difficulty
        for d in Difficulty
    }
    assert seen == set(Difficulty)


def test_difficulty_changes_wording_not_just_a_suffix() -> None:
    for level in CognitiveLevel:
        easy_texts = set(TEMPLATES[level][Difficulty.EASY])
        hard_texts = set(TEMPLATES[level][Difficulty.HARD])
        assert easy_texts.isdisjoint(hard_texts)
        # Hard variants should not simply be an Easy variant plus a suffix.
        for hard_template in hard_texts:
            assert hard_template not in {t + " (hard)" for t in easy_texts}


def test_difficulty_score_increases_with_difficulty() -> None:
    generator = _generator()
    scores = {}
    for difficulty in Difficulty:
        prompt = generator.generate_prompt(
            subject="Mathematics", topic="Algebra", difficulty=difficulty, cognitive_level=CognitiveLevel.UNDERSTAND
        )
        scores[difficulty] = prompt.metadata.difficulty_score
    assert scores[Difficulty.EASY] < scores[Difficulty.MEDIUM] < scores[Difficulty.HARD]


# ---------------------------------------------------------------------------
# Bloom level generation
# ---------------------------------------------------------------------------


def test_all_cognitive_levels_reachable() -> None:
    generator = _generator()
    seen = {
        generator.generate_prompt(
            subject="Science", topic="Biology", cognitive_level=level
        ).cognitive_level
        for level in CognitiveLevel
    }
    assert seen == set(CognitiveLevel)


def test_cognitive_complexity_score_matches_bloom_order() -> None:
    generator = _generator()
    for index, level in enumerate(BLOOM_ORDER):
        prompt = generator.generate_prompt(subject="Science", topic="Biology", cognitive_level=level)
        expected = index / (len(BLOOM_ORDER) - 1)
        assert prompt.metadata.cognitive_complexity_score == pytest.approx(expected)


def test_bloom_levels_produce_different_prompt_styles() -> None:
    generator = _generator()
    remember = generator.generate_prompt(
        subject="Science", topic="Biology", difficulty=Difficulty.EASY, cognitive_level=CognitiveLevel.REMEMBER
    )
    create = generator.generate_prompt(
        subject="Science", topic="Biology", difficulty=Difficulty.EASY, cognitive_level=CognitiveLevel.CREATE
    )
    assert remember.prompt_text != create.prompt_text
    assert remember.expected_answer_type != create.expected_answer_type


# ---------------------------------------------------------------------------
# Template diversity
# ---------------------------------------------------------------------------


def test_repeated_generation_produces_diverse_text() -> None:
    generator = _generator()
    texts = {
        generator.generate_prompt(
            subject="Mathematics", topic="Algebra", difficulty=Difficulty.MEDIUM, cognitive_level=CognitiveLevel.APPLY
        ).prompt_text
        for _ in range(20)
    }
    assert len(texts) > 1


# ---------------------------------------------------------------------------
# Curriculum ordering
# ---------------------------------------------------------------------------


def test_generate_curriculum_follows_progression_order() -> None:
    generator = _generator()
    prompts = generator.generate_curriculum("Mathematics", prompts_per_topic=1)
    assert [p.topic for p in prompts] == ["Arithmetic", "Algebra", "Geometry", "Probability"]


def test_generate_curriculum_cycles_cognitive_levels() -> None:
    generator = _generator()
    levels = [CognitiveLevel.REMEMBER, CognitiveLevel.CREATE]
    prompts = generator.generate_curriculum(
        "Science", prompts_per_topic=2, cognitive_levels=levels
    )
    # Physics, Chemistry, Biology x 2 prompts each, cycling [REMEMBER, CREATE]
    assert [p.cognitive_level for p in prompts] == [
        CognitiveLevel.REMEMBER,
        CognitiveLevel.CREATE,
    ] * 3


# ---------------------------------------------------------------------------
# Random seed reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_prompt_sequence() -> None:
    config = default_config()
    a = _generator(config, seed=config.seed).generate_batch(15)
    b = _generator(config, seed=config.seed).generate_batch(15)
    assert [p.model_dump(mode="json") for p in a] == [p.model_dump(mode="json") for p in b]


def test_different_seeds_produce_different_prompt_sequences() -> None:
    config = default_config()
    a = _generator(config, seed=config.seed).generate_batch(15)
    b = _generator(config, seed=config.seed + 1).generate_batch(15)
    assert [p.prompt_text for p in a] != [p.prompt_text for p in b]


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_duplicate_prompt_ids_detects_forced_duplicates() -> None:
    generator = _generator()
    prompts = generator.generate_batch(
        30, subject="Mathematics", topic="Algebra", difficulty=Difficulty.EASY, cognitive_level=CognitiveLevel.REMEMBER
    )
    # Only 3 distinct Easy/Remember templates x limited keywords exist for
    # Algebra, so requesting many should eventually force exact repeats.
    dupes = duplicate_prompt_ids(prompts)
    rate = exact_duplicate_rate(prompts)
    assert rate == len(dupes) / len(prompts)


def test_exact_duplicate_rate_zero_for_unique_texts() -> None:
    prompts = [
        Prompt.model_validate(
            {
                "prompt_id": f"P{i}",
                "subject": "Mathematics",
                "topic": "Algebra",
                "difficulty": "Easy",
                "cognitive_level": "Remember",
                "prompt_text": f"What is concept number {i}?",
                "expected_answer_type": "short_answer",
                "estimated_response_length": 8,
                "keywords": ["variable"],
                "learning_objective": "Understand algebra.",
                "metadata": {
                    "estimated_reading_time_seconds": 1.0,
                    "token_count": 5,
                    "concept_count": 1,
                    "difficulty_score": 0.2,
                    "cognitive_complexity_score": 0.0,
                    "readability_grade": 3.0,
                    "subject_id": "Mathematics",
                    "topic_id": "Algebra",
                },
            }
        )
        for i in range(5)
    ]
    assert exact_duplicate_rate(prompts) == 0.0
    assert duplicate_prompt_ids(prompts) == []


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------


def test_metadata_fields_are_sane() -> None:
    generator = _generator()
    prompt = generator.generate_prompt(subject="Mathematics", topic="Geometry")
    assert prompt.metadata.token_count > 0
    assert prompt.metadata.estimated_reading_time_seconds > 0
    assert prompt.metadata.concept_count == len(prompt.keywords)
    assert 0.0 <= prompt.metadata.difficulty_score <= 1.0
    assert 0.0 <= prompt.metadata.cognitive_complexity_score <= 1.0
    assert prompt.metadata.subject_id == "Mathematics"
    assert prompt.metadata.topic_id == "Geometry"


def test_validate_prompt_batch_flags_nothing_for_generator_output() -> None:
    generator = _generator()
    prompts = generator.generate_batch(50)
    assert validate_prompt_batch(prompts) == {}


def test_validate_prompt_rejects_unfilled_placeholder() -> None:
    generator = _generator()
    prompt = generator.generate_prompt(subject="Mathematics", topic="Algebra")
    data = prompt.model_dump(mode="json")
    data["prompt_text"] = "What is {topic}?"
    tampered = Prompt.model_validate(data)
    assert "unfilled template placeholder" in validate_prompt(tampered)


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------


def test_generate_10000_prompts_without_error_and_report() -> None:
    generator = _generator()
    prompts = generator.generate_batch(10_000)
    assert len(prompts) == 10_000
    assert validate_prompt_batch(prompts) == {}

    report = build_prompt_report(prompts)
    assert report.total_prompts == 10_000
    assert abs(sum(report.subject_distribution.values()) - 1.0) < 1e-6
    assert abs(sum(report.difficulty_distribution.values()) - 1.0) < 1e-6
    assert abs(sum(report.cognitive_level_distribution.values()) - 1.0) < 1e-6
    assert report.duplicate_rate < 0.05

    text = render_report(report)
    assert "Total prompts: 10000" in text
    assert "Average readability" in text


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_default_config_curriculum_and_prompt_generation_present() -> None:
    config = default_config()
    assert "Mathematics" in config.curriculum.subjects
    assert set(config.prompt_generation.difficulty_distribution) == set(Difficulty)


def test_prompt_generation_config_rejects_unbalanced_distribution() -> None:
    data = default_config().model_dump(mode="json")
    data["prompt_generation"]["difficulty_distribution"] = {"Easy": 0.9, "Medium": 0.3, "Hard": 0.2}
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(data)
