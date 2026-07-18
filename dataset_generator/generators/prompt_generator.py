"""Module 3: Prompt Generator.

Produces `Prompt` objects from `CurriculumConfig` + `PromptGenerationConfig`
+ the wording templates in `prompt_templates.py`. Knows nothing about BAS,
attention states, students, or reinforcement learning — its only job is
educational content.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config import GeneratorConfig
from dataset_generator.config.curriculum import TopicDefinition
from dataset_generator.config.prompt_generation import BLOOM_ORDER, CognitiveLevel, Difficulty
from dataset_generator.generators.prompt_templates import (
    COGNITIVE_TO_ANSWER_TYPE,
    TEMPLATES,
    difficulty_score,
    estimated_response_length,
)
from dataset_generator.models.prompt import Prompt, PromptMetadata
from dataset_generator.utils.text_metrics import (
    estimate_reading_time_seconds,
    flesch_kincaid_grade,
    word_count,
)
from dataset_generator.validators.prompt_validator import validate_prompt


class PromptGenerator:
    """Stateful prompt generator: samples prompts from a curriculum + config.

    Statefulness is limited to (a) an incrementing `prompt_id` counter and
    (b) a set of previously produced `prompt_text`s used for duplicate
    avoidance — no behavioural, session, or student state is ever touched.
    """

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator) -> None:
        self._curriculum = config.curriculum
        self._settings = config.prompt_generation
        self._rng = rng
        self._counter = 0
        self._seen_texts: set[str] = set()
        self._forced_duplicate_count = 0
        self._generated: list[Prompt] = []

    @property
    def generated(self) -> list[Prompt]:
        """All prompts produced by this generator instance so far."""

        return list(self._generated)

    @property
    def forced_duplicate_count(self) -> int:
        """How many prompts were accepted as exact duplicates after exhausting retries."""

        return self._forced_duplicate_count

    def generate_prompt(
        self,
        *,
        subject: str | None = None,
        topic: str | None = None,
        difficulty: Difficulty | None = None,
        cognitive_level: CognitiveLevel | None = None,
    ) -> Prompt:
        """Generate one `Prompt`, sampling any unspecified dimension.

        `subject`/`topic` are curriculum keys (e.g. `"Mathematics"` /
        `"Algebra"`), not display names. Retries template/keyword choice up
        to `prompt_generation.duplicate_retry_limit` times to avoid exact
        text duplicates before falling back to accepting one (counted in
        `forced_duplicate_count`) so generation always terminates.
        """

        subject_key = subject if subject is not None else self._sample_subject()
        topic_def, topic_key = self._resolve_topic(subject_key, topic)
        chosen_difficulty = difficulty if difficulty is not None else self._sample_difficulty()
        chosen_level = cognitive_level if cognitive_level is not None else self._sample_cognitive_level()

        prompt = self._build_prompt(subject_key, topic_key, topic_def, chosen_difficulty, chosen_level)

        issues = validate_prompt(prompt, min_words=self._settings.min_prompt_words)
        if issues:
            raise RuntimeError(
                f"generated prompt {prompt.prompt_id} failed validation: {issues}"
            )

        self._seen_texts.add(prompt.prompt_text)
        self._generated.append(prompt)
        return prompt

    def generate_batch(
        self,
        n: int,
        *,
        subject: str | None = None,
        topic: str | None = None,
        difficulty: Difficulty | None = None,
        cognitive_level: CognitiveLevel | None = None,
    ) -> list[Prompt]:
        """Generate `n` prompts, each independently sampling unspecified dimensions."""

        return [
            self.generate_prompt(
                subject=subject, topic=topic, difficulty=difficulty, cognitive_level=cognitive_level
            )
            for _ in range(n)
        ]

    def generate_curriculum(
        self,
        subject: str,
        *,
        prompts_per_topic: int = 1,
        cognitive_levels: list[CognitiveLevel] | None = None,
        difficulty: Difficulty | None = None,
    ) -> list[Prompt]:
        """Generate prompts across `subject`'s topics in learning-progression order.

        If `cognitive_levels` is given, it's cycled through per topic
        (giving deterministic coverage of Bloom levels along the
        progression); otherwise each prompt samples its own level.
        """

        prompts: list[Prompt] = []
        for topic_def in self._curriculum.ordered_topics(subject):
            topic_key = self._curriculum.topic_key_for(subject, topic_def.name)
            for i in range(prompts_per_topic):
                level = (
                    cognitive_levels[i % len(cognitive_levels)]
                    if cognitive_levels
                    else None
                )
                prompts.append(
                    self.generate_prompt(
                        subject=subject, topic=topic_key, difficulty=difficulty, cognitive_level=level
                    )
                )
        return prompts

    # -- sampling helpers -----------------------------------------------

    def _sample_subject(self) -> str:
        subject_keys = sorted(self._curriculum.subjects)
        return str(self._rng.choice(subject_keys))

    def _resolve_topic(
        self, subject_key: str, topic_key: str | None
    ) -> tuple[TopicDefinition, str]:
        subject = self._curriculum.subjects[subject_key]
        if topic_key is not None:
            return subject.topics[topic_key], topic_key
        keys = sorted(subject.topics)
        chosen_key = str(self._rng.choice(keys))
        return subject.topics[chosen_key], chosen_key

    def _sample_difficulty(self) -> Difficulty:
        options = list(self._settings.difficulty_distribution)
        weights = np.array([self._settings.difficulty_distribution[o] for o in options])
        return options[self._rng.choice(len(options), p=weights / weights.sum())]

    def _sample_cognitive_level(self) -> CognitiveLevel:
        options = list(self._settings.cognitive_level_distribution)
        weights = np.array([self._settings.cognitive_level_distribution[o] for o in options])
        return options[self._rng.choice(len(options), p=weights / weights.sum())]

    # -- construction -----------------------------------------------------

    def _build_prompt(
        self,
        subject_key: str,
        topic_key: str,
        topic_def: TopicDefinition,
        difficulty: Difficulty,
        cognitive_level: CognitiveLevel,
    ) -> Prompt:
        template_pool = TEMPLATES[cognitive_level][difficulty]
        keyword_count = min(self._settings.keywords_per_prompt, len(topic_def.keywords))

        text = ""
        for attempt in range(self._settings.duplicate_retry_limit):
            template = str(self._rng.choice(template_pool))
            keyword = str(self._rng.choice(topic_def.keywords))
            candidate = template.format(topic=topic_def.name, keyword=keyword)
            if candidate not in self._seen_texts:
                text = candidate
                break
            text = candidate
        else:
            self._forced_duplicate_count += 1

        selected_keywords = self._rng.choice(
            topic_def.keywords, size=keyword_count, replace=False
        ).tolist()

        self._counter += 1
        prompt_id = f"P{self._counter:06d}"

        token_count = word_count(text)

        metadata = PromptMetadata(
            estimated_reading_time_seconds=estimate_reading_time_seconds(text),
            token_count=token_count,
            concept_count=len(selected_keywords),
            difficulty_score=difficulty_score(difficulty, token_count),
            cognitive_complexity_score=BLOOM_ORDER.index(cognitive_level) / (len(BLOOM_ORDER) - 1),
            readability_grade=flesch_kincaid_grade(text),
            subject_id=subject_key,
            topic_id=topic_key,
        )

        return Prompt(
            prompt_id=prompt_id,
            subject=subject_key,
            topic=topic_key,
            difficulty=difficulty,
            cognitive_level=cognitive_level,
            prompt_text=text,
            expected_answer_type=COGNITIVE_TO_ANSWER_TYPE[cognitive_level],
            estimated_response_length=estimated_response_length(difficulty, cognitive_level),
            keywords=selected_keywords,
            learning_objective=topic_def.learning_objective_template.format(topic=topic_def.name),
            metadata=metadata,
        )
