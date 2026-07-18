"""Module 4: Response Generator.

Pipeline (Module 4, Step 1):

    Teacher Prompt -> Prompt Analysis -> Student Profile -> Attention State
    -> Behaviour Modifiers -> Response Generator -> Feature Extraction

Concretely:
- "Prompt Analysis": `PromptAnalyzer.analyze()` converts a `Prompt` into a
  `PromptAnalysis` once; nothing here touches `Prompt` fields again.
- "Student Profile" + "Attention State" + "Behaviour Modifiers": `Student` +
  the caller-supplied `AttentionState` (resolved to a `ResponseStrategy` via
  `ResponseStrategyFactory`) + `SessionContext`, fed into
  `response_scoring.sample_correctness_score`.
- "Response Generator" (this class): samples wording from the strategy's
  template pool, conditioned on the sampled correctness score.
- "Feature Extraction": every `Response`/`ResponseFeatures` field is computed
  from the *generated text itself* (lexical diversity, repetition, sentiment,
  semantic similarity, coherence, hesitation markers, engagement, confidence)
  rather than drawn as independent random numbers — Module 4, Step 6.

This module does not decide attention states, run a Markov chain, or track
a Behavioural Attention Score — those belong to the temporal simulator and
BAS tracker (later roadmap steps), which call this generator once per
interaction with whatever `attention_state`/`SessionContext` they have
already determined.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config import GeneratorConfig
from dataset_generator.config.attention_state import AttentionState
from dataset_generator.generators.prompt_analyzer import PromptAnalysis, PromptAnalyzer
from dataset_generator.generators.response_scoring import (
    coherence_score,
    confidence_score,
    engagement_proxy,
    sample_correctness_score,
)
from dataset_generator.generators.response_strategies import ResponseStrategy, ResponseStrategyFactory
from dataset_generator.models.prompt import Prompt
from dataset_generator.models.response import Response, ResponseFeatures, ResponseMetadata
from dataset_generator.models.session_context import SessionContext
from dataset_generator.models.student import Student
from dataset_generator.utils.heuristic_nlp import (
    concept_coverage,
    find_hesitation_markers,
    repetition_ratio,
    simple_sentiment,
)
from dataset_generator.utils.text_metrics import token_jaccard_similarity, word_tokenize
from dataset_generator.validators.response_validator import is_immediate_repeat, validate_response


class ResponseGenerator:
    """Generates one `Response` per call from a `Prompt` + `Student` + context.

    Stateless across calls except for the shared `response_rng` stream —
    dedicated per Step 7, so changing response randomness never perturbs
    prompt generation or student sampling (each has its own RNG stream from
    `utils.rng.build_rng_streams`). No student, session, or behavioural
    state is cached on this object.
    """

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator) -> None:
        self._settings = config.response_generation
        self._rng = rng
        self._analyzer = PromptAnalyzer()

    def generate_response(
        self,
        *,
        prompt: Prompt,
        student: Student,
        attention_state: AttentionState,
        session_context: SessionContext,
    ) -> Response:
        """Generate one `Response` to `prompt` from `student` under `attention_state`."""

        analysis = self._analyzer.analyze(prompt)
        strategy = ResponseStrategyFactory.for_state(attention_state)

        score, expected_value = sample_correctness_score(
            analysis.difficulty, strategy, student, session_context, self._settings, self._rng
        )
        correct = score >= 0.5

        keyword = str(self._rng.choice(analysis.keywords))
        text = self._sample_text(analysis, strategy, correct, keyword, session_context)

        tokens = word_tokenize(text)
        response_length = len(tokens)
        unique_tokens = {t.lower() for t in tokens}
        lexical_diversity = len(unique_tokens) / response_length if response_length else 0.0

        rep_ratio = repetition_ratio(text)
        hesitation_markers = find_hesitation_markers(text, self._settings.hesitation_phrases)
        sentiment_score = simple_sentiment(text)

        # Concept coverage against [keyword, topic] rather than a Jaccard
        # similarity over the whole prompt sentence — see response_templates.py's
        # module docstring for why the latter drives similarity toward zero
        # regardless of how on-topic a response actually is.
        semantic_similarity = concept_coverage(text, [keyword, analysis.topic_display])

        topic_shift = (
            1.0 - token_jaccard_similarity(session_context.previous_response_text, text)
            if session_context.previous_response_text
            else 0.0
        )

        coherence = coherence_score(lexical_diversity, rep_ratio)
        confidence = confidence_score(strategy, hesitation_markers, self._settings)
        target_length = strategy.target_length(analysis.expected_response_length)
        engagement = engagement_proxy(
            response_length,
            target_length,
            semantic_similarity,
            lexical_diversity,
            rep_ratio,
            self._settings.engagement_proxy_weights,
        )

        response = Response(
            response_id=f"{student.student_id}_{session_context.interaction_number:04d}",
            student_id=student.student_id,
            prompt_id=analysis.prompt_id,
            response_text=text,
            correctness_score=score,
            response_length=response_length,
            semantic_similarity=semantic_similarity,
            lexical_diversity=lexical_diversity,
            sentiment=sentiment_score,
            engagement_proxy=engagement,
            confidence=confidence,
            hesitation_markers=hesitation_markers,
            features=ResponseFeatures(
                token_count=response_length,
                repetition_ratio=rep_ratio,
                coherence_score=coherence,
                topic_shift=topic_shift,
            ),
            metadata=ResponseMetadata(
                correctness_probability=expected_value,
                strategy_used=type(strategy).__name__,
                difficulty=analysis.difficulty,
                cognitive_level=analysis.cognitive_level,
                subject=analysis.subject,
                topic=analysis.topic,
                attention_state=attention_state,
                student_profile=student.profile_name,
                intervention_applied=session_context.intervention_applied,
                session_progress=session_context.session_progress,
            ),
        )

        issues = validate_response(response, max_words=self._settings.max_response_words)
        if issues:
            raise RuntimeError(f"generated response {response.response_id} failed validation: {issues}")

        return response

    def _sample_text(
        self,
        analysis: PromptAnalysis,
        strategy: ResponseStrategy,
        correct: bool,
        keyword: str,
        session_context: SessionContext,
    ) -> str:
        """Sample wording, retrying if it would verbatim-repeat the student's last turn."""

        text = ""
        for _ in range(self._settings.duplicate_retry_limit):
            candidate = strategy.generate_text(self._rng, correct, keyword, analysis.topic_display)
            text = candidate
            if not is_immediate_repeat(candidate, session_context.previous_response_text):
                break
        return text
