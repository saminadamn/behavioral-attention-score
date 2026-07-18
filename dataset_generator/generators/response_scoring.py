"""Pure scoring functions for response generation (Module 4, Steps 3/4/5/6).

Kept separate from `response_generator.py` (orchestration) and
`response_templates.py`/`response_strategies.py` (wording/traits) per the
module's OOP-separation requirement: every function here is a deterministic
formula over already-known inputs (plus, for `correctness_score`, one
explicit noise draw from the caller-supplied RNG) — no hidden state, no I/O,
trivial to unit test in isolation.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config.prompt_generation import Difficulty
from dataset_generator.config.response_generation import ResponseGenerationConfig
from dataset_generator.generators.response_strategies import ResponseStrategy
from dataset_generator.models.session_context import SessionContext
from dataset_generator.models.student import Student


def expected_correctness(
    difficulty: Difficulty,
    strategy: ResponseStrategy,
    student: Student,
    session_context: SessionContext,
    config: ResponseGenerationConfig,
) -> float:
    """The deterministic (noise-free) expected correctness in [0, 1].

    Every term traces to a specific Module 4 design point:
    - `base`: Step 3 (difficulty sets the baseline).
    - `attention_term`: Step 3 (Focused+Hard can still succeed; Distracted+Easy
      can still fail — the interaction, not just the main effects, matters).
    - `strategy_term`: Step 3 (the strategy's own `error_tendency` trait, on
      top of — not instead of — the config's attention-state modifier).
    - `engagement_term`: Step 4 (a student's baseline engagement tendency).
    - `fatigue_term`: Step 4/Step 5 (Gradually_Fatigued-style decay over the session).
    - `intervention_term`: Step 4 (Recovering_Learner-style post-intervention boost).
    - `momentum_term`: Step 5 (session memory — repeating one's own attention state).
    - `rolling_engagement_term`: Step 5 (session memory — the running engagement trend).
    """

    base = config.base_correctness_by_difficulty[difficulty]
    attention_term = config.attention_state_correctness_modifier[strategy.attention_state]
    strategy_term = -strategy.error_tendency * config.strategy_error_weight
    engagement_term = (student.engagement_tendency - 0.5) * config.engagement_tendency_weight
    fatigue_term = -session_context.session_progress * student.fatigue_rate * config.fatigue_penalty_weight
    intervention_term = (
        student.intervention_sensitivity * config.intervention_bonus_weight
        if session_context.intervention_applied
        else 0.0
    )
    momentum_term = (
        config.momentum_bonus
        if session_context.previous_attention_state == strategy.attention_state
        else 0.0
    )
    rolling_engagement_term = (session_context.rolling_engagement - 0.5) * config.rolling_engagement_weight

    value = (
        base
        + attention_term
        + strategy_term
        + engagement_term
        + fatigue_term
        + intervention_term
        + momentum_term
        + rolling_engagement_term
    )
    return min(config.correctness_prob_max, max(config.correctness_prob_min, value))


def sample_correctness_score(
    difficulty: Difficulty,
    strategy: ResponseStrategy,
    student: Student,
    session_context: SessionContext,
    config: ResponseGenerationConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Draw a continuous `correctness_score`, returning `(score, expected_value)`.

    `expected_value` (the noise-free `expected_correctness`) is also
    returned so callers can record it in `ResponseMetadata.correctness_probability`
    for explainability, while `score` — `expected_value` plus a small
    Gaussian nudge — is what actually determines which template bank
    (correct-sounding vs. incorrect-sounding) gets sampled from.
    """

    expected_value = expected_correctness(difficulty, strategy, student, session_context, config)
    noise = rng.normal(0.0, config.correctness_noise_std) if config.correctness_noise_std > 0 else 0.0
    score = min(1.0, max(0.0, expected_value + noise))
    return score, expected_value


def confidence_score(
    strategy: ResponseStrategy,
    hesitation_markers: list[str],
    config: ResponseGenerationConfig,
) -> float:
    """How assertive a response sounds, inferred from attention state + detected hesitation.

    Deliberately independent of `correctness_score` — an Impulsive answer is
    typically confident whether or not it's right; a Distracted one hedges
    even when it happens to be correct. `hesitation_markers` is the actual
    detected list (Step 6: inferred from the generated text, not guessed).
    """

    base = config.confidence_by_attention_state[strategy.attention_state]
    penalty = config.hesitation_confidence_penalty * len(hesitation_markers)
    return min(1.0, max(0.0, base - penalty))


def coherence_score(lexical_diversity: float, repetition_ratio: float) -> float:
    """A simple text-derived coherence proxy: richer vocabulary, less repetition."""

    score = 0.5 * lexical_diversity + 0.5 * (1.0 - repetition_ratio)
    return min(1.0, max(0.0, score))


def engagement_proxy(
    response_length: int,
    target_length: int,
    semantic_similarity: float,
    lexical_diversity: float,
    repetition_ratio: float,
    weights: tuple[float, float, float, float],
) -> float:
    """The Stage 2 Section 4.8 composite engagement score, reused for responses:
    `w1*length + w2*similarity + w3*diversity + w4*(1 - repetition)`.
    """

    normalized_length = min(1.0, response_length / max(1, target_length))
    w_length, w_similarity, w_diversity, w_repetition = weights
    score = (
        w_length * normalized_length
        + w_similarity * semantic_similarity
        + w_diversity * lexical_diversity
        + w_repetition * (1.0 - repetition_ratio)
    )
    return min(1.0, max(0.0, score))
