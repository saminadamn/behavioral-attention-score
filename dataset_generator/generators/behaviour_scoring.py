"""Pure scoring functions for behaviour generation (Module 5, Steps 3/4/5/6).

Same separation-of-concerns rationale as `response_scoring.py`: every
function here is a deterministic formula (plus, where noted, one explicit
RNG draw) over already-known inputs — no hidden state, trivial to unit test.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config.attention_state import AttentionState
from dataset_generator.config.behaviour_generation import BehaviourGenerationConfig
from dataset_generator.config.schema import FeatureDistributionParams
from dataset_generator.distributions.sampler import sample as sample_distribution
from dataset_generator.models.session_context import SessionContext
from dataset_generator.models.student import Student


def fatigue_level(
    student: Student,
    session_context: SessionContext,
    config: BehaviourGenerationConfig,
) -> float:
    """Accumulated session fatigue in [0, 1].

    Grows with `session_progress x student.fatigue_rate` (Step 4:
    Gradually_Fatigued students have a high `fatigue_rate` from Module 2) and
    is *reduced* by a recent intervention, scaled by the student's own
    `intervention_sensitivity` (Step 4: Recovering_Learner students recover
    the most) — both fields reused from Module 2's `Student`, not re-derived.
    """

    raw = session_context.session_progress * student.fatigue_rate * config.fatigue_accumulation_weight
    if session_context.intervention_applied:
        raw *= 1.0 - min(1.0, student.intervention_sensitivity * config.intervention_recovery_weight)
    return min(1.0, max(0.0, raw))


def sample_response_latency(
    rng: np.random.Generator,
    student: Student,
    attention_state: AttentionState,
    fatigue: float,
    config: BehaviourGenerationConfig,
) -> float:
    """Sample `response_latency`, personalized to `student` rather than a
    population-level distribution (see Module 5's design-decision summary:
    Module 2 already derived per-student `baseline_latency`/`latency_variance`
    from profile multipliers, so reusing the population Stage 2 distribution
    directly here would throw that personalization away).

    `attention_state` scales the mean/variance around the student's own
    baseline (Step 3); `fatigue` further inflates the mean (Step 4/5:
    fatigue increases latency over the session).
    """

    latency_multiplier = config.attention_state_latency_multiplier[attention_state]
    variance_multiplier = config.attention_state_variance_multiplier[attention_state]

    mean = student.baseline_latency * latency_multiplier * (1.0 + fatigue * config.fatigue_latency_weight)
    std = student.latency_variance * variance_multiplier

    params = FeatureDistributionParams(
        family="truncated_normal",
        params={"mean": mean, "std": std},
        clip_min=config.latency_clip_min,
        clip_max=config.latency_clip_max,
    )
    return sample_distribution(rng, params)


def normalized_latency(latency: float, student: Student) -> float:
    """Z-score of `latency` relative to the student's own baseline (Stage 2, Section 4.2)."""

    return (latency - student.baseline_latency) / max(student.latency_variance, 1e-6)


def update_rolling_value(previous: float | None, current: float, window: int) -> float:
    """Exponential-moving-average update: `previous` is the incoming rolling
    value (or `None`/absent for the first interaction), `current` is this
    interaction's fresh observation. `alpha = 1/window` approximates a
    window-sized moving average without storing per-interaction history.
    """

    if previous is None:
        return current
    alpha = 1.0 / max(1, window)
    return alpha * current + (1.0 - alpha) * previous
