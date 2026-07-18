"""Module 5: Behaviour Generator.

Causal pipeline:

    Student Profile -> Attention State -> Distribution Sampling
    -> Fatigue Adjustment -> Intervention Adjustment -> Rolling Statistics
    -> Derived Behaviour Features -> Validation

Concretely, each stage maps onto one step below:
1. `student` + `attention_state` (both caller-supplied, same pattern as
   `ResponseGenerator` — this module does not decide attention states or run
   the Markov chain; that's the temporal simulator's job) select which
   per-state latency/hesitation/interaction-duration parameters apply.
2. `behaviour_scoring.sample_response_latency` draws the personalized latency
   (Distribution Sampling stage, itself folding in the Fatigue Adjustment via
   its `fatigue` argument).
3. `behaviour_scoring.fatigue_level` computes the Fatigue/Intervention
   Adjustment (a student's `fatigue_rate` accumulates over the session;
   `intervention_applied` + `intervention_sensitivity` pull it back down).
4. `behaviour_scoring.update_rolling_value` computes the Rolling Statistics
   stage (EMA over `SessionContext.rolling_latency`/`rolling_engagement`).
5. Derived Behaviour Features (`normalized_latency`, `transition_occurred`)
   are computed, never sampled.
6. `validators.behaviour_validator.validate_behaviour_record` is the
   Validation stage, run before the record is returned.

Per Step 9, this module does **not** resample `response_length`,
`repetition_ratio`, `topic_shift`, or engagement — those are read directly
from the `Response` the Response Generator already computed from real text.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config import GeneratorConfig
from dataset_generator.config.attention_state import AttentionState
from dataset_generator.distributions.sampler import sample as sample_distribution
from dataset_generator.generators.behaviour_scoring import (
    fatigue_level,
    normalized_latency,
    sample_response_latency,
    update_rolling_value,
)
from dataset_generator.models.behaviour import BehaviourFeatures, BehaviourMetadata, BehaviourRecord
from dataset_generator.models.prompt import Prompt
from dataset_generator.models.response import Response
from dataset_generator.models.session_context import SessionContext
from dataset_generator.models.student import Student
from dataset_generator.validators.behaviour_validator import validate_behaviour_record


class BehaviourGenerator:
    """Generates one `BehaviourRecord` per call from `Student`/`Prompt`/`Response`/`SessionContext`.

    Uses a dedicated `noise_rng` stream (see `utils.rng.build_rng_streams`),
    so behavioural sampling never perturbs prompt, student, or response
    randomness — the same RNG-stream-separation discipline established
    since Module 1.
    """

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._behaviour_settings = config.behaviour_generation
        self._rolling_window = config.rolling_window
        self._rng = rng

    def generate_behaviour(
        self,
        *,
        student: Student,
        prompt: Prompt,
        response: Response,
        attention_state: AttentionState,
        session_context: SessionContext,
    ) -> BehaviourRecord:
        """Generate one `BehaviourRecord` for this interaction."""

        fatigue = fatigue_level(student, session_context, self._behaviour_settings)

        latency = sample_response_latency(
            self._rng, student, attention_state, fatigue, self._behaviour_settings
        )

        state_distribution = self._config.distributions.for_state(attention_state)
        hesitation_duration = sample_distribution(self._rng, state_distribution.hesitation)
        interaction_duration = sample_distribution(self._rng, state_distribution.interaction_duration)
        interaction_duration = max(interaction_duration, latency + hesitation_duration)

        rolling_latency = update_rolling_value(
            session_context.rolling_latency, latency, self._rolling_window
        )
        rolling_engagement = update_rolling_value(
            session_context.rolling_engagement, response.engagement_proxy, self._rolling_window
        )

        previous_state = session_context.previous_attention_state
        transition_occurred = previous_state is not None and previous_state != attention_state

        record = BehaviourRecord(
            student_id=student.student_id,
            session_id=session_context.session_id,
            interaction_number=session_context.interaction_number,
            attention_state=attention_state,
            response_latency=latency,
            interaction_duration=interaction_duration,
            hesitation_duration=hesitation_duration,
            response_length=response.response_length,
            engagement_score=response.engagement_proxy,
            repetition_ratio=response.features.repetition_ratio,
            topic_shift=response.features.topic_shift,
            rolling_latency=rolling_latency,
            rolling_engagement=rolling_engagement,
            fatigue_level=fatigue,
            intervention_applied=session_context.intervention_applied,
            features=BehaviourFeatures(
                normalized_latency=normalized_latency(latency, student),
                fatigue_progression=fatigue,
                rolling_latency=rolling_latency,
                rolling_engagement=rolling_engagement,
                transition_occurred=transition_occurred,
            ),
            metadata=BehaviourMetadata(
                prompt_id=prompt.prompt_id,
                subject=prompt.subject,
                topic=prompt.topic,
                student_profile=student.profile_name,
                previous_attention_state=previous_state,
                session_progress=session_context.session_progress,
                correctness_score=response.correctness_score,
            ),
        )

        issues = validate_behaviour_record(record)
        if issues:
            raise RuntimeError(
                f"generated behaviour record for {record.student_id}:{record.interaction_number} "
                f"failed validation: {issues}"
            )

        return record
