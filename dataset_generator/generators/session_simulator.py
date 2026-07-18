"""Module 6: Temporal Session Simulator.

Orchestrates the Prompt/Response/Behaviour generators and the
`TransitionEngine` into one coherent session. Every dependency is injected
through `__init__` (`PromptGenerator`, `ResponseGenerator`,
`BehaviourGenerator`, `TransitionEngine`) — this class coordinates them, it
does not construct them and does not duplicate any of their internal logic.

Interaction pipeline (Step 4), run once per interaction:

    Generate Prompt -> Generate Response -> Generate Behaviour
    -> Update Session Context -> Sample Next Attention State -> Continue

Fatigue dynamics (Step 5) and intervention-driven correctness/engagement
effects (Step 6) require **no new logic here** — they're already implemented
in `BehaviourGenerator.generate_behaviour` (fatigue) and
`ResponseGenerator.generate_response` (correctness/engagement), both of
which read `SessionContext.intervention_applied`. This class's only new
contribution to intervention dynamics is the transition-probability boost in
`TransitionEngine.sample_next_state` (Step 6: never overwrite the state
directly).
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config import GeneratorConfig
from dataset_generator.config.attention_state import AttentionState
from dataset_generator.generators.prompt_generator import PromptGenerator
from dataset_generator.generators.response_generator import ResponseGenerator
from dataset_generator.generators.behaviour_generator import BehaviourGenerator
from dataset_generator.generators.session_statistics import SessionStatisticsBuilder
from dataset_generator.generators.transition_engine import TransitionEngine
from dataset_generator.models.session import (
    InteractionRecord,
    InterventionEvent,
    SessionRecord,
    SessionStatistics,
    SessionSummary,
    TransitionEvent,
)
from dataset_generator.models.session_context import SessionContext
from dataset_generator.models.student import Student
from dataset_generator.validators.session_validator import validate_session


class SessionSimulator:
    """Simulates one complete session for one student.

    `rng` is used only for decisions this class itself owns (session length,
    intervention triggering) — never for prompt/response/behaviour text or
    sampling, which stay entirely inside their own injected generators and
    RNG streams (Step 7 of Module 4/5's design carried forward here).
    """

    def __init__(
        self,
        config: GeneratorConfig,
        prompt_generator: PromptGenerator,
        response_generator: ResponseGenerator,
        behaviour_generator: BehaviourGenerator,
        transition_engine: TransitionEngine,
        rng: np.random.Generator,
    ) -> None:
        self._config = config
        self._prompt_generator = prompt_generator
        self._response_generator = response_generator
        self._behaviour_generator = behaviour_generator
        self._transition_engine = transition_engine
        self._rng = rng

    def simulate_session(self, student: Student, session_id: str) -> SessionRecord:
        """Simulate one full session for `student`, returning a `SessionRecord`."""

        session_length = self._sample_session_length()
        stats_builder = SessionStatisticsBuilder(self._config.rolling_window)

        interactions: list[InteractionRecord] = []
        transition_history: list[TransitionEvent] = []
        intervention_history: list[InterventionEvent] = []

        current_state = self._transition_engine.sample_initial_state()
        transition_history.append(
            TransitionEvent(interaction_number=1, from_state=None, to_state=current_state, transitioned=False)
        )
        stats_builder.record_transition(None, current_state)

        previous_response_text: str | None = None
        previous_attention_state: AttentionState | None = None
        rolling_engagement = 0.5
        rolling_latency: float | None = None
        previous_fatigue = 0.0

        for interaction_number in range(1, session_length + 1):
            intervention_applied = self._decide_intervention(rolling_engagement)

            prompt = self._prompt_generator.generate_prompt()
            session_context = SessionContext(
                session_id=session_id,
                interaction_number=interaction_number,
                session_length=session_length,
                previous_response_text=previous_response_text,
                previous_attention_state=previous_attention_state,
                rolling_engagement=rolling_engagement,
                rolling_latency=rolling_latency,
                intervention_applied=intervention_applied,
            )

            response = self._response_generator.generate_response(
                prompt=prompt, student=student, attention_state=current_state, session_context=session_context
            )
            behaviour = self._behaviour_generator.generate_behaviour(
                student=student, prompt=prompt, response=response,
                attention_state=current_state, session_context=session_context,
            )

            interactions.append(
                InteractionRecord(
                    interaction_number=interaction_number,
                    prompt=prompt,
                    response=response,
                    behaviour=behaviour,
                )
            )
            if intervention_applied:
                intervention_history.append(
                    InterventionEvent(
                        interaction_number=interaction_number,
                        triggered_by=f"rolling_engagement<{self._config.session_simulation.intervention_engagement_threshold}",
                        pre_fatigue=previous_fatigue,
                        post_fatigue=behaviour.fatigue_level,
                    )
                )
            stats_builder.record_interaction(response, behaviour, current_state, intervention_applied)

            previous_response_text = response.response_text
            previous_attention_state = current_state
            previous_fatigue = behaviour.fatigue_level
            rolling_engagement = behaviour.rolling_engagement
            rolling_latency = behaviour.rolling_latency

            next_state = self._transition_engine.sample_next_state(
                current_state,
                student.profile_name,
                intervention_applied=intervention_applied,
                intervention_sensitivity=student.intervention_sensitivity,
            )
            stats_builder.record_transition(current_state, next_state)
            if interaction_number < session_length:
                transition_history.append(
                    TransitionEvent(
                        interaction_number=interaction_number + 1,
                        from_state=current_state,
                        to_state=next_state,
                        transitioned=next_state != current_state,
                    )
                )
            current_state = next_state

        statistics = stats_builder.finalize()
        summary = self._build_summary(student, session_id, interactions, statistics)

        record = SessionRecord(
            session_id=session_id,
            student_id=student.student_id,
            student_profile=student.profile_name,
            interactions=interactions,
            transition_history=transition_history,
            intervention_history=intervention_history,
            statistics=statistics,
            summary=summary,
        )

        issues = validate_session(record)
        if issues:
            raise RuntimeError(f"generated session {session_id} failed validation: {issues}")

        return record

    def _sample_session_length(self) -> int:
        lo, hi = self._config.interactions_per_session
        return int(self._rng.integers(lo, hi + 1))

    def _decide_intervention(self, rolling_engagement: float) -> bool:
        """Trigger an intervention when engagement has dropped, at the configured rate.

        Reuses `GeneratorConfig.intervention_probability` (already defined in
        Module 1) as the base trigger rate rather than introducing a second
        probability knob; `session_simulation.intervention_engagement_threshold`
        is the one genuinely new policy parameter this module needs.
        """

        if rolling_engagement >= self._config.session_simulation.intervention_engagement_threshold:
            return False
        return bool(self._rng.random() < self._config.intervention_probability)

    def _build_summary(
        self,
        student: Student,
        session_id: str,
        interactions: list[InteractionRecord],
        statistics: SessionStatistics,
    ) -> SessionSummary:
        n = len(interactions)
        dominant_state = max(statistics.state_frequencies, key=lambda s: statistics.state_frequencies[s])
        return SessionSummary(
            student_id=student.student_id,
            student_profile=student.profile_name,
            session_id=session_id,
            total_interactions=n,
            final_fatigue=interactions[-1].behaviour.fatigue_level,
            average_engagement=sum(i.response.engagement_proxy for i in interactions) / n,
            average_correctness=sum(i.response.correctness_score for i in interactions) / n,
            average_latency=sum(i.behaviour.response_latency for i in interactions) / n,
            dominant_attention_state=dominant_state,
            intervention_count=statistics.intervention_count,
        )
