"""Module 6, Step 7: Rolling Session Statistics.

A mutable internal accumulator — not a Pydantic model, since it exists only
to be updated incrementally during simulation and then frozen once into a
`SessionStatistics` snapshot via `finalize()`. Updated after every
interaction; nothing here is recomputed from history once the session ends.

`rolling_latency`/`rolling_engagement` are **not** re-derived here — Module 5
already EMA-tracks both on every `BehaviourRecord` (`update_rolling_value`),
so this builder just carries the latest `BehaviourRecord`'s values forward.
Only `rolling_correctness`/`rolling_similarity` are genuinely new rolling
statistics (no earlier module tracks them), computed with the *same*
`update_rolling_value` helper Module 5 defined, not a re-implementation of it.
"""

from __future__ import annotations

from collections import Counter

from dataset_generator.config.attention_state import AttentionState
from dataset_generator.generators.behaviour_scoring import update_rolling_value
from dataset_generator.models.behaviour import BehaviourRecord
from dataset_generator.models.response import Response
from dataset_generator.models.session import SessionStatistics


class SessionStatisticsBuilder:
    """Incrementally builds one session's `SessionStatistics`."""

    def __init__(self, rolling_window: int) -> None:
        self._rolling_window = rolling_window
        self._count = 0
        self._rolling_latency = 0.0
        self._rolling_engagement = 0.5
        self._rolling_correctness: float | None = None
        self._rolling_similarity: float | None = None
        self._transition_counts: Counter[str] = Counter()
        self._state_frequencies: Counter[str] = Counter()
        self._total_duration = 0.0
        self._intervention_count = 0

    def record_interaction(
        self,
        response: Response,
        behaviour: BehaviourRecord,
        attention_state: AttentionState,
        intervention_applied: bool,
    ) -> None:
        """Fold one interaction's results into the running statistics."""

        self._count += 1
        self._rolling_latency = behaviour.rolling_latency
        self._rolling_engagement = behaviour.rolling_engagement
        self._rolling_correctness = update_rolling_value(
            self._rolling_correctness, response.correctness_score, self._rolling_window
        )
        self._rolling_similarity = update_rolling_value(
            self._rolling_similarity, response.semantic_similarity, self._rolling_window
        )
        self._state_frequencies[attention_state.value] += 1
        self._total_duration += behaviour.interaction_duration
        if intervention_applied:
            self._intervention_count += 1

    def record_transition(self, from_state: AttentionState | None, to_state: AttentionState) -> None:
        """Fold one observed transition into the running transition counts."""

        if from_state is None:
            return
        key = f"{from_state.value}->{to_state.value}"
        self._transition_counts[key] += 1

    def finalize(self) -> SessionStatistics:
        """Freeze the running accumulators into a `SessionStatistics` snapshot."""

        return SessionStatistics(
            interaction_count=self._count,
            rolling_latency=self._rolling_latency,
            rolling_engagement=self._rolling_engagement,
            rolling_correctness=self._rolling_correctness if self._rolling_correctness is not None else 0.0,
            rolling_similarity=self._rolling_similarity if self._rolling_similarity is not None else 0.0,
            transition_counts=dict(self._transition_counts),
            state_frequencies=dict(self._state_frequencies),
            total_duration_seconds=self._total_duration,
            intervention_count=self._intervention_count,
        )
