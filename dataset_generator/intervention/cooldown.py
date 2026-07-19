"""Module 11, Step 8: Cooldown Management.

Per-session, stateful — reset once per session, following the same pattern
`SessionStatisticsBuilder`/`BASEngine`/`RewardEngine` use elsewhere in this
project. Enforces minimum spacing between real interventions, a maximum
count per session, and short-window duplicate prevention (recommending the
same policy twice in quick succession even once general cooldown has
elapsed).
"""

from __future__ import annotations

from dataset_generator.intervention.config import InterventionConfig
from dataset_generator.intervention.models import InterventionCandidate

NO_INTERVENTION_POLICY_NAME = "NoInterventionPolicy"


class CooldownManager:
    """Tracks one session's intervention history and filters candidates accordingly."""

    def __init__(self, config: InterventionConfig) -> None:
        self._config = config
        self._last_intervention_interaction: int | None = None
        self._intervention_count = 0
        self._policy_history: list[tuple[int, str]] = []

    def can_intervene(self, interaction_number: int) -> bool:
        """Whether a real (non-`NoInterventionPolicy`) intervention is allowed here."""

        config = self._config
        if interaction_number < config.min_interactions_before_intervention:
            return False
        if self._intervention_count >= config.max_interventions_per_session:
            return False
        if self._last_intervention_interaction is not None:
            spacing = interaction_number - self._last_intervention_interaction
            if spacing <= config.cooldown_length:
                return False
        return True

    def filter_candidates(
        self, candidates: list[InterventionCandidate], interaction_number: int
    ) -> tuple[list[InterventionCandidate], bool]:
        """Filter `candidates` for cooldown/limits/duplicate prevention.

        Returns `(allowed, was_suppressed)`, where `was_suppressed` is True
        only when a real intervention would otherwise have been the top
        candidate but was blocked.
        """

        real_candidates = [c for c in candidates if c.policy_name != NO_INTERVENTION_POLICY_NAME]
        no_intervention = [c for c in candidates if c.policy_name == NO_INTERVENTION_POLICY_NAME]

        if not real_candidates:
            return candidates, False

        if not self.can_intervene(interaction_number):
            return no_intervention or candidates, True

        allowed_real = [
            c for c in real_candidates if not self._is_recent_duplicate(c.policy_name, interaction_number)
        ]
        if not allowed_real:
            return no_intervention or candidates, True

        return sorted(allowed_real + no_intervention, key=lambda c: c.score, reverse=True), False

    def _is_recent_duplicate(self, policy_name: str, interaction_number: int) -> bool:
        window = self._config.duplicate_prevention_window
        for past_interaction, past_policy in reversed(self._policy_history):
            if interaction_number - past_interaction > window:
                break
            if past_policy == policy_name:
                return True
        return False

    def record_intervention(self, interaction_number: int, policy_name: str) -> None:
        """Record that `policy_name` was chosen at `interaction_number`."""

        self._policy_history.append((interaction_number, policy_name))
        if policy_name != NO_INTERVENTION_POLICY_NAME:
            self._last_intervention_interaction = interaction_number
            self._intervention_count += 1
