"""Module 6, Steps 2/3: Temporal State Machine / Transition Engine.

Samples attention states from the *configured* transition matrix only —
`config.transition_matrix` combined with a profile's `transition_modifiers`
via `config.attention_state.combine_transition_matrix` (the same function
`GeneratorConfig`'s own validation uses, so the matrix this engine samples
from is guaranteed identical to the one already checked for reachability at
config-load time). No attention state is ever chosen independently of this
matrix — that would defeat the entire point of Module 6.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from dataset_generator.config import GeneratorConfig
from dataset_generator.config.attention_state import AttentionState, combine_transition_matrix


class TransitionEngine:
    """Samples initial and next attention states for one simulation run.

    Effective (profile-adjusted) matrices are computed once per profile and
    cached — every student sharing a profile reuses the same matrix, since
    it depends only on `config.profiles[profile_key]`, never on a specific
    student.
    """

    _PROBABILITY_TOLERANCE: ClassVar[float] = 1e-6

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._effective_matrix_cache: dict[str, dict[AttentionState, dict[AttentionState, float]]] = {}

    def effective_matrix(self, profile_key: str) -> dict[AttentionState, dict[AttentionState, float]]:
        """The profile-adjusted transition matrix for `profile_key`, cached after first use."""

        if profile_key not in self._effective_matrix_cache:
            profile_cfg = self._config.profiles[profile_key]
            self._effective_matrix_cache[profile_key] = combine_transition_matrix(
                self._config.transition_matrix.matrix, profile_cfg.transition_modifiers
            )
        return self._effective_matrix_cache[profile_key]

    def sample_initial_state(self) -> AttentionState:
        """Sample the session's first state from `config.class_balance`.

        The transition matrix samples a *next* state given a *current* one —
        the first interaction has no prior state, so it instead draws from
        the population-level class balance (Stage 2's target marginal
        distribution over states), which is exactly what that config field
        is for.
        """

        states = list(self._config.class_balance)
        weights = np.array([self._config.class_balance[s] for s in states])
        index = self._rng.choice(len(states), p=weights / weights.sum())
        return states[index]

    def sample_next_state(
        self,
        current_state: AttentionState,
        profile_key: str,
        intervention_applied: bool = False,
        intervention_sensitivity: float = 0.0,
    ) -> AttentionState:
        """Sample the next state from `current_state`'s row of the effective matrix.

        Per Module 6 Step 6, an applied intervention never overwrites the
        state directly — it reshapes this row, shifting probability mass
        toward Focused proportional to `intervention_sensitivity` (Module 2's
        per-student trait) before sampling, then samples normally.
        """

        matrix = self.effective_matrix(profile_key)
        row = dict(matrix[current_state])

        if intervention_applied and intervention_sensitivity > 0.0:
            row = self._apply_intervention_boost(row, intervention_sensitivity)

        states = list(row)
        weights = np.array([row[s] for s in states])
        total = weights.sum()
        if abs(total - 1.0) > self._PROBABILITY_TOLERANCE:
            weights = weights / total
        index = self._rng.choice(len(states), p=weights)
        return states[index]

    def _apply_intervention_boost(
        self, row: dict[AttentionState, float], intervention_sensitivity: float
    ) -> dict[AttentionState, float]:
        """Shift probability mass toward Focused, scaled by `intervention_sensitivity`.

        Reuses `BehaviourGenerationConfig.intervention_recovery_weight` — the
        same "how strongly does this student respond to an intervention"
        coefficient Module 5 uses for fatigue recovery, applied here to
        transition probabilities instead, rather than introducing a second,
        near-duplicate config knob for the same underlying idea.
        """

        weight = self._config.behaviour_generation.intervention_recovery_weight
        boost = min(1.0, intervention_sensitivity * weight)

        boosted = {state: value * (1.0 - boost) for state, value in row.items()}
        boosted[AttentionState.FOCUSED] += boost

        total = sum(boosted.values())
        return {state: value / total for state, value in boosted.items()}
