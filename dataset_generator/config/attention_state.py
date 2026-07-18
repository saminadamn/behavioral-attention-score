"""`AttentionState` and Markov-reachability checking.

Split out from `schema.py` so both `schema.py` (which needs it for
`GeneratorConfig`) and sibling config modules that `schema.py` imports
(e.g. `response_generation.py`, which needs it for correctness modifiers)
can depend on it without a circular import.
"""

from __future__ import annotations

from enum import Enum


class AttentionState(str, Enum):
    """The three behavioural attention states. Not a clinical classification."""

    FOCUSED = "Focused"
    DISTRACTED = "Distracted"
    IMPULSIVE = "Impulsive"


# The dynamic behavioural features whose per-state distributions must be
# configured (Stage 2, Section 5 / Section 7 in the revised schema).
# `interaction_duration` was not part of the original Stage 2 feature list;
# it was added for Module 5 (Behaviour Generator), which needs a per-state
# distribution for it (Step 4: Highly Impulsive students have short
# interaction durations, distinct from other states).
BEHAVIOURAL_FEATURES: tuple[str, ...] = (
    "response_latency",
    "response_length",
    "topic_similarity",
    "sentiment",
    "engagement",
    "lexical_diversity",
    "topic_shift",
    "hesitation",
    "repetition_ratio",
    "interaction_duration",
)


def combine_transition_matrix(
    base: dict[AttentionState, dict[AttentionState, float]],
    modifiers: dict[AttentionState, dict[AttentionState, float]],
) -> dict[AttentionState, dict[AttentionState, float]]:
    """Apply additive `modifiers` to `base`, clip negatives to 0, renormalize each row.

    Shared by `GeneratorConfig`'s own validation (checking a profile's
    effective matrix stays reachable) and `TransitionEngine` (Module 6),
    which needs the identical computation at sampling time — extracted here,
    rather than duplicated in both places, specifically so those two call
    sites can never drift apart.

    Raises `ValueError` if any row's probabilities are all <= 0 after
    modifiers are applied (a degenerate row has nothing left to renormalize).
    """

    effective: dict[AttentionState, dict[AttentionState, float]] = {
        state: dict(base[state]) for state in AttentionState
    }
    for from_state, deltas in modifiers.items():
        for to_state, delta in deltas.items():
            effective[from_state][to_state] += delta

    for from_state, row in effective.items():
        clipped = {state: max(0.0, value) for state, value in row.items()}
        total = sum(clipped.values())
        if total <= 0:
            raise ValueError(
                f"degenerate transition row from {from_state.value} after modifiers "
                "(all probabilities <= 0)"
            )
        effective[from_state] = {state: value / total for state, value in clipped.items()}

    return effective


def reachability_violations(
    matrix: dict[AttentionState, dict[AttentionState, float]], tolerance: float = 1e-9
) -> list[str]:
    """Return a human-readable list of Markov-reachability violations in `matrix`.

    Every attention state must have at least one incoming transition from a
    *different* state and at least one outgoing transition to a *different*
    state (self-loops don't count). Otherwise the state is either
    unreachable or a trap once entered — an impossible/degenerate behavioural
    dynamic for this project. Returns an empty list if `matrix` is valid.
    """

    violations: list[str] = []
    states = list(AttentionState)
    for state in states:
        incoming = sum(matrix[other][state] for other in states if other != state)
        outgoing = sum(matrix[state][other] for other in states if other != state)
        if incoming <= tolerance:
            violations.append(f"{state.value} has no incoming transition from another state")
        if outgoing <= tolerance:
            violations.append(f"{state.value} has no outgoing transition to another state")
    return violations
