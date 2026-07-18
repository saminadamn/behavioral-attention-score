"""Session validation (Module 6, Step 8).

Checks that need cross-record context spanning the whole session — ordering,
transition consistency, fatigue monotonicity, and history/statistics
agreement — none of which a single `BehaviourRecord`'s own field
constraints can express.
"""

from __future__ import annotations

from dataset_generator.models.session import SessionRecord

_FATIGUE_TOLERANCE = 1e-9


def validate_session(record: SessionRecord, fatigue_tolerance: float = _FATIGUE_TOLERANCE) -> list[str]:
    """Return a list of validation issues for `record`; empty means valid."""

    issues: list[str] = []
    interactions = record.interactions

    # Interaction ordering: contiguous 1..N, strictly increasing.
    expected_numbers = list(range(1, len(interactions) + 1))
    actual_numbers = [i.interaction_number for i in interactions]
    if actual_numbers != expected_numbers:
        issues.append(f"interaction_number sequence {actual_numbers} is not contiguous from 1")

    # Duplicate interaction IDs (response_id is unique per interaction).
    response_ids = [i.response.response_id for i in interactions]
    if len(response_ids) != len(set(response_ids)):
        issues.append("duplicate response_id found across interactions")

    # Transition validity: transition_history's to_state must match the
    # corresponding interaction's recorded attention_state.
    transitions_by_number = {t.interaction_number: t for t in record.transition_history}
    for interaction in interactions:
        transition = transitions_by_number.get(interaction.interaction_number)
        if transition is None:
            issues.append(f"interaction {interaction.interaction_number} has no matching transition event")
            continue
        if transition.to_state != interaction.behaviour.attention_state:
            issues.append(
                f"interaction {interaction.interaction_number}: transition.to_state "
                f"({transition.to_state}) != behaviour.attention_state ({interaction.behaviour.attention_state})"
            )

    # Monotonic fatigue: only allowed to decrease on an interaction where an
    # intervention was applied (Module 5: fatigue never decreases spontaneously).
    for previous, current in zip(interactions, interactions[1:]):
        if current.behaviour.intervention_applied:
            continue
        if current.behaviour.fatigue_level < previous.behaviour.fatigue_level - fatigue_tolerance:
            issues.append(
                f"fatigue decreased without intervention between interactions "
                f"{previous.interaction_number} and {current.interaction_number}"
            )

    # History / statistics consistency.
    if record.statistics.interaction_count != len(interactions):
        issues.append("statistics.interaction_count does not match len(interactions)")
    if record.summary.total_interactions != len(interactions):
        issues.append("summary.total_interactions does not match len(interactions)")
    if interactions and record.summary.final_fatigue != interactions[-1].behaviour.fatigue_level:
        issues.append("summary.final_fatigue does not match the last interaction's fatigue_level")

    return issues


def validate_session_batch(records: list[SessionRecord]) -> dict[str, list[str]]:
    """Validate every session; returns `{session_id: issues}` for invalid ones only."""

    return {
        record.session_id: issues
        for record in records
        if (issues := validate_session(record))
    }
