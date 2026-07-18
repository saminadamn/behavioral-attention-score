"""Quality validation for generated behavioural records (Module 5, Step 7).

Most impossible values (negative latency, engagement outside [0,1], ...)
are already prevented by `BehaviourRecord`'s Pydantic field constraints —
this module's job is the checks that need *cross-field* or *cross-record*
context Pydantic's per-field validation can't express, and to expose a
configurable, reusable entry point (`validate_behaviour_record`) rather than
scattering these rules through the generator.
"""

from __future__ import annotations

from dataset_generator.models.behaviour import BehaviourRecord


def validate_behaviour_record(
    record: BehaviourRecord, max_latency: float | None = None, max_interaction_duration: float | None = None
) -> list[str]:
    """Return a list of validation issues for `record`; empty means valid."""

    issues: list[str] = []

    if max_latency is not None and record.response_latency > max_latency:
        issues.append(f"response_latency ({record.response_latency:.2f}) exceeds max_latency ({max_latency})")
    if max_interaction_duration is not None and record.interaction_duration > max_interaction_duration:
        issues.append(
            f"interaction_duration ({record.interaction_duration:.2f}) exceeds "
            f"max_interaction_duration ({max_interaction_duration})"
        )
    if record.hesitation_duration > record.interaction_duration:
        issues.append("hesitation_duration exceeds interaction_duration")

    # "Invalid transitions": `transition_occurred` must agree with whether
    # `previous_attention_state` actually differs from this record's state.
    previous_state = record.metadata.previous_attention_state
    actually_transitioned = previous_state is not None and previous_state != record.attention_state
    if record.features.transition_occurred != actually_transitioned:
        issues.append("features.transition_occurred inconsistent with attention_state history")

    if record.features.fatigue_progression != record.fatigue_level:
        issues.append("features.fatigue_progression inconsistent with fatigue_level")
    if record.features.rolling_latency != record.rolling_latency:
        issues.append("features.rolling_latency inconsistent with rolling_latency")
    if record.features.rolling_engagement != record.rolling_engagement:
        issues.append("features.rolling_engagement inconsistent with rolling_engagement")

    return issues


def validate_behaviour_batch(
    records: list[BehaviourRecord],
    max_latency: float | None = None,
    max_interaction_duration: float | None = None,
) -> dict[str, list[str]]:
    """Validate every record; returns `{"student_id:interaction_number": issues}` for invalid ones."""

    result: dict[str, list[str]] = {}
    for record in records:
        issues = validate_behaviour_record(record, max_latency, max_interaction_duration)
        if issues:
            result[f"{record.student_id}:{record.interaction_number}"] = issues
    return result
