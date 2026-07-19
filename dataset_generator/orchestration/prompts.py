"""Module 12, Step 6 support: static Tutor action-type labels and message templates.

No LLM is involved anywhere in this module — these are deterministic string
templates keyed by policy name, not generated text. Keeping them here (not
inline in `agents.py`) means `TutorAgent` stays a pure translation step: read
a decision, look up a template, format it.
"""

from __future__ import annotations

POLICY_ACTION_TYPES: dict[str, str] = {
    "NoInterventionPolicy": "No Intervention",
    "HintPolicy": "Hint",
    "ConceptReviewPolicy": "Concept Review",
    "DifficultyReductionPolicy": "Difficulty Reduction",
    "MotivationalPromptPolicy": "Encouragement",
    "BreakRecommendationPolicy": "Break Suggestion",
    "EncouragementPolicy": "Encouragement",
    "QuestionReframingPolicy": "Question Reframing",
}


def action_type_for_policy(policy_name: str) -> str:
    """The human-readable action-type label for `policy_name`.

    Falls back to the raw policy name for any policy not in the table above
    (defensive only — `test_orchestration.py` asserts the table covers every
    policy `InterventionPolicyFactory` currently registers).
    """

    return POLICY_ACTION_TYPES.get(policy_name, policy_name)


def format_tutor_message(policy_name: str, chosen_reason: str) -> str:
    """Format a tutor-facing message from an already-decided policy + reason.

    `chosen_reason` is `InterventionDecision.chosen_reason` — text the
    intervention policy already generated in Module 11. This function does
    not add new justification, only a consistent action-type prefix.
    """

    action_type = action_type_for_policy(policy_name)
    return f"[{action_type}] {chosen_reason}"
