"""Quality validation for generated responses (Module 4, Step 8).

Rule-based, same scope caveat as `prompt_validator.py`: well-formedness
checks appropriate for templated synthetic text, not a grammar parser.

"Inconsistent with difficulty/Bloom level" is deliberately implemented as a
*wiring* check (does `response.metadata` actually match the `Prompt` it was
generated from), not a stylistic judgement of the text. A stylistic
minimum-length-by-difficulty rule would falsely reject the system's own
intended behaviour — Distracted/Impulsive strategies are designed to
produce short answers regardless of difficulty, so penalizing short Hard
answers would contradict Module 4 Step 3/4 by design.
"""

from __future__ import annotations

from collections import Counter

from dataset_generator.models.prompt import Prompt
from dataset_generator.models.response import Response


def validate_response(
    response: Response, prompt: Prompt | None = None, max_words: int = 200
) -> list[str]:
    """Return a list of validation issues for `response`; empty means valid.

    If `prompt` is given, also checks that `response.metadata` (difficulty,
    cognitive level, prompt_id) is actually consistent with it.
    """

    issues: list[str] = []
    text = response.response_text.strip()

    if not text:
        issues.append("empty response_text")
        return issues

    words = text.split()
    if len(words) > max_words:
        issues.append(f"impossible response length ({len(words)} words > {max_words})")
    if "{" in text or "}" in text:
        issues.append("unfilled template placeholder")
    if "  " in text:
        issues.append("response_text contains a double space")

    if prompt is not None:
        if response.prompt_id != prompt.prompt_id:
            issues.append("prompt_id does not match the prompt this response was generated from")
        if response.metadata.difficulty != prompt.difficulty:
            issues.append("metadata.difficulty inconsistent with prompt.difficulty")
        if response.metadata.cognitive_level != prompt.cognitive_level:
            issues.append("metadata.cognitive_level inconsistent with prompt.cognitive_level")

    return issues


def validate_response_batch(
    responses: list[Response], max_words: int = 200
) -> dict[str, list[str]]:
    """Validate every response; returns `{response_id: issues}` for invalid ones only."""

    return {
        response.response_id: issues
        for response in responses
        if (issues := validate_response(response, max_words=max_words))
    }


def is_immediate_repeat(candidate_text: str, previous_response_text: str | None) -> bool:
    """Whether `candidate_text` is a verbatim repeat of the student's own previous turn.

    Used for retry-avoidance during generation — a student repeating a short
    reaction like "I don't know." across *different* interactions is
    realistic and must not be penalized; repeating the *exact same* turn
    twice in a row is what Step 8 means by "duplicate responses" at
    generation time.
    """

    return previous_response_text is not None and candidate_text == previous_response_text


def duplicate_response_ids(responses: list[Response]) -> list[tuple[str, str]]:
    """Pairs of `(response_id, response_id)` sharing byte-identical `response_text`."""

    seen_text_to_id: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for response in responses:
        original_id = seen_text_to_id.get(response.response_text)
        if original_id is not None:
            duplicates.append((original_id, response.response_id))
        else:
            seen_text_to_id[response.response_text] = response.response_id
    return duplicates


def exact_duplicate_rate(responses: list[Response]) -> float:
    """Fraction of `responses` whose `response_text` is an exact repeat of an earlier one."""

    if not responses:
        return 0.0
    text_counts = Counter(r.response_text for r in responses)
    duplicate_count = sum(count - 1 for count in text_counts.values() if count > 1)
    return duplicate_count / len(responses)
