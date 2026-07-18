"""Quality validation for generated prompts (Module 3, Step 9).

Intentionally rule-based, not a real grammar/NLP checker (see
`utils.text_metrics`'s module docstring for the same caveat) — these are
well-formedness checks appropriate for templated synthetic text: is the
placeholder filled in, is it long enough, does it look like a sentence.
"""

from __future__ import annotations

from collections import Counter

from dataset_generator.models.prompt import Prompt


def validate_prompt(prompt: Prompt, min_words: int = 3) -> list[str]:
    """Return a list of validation issues for `prompt`; empty means valid."""

    issues: list[str] = []
    text = prompt.prompt_text.strip()

    if not text:
        issues.append("empty prompt_text")
        return issues

    words = text.split()
    if len(words) < min_words:
        issues.append(f"prompt_text too short ({len(words)} words < {min_words})")
    if "{" in text or "}" in text:
        issues.append("unfilled template placeholder")
    if not text[0].isupper():
        issues.append("prompt_text does not start with a capital letter")
    if text[-1] not in ".?!":
        issues.append("prompt_text does not end with terminal punctuation")
    if "  " in text:
        issues.append("prompt_text contains a double space")
    if not prompt.keywords:
        issues.append("missing keywords")
    if not prompt.learning_objective.strip():
        issues.append("missing learning_objective")
    if prompt.metadata.token_count <= 0:
        issues.append("metadata.token_count is not positive")

    return issues


def validate_prompt_batch(prompts: list[Prompt], min_words: int = 3) -> dict[str, list[str]]:
    """Validate every prompt; returns `{prompt_id: issues}` for invalid prompts only."""

    return {
        prompt.prompt_id: issues
        for prompt in prompts
        if (issues := validate_prompt(prompt, min_words=min_words))
    }


def duplicate_prompt_ids(prompts: list[Prompt]) -> list[tuple[str, str]]:
    """Pairs of `(prompt_id, prompt_id)` sharing byte-identical `prompt_text`.

    Only the first prompt_id seen for a given text is treated as the
    "original"; every later prompt_id with the same text is paired with it.
    """

    seen_text_to_id: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for prompt in prompts:
        original_id = seen_text_to_id.get(prompt.prompt_text)
        if original_id is not None:
            duplicates.append((original_id, prompt.prompt_id))
        else:
            seen_text_to_id[prompt.prompt_text] = prompt.prompt_id
    return duplicates


def exact_duplicate_rate(prompts: list[Prompt]) -> float:
    """Fraction of `prompts` whose `prompt_text` is an exact repeat of an earlier one."""

    if not prompts:
        return 0.0
    text_counts = Counter(p.prompt_text for p in prompts)
    duplicate_count = sum(count - 1 for count in text_counts.values() if count > 1)
    return duplicate_count / len(prompts)
