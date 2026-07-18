"""Lightweight, lexicon/heuristic text analysis for response scoring.

Like `text_metrics.py`, these are deliberately simple heuristics, not real
NLP models — a tiny fixed-lexicon sentiment score and an n-gram repetition
ratio. Documented here so they're never mistaken for a trained sentiment
classifier or a semantic repetition detector.
"""

from __future__ import annotations

from dataset_generator.utils.text_metrics import word_tokenize

_POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "good", "great", "clear", "easy", "correct", "confident", "interesting",
        "excellent", "understand", "understood", "makes", "sense", "sure", "yes",
        "love", "like", "enjoy", "fun", "obviously", "definitely",
    }
)

_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "bad", "hard", "confusing", "wrong", "unsure", "difficult", "lost",
        "boring", "sorry", "no", "not", "cant", "confused", "whatever",
    }
)


def simple_sentiment(text: str) -> float:
    """A fixed-lexicon sentiment score in [-1, 1]; 0.0 if no lexicon words match."""

    tokens = [t.lower() for t in word_tokenize(text)]
    if not tokens:
        return 0.0

    positive_count = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    negative_count = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    total = positive_count + negative_count
    if total == 0:
        return 0.0

    return (positive_count - negative_count) / total


def repetition_ratio(text: str) -> float:
    """Fraction of consecutive-word-pairs (bigrams) that repeat elsewhere in `text`."""

    tokens = [t.lower() for t in word_tokenize(text)]
    if len(tokens) < 2:
        return 0.0

    bigrams = list(zip(tokens, tokens[1:]))
    unique_bigrams = set(bigrams)
    repeated = len(bigrams) - len(unique_bigrams)
    return repeated / len(bigrams)


def hesitation_marker_count(text: str, phrases: list[str]) -> int:
    """Count (case-insensitive, substring) occurrences of any `phrases` in `text`."""

    lowered = text.lower()
    return sum(lowered.count(phrase.lower()) for phrase in phrases)


def concept_coverage(text: str, concepts: list[str]) -> float:
    """Fraction of `concepts` (case-insensitive substring match) present in `text`.

    Deliberately *not* a Jaccard similarity over the full prompt sentence:
    comparing a response's token set against an entire prompt sentence
    (which is mostly grammatical scaffolding — "explain", "why", "is",
    "important", "for", "understanding", ...) dilutes the union with words
    that were never going to appear in any response, driving similarity
    toward zero regardless of how on-topic the response actually is. This
    instead measures "how many of the prompt's core concepts (its keyword
    and topic) did the response actually reference" — a small, meaningful
    concept set — so a short but on-topic answer scores fairly and a
    genuinely unrelated answer ("I don't know") correctly scores near zero.
    """

    unique_concepts = list(dict.fromkeys(c.strip().lower() for c in concepts if c.strip()))
    if not unique_concepts:
        return 0.0
    lowered = text.lower()
    matched = sum(1 for concept in unique_concepts if concept in lowered)
    return matched / len(unique_concepts)


def find_hesitation_markers(text: str, phrases: list[str]) -> list[str]:
    """Which `phrases` (case-insensitive, substring match) actually appear in `text`.

    Returns each matching phrase once (in `phrases`' order), not per
    occurrence — e.g. "um... um, maybe" with `phrases=["um", "maybe"]`
    returns `["um", "maybe"]`, not `["um", "um", "maybe"]`.
    """

    lowered = text.lower()
    return [phrase for phrase in phrases if phrase.lower() in lowered]
