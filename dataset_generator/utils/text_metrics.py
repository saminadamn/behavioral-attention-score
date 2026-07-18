"""Lightweight, dependency-free text metrics used by prompt (and later
response) generation and validation.

These are heuristic, not linguistically rigorous: `count_syllables` is a
vowel-group heuristic and `grammar`-adjacent checks elsewhere are rule-based
well-formedness checks, not a real parser. That's a deliberate scope choice
to avoid a heavy NLP dependency for synthetic, templated text — documented
here so it's never mistaken for a claim of true grammatical analysis.
"""

from __future__ import annotations

import re

_WORD_PATTERN = re.compile(r"[A-Za-z']+")
_SENTENCE_END_PATTERN = re.compile(r"[.!?]+")
_VOWEL_GROUP_PATTERN = re.compile(r"[aeiouyAEIOUY]+")


def word_tokenize(text: str) -> list[str]:
    """Extract word tokens (letters/apostrophes only) from `text`."""

    return _WORD_PATTERN.findall(text)


def word_count(text: str) -> int:
    return len(word_tokenize(text))


def sentence_count(text: str) -> int:
    """Count sentences by terminal punctuation, minimum 1 for non-empty text."""

    if not text.strip():
        return 0
    return max(1, len(_SENTENCE_END_PATTERN.findall(text)))


def count_syllables(word: str) -> int:
    """Heuristic syllable count: number of vowel groups, silent-e adjusted, min 1."""

    word = word.lower()
    groups = _VOWEL_GROUP_PATTERN.findall(word)
    count = len(groups)
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease score (higher = easier to read). 0.0 for empty text."""

    words = word_tokenize(text)
    if not words:
        return 0.0
    sentences = sentence_count(text)
    syllables = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level (approximate US school grade). 0.0 for empty text."""

    words = word_tokenize(text)
    if not words:
        return 0.0
    sentences = sentence_count(text)
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59


def estimate_reading_time_seconds(text: str, words_per_minute: float = 200.0) -> float:
    """Estimated silent-reading time for `text` at `words_per_minute`."""

    return (word_count(text) / words_per_minute) * 60.0


def token_jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity of `a` and `b`'s lowercase word-token sets. 0.0 if both empty."""

    tokens_a = {w.lower() for w in word_tokenize(a)}
    tokens_b = {w.lower() for w in word_tokenize(b)}
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)
