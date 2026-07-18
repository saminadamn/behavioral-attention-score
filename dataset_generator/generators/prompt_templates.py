"""Prompt wording templates, keyed by (cognitive level, difficulty).

Design decision: templates are parameterized by `{topic}` and `{keyword}`
rather than written per-subject — a subject/topic is just data substituted
into a wording pattern, so "every subject x difficulty x cognitive level"
coverage (Module 3, Step 5) comes from the cross product of curriculum
content and this template table, not from N*M*K hand-written strings.

Difficulty changes *which* template variant is chosen (different sentence
structure, more clauses, more reasoning demanded at Hard) rather than
appending a word like "(hard)" — see the Hard-tier stems below, which add
conditions, exceptions, or multi-step framing that Easy/Medium stems don't
have.

Each (cognitive level, difficulty) cell is built from one "topic-only" stem
(no `{keyword}` — matches the Bloom-level examples in the spec, e.g. "What
is photosynthesis?") plus three keyword-bearing stems, then multiplied by
four lead-in framings (`_LEAD_INS`) applied to the keyword-bearing stems.
That gives 16 raw templates per cell; combined with each topic's ~8
keywords, a cell's *text* capacity is large enough that generating
thousands of prompts stays low-duplicate (see `prompt_report`'s stress
test) without hand-writing over a hundred bespoke sentences.
"""

from __future__ import annotations

from dataset_generator.config.prompt_generation import CognitiveLevel, Difficulty

_LEAD_INS: tuple[str, ...] = (
    "Thinking carefully, {stem}",
    "As part of today's lesson, {stem}",
    "Here's a question for you: {stem}",
    "Take a moment to consider this: {stem}",
)


def _expand(topic_only_stem: str, keyword_stems: list[str]) -> list[str]:
    """Build one cell's full template list from a topic-only stem + keyword stems.

    Result = [topic_only_stem] + keyword_stems + every (lead_in, keyword_stem)
    combination, i.e. `1 + len(keyword_stems) * (1 + len(_LEAD_INS))` templates.
    """

    variants = [topic_only_stem, *keyword_stems]
    for lead_in in _LEAD_INS:
        for stem in keyword_stems:
            lowered = stem[0].lower() + stem[1:]
            variants.append(lead_in.format(stem=lowered))
    return variants


# ---------------------------------------------------------------------------
# Wording templates
# ---------------------------------------------------------------------------

TEMPLATES: dict[CognitiveLevel, dict[Difficulty, list[str]]] = {
    CognitiveLevel.REMEMBER: {
        Difficulty.EASY: _expand(
            "What is {topic}?",
            [
                "Define {keyword} in the context of {topic}.",
                "Name one key fact you recall about {keyword} in {topic}.",
                "Recall the meaning of {keyword} used in {topic}.",
            ],
        ),
        Difficulty.MEDIUM: _expand(
            "Recall the main definition of {topic} covered in class.",
            [
                "What is {keyword} and how does it relate to {topic}?",
                "Restate, in precise terms, what {keyword} means within {topic}.",
                "Recall how {keyword} was defined during the lesson on {topic}.",
            ],
        ),
        Difficulty.HARD: _expand(
            "State every essential fact about {topic} you can remember, in order.",
            [
                "Recall and precisely state the formal definition of {topic}, including {keyword}.",
                "Reconstruct, from memory, the full definition of {keyword} as it applies to {topic}.",
                "Recall every detail you were taught about {keyword} within {topic}, without omission.",
            ],
        ),
    },
    CognitiveLevel.UNDERSTAND: {
        Difficulty.EASY: _expand(
            "Explain what {topic} means in your own words.",
            [
                "Describe how {keyword} works in {topic}.",
                "In simple terms, explain what {keyword} means within {topic}.",
                "Describe, in your own words, the role of {keyword} in {topic}.",
            ],
        ),
        Difficulty.MEDIUM: _expand(
            "Summarize the main idea behind {topic}.",
            [
                "Explain why {keyword} is important for understanding {topic}.",
                "Explain how {keyword} helps clarify the concept of {topic}.",
                "Summarize, using {keyword} as an example, the main idea behind {topic}.",
            ],
        ),
        Difficulty.HARD: _expand(
            "Explain, in careful detail, the underlying logic of {topic}.",
            [
                "Explain in detail how {keyword} influences {topic}, including any exceptions.",
                "Summarize {topic}, addressing common misconceptions about {keyword}.",
                "Explain the relationship between {keyword} and {topic}, accounting for edge cases.",
            ],
        ),
    },
    CognitiveLevel.APPLY: {
        Difficulty.EASY: _expand(
            "Show how {topic} applies to an everyday example.",
            [
                "Use {keyword} to solve a simple problem about {topic}.",
                "Apply {keyword} to a basic example involving {topic}.",
                "Try using {keyword} to work through a short {topic} problem.",
            ],
        ),
        Difficulty.MEDIUM: _expand(
            "Demonstrate how to use {topic} in a new situation.",
            [
                "Apply {keyword} to solve a problem involving {topic}.",
                "Use {keyword} to work through an unfamiliar {topic} problem.",
                "Demonstrate step by step how {keyword} solves a {topic} problem.",
            ],
        ),
        Difficulty.HARD: _expand(
            "Solve a challenging, multi-step problem involving {topic}, showing all your work.",
            [
                "Apply {keyword} to solve a multi-step problem involving {topic}, showing all steps.",
                "Use {topic} to solve a complex, unfamiliar problem involving {keyword}.",
                "Apply {keyword} rigorously to an advanced {topic} problem, justifying each step.",
            ],
        ),
    },
    CognitiveLevel.ANALYZE: {
        Difficulty.EASY: _expand(
            "Break {topic} down into its main parts.",
            [
                "Compare {topic} and {keyword}.",
                "What are the differences between {topic} and {keyword}?",
                "Identify how {keyword} fits into the bigger picture of {topic}.",
            ],
        ),
        Difficulty.MEDIUM: _expand(
            "Analyze the underlying structure of {topic}.",
            [
                "Analyze how {keyword} affects {topic}.",
                "Break down {topic} into its key components involving {keyword}.",
                "Analyze the connection between {keyword} and the rest of {topic}.",
            ],
        ),
        Difficulty.HARD: _expand(
            "Critically analyze {topic} from multiple perspectives.",
            [
                "Critically analyze the relationship between {keyword} and {topic}, citing evidence.",
                "Compare and contrast {topic} with {keyword}, evaluating the strengths and weaknesses of each.",
                "Analyze how {keyword} changes your interpretation of {topic}, with justification.",
            ],
        ),
    },
    CognitiveLevel.EVALUATE: {
        Difficulty.EASY: _expand(
            "Judge whether {topic} is easy or hard to understand.",
            [
                "Do you think {keyword} is important for {topic}? Why?",
                "In your opinion, does {keyword} make {topic} easier to understand?",
                "Judge whether {keyword} is essential to understanding {topic}.",
            ],
        ),
        Difficulty.MEDIUM: _expand(
            "Assess the strengths and weaknesses of {topic}.",
            [
                "Evaluate the effectiveness of {keyword} in explaining {topic}.",
                "Assess how well {keyword} explains {topic}.",
                "Evaluate whether {keyword} is the best way to approach {topic}.",
            ],
        ),
        Difficulty.HARD: _expand(
            "Critically evaluate {topic} as a whole, and justify your judgment.",
            [
                "Critically evaluate {topic} using {keyword} as supporting evidence, and justify your conclusion.",
                "Assess and defend a position on {topic}, considering {keyword} and possible counterarguments.",
                "Critically evaluate how convincing {keyword} is as evidence about {topic}.",
            ],
        ),
    },
    CognitiveLevel.CREATE: {
        Difficulty.EASY: _expand(
            "Come up with your own example of {topic}.",
            [
                "Create your own example of {keyword} within {topic}.",
                "Invent a simple example that shows {keyword} in {topic}.",
                "Draw or describe a simple diagram showing {keyword} in {topic}.",
            ],
        ),
        Difficulty.MEDIUM: _expand(
            "Create a real-world scenario that uses {topic}.",
            [
                "Design a short activity that teaches {keyword} within {topic}.",
                "Create a scenario that demonstrates {keyword} in the context of {topic}.",
                "Design a simple game that helps someone practice {keyword} in {topic}.",
            ],
        ),
        Difficulty.HARD: _expand(
            "Design an original, comprehensive project that demonstrates mastery of {topic}.",
            [
                "Design an original experiment or project demonstrating {topic}, incorporating {keyword}.",
                "Create a comprehensive plan that applies {topic} and {keyword} to solve a novel problem.",
                "Design an innovative solution to a real problem, using {keyword} and {topic} together.",
            ],
        ),
    },
}

# ---------------------------------------------------------------------------
# Cognitive level -> expected answer shape (Step 1's `expected_answer_type`)
# ---------------------------------------------------------------------------

COGNITIVE_TO_ANSWER_TYPE: dict[CognitiveLevel, str] = {
    CognitiveLevel.REMEMBER: "short_answer",
    CognitiveLevel.UNDERSTAND: "explanation",
    CognitiveLevel.APPLY: "worked_example",
    CognitiveLevel.ANALYZE: "comparison",
    CognitiveLevel.EVALUATE: "justified_judgment",
    CognitiveLevel.CREATE: "design_proposal",
}

# ---------------------------------------------------------------------------
# Difficulty / cognitive-level -> target response length (tokens)
# ---------------------------------------------------------------------------

_DIFFICULTY_BASE_LENGTH: dict[Difficulty, int] = {
    Difficulty.EASY: 8,
    Difficulty.MEDIUM: 15,
    Difficulty.HARD: 24,
}

_COGNITIVE_LENGTH_BONUS: dict[CognitiveLevel, int] = {
    CognitiveLevel.REMEMBER: 0,
    CognitiveLevel.UNDERSTAND: 2,
    CognitiveLevel.APPLY: 4,
    CognitiveLevel.ANALYZE: 6,
    CognitiveLevel.EVALUATE: 8,
    CognitiveLevel.CREATE: 10,
}

_DIFFICULTY_SCORE_BASE: dict[Difficulty, float] = {
    Difficulty.EASY: 0.20,
    Difficulty.MEDIUM: 0.50,
    Difficulty.HARD: 0.80,
}


def estimated_response_length(difficulty: Difficulty, cognitive_level: CognitiveLevel) -> int:
    """Target answer length (tokens) for a "good" response to this prompt."""

    return _DIFFICULTY_BASE_LENGTH[difficulty] + _COGNITIVE_LENGTH_BONUS[cognitive_level]


def difficulty_score(difficulty: Difficulty, token_count: int) -> float:
    """A 0..1 difficulty score: enum base plus a small bump for longer/denser prompts."""

    base = _DIFFICULTY_SCORE_BASE[difficulty]
    bump = 0.02 * max(0, token_count - 8)
    return min(1.0, base + bump)
