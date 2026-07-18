"""Module 4, Step 2: Prompt Analysis.

`Prompt` (Module 3) already stores subject/topic/difficulty/cognitive
level/keywords/etc. as structured fields — there is no raw text to parse.
`PromptAnalyzer` exists anyway as a deliberate architectural seam: it is the
*one* place that reads a `Prompt` into the shape the Response Generator and
its strategies consume (`PromptAnalysis`), computed once per prompt. Nothing
downstream re-reads `Prompt` fields directly or re-derives `topic_display`
independently — avoiding both repeated parsing and inconsistent derivations.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_generator.config.prompt_generation import CognitiveLevel, Difficulty
from dataset_generator.models.prompt import Prompt


@dataclass(frozen=True)
class PromptAnalysis:
    """Everything a Response Generator needs from one `Prompt`, extracted once."""

    prompt_id: str
    subject: str
    topic: str
    topic_display: str
    difficulty: Difficulty
    cognitive_level: CognitiveLevel
    expected_answer_type: str
    expected_response_length: int
    keywords: list[str]
    learning_objective: str
    prompt_text: str


class PromptAnalyzer:
    """Extracts a `PromptAnalysis` from a `Prompt`."""

    def analyze(self, prompt: Prompt) -> PromptAnalysis:
        """Read `prompt`'s structured fields into a `PromptAnalysis`, once."""

        return PromptAnalysis(
            prompt_id=prompt.prompt_id,
            subject=prompt.subject,
            topic=prompt.topic,
            topic_display=prompt.topic.replace("_", " "),
            difficulty=prompt.difficulty,
            cognitive_level=prompt.cognitive_level,
            expected_answer_type=prompt.expected_answer_type,
            expected_response_length=prompt.estimated_response_length,
            keywords=list(prompt.keywords),
            learning_objective=prompt.learning_objective,
            prompt_text=prompt.prompt_text,
        )
