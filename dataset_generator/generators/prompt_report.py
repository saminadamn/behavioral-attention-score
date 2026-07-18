"""Prompt Validation Report (Module 3 improvement requested in review).

Built from an exported list of `Prompt`s, independent of any single
`PromptGenerator` instance's internal counters — so it can validate a batch
assembled from multiple generators, a loaded dataset, or a curriculum
snapshot equally well. Doubles as a debugging tool and a methodology-section
figure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from dataset_generator.models.prompt import Prompt
from dataset_generator.validators.prompt_validator import validate_prompt_batch


@dataclass(frozen=True)
class PromptValidationReport:
    total_prompts: int
    subject_distribution: dict[str, float]
    difficulty_distribution: dict[str, float]
    cognitive_level_distribution: dict[str, float]
    average_prompt_length_words: float
    duplicate_count: int
    duplicate_rate: float
    average_readability_grade: float
    invalid_prompt_count: int


def build_prompt_report(prompts: list[Prompt]) -> PromptValidationReport:
    """Compute a `PromptValidationReport` over `prompts`."""

    total = len(prompts)
    if total == 0:
        raise ValueError("cannot build a report for an empty prompt list")

    subject_counts = Counter(p.subject for p in prompts)
    difficulty_counts = Counter(p.difficulty.value for p in prompts)
    cognitive_counts = Counter(p.cognitive_level.value for p in prompts)
    text_counts = Counter(p.prompt_text for p in prompts)

    duplicate_count = sum(count - 1 for count in text_counts.values() if count > 1)
    invalid_count = len(validate_prompt_batch(prompts))

    return PromptValidationReport(
        total_prompts=total,
        subject_distribution={k: v / total for k, v in subject_counts.items()},
        difficulty_distribution={k: v / total for k, v in difficulty_counts.items()},
        cognitive_level_distribution={k: v / total for k, v in cognitive_counts.items()},
        average_prompt_length_words=sum(p.metadata.token_count for p in prompts) / total,
        duplicate_count=duplicate_count,
        duplicate_rate=duplicate_count / total,
        average_readability_grade=sum(p.metadata.readability_grade for p in prompts) / total,
        invalid_prompt_count=invalid_count,
    )


def render_report(report: PromptValidationReport) -> str:
    """Render `report` as a plain-text summary suitable for logs or a paper figure."""

    lines = [f"Total prompts: {report.total_prompts}", ""]

    lines.append("Subjects:")
    for key, value in sorted(report.subject_distribution.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    lines.append("Difficulty:")
    for key, value in sorted(report.difficulty_distribution.items()):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    lines.append("Bloom Levels:")
    for key, value in sorted(report.cognitive_level_distribution.items()):
        lines.append(f"  {key}: {value:.0%}")
    lines.append("")

    lines.append(f"Average prompt length: {report.average_prompt_length_words:.1f} words")
    lines.append(f"Duplicate rate: {report.duplicate_rate:.2%}")
    lines.append(f"Average readability: Grade {report.average_readability_grade:.1f}")

    return "\n".join(lines)
