"""Module 9, Step 9: Explanation Engine.

Two explanation modes, chosen automatically by whether a previous
interaction's contributions are available:
- **Snapshot** (no previous record — the session's first interaction):
  describes this interaction's top positive/negative contributors.
- **Trend** (previous contributions given): compares each feature's
  contribution to the previous interaction and narrates which features
  moved the score up or down — the "BAS decreased because X, Y increased
  while Z decreased" style from the brief's example.
"""

from __future__ import annotations

from dataset_generator.bas.models import BASContribution

_READABLE_NAMES: dict[str, str] = {
    "correctness": "correctness",
    "semantic_similarity": "semantic similarity",
    "coherence": "coherence",
    "lexical_diversity": "lexical diversity",
    "confidence": "confidence",
    "hesitation": "hesitation",
    "topic_shift": "topic shift",
    "repetition_ratio": "repetition",
    "fatigue": "fatigue",
    "abs_normalized_latency": "response latency",
    "classifier_confidence": "classifier confidence",
}


def _readable(feature: str) -> str:
    return _READABLE_NAMES.get(feature, feature.replace("_", " "))


def top_contributors(
    contributions: list[BASContribution], k: int, positive: bool
) -> list[BASContribution]:
    """The `k` contributions with the largest (`positive=True`) or smallest
    (`positive=False`) signed `contribution`, i.e. the strongest boosters
    or the strongest drags on the score.
    """

    ordered = sorted(contributions, key=lambda c: c.contribution, reverse=positive)
    if positive:
        ordered = [c for c in ordered if c.contribution > 0]
    else:
        ordered = [c for c in ordered if c.contribution < 0]
    return ordered[:k]


def generate_explanation(
    contributions: list[BASContribution],
    previous_contributions: list[BASContribution] | None = None,
    top_k: int = 3,
) -> str:
    """Generate a natural-language explanation for `contributions`.

    `previous_contributions` (this feature's contribution values at the
    prior interaction) enables the trend-style explanation; without it,
    falls back to a static snapshot of the current interaction's strongest
    contributors.
    """

    if previous_contributions is None:
        positives = top_contributors(contributions, top_k, positive=True)
        negatives = top_contributors(contributions, top_k, positive=False)

        parts = []
        if positives:
            names = ", ".join(_readable(c.feature) for c in positives)
            parts.append(f"positively driven by {names}")
        if negatives:
            names = ", ".join(_readable(c.feature) for c in negatives)
            parts.append(f"negatively affected by {names}")

        if not parts:
            return "BAS reflects a mix of features with no single strong driver."
        return "BAS is " + " and ".join(parts) + "."

    previous_by_feature = {c.feature: c.contribution for c in previous_contributions}
    increased: list[str] = []
    decreased: list[str] = []

    for contribution in contributions:
        previous_value = previous_by_feature.get(contribution.feature)
        if previous_value is None:
            continue
        delta = contribution.contribution - previous_value
        if delta > 1e-6:
            increased.append(contribution.feature)
        elif delta < -1e-6:
            decreased.append(contribution.feature)

    total_delta = sum(c.contribution for c in contributions) - sum(previous_by_feature.values())
    direction = "increased" if total_delta > 1e-6 else "decreased" if total_delta < -1e-6 else "held steady"

    if not increased and not decreased:
        return f"BAS {direction}; no individual feature changed meaningfully."

    clauses = []
    if increased:
        clauses.append(f"{', '.join(_readable(f) for f in increased[:top_k])} improved")
    if decreased:
        clauses.append(f"{', '.join(_readable(f) for f in decreased[:top_k])} worsened")

    return f"BAS {direction} because " + " while ".join(clauses) + "."
