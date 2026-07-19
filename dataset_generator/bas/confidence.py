"""Module 9, Step 8: Confidence Estimation.

Confidence combines three signals, each independently meaningful:
- **Coverage**: fraction of scored features that were actually observed
  (`1 - missing_feature_ratio`) — an interaction missing half its evidence
  shouldn't produce as trustworthy a score as one with full evidence.
- **Agreement**: low variance across the evidence values means the
  features agree on the interpretation (all "good" or all "bad"); high
  variance (some features say "focused", others say "distracted") means
  conflicting evidence, which should lower confidence even with full coverage.
- **Classifier confidence** (optional): if `classifier_confidence` was
  observed, blended in at `config.confidence_classifier_weight`.
"""

from __future__ import annotations

import statistics

from dataset_generator.bas.config import BASConfig
from dataset_generator.bas.models import BASEvidence


def compute_confidence(
    evidence: BASEvidence, config: BASConfig, classifier_confidence: float | None = None
) -> tuple[float, float, str]:
    """Return `(confidence, uncertainty, reliability_label)` for one interaction."""

    total_features = len(evidence.values) + len(evidence.missing_features)
    coverage = 1.0 if total_features == 0 else len(evidence.values) / total_features

    agreement_values = list(evidence.values.values())
    variance = statistics.pvariance(agreement_values) if len(agreement_values) >= 2 else 0.0

    base_confidence = coverage * (1.0 - min(1.0, config.confidence_variance_weight * variance))

    if classifier_confidence is not None:
        base_confidence = (
            (1.0 - config.confidence_classifier_weight) * base_confidence
            + config.confidence_classifier_weight * classifier_confidence
        )

    confidence = min(1.0, max(0.0, base_confidence))
    uncertainty = 1.0 - confidence

    if confidence >= 0.75:
        reliability = "high"
    elif confidence >= 0.40:
        reliability = "medium"
    else:
        reliability = "low"

    return confidence, uncertainty, reliability
