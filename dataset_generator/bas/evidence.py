"""Module 9, Step 5: Evidence Mapping.

Converts normalized values (each in the feature's own [clip_min, clip_max]
range, typically [0,1]) into evidence on one consistent "more evidence of
attention" scale: `positive` polarity passes the value through unchanged,
`negative` polarity flips it (`1 - value`, i.e. "higher fatigue -> lower
evidence of attention"), and `neutral` features are excluded from
`BASEvidence.values` entirely — they're observed, not scored.
"""

from __future__ import annotations

from dataset_generator.bas.config import BASConfig, FeaturePolarity
from dataset_generator.bas.models import BASEvidence


def map_to_evidence(normalized: dict[str, float | None], config: BASConfig) -> BASEvidence:
    """Map `normalized` feature values to polarity-adjusted evidence."""

    values: dict[str, float] = {}
    missing: list[str] = []

    for feature, feature_config in config.feature_configs.items():
        if feature_config.polarity == FeaturePolarity.NEUTRAL:
            continue

        normalized_value = normalized.get(feature)
        if normalized_value is None:
            missing.append(feature)
            continue

        if feature_config.polarity == FeaturePolarity.POSITIVE:
            values[feature] = normalized_value
        else:  # NEGATIVE
            values[feature] = 1.0 - normalized_value

    return BASEvidence(values=values, missing_features=missing)
