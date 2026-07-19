"""Module 9, Step 4: Normalization.

Each feature is normalized independently using its own
`FeatureNormalizationConfig` — no cross-feature interaction here. A final
clip to `[clip_min, clip_max]` is applied after every strategy (including
`IDENTITY`), so a feature nominally "already in [0,1]" is still protected
against a stray out-of-range value.
"""

from __future__ import annotations

from dataset_generator.bas.config import BASConfig, FeatureNormalizationConfig, NormalizationStrategy
from dataset_generator.bas.models import BASObservation


def normalize_value(raw_value: float, config: FeatureNormalizationConfig) -> float:
    """Normalize one raw value per `config`'s strategy, then clip."""

    if config.strategy == NormalizationStrategy.IDENTITY:
        value = raw_value
    elif config.strategy == NormalizationStrategy.CLIP:
        value = raw_value
    elif config.strategy == NormalizationStrategy.MIN_MAX:
        span = config.max_value - config.min_value  # type: ignore[operator]
        value = (raw_value - config.min_value) / span  # type: ignore[operator]
    elif config.strategy == NormalizationStrategy.Z_SCORE:
        value = (raw_value - config.mean) / config.std  # type: ignore[operator]
        # z-scores are unbounded; rescale roughly onto [0,1] around 0 = mean,
        # +-3 std -> [0,1] edges, before the final clip below.
        value = (value + 3.0) / 6.0
    else:
        raise ValueError(f"unknown normalization strategy {config.strategy!r}")

    return min(config.clip_max, max(config.clip_min, value))


class Normalizer:
    """Normalizes a `BASObservation`'s raw values per `BASConfig.feature_configs`."""

    def __init__(self, config: BASConfig) -> None:
        self._config = config

    def normalize(self, observation: BASObservation) -> dict[str, float | None]:
        """Return `{feature: normalized_value}`; missing raw values stay `None`."""

        normalized: dict[str, float | None] = {}
        for feature, feature_config in self._config.feature_configs.items():
            raw_value = observation.raw_values.get(feature)
            if raw_value is None:
                normalized[feature] = None
                continue
            normalized[feature] = normalize_value(raw_value, feature_config.normalization)
        return normalized

    def metadata(self) -> dict[str, str]:
        """`{feature: strategy_name}` — recorded on `BASRecordMetadata` for traceability."""

        return {
            feature: cfg.normalization.strategy.value
            for feature, cfg in self._config.feature_configs.items()
        }
