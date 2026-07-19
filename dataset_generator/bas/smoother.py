"""Module 9, Step 7: Temporal Smoothing.

A pure function, not a stateful object: `smooth()` takes the previous
smoothed value (or `None` for the first interaction) explicitly, matching
the `SessionContext`-threading pattern already used throughout this project
(Modules 4-6) rather than an object that must be reset between sessions.
"""

from __future__ import annotations

from dataset_generator.bas.config import BASConfig, SmoothingStrategy


def smooth(
    raw_score: float,
    previous_score: float | None,
    config: BASConfig,
    history: list[float] | None = None,
) -> float:
    """Compute this interaction's smoothed BAS.

    - `identity`: no smoothing, `raw_score` unchanged.
    - `ema`: exponential moving average with `config.ema_alpha`
      (`alpha * raw_score + (1 - alpha) * previous_score`); the first
      interaction of a session (`previous_score is None`) has nothing to
      smooth against, so it returns `raw_score` unchanged.
    - `rolling_average`: mean of the last `config.rolling_window` raw
      scores, `history` (including the current `raw_score`, most recent
      last) supplied by the caller.
    """

    if config.smoothing_strategy == SmoothingStrategy.IDENTITY:
        return raw_score

    if config.smoothing_strategy == SmoothingStrategy.EMA:
        if previous_score is None:
            return raw_score
        return config.ema_alpha * raw_score + (1.0 - config.ema_alpha) * previous_score

    if config.smoothing_strategy == SmoothingStrategy.ROLLING_AVERAGE:
        if not history:
            return raw_score
        window = history[-config.rolling_window :]
        return sum(window) / len(window)

    raise ValueError(f"unknown smoothing strategy {config.smoothing_strategy!r}")
