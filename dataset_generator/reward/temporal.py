"""Module 10, Step 6: Temporal Credit Assignment.

Deterministic arithmetic over an already-computed sequence of raw rewards
— not a reinforcement-learning algorithm, just the credit-assignment
transform an eventual RL-based intervention engine would consume.

`discounted` and `moving_average` look in **opposite directions**, which is
worth being precise about:
- `discounted` computes a standard backward discounted return
  (`G_t = r_t + gamma * G_{t+1}`, normalized by `(1 - gamma)` to stay
  bounded) — interaction t's credit-assigned reward is influenced by
  *future* interactions in the same session, decayed by `gamma`.
- `moving_average` is causal smoothing over *past* raw rewards only (the
  same rolling-window idea as `bas.smoother`, applied here to rewards).
- `immediate` is a no-op passthrough.
"""

from __future__ import annotations

from dataset_generator.reward.config import RewardConfig, TemporalMode


def apply_temporal_credit_assignment(raw_rewards: list[float], config: RewardConfig) -> list[float]:
    """Apply `config.temporal_mode`'s credit assignment across one session's `raw_rewards`
    (already ordered by `interaction_number`).
    """

    if config.temporal_mode == TemporalMode.IMMEDIATE:
        return list(raw_rewards)

    if config.temporal_mode == TemporalMode.DISCOUNTED:
        return _discounted_returns(raw_rewards, config.discount_factor)

    if config.temporal_mode == TemporalMode.MOVING_AVERAGE:
        return _moving_averages(raw_rewards, config.rolling_window)

    raise ValueError(f"unknown temporal mode {config.temporal_mode!r}")


def _discounted_returns(raw_rewards: list[float], gamma: float) -> list[float]:
    """Backward discounted return, normalized by `(1 - gamma**remaining)` to stay
    within the range of the underlying `raw_rewards` regardless of session length.
    """

    n = len(raw_rewards)
    returns = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        running = raw_rewards[t] + gamma * running
        remaining = n - t
        normalizer = (1.0 - gamma**remaining) / (1.0 - gamma) if gamma < 1.0 else float(remaining)
        returns[t] = running / normalizer if normalizer > 0 else raw_rewards[t]
    return returns


def _moving_averages(raw_rewards: list[float], window: int) -> list[float]:
    """Causal rolling mean: interaction t's value is the mean of the last
    `window` raw rewards ending at (and including) t.
    """

    result = []
    for t in range(len(raw_rewards)):
        start = max(0, t - window + 1)
        segment = raw_rewards[start : t + 1]
        result.append(sum(segment) / len(segment))
    return result
