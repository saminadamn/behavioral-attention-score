"""Experimental DQN hyperparameters — a typed, fingerprinted config, same
convention as every other config in this repository.

Covers three algorithmic pieces, each independently toggleable so an
ablation can isolate what each one contributes:

- Double DQN (Van Hasselt, Guez & Silver, 2016) — decouples action
  selection (online network) from action evaluation (target network) in
  the Bellman target, correcting vanilla DQN's maximization bias.
- Prioritized Experience Replay (Schaul et al., 2016) — samples
  high-TD-error transitions more often, via a sum-tree.
- An LSTM encoder (Hausknecht & Stone, 2015 — "DRQN") reading a short
  window of past states before the Q-head, so the agent sees temporal
  context (a trend) rather than one isolated interaction.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class DQNConfig(BaseModel):
    """Everything that controls training. Nothing here touches the
    intervention rule engine's own `InterventionConfig`.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 42

    # -- Recurrent (DRQN) encoder --
    sequence_length: int = Field(default=5, gt=0)
    lstm_hidden_dim: int = Field(default=16, gt=0)

    learning_rate: float = Field(default=0.01, gt=0.0)
    gamma: float = Field(default=0.9, ge=0.0, le=1.0)

    replay_capacity: int = Field(default=5000, gt=0)
    batch_size: int = Field(default=32, gt=0)
    min_replay_size: int = Field(default=64, gt=0)

    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay_steps: int = Field(default=2000, gt=0)

    target_sync_every: int = Field(default=50, gt=0)
    epochs: int = Field(default=5, gt=0)

    # -- Double DQN --
    use_double_dqn: bool = True

    # -- Prioritized Experience Replay --
    use_prioritized_replay: bool = True
    per_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    per_beta_start: float = Field(default=0.4, ge=0.0, le=1.0)
    per_beta_end: float = Field(default=1.0, ge=0.0, le=1.0)
    per_beta_anneal_steps: int = Field(default=2000, gt=0)
    per_priority_epsilon: float = Field(default=1e-3, gt=0.0)


def compute_dqn_config_fingerprint(config: DQNConfig) -> str:
    """A deterministic SHA-256 fingerprint of `config`, matching every
    other module's provenance convention.
    """

    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


def default_dqn_config() -> DQNConfig:
    return DQNConfig()
