"""DQN agent over a recurrent (LSTM) Q-network, with two independently
toggleable upgrades over vanilla DQN:

- **Double DQN** (Van Hasselt, Guez & Silver, 2016): the online network
  selects the greedy next action, the target network evaluates it. Plain
  DQN uses `max_a' Q_target(s',a')` for both, which systematically
  overestimates action values whenever the target network's noise is
  correlated with which action looks best — decoupling selection from
  evaluation corrects that bias.
- **Prioritized Experience Replay** (Schaul et al., 2016): transitions are
  sampled proportional to their last-seen TD error rather than uniformly,
  with importance-sampling weights correcting the resulting bias.

Both default on (`DQNConfig.use_double_dqn` / `use_prioritized_replay`),
each independently switchable off for ablation.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.rl_experimental.config import DQNConfig
from dataset_generator.rl_experimental.network import RecurrentQNetwork
from dataset_generator.rl_experimental.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig) -> None:
        self._config = config
        self.action_dim = action_dim
        self.online_network = RecurrentQNetwork(
            state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed
        )
        self.target_network = RecurrentQNetwork(
            state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed
        )
        self.target_network.copy_weights_from(self.online_network)

        self.replay_buffer: PrioritizedReplayBuffer | ReplayBuffer
        if config.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                config.replay_capacity,
                seed=config.seed,
                alpha=config.per_alpha,
                beta_start=config.per_beta_start,
                beta_end=config.per_beta_end,
                beta_anneal_steps=config.per_beta_anneal_steps,
                priority_epsilon=config.per_priority_epsilon,
            )
        else:
            self.replay_buffer = ReplayBuffer(config.replay_capacity, seed=config.seed)

        self._rng = np.random.default_rng(config.seed)
        self._step_count = 0

    def epsilon(self) -> float:
        cfg = self._config
        fraction = min(1.0, self._step_count / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)

    def select_action(self, state_sequence: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon():
            return int(self._rng.integers(0, self.action_dim))
        q_values = self.online_network.predict(state_sequence[np.newaxis, ...])[0]
        return int(np.argmax(q_values))

    def bellman_targets(
        self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        """The (Double-)DQN bootstrap target `r + gamma * Q(s', argmax_a' ...) * (1-done)`.
        Public so callers (e.g. the trainer, computing post-hoc TD errors for
        reporting) don't need to reach into a private method.
        """
        cfg = self._config
        if cfg.use_double_dqn:
            online_next_q = self.online_network.predict(next_states)
            best_actions = np.argmax(online_next_q, axis=1)
            target_next_q = self.target_network.predict(next_states)
            selected_q = target_next_q[np.arange(len(best_actions)), best_actions]
        else:
            target_next_q = self.target_network.predict(next_states)
            selected_q = target_next_q.max(axis=1)

        return rewards + cfg.gamma * selected_q * (1.0 - dones)

    def train_on_batch(self) -> float | None:
        cfg = self._config
        if len(self.replay_buffer) < cfg.min_replay_size:
            return None

        batch, tree_indices, is_weights = self.replay_buffer.sample(cfg.batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float64)

        targets = self.bellman_targets(rewards, next_states, dones)
        loss, td_errors = self.online_network.train_step(
            states, actions, targets, cfg.learning_rate, sample_weights=is_weights
        )
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        self._step_count += 1
        if self._step_count % cfg.target_sync_every == 0:
            self.target_network.copy_weights_from(self.online_network)
        return loss
