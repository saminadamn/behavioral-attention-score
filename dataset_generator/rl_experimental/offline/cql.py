"""Conservative Q-Learning (Kumar, Zhou, Tucker & Levine, 2020) — the
first of three algorithms in this package purpose-built for offline data,
as opposed to Double DQN/PER/LSTM, which improve *how fast and stably* a
network fits logged transitions without addressing *what happens on
actions the logging policy never took*. CQL directly targets that gap: it
adds a penalty that pushes Q-values for out-of-distribution actions down
and Q-values for the logged (in-distribution) action up, producing a
provably-conservative (lower-bounded) value estimate — the standard
counter to the offline extrapolation-error problem documented in
`docs/EXPERIMENTAL_DQN.md`.

Discrete-action CQL(H) loss (Eq. 4 of the paper, discrete-action case):

    L(theta) = alpha * E_s[ log sum_a exp(Q(s,a)) - Q(s, a_data) ]  +  L_TD(theta)

The first term is minimized when Q is large only at the logged action and
small everywhere else — it does not require sampling or importance
weights the way the continuous-action version does, which is why the
discrete case is implemented directly here rather than approximated.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.bas.models import BASArtifact
from dataset_generator.intervention.models import InterventionArtifact
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.reward.models import RewardArtifact
from dataset_generator.rl_experimental.agent import double_dqn_bellman_target
from dataset_generator.rl_experimental.environment import ACTION_NAMES, STATE_DIM, collect_transitions
from dataset_generator.rl_experimental.network import RecurrentQNetwork
from dataset_generator.rl_experimental.offline.common import compute_config_fingerprint
from dataset_generator.rl_experimental.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

SCHEMA_VERSION = "1.0"


class CQLConfig(BaseModel):
    """Same recurrent-network / replay / Double-DQN machinery as `DQNConfig`,
    plus `cql_alpha` — the one new hyperparameter CQL introduces.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    sequence_length: int = Field(default=5, gt=0)
    lstm_hidden_dim: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=0.01, gt=0.0)
    gamma: float = Field(default=0.9, ge=0.0, le=1.0)

    replay_capacity: int = Field(default=5000, gt=0)
    batch_size: int = Field(default=32, gt=0)
    min_replay_size: int = Field(default=64, gt=0)
    target_sync_every: int = Field(default=50, gt=0)
    epochs: int = Field(default=5, gt=0)

    use_double_dqn: bool = True
    use_prioritized_replay: bool = True
    per_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    per_beta_start: float = Field(default=0.4, ge=0.0, le=1.0)
    per_beta_end: float = Field(default=1.0, ge=0.0, le=1.0)
    per_beta_anneal_steps: int = Field(default=2000, gt=0)
    per_priority_epsilon: float = Field(default=1e-3, gt=0.0)

    cql_alpha: float = Field(default=1.0, ge=0.0)


def default_cql_config() -> CQLConfig:
    return CQLConfig()


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _logsumexp(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).squeeze(axis=1)


class CQLAgent:
    """A Double-DQN agent (same target computation, same replay mechanics)
    with the CQL conservative penalty added to the training gradient.
    Deliberately not built as a subclass of `DQNAgent` — CQL is a distinct
    algorithm with its own gradient formula, and keeping it in its own
    class means this file alone tells the whole story for a thesis
    chapter, at the cost of a small amount of bookkeeping duplication
    with `DQNAgent` (replay buffer setup, target-network sync).
    """

    def __init__(self, state_dim: int, action_dim: int, config: CQLConfig) -> None:
        self._config = config
        self.action_dim = action_dim
        self.online_network = RecurrentQNetwork(state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed)
        self.target_network = RecurrentQNetwork(state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed)
        self.target_network.copy_weights_from(self.online_network)

        self.replay_buffer: PrioritizedReplayBuffer | ReplayBuffer
        if config.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                config.replay_capacity, seed=config.seed, alpha=config.per_alpha,
                beta_start=config.per_beta_start, beta_end=config.per_beta_end,
                beta_anneal_steps=config.per_beta_anneal_steps, priority_epsilon=config.per_priority_epsilon,
            )
        else:
            self.replay_buffer = ReplayBuffer(config.replay_capacity, seed=config.seed)

        self._step_count = 0

    def bellman_targets(self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> np.ndarray:
        cfg = self._config
        return double_dqn_bellman_target(
            self.online_network, self.target_network, rewards, next_states, dones,
            gamma=cfg.gamma, use_double_dqn=cfg.use_double_dqn,
        )

    def train_on_batch(self) -> tuple[float, float] | None:
        """Returns `(td_loss, cql_penalty)`, or `None` if the buffer is still warming up."""

        cfg = self._config
        if len(self.replay_buffer) < cfg.min_replay_size:
            return None

        batch, tree_indices, is_weights = self.replay_buffer.sample(cfg.batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float64)
        batch_size = len(batch)

        targets = self.bellman_targets(rewards, next_states, dones)

        q, cache = self.online_network.forward(states)
        predicted = q[np.arange(batch_size), actions]
        td_errors = predicted - targets
        td_loss = float(np.mean(is_weights * td_errors**2))
        grad_q_td = np.zeros_like(q)
        grad_q_td[np.arange(batch_size), actions] = 2.0 * is_weights * td_errors / batch_size

        # CQL(H) penalty: alpha * (logsumexp_a Q(s,a) - Q(s,a_data)), averaged
        # over the batch. Gradient wrt Q(s,a'): softmax(Q(s,.))[a'] - 1{a'=a_data}.
        cql_penalty = float(np.mean(_logsumexp(q) - predicted))
        grad_q_cql = _softmax(q).copy()
        grad_q_cql[np.arange(batch_size), actions] -= 1.0
        grad_q_cql *= cfg.cql_alpha / batch_size

        total_grad_q = grad_q_td + grad_q_cql
        self.online_network.apply_output_gradient(cache, total_grad_q, cfg.learning_rate)
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        self._step_count += 1
        if self._step_count % cfg.target_sync_every == 0:
            self.target_network.copy_weights_from(self.online_network)
        return td_loss, cql_penalty


class CQLTrainingArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    transitions_used: int = Field(ge=0)
    epochs_run: int = Field(ge=0)
    batches_trained: int = Field(ge=0)
    sequence_length: int = Field(gt=0)

    td_loss_per_epoch: list[float]
    cql_penalty_per_epoch: list[float]

    mean_q_value: float
    mean_ood_action_gap: float = Field(
        description="mean over states of (Q(logged action) - mean Q(other actions)) — "
        "how much CQL has separated in-distribution from out-of-distribution actions"
    )
    action_distribution: dict[str, float]
    greedy_policy_agreement_rate: float = Field(ge=0.0, le=1.0)

    config_fingerprint: str
    schema_version: str
    generation_timestamp: str

    disclaimer: str = (
        "Trained offline on logged transitions from the rule-based InterventionPlanner "
        "and a synthetic reward. The CQL penalty makes Q-values conservative with respect "
        "to actions absent from the logs, but does not create information about what those "
        "actions would have done. Not validated against any real outcome. Not used by the "
        "default pipeline."
    )


class CQLTrainer:
    def __init__(self, config: CQLConfig | None = None) -> None:
        self._config = config or default_cql_config()
        self._fingerprint = compute_config_fingerprint(self._config)

    def train(
        self,
        dataset_artifact: DatasetArtifact,
        bas_artifact: BASArtifact,
        reward_artifact: RewardArtifact,
        intervention_artifact: InterventionArtifact,
    ) -> CQLTrainingArtifact:
        cfg = self._config
        transitions = collect_transitions(
            dataset_artifact, bas_artifact, reward_artifact, intervention_artifact,
            sequence_length=cfg.sequence_length,
        )
        if not transitions:
            raise ValueError("No transitions could be built — check the intervention artifact is non-empty.")

        agent = CQLAgent(state_dim=STATE_DIM, action_dim=len(ACTION_NAMES), config=cfg)
        for t in transitions:
            agent.replay_buffer.push(t)

        td_loss_per_epoch: list[float] = []
        cql_penalty_per_epoch: list[float] = []
        batches_trained = 0

        for _ in range(cfg.epochs):
            td_losses: list[float] = []
            penalties: list[float] = []
            steps_this_epoch = max(1, len(transitions) // cfg.batch_size)
            for _ in range(steps_this_epoch):
                result = agent.train_on_batch()
                if result is not None:
                    td_loss, penalty = result
                    td_losses.append(td_loss)
                    penalties.append(penalty)
                    batches_trained += 1
            td_loss_per_epoch.append(float(np.mean(td_losses)) if td_losses else 0.0)
            cql_penalty_per_epoch.append(float(np.mean(penalties)) if penalties else 0.0)

        states = np.stack([t.state for t in transitions])
        logged_actions = np.array([t.action for t in transitions])
        q_values = agent.online_network.predict(states)
        greedy_actions = np.argmax(q_values, axis=1)

        logged_q = q_values[np.arange(len(transitions)), logged_actions]
        other_mask = np.ones_like(q_values, dtype=bool)
        other_mask[np.arange(len(transitions)), logged_actions] = False
        mean_other_q = np.where(other_mask, q_values, np.nan)
        mean_other_q = np.nanmean(mean_other_q, axis=1)
        ood_gap = float(np.mean(logged_q - mean_other_q))

        agreement_rate = float(np.mean(greedy_actions == logged_actions))
        action_counts = Counter(ACTION_NAMES[a] for a in greedy_actions)
        action_distribution = {name: action_counts.get(name, 0) / len(greedy_actions) for name in ACTION_NAMES}

        return CQLTrainingArtifact(
            algorithm=f"CQL (alpha={cfg.cql_alpha}) + Double DQN + PER + LSTM(seq_len={cfg.sequence_length})",
            transitions_used=len(transitions),
            epochs_run=cfg.epochs,
            batches_trained=batches_trained,
            sequence_length=cfg.sequence_length,
            td_loss_per_epoch=td_loss_per_epoch,
            cql_penalty_per_epoch=cql_penalty_per_epoch,
            mean_q_value=float(np.mean(q_values)),
            mean_ood_action_gap=ood_gap,
            action_distribution=action_distribution,
            greedy_policy_agreement_rate=agreement_rate,
            config_fingerprint=self._fingerprint,
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )
