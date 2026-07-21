"""Implicit Q-Learning (Kostrikov, Nair & Levine, 2021) — the second
offline-RL algorithm in this package. Where CQL makes Q-values conservative
by penalty, IQL sidesteps the extrapolation problem structurally: it never
evaluates Q at an action the critic itself chooses. Instead it fits a
state-value function V(s) as an *expectile* of Q(s,a) over actions that
actually appear in the data, then bootstraps Q's regression target off V(s')
— never off max_a' Q(s',a') or any action selection at all.

Two networks, two losses:

    Expectile value regression (Eq. 5/7 of the paper):
        L_V(psi) = E_(s,a)~D[ L2_tau( Q_target(s,a) - V_psi(s) ) ]
        L2_tau(u) = |tau - 1{u<0}| * u^2

    Q regression (ordinary MSE, target built from V — no max, no argmax):
        L_Q(theta) = E_(s,a,r,s')~D[ (Q_theta(s,a) - (r + gamma*(1-done)*V(s')))^2 ]

`tau > 0.5` (default 0.7, the paper's recommendation) makes V lean toward
the upper expectile of Q(s,a) over the data's action distribution — an
implicit, in-distribution approximation of "the best available action's
value" without ever asking Q to score an action nobody took.

**Simplification versus the full IQL paper, stated explicitly:** the
paper additionally extracts a policy via advantage-weighted regression
(AWR) so the *policy* itself never proposes an out-of-distribution
action. This package's action space is a small, fixed discrete set (8
policies), so the greedy policy here is `argmax_a Q(s,a)` directly — the
critic's targets are OOD-free by construction (that's what V achieves),
but this argmax step can still, in principle, select an action Q was
never trained to evaluate at that state. A full discrete AWR policy head
is listed as future work in `docs/EXPERIMENTAL_DQN.md`.
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
from dataset_generator.rl_experimental.environment import ACTION_NAMES, STATE_DIM, collect_transitions
from dataset_generator.rl_experimental.network import RecurrentQNetwork
from dataset_generator.rl_experimental.offline.common import compute_config_fingerprint
from dataset_generator.rl_experimental.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

SCHEMA_VERSION = "1.0"


class IQLConfig(BaseModel):
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

    use_prioritized_replay: bool = True
    per_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    per_beta_start: float = Field(default=0.4, ge=0.0, le=1.0)
    per_beta_end: float = Field(default=1.0, ge=0.0, le=1.0)
    per_beta_anneal_steps: int = Field(default=2000, gt=0)
    per_priority_epsilon: float = Field(default=1e-3, gt=0.0)

    iql_tau: float = Field(default=0.7, gt=0.0, lt=1.0, description="expectile — >0.5 leans toward optimism")


def default_iql_config() -> IQLConfig:
    return IQLConfig()


def expectile_weight(residual: np.ndarray, tau: float) -> np.ndarray:
    """`|tau - 1{residual<0}|` — the asymmetric weight that turns an
    ordinary MSE loss into an expectile regression when multiplied through
    the squared error. `residual = target - prediction`; weight is `tau`
    where the target exceeds the prediction, `1-tau` otherwise.
    """

    return np.where(residual > 0.0, tau, 1.0 - tau)


class IQLAgent:
    def __init__(self, state_dim: int, action_dim: int, config: IQLConfig) -> None:
        self._config = config
        self.action_dim = action_dim

        self.q_network = RecurrentQNetwork(state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed)
        self.q_target_network = RecurrentQNetwork(state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed)
        self.q_target_network.copy_weights_from(self.q_network)
        # V has a single output ("action_dim=1") — reusing RecurrentQNetwork
        # as a scalar value function, not a Q-function, is why every call
        # below indexes its output at action 0.
        self.v_network = RecurrentQNetwork(state_dim, 1, config.lstm_hidden_dim, seed=config.seed + 1)

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

    def predict_value(self, states: np.ndarray) -> np.ndarray:
        return self.v_network.predict(states)[:, 0]

    def train_on_batch(self) -> tuple[float, float] | None:
        """Returns `(value_loss, q_loss)`, or `None` while the buffer warms up."""

        cfg = self._config
        if len(self.replay_buffer) < cfg.min_replay_size:
            return None

        batch, tree_indices, is_weights = self.replay_buffer.sample(cfg.batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float64)

        # -- Value regression: V(s) toward the tau-expectile of Q_target(s,a) --
        q_target_all = self.q_target_network.predict(states)
        q_target_sa = q_target_all[np.arange(len(batch)), actions]

        v_pred = self.predict_value(states)
        residual = q_target_sa - v_pred
        weights = expectile_weight(residual, cfg.iql_tau) * is_weights

        value_loss, _ = self.v_network.train_step(
            states, np.zeros(len(batch), dtype=int), q_target_sa, cfg.learning_rate, sample_weights=weights
        )

        # -- Q regression: bootstrapped off V(s'), never off max/argmax Q --
        next_values = self.predict_value(next_states)
        q_targets = rewards + cfg.gamma * next_values * (1.0 - dones)
        q_loss, td_errors = self.q_network.train_step(
            states, actions, q_targets, cfg.learning_rate, sample_weights=is_weights
        )
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        self._step_count += 1
        if self._step_count % cfg.target_sync_every == 0:
            self.q_target_network.copy_weights_from(self.q_network)
        return value_loss, q_loss


class IQLTrainingArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    transitions_used: int = Field(ge=0)
    epochs_run: int = Field(ge=0)
    batches_trained: int = Field(ge=0)
    sequence_length: int = Field(gt=0)

    value_loss_per_epoch: list[float]
    q_loss_per_epoch: list[float]

    mean_value: float
    mean_q_value: float
    mean_advantage: float = Field(description="mean of Q(s, a_logged) - V(s) over the dataset")
    action_distribution: dict[str, float]
    greedy_policy_agreement_rate: float = Field(ge=0.0, le=1.0)

    config_fingerprint: str
    schema_version: str
    generation_timestamp: str

    disclaimer: str = (
        "Trained offline on logged transitions from the rule-based InterventionPlanner and "
        "a synthetic reward. The critic (V, Q) is trained without ever bootstrapping off an "
        "out-of-distribution action, per Implicit Q-Learning — but the extracted greedy "
        "policy (argmax_a Q) is a simplification of the paper's advantage-weighted policy "
        "extraction, documented in docs/EXPERIMENTAL_DQN.md. Not validated against any real "
        "outcome. Not used by the default pipeline."
    )


class IQLTrainer:
    def __init__(self, config: IQLConfig | None = None) -> None:
        self._config = config or default_iql_config()
        self._fingerprint = compute_config_fingerprint(self._config)

    def train(
        self,
        dataset_artifact: DatasetArtifact,
        bas_artifact: BASArtifact,
        reward_artifact: RewardArtifact,
        intervention_artifact: InterventionArtifact,
    ) -> IQLTrainingArtifact:
        cfg = self._config
        transitions = collect_transitions(
            dataset_artifact, bas_artifact, reward_artifact, intervention_artifact,
            sequence_length=cfg.sequence_length,
        )
        if not transitions:
            raise ValueError("No transitions could be built — check the intervention artifact is non-empty.")

        agent = IQLAgent(state_dim=STATE_DIM, action_dim=len(ACTION_NAMES), config=cfg)
        for t in transitions:
            agent.replay_buffer.push(t)

        value_loss_per_epoch: list[float] = []
        q_loss_per_epoch: list[float] = []
        batches_trained = 0

        for _ in range(cfg.epochs):
            v_losses: list[float] = []
            q_losses: list[float] = []
            steps_this_epoch = max(1, len(transitions) // cfg.batch_size)
            for _ in range(steps_this_epoch):
                result = agent.train_on_batch()
                if result is not None:
                    v_loss, q_loss = result
                    v_losses.append(v_loss)
                    q_losses.append(q_loss)
                    batches_trained += 1
            value_loss_per_epoch.append(float(np.mean(v_losses)) if v_losses else 0.0)
            q_loss_per_epoch.append(float(np.mean(q_losses)) if q_losses else 0.0)

        states = np.stack([t.state for t in transitions])
        logged_actions = np.array([t.action for t in transitions])
        q_values = agent.q_network.predict(states)
        values = agent.predict_value(states)
        greedy_actions = np.argmax(q_values, axis=1)

        logged_q = q_values[np.arange(len(transitions)), logged_actions]
        mean_advantage = float(np.mean(logged_q - values))

        agreement_rate = float(np.mean(greedy_actions == logged_actions))
        action_counts = Counter(ACTION_NAMES[a] for a in greedy_actions)
        action_distribution = {name: action_counts.get(name, 0) / len(greedy_actions) for name in ACTION_NAMES}

        return IQLTrainingArtifact(
            algorithm=f"IQL (tau={cfg.iql_tau}) + LSTM(seq_len={cfg.sequence_length})",
            transitions_used=len(transitions),
            epochs_run=cfg.epochs,
            batches_trained=batches_trained,
            sequence_length=cfg.sequence_length,
            value_loss_per_epoch=value_loss_per_epoch,
            q_loss_per_epoch=q_loss_per_epoch,
            mean_value=float(np.mean(values)),
            mean_q_value=float(np.mean(q_values)),
            mean_advantage=mean_advantage,
            action_distribution=action_distribution,
            greedy_policy_agreement_rate=agreement_rate,
            config_fingerprint=self._fingerprint,
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )
