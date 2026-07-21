"""Batch-Constrained Q-Learning (Fujimoto, Meger & Precup, 2019), discrete-
action variant (following the simplified formulation in Fujimoto, Conti,
Ghavamzadeh & Pineau, "Benchmarking Batch Deep Reinforcement Learning
Algorithms," 2019, sec. 3 — the original BCQ paper's actor-based
perturbation model applies to continuous actions; the discrete case only
needs a behavior model and a support threshold).

BCQ's idea, third and last of the "genuinely offline" family in this
package: constrain the actions Q is ever *asked to maximize over* to
those a behavior-cloning model judges plausible under the logged data,
rather than penalizing Q afterward (CQL) or avoiding the bootstrap
altogether (IQL).

Two models:

    Behavior model  G_omega(a|s): a softmax classifier over the 8 policies,
                     trained by ordinary cross-entropy on (s, a_logged) pairs
                     — literally "what would the rule engine have done here."

    Q(s,a):          trained exactly as Double DQN, EXCEPT the target's
                     max/argmax is restricted to the support set
                         A_support(s) = { a : G_omega(a|s) / max_a' G_omega(a'|s) >= threshold }
                     i.e. actions at least `threshold` as likely, relative to
                     the behavior model's own top choice, as the behavior
                     model's most likely action at that state. Actions
                     outside this set are masked to -infinity before the
                     max, so Q is never bootstrapped off (and the greedy
                     policy never selects) an action the data essentially
                     never supports at that state.

`threshold=0.3` is the paper's stated default for the discrete case.
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


class BCQConfig(BaseModel):
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
    behavior_epochs: int = Field(
        default=5, gt=0, description="epochs to pretrain the behavior-cloning model before Q training starts"
    )

    use_prioritized_replay: bool = True
    per_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    per_beta_start: float = Field(default=0.4, ge=0.0, le=1.0)
    per_beta_end: float = Field(default=1.0, ge=0.0, le=1.0)
    per_beta_anneal_steps: int = Field(default=2000, gt=0)
    per_priority_epsilon: float = Field(default=1e-3, gt=0.0)

    bcq_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


def default_bcq_config() -> BCQConfig:
    return BCQConfig()


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def support_mask(behavior_probs: np.ndarray, threshold: float) -> np.ndarray:
    """`True` where an action is at least `threshold` as likely as the
    behavior model's own top action at that state — the BCQ support set.
    The behavior model's own argmax always passes (ratio 1.0 >= threshold),
    so the mask is never entirely empty.
    """

    max_prob = behavior_probs.max(axis=1, keepdims=True)
    return behavior_probs >= threshold * max_prob


def masked_bellman_targets(
    online_network: RecurrentQNetwork,
    target_network: RecurrentQNetwork,
    behavior_network: RecurrentQNetwork,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    threshold: float,
) -> np.ndarray:
    """Double-DQN-style target, but `argmax_a'` is restricted to the
    behavior model's support set at `s'` — the one mechanical difference
    from `double_dqn_bellman_target`.
    """

    behavior_logits = behavior_network.predict(next_states)
    behavior_probs = _softmax(behavior_logits)
    mask = support_mask(behavior_probs, threshold)

    online_next_q = online_network.predict(next_states)
    masked_online_q = np.where(mask, online_next_q, -np.inf)
    best_actions = np.argmax(masked_online_q, axis=1)

    target_next_q = target_network.predict(next_states)
    selected_q = target_next_q[np.arange(len(best_actions)), best_actions]
    return rewards + gamma * selected_q * (1.0 - dones)


class BCQAgent:
    def __init__(self, state_dim: int, action_dim: int, config: BCQConfig) -> None:
        self._config = config
        self.action_dim = action_dim

        self.online_network = RecurrentQNetwork(state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed)
        self.target_network = RecurrentQNetwork(state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed)
        self.target_network.copy_weights_from(self.online_network)
        self.behavior_network = RecurrentQNetwork(state_dim, action_dim, config.lstm_hidden_dim, seed=config.seed + 1)

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

    def train_behavior_on_batch(self) -> float | None:
        """Cross-entropy step for the behavior-cloning model — pretrained
        before Q-learning starts, since the support set it defines is what
        Q's targets are constrained by from the very first Q update.
        """

        cfg = self._config
        if len(self.replay_buffer) < cfg.min_replay_size:
            return None

        batch, _, _ = self.replay_buffer.sample(cfg.batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        batch_size = len(batch)

        logits, cache = self.behavior_network.forward(states)
        probs = _softmax(logits)
        loss = float(-np.mean(np.log(np.clip(probs[np.arange(batch_size), actions], 1e-12, 1.0))))

        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), actions] -= 1.0
        grad_logits /= batch_size

        self.behavior_network.apply_output_gradient(cache, grad_logits, cfg.learning_rate)
        return loss

    def train_q_on_batch(self) -> float | None:
        cfg = self._config
        if len(self.replay_buffer) < cfg.min_replay_size:
            return None

        batch, tree_indices, is_weights = self.replay_buffer.sample(cfg.batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float64)

        targets = masked_bellman_targets(
            self.online_network, self.target_network, self.behavior_network,
            rewards, next_states, dones, gamma=cfg.gamma, threshold=cfg.bcq_threshold,
        )
        loss, td_errors = self.online_network.train_step(
            states, actions, targets, cfg.learning_rate, sample_weights=is_weights
        )
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        self._step_count += 1
        if self._step_count % cfg.target_sync_every == 0:
            self.target_network.copy_weights_from(self.online_network)
        return loss

    def masked_greedy_action(self, state_sequence: np.ndarray) -> int:
        """The action selection BCQ actually licenses: argmax Q restricted
        to the behavior model's support set at this state — used for
        reporting the induced policy, not just Q's unconstrained argmax.
        """

        behavior_probs = _softmax(self.behavior_network.predict(state_sequence[np.newaxis, ...]))[0]
        mask = behavior_probs >= self._config.bcq_threshold * behavior_probs.max()
        q = self.online_network.predict(state_sequence[np.newaxis, ...])[0]
        masked_q = np.where(mask, q, -np.inf)
        return int(np.argmax(masked_q))


class BCQTrainingArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    transitions_used: int = Field(ge=0)
    behavior_epochs_run: int = Field(ge=0)
    epochs_run: int = Field(ge=0)
    batches_trained: int = Field(ge=0)
    sequence_length: int = Field(gt=0)

    behavior_loss_per_epoch: list[float]
    q_loss_per_epoch: list[float]

    mean_q_value: float
    mean_support_set_size: float = Field(
        ge=0.0, description="mean number of actions per state that pass the BCQ support threshold"
    )
    action_distribution: dict[str, float]
    unconstrained_greedy_agreement_rate: float = Field(
        ge=0.0, le=1.0, description="argmax_a Q(s,a) agreement with the logged action, ignoring the support mask"
    )
    constrained_greedy_agreement_rate: float = Field(
        ge=0.0, le=1.0, description="support-masked argmax agreement with the logged action — BCQ's actual induced policy"
    )

    config_fingerprint: str
    schema_version: str
    generation_timestamp: str

    disclaimer: str = (
        "Trained offline on logged transitions from the rule-based InterventionPlanner and "
        "a synthetic reward. Q is never bootstrapped off, and the induced policy never "
        "selects, an action outside the behavior model's learned support set at a given "
        "state — but the support set is only as good as the behavior model's own fit to the "
        "logged data. Not validated against any real outcome. Not used by the default pipeline."
    )


class BCQTrainer:
    def __init__(self, config: BCQConfig | None = None) -> None:
        self._config = config or default_bcq_config()
        self._fingerprint = compute_config_fingerprint(self._config)

    def train(
        self,
        dataset_artifact: DatasetArtifact,
        bas_artifact: BASArtifact,
        reward_artifact: RewardArtifact,
        intervention_artifact: InterventionArtifact,
    ) -> BCQTrainingArtifact:
        cfg = self._config
        transitions = collect_transitions(
            dataset_artifact, bas_artifact, reward_artifact, intervention_artifact,
            sequence_length=cfg.sequence_length,
        )
        if not transitions:
            raise ValueError("No transitions could be built — check the intervention artifact is non-empty.")

        agent = BCQAgent(state_dim=STATE_DIM, action_dim=len(ACTION_NAMES), config=cfg)
        for t in transitions:
            agent.replay_buffer.push(t)

        # Pretrain the behavior model first — Q's targets depend on its
        # support set from the first Q update onward.
        behavior_loss_per_epoch: list[float] = []
        steps_per_epoch = max(1, len(transitions) // cfg.batch_size)
        for _ in range(cfg.behavior_epochs):
            losses = [loss for _ in range(steps_per_epoch) if (loss := agent.train_behavior_on_batch()) is not None]
            behavior_loss_per_epoch.append(float(np.mean(losses)) if losses else 0.0)

        q_loss_per_epoch: list[float] = []
        batches_trained = 0
        for _ in range(cfg.epochs):
            losses = [loss for _ in range(steps_per_epoch) if (loss := agent.train_q_on_batch()) is not None]
            batches_trained += len(losses)
            q_loss_per_epoch.append(float(np.mean(losses)) if losses else 0.0)

        states = np.stack([t.state for t in transitions])
        logged_actions = np.array([t.action for t in transitions])
        q_values = agent.online_network.predict(states)
        behavior_probs = _softmax(agent.behavior_network.predict(states))
        mask = support_mask(behavior_probs, cfg.bcq_threshold)

        unconstrained_greedy = np.argmax(q_values, axis=1)
        masked_q = np.where(mask, q_values, -np.inf)
        constrained_greedy = np.argmax(masked_q, axis=1)

        unconstrained_agreement = float(np.mean(unconstrained_greedy == logged_actions))
        constrained_agreement = float(np.mean(constrained_greedy == logged_actions))
        mean_support_set_size = float(np.mean(mask.sum(axis=1)))

        action_counts = Counter(ACTION_NAMES[a] for a in constrained_greedy)
        action_distribution = {name: action_counts.get(name, 0) / len(constrained_greedy) for name in ACTION_NAMES}

        return BCQTrainingArtifact(
            algorithm=f"Discrete BCQ (threshold={cfg.bcq_threshold}) + LSTM(seq_len={cfg.sequence_length})",
            transitions_used=len(transitions),
            behavior_epochs_run=cfg.behavior_epochs,
            epochs_run=cfg.epochs,
            batches_trained=batches_trained,
            sequence_length=cfg.sequence_length,
            behavior_loss_per_epoch=behavior_loss_per_epoch,
            q_loss_per_epoch=q_loss_per_epoch,
            mean_q_value=float(np.mean(q_values)),
            mean_support_set_size=mean_support_set_size,
            action_distribution=action_distribution,
            unconstrained_greedy_agreement_rate=unconstrained_agreement,
            constrained_greedy_agreement_rate=constrained_agreement,
            config_fingerprint=self._fingerprint,
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )
