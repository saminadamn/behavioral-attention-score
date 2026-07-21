"""Orchestrates offline DQN training over one dataset's logged transitions
and reports a `DQNTrainingArtifact` — mirrors every other module's
Engine/Planner entry-point shape (`__init__(config)`, one public method).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np

from dataset_generator.bas.models import BASArtifact
from dataset_generator.intervention.models import InterventionArtifact
from dataset_generator.intervention.observation import InterventionObservationExtractor
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.reward.models import RewardArtifact
from dataset_generator.rl_experimental.agent import DQNAgent
from dataset_generator.rl_experimental.config import DQNConfig, compute_dqn_config_fingerprint, default_dqn_config
from dataset_generator.rl_experimental.environment import ACTION_NAMES, STATE_DIM, build_transitions
from dataset_generator.rl_experimental.models import DQNTrainingArtifact

SCHEMA_VERSION = "1.1"


def _algorithm_label(config: DQNConfig) -> str:
    label = "Double DQN" if config.use_double_dqn else "DQN"
    if config.use_prioritized_replay:
        label += " + PER"
    label += f" + LSTM(seq_len={config.sequence_length}, hidden={config.lstm_hidden_dim})"
    return label


class DQNTrainer:
    def __init__(self, config: DQNConfig | None = None) -> None:
        self._config = config or default_dqn_config()
        self._fingerprint = compute_dqn_config_fingerprint(self._config)

    def train(
        self,
        dataset_artifact: DatasetArtifact,
        bas_artifact: BASArtifact,
        reward_artifact: RewardArtifact,
        intervention_artifact: InterventionArtifact,
    ) -> DQNTrainingArtifact:
        cfg = self._config
        observations = InterventionObservationExtractor().extract_batch(
            dataset_artifact, bas_artifact, reward_artifact
        )
        transitions = build_transitions(
            intervention_artifact, reward_artifact, observations, sequence_length=cfg.sequence_length
        )
        if not transitions:
            raise ValueError("No transitions could be built — check the intervention artifact is non-empty.")

        agent = DQNAgent(state_dim=STATE_DIM, action_dim=len(ACTION_NAMES), config=cfg)
        for t in transitions:
            agent.replay_buffer.push(t)

        loss_per_epoch: list[float] = []
        batches_trained = 0
        last_td_errors: np.ndarray = np.zeros(1)

        for _ in range(cfg.epochs):
            epoch_losses: list[float] = []
            steps_this_epoch = max(1, len(transitions) // cfg.batch_size)
            for _ in range(steps_this_epoch):
                loss = agent.train_on_batch()
                if loss is not None:
                    epoch_losses.append(loss)
                    batches_trained += 1
            loss_per_epoch.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

        states = np.stack([t.state for t in transitions])
        logged_actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float64)

        q_values = agent.online_network.predict(states)
        greedy_actions = np.argmax(q_values, axis=1)

        targets = agent.bellman_targets(rewards, next_states, dones)
        predicted_at_logged_action = q_values[np.arange(len(transitions)), logged_actions]
        last_td_errors = predicted_at_logged_action - targets

        agreement_rate = float(np.mean(greedy_actions == logged_actions))
        action_counts = Counter(ACTION_NAMES[a] for a in greedy_actions)
        action_distribution = {
            name: action_counts.get(name, 0) / len(greedy_actions) for name in ACTION_NAMES
        }

        return DQNTrainingArtifact(
            algorithm=_algorithm_label(cfg),
            transitions_used=len(transitions),
            epochs_run=cfg.epochs,
            batches_trained=batches_trained,
            sequence_length=cfg.sequence_length,
            loss_per_epoch=loss_per_epoch,
            mean_q_value=float(np.mean(q_values)),
            mean_abs_td_error=float(np.mean(np.abs(last_td_errors))),
            action_distribution=action_distribution,
            greedy_policy_agreement_rate=agreement_rate,
            config_fingerprint=self._fingerprint,
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )
