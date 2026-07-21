"""Experimental offline Double DQN + Prioritized Experience Replay + LSTM
intervention policy.

**Not part of the default pipeline.** `InterventionPlanner` (Module 11)
remains the system's only intervention engine in `orchestration/` and in
every other module's default wiring. This package is an opt-in research
prototype that trains a DQN against the same synthetic reward the rule
engine already uses, so the two can be compared honestly.

Three algorithmic pieces, each independently toggleable in `DQNConfig`:

- **Double DQN** (Van Hasselt, Guez & Silver, 2016) — the online network
  selects the next action, the target network evaluates it, correcting
  vanilla DQN's maximization bias.
- **Prioritized Experience Replay** (Schaul et al., 2016) — a sum-tree
  buffer sampling high-TD-error transitions more often, with
  importance-sampling correction.
- **An LSTM encoder** (Hausknecht & Stone, 2015 — "DRQN") reading a short
  window of past interactions before the Q-head, so the agent sees a
  trend rather than one isolated snapshot.

Why this stays experimental rather than becoming the new default:

1. **Offline, not online.** There is no live student to act on — training
   replays already-generated `(state, action, reward, next_state)`
   transitions logged from the rule engine's own action choices. A DQN
   trained this way learns to approve of whatever the logging policy did;
   it has no signal about actions the rule engine never took.
2. **The reward is synthetic.** It was designed as an interpretable proxy
   (Module 10), not validated against real learning outcomes. Optimizing
   it harder does not mean producing better tutoring decisions.
3. **No counterfactual evaluation.** What is reported (Q-value spread,
   agreement rate with the rule engine) describes how well the network
   fits the logged data — not whether its policy would perform better in
   a real classroom. Proper off-policy evaluation (importance sampling,
   doubly-robust estimators) is future work, not implemented here.

See `docs/EXPERIMENTAL_DQN.md` for the full writeup.
"""

from dataset_generator.rl_experimental.agent import DQNAgent
from dataset_generator.rl_experimental.config import DQNConfig, default_dqn_config
from dataset_generator.rl_experimental.environment import (
    ACTION_NAMES,
    STATE_DIM,
    Transition,
    build_transitions,
    observation_to_state,
)
from dataset_generator.rl_experimental.models import DQNTrainingArtifact
from dataset_generator.rl_experimental.network import RecurrentQNetwork
from dataset_generator.rl_experimental.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from dataset_generator.rl_experimental.trainer import DQNTrainer

__all__ = [
    "ACTION_NAMES",
    "STATE_DIM",
    "DQNAgent",
    "DQNConfig",
    "DQNTrainer",
    "DQNTrainingArtifact",
    "PrioritizedReplayBuffer",
    "RecurrentQNetwork",
    "ReplayBuffer",
    "Transition",
    "build_transitions",
    "default_dqn_config",
    "observation_to_state",
]
