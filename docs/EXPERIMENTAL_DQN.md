# Experimental: Offline Double DQN + PER + LSTM Intervention Policy

`dataset_generator/rl_experimental/` is a research prototype, kept
deliberately separate from the default pipeline. `InterventionPlanner`
(Module 11) is still the only intervention engine `orchestration/` wires up,
and every other module's default behavior is unchanged by this package's
existence.

## Why this exists

The project's central architectural argument (see `docs/DESIGN_DECISIONS.md`)
is that a rule-based intervention engine is the *defensible* choice today,
because a trained policy has nothing but the same synthetic reward to
validate itself against. This package makes that argument checkable rather
than asserted: it actually trains a DQN against the logged reward and
reports, honestly, what that produces.

## What was built

A textbook DQN (Mnih et al., 2015) — experience replay, a target network,
epsilon-greedy exploration — implemented directly in NumPy, since this
repository has no PyTorch/TensorFlow dependency and adding one for a
deliberately-experimental module wasn't justified. Three upgrades sit on
top of that baseline, each independently toggleable in `DQNConfig` so an
ablation can isolate what each one contributes:

- **Double DQN** (Van Hasselt, Guez & Silver, 2016, `use_double_dqn`) —
  the *online* network selects the greedy next action, the *target*
  network evaluates that action's value. Vanilla DQN uses
  `max_a' Q_target(s',a')` for both selection and evaluation, which
  systematically overestimates values whenever the target network's
  noise happens to be correlated with which action looks best.
  Decoupling the two corrects that bias — see `DQNAgent.bellman_targets`.
- **Prioritized Experience Replay** (Schaul, Quan, Antonoglou & Silver,
  2016, `use_prioritized_replay`) — a sum-tree buffer
  (`replay_buffer.py::_SumTree`) samples transitions with probability
  proportional to `(|TD error| + eps)^alpha` instead of uniformly, so
  training time is spent on the transitions the network is currently
  most wrong about. Importance-sampling weights
  (`(N · P(i))^{-beta}`, `beta` annealed from 0.4 toward 1.0) correct the
  resulting sampling bias in the gradient.
- **An LSTM encoder** (Hausknecht & Stone, 2015 — "DRQN",
  `sequence_length` / `lstm_hidden_dim`) reads a short window of the last
  `sequence_length` interactions before a linear Q-head, instead of one
  interaction snapshot. Implemented as full backpropagation-through-time
  in NumPy (`network.py::RecurrentQNetwork`) — verified correct against
  numerical (finite-difference) gradients to ~1e-9 relative error before
  being wired into training.

| Component | File | Role |
|---|---|---|
| `DQNConfig` | `config.py` | Typed, fingerprinted hyperparameters; three upgrade toggles |
| `RecurrentQNetwork` | `network.py` | LSTM encoder + linear head, full BPTT, Adam |
| `PrioritizedReplayBuffer` / `ReplayBuffer` | `replay_buffer.py` | Sum-tree prioritized sampling, or uniform fallback |
| state-sequence encoding, transitions | `environment.py` | Turns logged artifacts into windowed `(s, a, r, s', done)` |
| `DQNAgent` | `agent.py` | Epsilon-greedy policy, Double-DQN targets, target-network sync |
| `DQNTrainer` | `trainer.py` | Orchestrates training, returns `DQNTrainingArtifact` |

## Why this is offline, not online

There is no live student to act on between updates. Training replays
**already-generated** transitions: the state at each interaction (from
`InterventionObservation`), the action the rule engine already chose
(`InterventionDecision.chosen_policy`), and the reward the reward engine
already computed (`RewardRecord.reward`) for that interaction. The rule
engine is, in RL terms, the *logging policy* — this is standard offline
(batch) reinforcement learning, not online control.

## A real finding from running this

On an 8-student / 2-session run (491 transitions, `sequence_length=5`,
`lstm_hidden_dim=16`), a fresh training run produced:

```
algorithm         : Double DQN + PER + LSTM(seq_len=5, hidden=16)
loss_per_epoch    : [0.1773, 0.0193, 0.0108, 0.0071, 0.0050, 0.0072]
mean_q_value      : 0.1674
mean_abs_td_error : 0.0615
agreement_rate    : 0.0244  (2.4%)
```

Loss falls sharply and by more, and faster, than the plain-DQN baseline
did on a comparable run (which needed ~4 epochs to reach a similar loss
band) — consistent with PER concentrating gradient steps on the
highest-error transitions rather than spending them uniformly. But the
core finding from the original (non-upgraded) DQN is **unchanged**: the
trained network's greedy action still agrees with the rule engine's
actual choice only ~2-3% of the time. The logged action distribution is
dominated by `NoInterventionPolicy` (most interactions don't need help),
so the greedy action is still pulled toward whichever *rarer* action the
network estimates highest value for — a textbook extrapolation problem
in offline RL (Fujimoto et al., 2019) that neither Double DQN, PER, nor
an LSTM addresses, because none of the three changes what data the
network is trained on. They make the network fit the logged data faster
and more stably; they do not give it data about actions the logging
policy never took.

This is reported, not hidden, in `DQNTrainingArtifact.greedy_policy_agreement_rate`
and `action_distribution` — and it is exactly the failure mode that makes
deploying this policy today indefensible, upgraded or not.

## What this does and does not demonstrate

**Does:** show that a genuine DQN can be trained end-to-end against this
project's artifacts, converges (falling loss), and is fully deterministic
given a seed — the same guarantees every other module provides.

**Does not:** demonstrate that the learned policy would make better
intervention decisions than the rule engine. Two reasons, both structural,
not implementation bugs:

1. **No counterfactual evaluation.** Comparing the learned policy's
   proposed actions to the logging policy's actual actions cannot say
   which one is *better* — only how much they agree. Off-policy evaluation
   (importance sampling, doubly-robust estimators) would be required, and
   is not implemented here.
2. **The reward itself is synthetic** (Module 10) — a designed proxy, not
   a validated measure of learning benefit. Optimizing it harder is not
   optimizing real tutoring quality.

## Beyond DQN: algorithms built specifically for offline data

Double DQN, PER, and the LSTM encoder above all improve *how well and how
fast* a network fits logged transitions — none of them addresses what
happens when the network is queried about an action the logging policy
rarely took. `docs/OFFLINE_RL.md` covers three algorithms
(`dataset_generator.rl_experimental.offline`) built specifically for that
gap — Conservative Q-Learning, Implicit Q-Learning, and Discrete
Batch-Constrained Q-Learning — each with a measured, reproducible effect
demonstrating its mechanism actually works (e.g., CQL's greedy/logged
agreement rate jumps from ~3% to ~86% when its penalty is turned on).

## Formal notation, the transition-reward table, and weight references

The formal MDP definition ($\mathcal{S}, \mathcal{A}, P, R, \gamma$), the
Bellman equation and DQN loss this code implements, a step-wise worked
numeric example of the reward computation (including the
missing-signal renormalization), the attention-state transition matrix
combined with expected reward per resulting state, and the literature
behind the manually-chosen reward category weights are all in
`docs/RL_FORMALIZATION.md` — kept separate from this file so the "what
and why" here stays readable without the notation.

## Running it

```python
from dataset_generator.orchestration import ObserverAgent, BASAgent, RewardAgent, InterventionAgent
from dataset_generator.rl_experimental import DQNTrainer, DQNConfig

dataset = ObserverAgent().generate(student_count=8, sessions_per_student=2)
bas = BASAgent().compute(dataset)
reward = RewardAgent().compute(dataset, bas)
intervention = InterventionAgent().plan(dataset, bas, reward)

artifact = DQNTrainer(DQNConfig(epochs=5)).train(dataset, bas, reward, intervention)
print(artifact.algorithm)
print(artifact.loss_per_epoch)
print(artifact.greedy_policy_agreement_rate)
print(artifact.disclaimer)
```

To ablate the three upgrades independently (each defaults to on):

```python
from dataset_generator.rl_experimental import DQNConfig, DQNTrainer

baseline = DQNConfig(use_double_dqn=False, use_prioritized_replay=False, sequence_length=1)
upgraded = DQNConfig()  # Double DQN + PER + LSTM(seq_len=5)

for config in (baseline, upgraded):
    artifact = DQNTrainer(config).train(dataset, bas, reward, intervention)
    print(config.use_double_dqn, config.use_prioritized_replay, config.sequence_length, "->", artifact.loss_per_epoch[-1])
```

Tests: `pytest tests/test_rl_experimental.py -q` (22 tests — LSTM forward/
convergence, sum-tree priority behavior, Double-DQN target computation,
determinism, and all four on/off combinations of the two toggles).

## References for this file

- V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 518, 529-533, 2015.
- H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning with Double Q-learning," *AAAI*, 2016.
- T. Schaul, J. Quan, I. Antonoglou, D. Silver, "Prioritized Experience Replay," *ICLR*, 2016.
- M. Hausknecht, P. Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs," *AAAI Fall Symposium*, 2015.
- S. Fujimoto, D. Meger, D. Precup, "Off-Policy Deep Reinforcement Learning without Exploration," *ICML*, 2019.

The formal equations behind each of these are in `docs/RL_FORMALIZATION.md`.

## What would need to change before this could replace the rule engine

Real logged feedback (student outcomes, not a synthetic proxy); an
online or safely-simulated environment instead of pure offline replay;
off-policy evaluation before any deployment; and a governance process —
the same conditions already listed as prerequisites in
`docs/DESIGN_DECISIONS.md` and the project roadmap. Until then, this
package stays a research prototype, not a default.
