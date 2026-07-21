# Offline-RL Algorithms: CQL, IQL, Discrete BCQ

`dataset_generator/rl_experimental/offline/` — three algorithms purpose-built
for offline data, as distinct from the Double DQN + PER + LSTM upgrades
covered in `docs/EXPERIMENTAL_DQN.md`. That upgrade set makes fitting logged
transitions faster and more stable; it does **not** address what happens
when the network is asked about an action the logging policy (the rule-based
`InterventionPlanner`) rarely or never took — the extrapolation problem
documented there, and the direct motivation for this file. None of these
three is part of the default pipeline; `InterventionPlanner` remains the
only intervention engine `orchestration/` wires up.

All three reuse the same `RecurrentQNetwork` (LSTM encoder, Section 1.5 of
`docs/RL_FORMALIZATION.md`) and the same `collect_transitions` /
`PrioritizedReplayBuffer` machinery as the DQN trainer — only the training
*objective* changes per algorithm. `network.py::apply_output_gradient` was
extracted specifically so the shared backprop-through-time implementation
has exactly one copy across all four algorithms in this package (DQN, CQL,
IQL, BCQ) rather than four slowly-diverging copies.

## Conservative Q-Learning (`cql.py`)

Kumar, Zhou, Tucker & Levine, 2020. Adds a penalty to the ordinary TD loss:

```
L(theta) = alpha * E_s[ logsumexp_a Q(s,a) - Q(s, a_logged) ]  +  L_TD(theta)
```

which is minimized when Q is large only at the action actually logged and
small everywhere else — directly counteracting the tendency of an
unconstrained Q-network to assign confidently-wrong high values to actions
it has little data about. `docs/RL_FORMALIZATION.md` §1.7 has the full
derivation and gradient.

**Measured effect** (6 students × 2 sessions, `cql_alpha` swept, everything
else held fixed): raising alpha from 0 to 2 increases the mean gap between
the logged action's Q-value and the mean of all other actions' Q-values,
and raises how often the greedy policy agrees with the logged policy —
exactly CQL's intended effect, verified with a real before/after comparison
rather than asserted:

| `cql_alpha` | mean OOD-action gap | greedy/logged agreement |
|---|---|---|
| 0.0 (no penalty — reduces to Double DQN) | ~0.07 | ~3% |
| 1.0 | ~2.6 | ~86% |

(`tests/test_rl_offline.py::test_cql_penalty_increases_ood_gap_vs_alpha_zero`
asserts this ordering holds on every run, not just the numbers above, which
came from one representative run and will vary with the seed/config.)

## Implicit Q-Learning (`iql.py`)

Kostrikov, Nair & Levine, 2021. Trains a state-value function $V(s)$ as an
**expectile** ($\tau > 0.5$, default 0.7) of $Q_{\bar\theta}(s,a)$ over the
actions that appear in the data, then regresses $Q$ toward
$r + \gamma(1-d)V(s')$ — never toward $\max_{a'} Q(s',a')$ or any action
selection. This is the one algorithm in the package that never evaluates
$Q$ at an action the critic itself proposes, at any point during training.

**Explicit limitation, stated rather than hidden:** the published IQL
algorithm also extracts its final policy via advantage-weighted regression
(AWR), so the *policy* — not just the *critic* — never proposes an
out-of-distribution action. This package's action space is a small fixed
discrete set (8 policies), and the induced policy here is `argmax_a Q(s,a)`
directly. The critic's training targets are OOD-free by construction; the
final argmax step is a documented simplification, not full AWR. A discrete
AWR policy head is listed as future work below.

**Measured effect:** both losses decrease monotonically
(`tests/test_rl_offline.py::test_iql_losses_decrease`), and raising $\tau$
from 0.5 to 0.9 does not decrease the mean learned value — the defining
behavior of expectile regression versus ordinary least-squares (which
would be insensitive to $\tau$ entirely, since $\tau=0.5$ *is* ordinary
regression).

## Discrete Batch-Constrained Q-Learning (`bcq.py`)

Fujimoto, Meger & Precup, 2019 (discrete-action simplification per
Fujimoto, Conti, Ghavamzadeh & Pineau, 2019). Trains a behavior-cloning
classifier $G_\omega(a\mid s)$ (cross-entropy on logged `(s,a)` pairs), then
restricts every $\max_{a'}$ in the Bellman target — and the final induced
policy — to the **support set**

```
A_support(s) = { a : G(a|s) / max_a' G(a'|s) >= threshold }
```

so Q is never bootstrapped off, and the agent never selects, an action the
behavior model considers implausible at that state.

**Measured effect** (behavior model pretrained 5 epochs, `threshold=0.3`):
restricting the greedy action to the support set raises agreement with the
logged policy dramatically compared to the unconstrained argmax —

| | agreement with logged policy |
|---|---|
| Unconstrained `argmax_a Q(s,a)` | ~14% |
| Support-masked (BCQ's actual policy) | ~87% |

confirming the mask is doing real work, not passing everything through
(`tests/test_rl_offline.py::test_bcq_constrained_agreement_beats_unconstrained`
asserts this ordering on every run; the specific percentages above are one
representative run).

## Comparing all four algorithms honestly

| Algorithm | Mechanism | What it demonstrably changes | What it doesn't |
|---|---|---|---|
| Double DQN + PER + LSTM | Faster/more stable fitting | Convergence speed, target-value bias | Nothing about unseen actions |
| CQL | Penalize OOD Q-values | Q-value gap, greedy/logged agreement (↑ sharply) | Doesn't know if the *penalized* actions were actually bad |
| IQL | Never bootstrap off OOD actions | Critic training never queries an OOD action | Final argmax can still, in principle, pick one |
| Discrete BCQ | Mask OOD actions from selection | Both training and inference restricted to the support set | Support quality bounded by the behavior model's own fit |

None of the four turns this into a validated tutoring policy — that still
requires real outcome data, exactly as `docs/EXPERIMENTAL_DQN.md` and
`docs/DESIGN_DECISIONS.md` state. What the offline-RL family adds over the
plain-DQN baseline is a *stronger, more defensible* attempt at the same
offline problem, with each algorithm's specific mechanism independently
verified to do what its literature claims.

## Running it

```python
from dataset_generator.orchestration import ObserverAgent, BASAgent, RewardAgent, InterventionAgent
from dataset_generator.rl_experimental.offline import CQLTrainer, CQLConfig, IQLTrainer, IQLConfig, BCQTrainer, BCQConfig

dataset = ObserverAgent().generate(student_count=8, sessions_per_student=2)
bas = BASAgent().compute(dataset)
reward = RewardAgent().compute(dataset, bas)
intervention = InterventionAgent().plan(dataset, bas, reward)

cql = CQLTrainer(CQLConfig(cql_alpha=1.0)).train(dataset, bas, reward, intervention)
iql = IQLTrainer(IQLConfig(iql_tau=0.7)).train(dataset, bas, reward, intervention)
bcq = BCQTrainer(BCQConfig(bcq_threshold=0.3)).train(dataset, bas, reward, intervention)
```

Tests: `pytest tests/test_rl_offline.py -q` (19 tests: artifact shape,
determinism, and — for each algorithm — a test asserting its
distinguishing mechanism actually produces the directional effect the
literature predicts, not just that training completes).

## Future work

- Discrete advantage-weighted regression (AWR) for IQL's policy extraction,
  closing the one gap noted above.
- Off-policy evaluation (fitted Q evaluation, importance sampling) to
  compare CQL/IQL/BCQ's induced policies against the rule engine on
  expected return, not just action-agreement rate.
- A joint ablation sweep (`cql_alpha`, `iql_tau`, `bcq_threshold`) through
  the Module 13 sensitivity framework (`docs/EVALUATION.md`), to report
  these effects as reproducible curves rather than single-run numbers.

## References

- A. Kumar, A. Zhou, G. Tucker, S. Levine, "Conservative Q-Learning for
  Offline Reinforcement Learning," *NeurIPS*, 2020.
- I. Kostrikov, A. Nair, S. Levine, "Offline Reinforcement Learning with
  Implicit Q-Learning," *ICLR*, 2022 (arXiv 2021).
- S. Fujimoto, D. Meger, D. Precup, "Off-Policy Deep Reinforcement Learning
  without Exploration," *ICML*, 2019. (Continuous-action BCQ.)
- S. Fujimoto, E. Conti, M. Ghavamzadeh, J. Pineau, "Benchmarking Batch
  Deep Reinforcement Learning Algorithms," arXiv:1910.01708, 2019.
  (Discrete-action BCQ simplification used here.)
