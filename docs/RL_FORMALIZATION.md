# Formal RL Notation for the Experimental DQN

This document gives the formal MDP notation behind `dataset_generator/rl_experimental/`
and `dataset_generator/reward/`, a worked step-wise numeric example using the
project's actual default weights and transition probabilities, and the
literature basis for the reward weights that were set by hand rather than
learned. It is a companion to `docs/EXPERIMENTAL_DQN.md` (what was built and
why it stays experimental) — this file is the math underneath it.

## 1. MDP definition

The intervention problem is formalized as a Markov Decision Process
$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- **State** $s_t \in \mathcal{S} \subset \mathbb{R}^{12}$ — the 12-dimensional
  vector `observation_to_state()` builds from an `InterventionObservation`:
  current BAS, BAS trend, current reward, reward trend, fatigue, engagement,
  normalized latency deviation, correctness, confidence, semantic similarity,
  prompt difficulty, session progress. All twelve components are already
  bounded to $[0,1]$ (or clipped into it) by the upstream engines, so no
  additional scaling is fit here.
- **Action** $a_t \in \mathcal{A} = \{0, \dots, 7\}$ — the 8 intervention
  policies in `ACTION_NAMES` (`NoInterventionPolicy` through
  `QuestionReframingPolicy`), a discrete action space.
- **Transition kernel** $P(s_{t+1} \mid s_t, a_t)$ — realized by the Module 4
  Markov attention-state model plus the per-student generative process; not
  estimated by the DQN code itself, only sampled through via the logged
  dataset (Section 3 gives its concrete numbers).
- **Reward function** $R(s_t, a_t, s_{t+1})$ — computed by
  `RewardAggregator.aggregate` (Module 10), given in full in Section 2.
- **Discount** $\gamma = 0.9$ — `RewardConfig.discount_factor`'s default.

### 1.1 Return and value functions

Discounted return from step $t$ in a finite session of length $T$:

$$
G_t = \sum_{k=0}^{T-t} \gamma^{k} r_{t+k}
$$

which is exactly `RewardConfig.temporal_mode = "discounted"`'s definition
(implemented in `reward/temporal.py::_discounted_returns`), normalized by
$\frac{1-\gamma^{T-t+1}}{1-\gamma}$ so it stays on the same scale as the raw
per-interaction rewards regardless of how many interactions remain.

State-value and action-value functions under policy $\pi$:

$$
V^{\pi}(s) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s\right], \qquad
Q^{\pi}(s,a) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s,\, a_t = a\right]
$$

### 1.2 Bellman optimality equation

$$
Q^{*}(s,a) = \mathbb{E}_{s'}\!\left[\, R(s,a,s') + \gamma \max_{a'} Q^{*}(s', a') \,\right]
$$

Two networks are involved in estimating the right-hand side: the
**online** network $Q_\theta$ (trained every step) and the **target**
network $Q_{\bar\theta}$ (a periodically-synced copy, `target_sync_every`
steps) — bootstrapping a network off its own live, rapidly-changing
estimate is unstable (Mnih et al., 2015), so the target is held fixed for
several steps at a time.

### 1.3 Double DQN target

Plain DQN reuses the target network for both selecting *and* evaluating
the next action, $y = r + \gamma \max_{a'} Q_{\bar\theta}(s', a')$. Because
$\max$ is applied to a noisy estimate, this systematically overestimates
action values (Thrun & Schwartz, 1993; formalized for DQN by Van Hasselt,
Guez & Silver, 2016). **Double DQN** decouples the two roles:

$$
a^{*} = \arg\max_{a'} Q_\theta(s', a'), \qquad
y_{\text{double}} = r + \gamma\,(1-d)\, Q_{\bar\theta}\!\left(s', a^{*}\right)
$$

the online network picks the action it currently believes is best, the
target network supplies *that specific action's* value — implemented in
`DQNAgent.bellman_targets`:

```python
online_next_q = self.online_network.predict(next_states)
best_actions = np.argmax(online_next_q, axis=1)
target_next_q = self.target_network.predict(next_states)
selected_q = target_next_q[np.arange(len(best_actions)), best_actions]
```

with `DQNConfig.use_double_dqn=False` falling back to the plain
$\max_{a'} Q_{\bar\theta}(s',a')$ target for ablation.

### 1.4 Loss, with Prioritized Experience Replay and importance-sampling correction

For a minibatch $\mathcal{B}$ of transitions:

$$
\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|}\sum_{(s,a,r,s',d) \in \mathcal{B}}
\omega_i \Big( Q_\theta(s,a) - y \Big)^2
$$

Under **uniform** replay, $\omega_i = 1$ for every sample. Under
**Prioritized Experience Replay** (Schaul et al., 2016), transition $i$ is
sampled with probability

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}, \qquad
p_i = |\delta_i| + \epsilon_{\text{PER}}
$$

where $\delta_i = Q_\theta(s_i,a_i) - y_i$ is transition $i$'s most
recently observed TD error (new transitions enter at the buffer's current
max priority, so they are sampled at least once before their real error is
known) and $\alpha \in [0,1]$ interpolates between uniform ($\alpha=0$)
and fully-greedy-on-error ($\alpha=1$) sampling. Sampling non-uniformly
biases the gradient toward high-error transitions, corrected by
importance-sampling weights

$$
\omega_i = \left(\frac{1}{N \cdot P(i)}\right)^{\beta} \Big/ \max_j \omega_j,
\qquad \beta: \beta_0 \to 1 \text{ over training}
$$

($\beta$ annealed toward 1 — full correction — as training progresses;
early on, partial correction is accepted in exchange for prioritizing
learning on the most surprising transitions). Implemented as a sum-tree
(`replay_buffer.py::_SumTree`) for $O(\log n)$ sampling and priority
update rather than an $O(n)$ rescan of a flat array; after each training
step, `PrioritizedReplayBuffer.update_priorities` writes
$p_i = (|\delta_i| + \epsilon_{\text{PER}})^{\alpha}$ for every transition
just trained on.

### 1.5 Recurrent (DRQN) state encoding

Rather than feeding a single-interaction state directly to the Q-head, a
window of the last `sequence_length` interactions
$s_{t-k+1}, \dots, s_t$ is passed through a single-layer LSTM
(Hausknecht & Stone, 2015 — this architecture, DQN's Q-head fed by an LSTM
instead of a flat vector, is literally named "DRQN" in that paper):

$$
\begin{aligned}
i_\tau &= \sigma(W_i [x_\tau, h_{\tau-1}] + b_i), &
f_\tau &= \sigma(W_f [x_\tau, h_{\tau-1}] + b_f) \\
g_\tau &= \tanh(W_g [x_\tau, h_{\tau-1}] + b_g), &
o_\tau &= \sigma(W_o [x_\tau, h_{\tau-1}] + b_o) \\
c_\tau &= f_\tau \odot c_{\tau-1} + i_\tau \odot g_\tau, &
h_\tau &= o_\tau \odot \tanh(c_\tau)
\end{aligned}
$$

for $\tau = 1, \dots, k$ with $h_0 = c_0 = \mathbf{0}$, and
$Q_\theta(s,\cdot) = h_k W_{\text{out}} + b_{\text{out}}$ — the final
hidden state, not the raw current-interaction features, is what the
Q-head sees. Gradients flow back through every gate at every timestep via
backpropagation-through-time (the standard LSTM BPTT equations,
implemented in `network.py::RecurrentQNetwork.train_step`); this
implementation was checked against numerical (finite-difference)
gradients on $W_i$, $W_f$, $W_{\text{out}}$, and $b_g$ before being wired
into training, matching to a relative error of $\sim 10^{-9}$–$10^{-12}$.

`sequence_length=1` degenerates to a single-frame LSTM (no real temporal
context, but still a valid recurrent cell) and is the closest available
ablation baseline against a non-recurrent encoder without maintaining two
separate network implementations.

### 1.6 Behavior policy and the offline-RL caveat

The replay buffer is populated once, before training, from transitions
**logged by the rule-based `InterventionPlanner`** — call this behavior
policy $\mu$. Every gradient step in this codebase optimizes
$Q_\theta$ against transitions drawn from $\mu$'s state-action distribution,
never against $\pi_\theta$'s own choices, since there is no live environment
to act in. This is why `docs/EXPERIMENTAL_DQN.md` reports
`greedy_policy_agreement_rate` against $\mu$ instead of any claim about
$Q_\theta$'s real-world value — under a fixed behavior policy, actions
$\mu$ rarely or never took are exactly where $Q_\theta$'s estimates are
least constrained by data (Fujimoto et al., 2019 formalize this as
*extrapolation error* in batch RL).

## 2. Reward function, step by step

`RewardAggregator.aggregate` computes, per interaction, for each scored
signal $i$ with configured weight $w_i$ and observed normalized evidence
$e_i \in [0,1]$:

$$
\text{signed}_i = 2e_i - 1 \in [-1, 1], \qquad
w_i' = \frac{w_i}{\sum_{j \in \text{observed}} w_j}, \qquad
c_i = w_i' \cdot \text{signed}_i
$$

($w_i'$ is the weight *renormalized* over whichever signals were actually
observed this interaction — Section 2.2 shows why this matters.) The raw
reward is $r_{\text{raw}} = \sum_i c_i$, clipped to $[-1, 1]$, and splits
by construction into

$$
r_{\text{raw}} = \underbrace{\textstyle\sum_{i \in \text{PERFORMANCE}} c_i}_{R_{\text{perf}}}
+ \underbrace{\textstyle\sum_{i \in \text{BEHAVIOUR}} c_i}_{R_{\text{behav}}}
- \underbrace{\Big(-\textstyle\sum_{i \in \text{COST}} c_i\Big)}_{R_{\text{cost}}}
$$

### 2.1 Default weight table (`reward/config.py::default_reward_config`)

| Signal | Category | Weight $w_i$ | Polarity |
|---|---|---|---|
| `delta_bas` | Performance | 0.30 | positive |
| `delta_correctness` | Performance | 0.15 | positive |
| `delta_confidence` | Performance | 0.10 | positive |
| `delta_engagement` | Behaviour | 0.15 | positive |
| `delta_latency_deviation` | Behaviour | 0.10 | negative |
| `delta_fatigue` | Behaviour | 0.10 | negative |
| `intervention_cost` | Cost | 0.10 | penalty |

Category totals: Performance $0.30+0.15+0.10=0.55$, Behaviour
$0.15+0.10+0.10=0.35$, Cost $0.10$ — summing to $1.00$ before any signal is
missing, matching the $R = \text{Performance} + \text{Behaviour} -
\text{Cost}$ framing used throughout the project.

### 2.2 Worked numeric example

Take one interaction where every Performance/Behaviour signal is observed
but `intervention_cost` is not (the common case — cost is only observed
when an intervention actually fired), with these normalized evidence values:

| Signal | $w_i$ | $e_i$ | $\text{signed}_i = 2e_i-1$ |
|---|---|---|---|
| `delta_bas` | 0.30 | 0.70 | 0.40 |
| `delta_correctness` | 0.15 | 0.60 | 0.20 |
| `delta_confidence` | 0.10 | 0.55 | 0.10 |
| `delta_engagement` | 0.15 | 0.65 | 0.30 |
| `delta_latency_deviation` | 0.10 | 0.80 | 0.60 |
| `delta_fatigue` | 0.10 | 0.40 | $-0.20$ |

`intervention_cost` missing $\Rightarrow$ weight available
$\sum w_j = 0.30+0.15+0.10+0.15+0.10+0.10 = 0.90$ (not $1.00$) — this is
exactly the renormalization the aggregator docstring calls out as required
for the decomposition identity to hold exactly regardless of what's missing.

Renormalized weights $w_i' = w_i / 0.90$:

| Signal | $w_i'$ | $c_i = w_i' \cdot \text{signed}_i$ |
|---|---|---|
| `delta_bas` | 0.3333 | 0.1333 |
| `delta_correctness` | 0.1667 | 0.0333 |
| `delta_confidence` | 0.1111 | 0.0111 |
| `delta_engagement` | 0.1667 | 0.0500 |
| `delta_latency_deviation` | 0.1111 | 0.0667 |
| `delta_fatigue` | 0.1111 | $-0.0222$ |

$$
r_{\text{raw}} = 0.1333+0.0333+0.0111+0.0500+0.0667-0.0222 = 0.2722
$$

$$
R_{\text{perf}} = 0.1333+0.0333+0.0111 = 0.1778, \quad
R_{\text{behav}} = 0.0500+0.0667-0.0222 = 0.0944, \quad
R_{\text{cost}} = 0
$$

Check: $0.1778 + 0.0944 - 0 = 0.2722 = r_{\text{raw}}$ ✓ — the invariant
`test_evaluation.py` and `test_reward_model.py` assert holds here by hand
as well as by code.

### 2.3 The transition-reward table $R(s,a,s')$

Because $\mathcal{S}$ is continuous, $R(s,a,s')$ is not tabulated over raw
states in this codebase — the reward is computed per realized transition
(Section 2.2), not looked up. What *is* a finite, tabulated Markov object
is the **attention-state transition matrix** the transition itself is drawn
from (`config/defaults.py`, Module 4's `AttentionState` chain), which
determines *which* $s'$ (in terms of coarse attention state, one component
folded into the 12-d vector) a given interaction is likely to land in:

$$
P =
\begin{pmatrix}
& \text{Focused} & \text{Distracted} & \text{Impulsive} \\
\text{Focused} & 0.75 & 0.15 & 0.10 \\
\text{Distracted} & 0.30 & 0.55 & 0.15 \\
\text{Impulsive} & 0.25 & 0.35 & 0.40
\end{pmatrix}
$$

Combining $P$ with a fixed action's expected reward per resulting state
gives the expected reward of taking no intervention from Focused, as a
concrete step-wise calculation. Suppose the (illustrative, not measured)
per-resulting-state expected raw reward for no intervention is
$\bar r(\text{Focused})=0.30$, $\bar r(\text{Distracted})=-0.05$,
$\bar r(\text{Impulsive})=-0.20$ (Focused interactions tend to produce
positive `delta_bas`/`delta_correctness`; Impulsive ones tend to produce
negative `delta_fatigue`/`delta_engagement` evidence). Then:

$$
\mathbb{E}\big[R \mid s=\text{Focused}, a=\text{NoIntervention}\big]
= \sum_{s'} P(s' \mid \text{Focused}) \, \bar r(s')
$$

$$
= 0.75(0.30) + 0.15(-0.05) + 0.10(-0.20) = 0.225 - 0.0075 - 0.020 = 0.1975
$$

This is the calculation `DQNTrainer` approximates statistically (via sampled
transitions and a target-network bootstrap) rather than computing in closed
form, since the real per-state reward distribution isn't hand-specified
anywhere in the codebase — it emerges from the generators. The number above
is a worked illustration of the mechanics, not a value read out of a run.

## 3. References for the manually chosen reward weights

The category split and the specific numeric weights in
`default_reward_config()` are **engineering choices**, not values fit to
data — there is no ground truth to fit them against (see
`docs/EVALUATION.md`'s discussion of why no intervention-need labels
exist). They were set once, are versioned in `RewardConfig`, and are
exactly what the ablation framework (`docs/EVALUATION.md`) exists to
stress-test. Three bodies of work informed the *shape* of that choice
(category structure and relative priority), not the decimal values
themselves:

1. **Reward shaping invariance.** Ng, Harada & Russell (1999), *"Policy
   invariance under reward transformations,"* ICML — establishes that a
   potential-based reward shaping term does not change the optimal policy.
   This motivates the decomposition's role: performance/behaviour/cost are
   *interpretability* structure layered on top of a single scalar signal
   the theory says any consistent decomposition should preserve, not
   independent objectives that could each be separately optimized without
   changing the induced policy's optimum.
2. **Decomposed / hybrid reward architectures.** Van Seijen et al. (2017),
   *"Hybrid Reward Architecture for Reinforcement Learning,"* NeurIPS —
   demonstrates that splitting a reward into interpretable components and
   summing their value estimates can preserve solution quality while
   making the decision legible. This is the direct precedent for reporting
   $R_{\text{perf}}, R_{\text{behav}}, R_{\text{cost}}$ as first-class,
   separately-trackable quantities rather than an opaque scalar.
3. **Competence and behavioral-cost framing.** Self-determination theory
   (Deci & Ryan, 2000, *"The 'what' and 'why' of goal pursuits,"*
   Psychological Inquiry) distinguishes competence-based outcomes from
   autonomy/effort-based process signals — the conceptual basis for
   scoring *what a student got right* (Performance) separately from *how
   they behaved while doing it* (Behaviour), rather than folding both into
   one undifferentiated "performance" number. This motivated the category
   boundary, not the 0.30/0.15/0.10 split within it.

The Performance category receiving the largest total weight (0.55) reflects
a deliberate design stance — correctness-linked signals should dominate
behavioural ones — documented here so it is inspectable and contestable,
exactly as `docs/DESIGN_DECISIONS.md` asks every non-obvious constant in
this project to be.

## 4. Additional references

[8] V. Mnih et al., "Human-level control through deep reinforcement
learning," *Nature*, 518, 529-533, 2015.
[9] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An
Introduction*, 2nd ed. MIT Press, 2018.
[10] S. Fujimoto, D. Meger, D. Precup, "Off-Policy Deep Reinforcement
Learning without Exploration," *ICML*, 2019.
[11] A. Y. Ng, D. Harada, S. Russell, "Policy invariance under reward
transformations: Theory and application to reward shaping," *ICML*, 1999.
[12] H. van Seijen et al., "Hybrid Reward Architecture for Reinforcement
Learning," *NeurIPS*, 2017.
[13] E. L. Deci and R. M. Ryan, "The 'what' and 'why' of goal pursuits:
Human needs and the self-determination of behavior," *Psychological
Inquiry*, 11(4), 227-268, 2000.
[14] H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning
with Double Q-learning," *AAAI*, 2016.
[15] S. Thrun, A. Schwartz, "Issues in Using Function Approximation for
Reinforcement Learning," *Proc. of the 1993 Connectionist Models Summer
School*, 1993. (Earliest identification of Q-learning's maximization
bias, which Double DQN corrects.)
[16] T. Schaul, J. Quan, I. Antonoglou, D. Silver, "Prioritized
Experience Replay," *ICLR*, 2016.
[17] M. Hausknecht, P. Stone, "Deep Recurrent Q-Learning for Partially
Observable MDPs," *AAAI Fall Symposium Series*, 2015.
[18] S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory," *Neural
Computation*, 9(8), 1735-1780, 1997. (The LSTM cell itself.)
