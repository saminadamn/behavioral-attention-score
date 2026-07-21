# Comparison Study: Phases 1-4

`run_comparison_study.py` is the entry point for this document — one
command produces every table below, freshly, from the real pipeline:

```bash
python run_comparison_study.py --students 40 --sessions 3
```

```
outputs_comparison/<timestamp>/
    reproducibility.json               Phase 1
    baseline_policy_table.csv          Phase 2
    rl_evaluation_table.csv            Phase 3
    classifier_comparison_table.csv    Phase 4 (see docs/DEEP_LEARNING_COMPARISON.md)
    models/                            every trained artifact
```

Phases 5-10 (ablation, hyperparameter sweeps, multi-seed statistical
significance, additional visualizations, and future deep-learning work)
are scoped and documented, not yet implemented — see
`docs/PHASE_ROADMAP.md`.

## Phase 1 — Reproducibility

`dataset_generator/experiment_reproducibility.py` provides:

- `set_global_seed(seed)` — seeds Python's `random` and NumPy's legacy
  global generator. Deliberately separate from the core pipeline's own
  determinism mechanism (`RNGStreams`, seeded per-concern from
  `GeneratorConfig.seed`, see `docs/DESIGN_DECISIONS.md`), which remains
  the source of truth for dataset generation — this covers the global
  RNGs ad hoc experiment code might otherwise touch unseeded.
- `collect_environment_info()` — package versions (pydantic, numpy,
  pandas, scipy, pyarrow, scikit-learn, joblib, langgraph, matplotlib,
  pytest), Python version, platform, and a UTC timestamp.
- `build_reproducibility_report(seed, config_summary)` — bundles the above
  with the run's own config into `reproducibility.json`.
- Every trained model in Phases 3-4 is saved to `models/` (joblib bundles
  for classifiers via `save_training_artifact`, JSON artifacts for the
  RL agents) — Phase 1's "save trained models and experiment artifacts."

## Phase 2 — Baseline Evaluation: a real causal comparison

**Why this needed new code, not just re-reading existing artifacts:** the
synthetic generator's own ground-truth `intervention_applied` flag
(`SessionSimulator._decide_intervention`, Module 6) is entirely independent
of `InterventionPlanner` (Module 11) — Module 11's decisions are computed
*after* a dataset already exists and never feed back into it. Reusing the
one existing reward series and relabeling it "Random Policy" would compare
against a trajectory that policy never actually produced.

`SessionSimulator` now accepts an optional `intervention_policy` callback
(`dataset_generator/generators/session_simulator.py::InterventionPolicy`),
defaulting to `None` — which reproduces the exact prior behavior, verified
by `tests/test_session_simulator.py::test_default_intervention_policy_none_matches_prior_behavior`
and confirmed by the full regression suite staying green. When a policy is
supplied, it genuinely controls generation: the resulting BAS and reward
are computed over an actually-different, freshly-simulated trajectory.

Three policies (`dataset_generator/rl_experimental/baselines.py`):

| Policy | Mechanism |
|---|---|
| Rule-Based (generator heuristic) | `None` — the simulator's own built-in rolling-engagement-threshold heuristic (the reference row) |
| No Intervention | Always `False` |
| Random Policy | Bernoulli draw at the same `intervention_probability` the reference uses — isolates *timing*, not *frequency* |

**Table shape** (`baseline_policy_table.csv`):

| Policy | Record count | Interventions | Intervention rate | BAS mean | Reward mean | Accuracy/Precision/Recall/F1 | Generation time |
|---|---|---|---|---|---|---|---|
| Rule-Based | | | | | | N/A (not a classifier) | |
| No Intervention | | | | | | N/A (not a classifier) | |
| Random Policy | | | | | | N/A (not a classifier) | |

The classifier-metric columns are explicitly `N/A` — these are generation
policies, not classifiers, and filling them with a placeholder number would
misrepresent what was measured. Note that record counts differ slightly
across policies: session length is drawn from the same shared RNG stream
policy decisions also draw from, so a policy that consumes a different
number of random draws per interaction (the built-in heuristic calls
`self._rng.random()` per decision; an externally-injected policy does not)
shifts later sessions' length draws. Reported statistics are per-record
means, which stay valid and comparable despite this.

### On Random Policy's slightly higher BAS mean

A representative run (30 students × 3 sessions, seed 42) measured
Rule-Based BAS mean 0.7196 against Random Policy BAS mean 0.7237 — Random
*numerically* higher. Taken at face value this looks backwards: why would
randomly-timed interventions outscore the engagement-triggered heuristic?

Checked directly rather than left as an eyeballed gap: a Welch's t-test
between the two policies' per-record BAS distributions (n≈2,690 vs.
n≈2,647) gives $t = 2.04$, $p = 0.042$ — technically below the
conventional 0.05 threshold — but Cohen's $d = 0.056$, an order of
magnitude below the conventional "small effect" cutoff of 0.2. This is
the standard large-sample-size trap: with ~2,700 observations per group,
even a practically negligible difference in means becomes statistically
"detectable." The honest reading is that Random and Rule-Based produce
**indistinguishable mean BAS at this scale and configuration** — not that
randomly-timed intervention outperforms the engagement-triggered
heuristic. Reported this way rather than as an unqualified "Random scores
higher," which the raw numbers alone would misleadingly suggest.

## Phase 3 — RL Evaluation

`rl_evaluation_table.csv` compares the rule-based reference against seven
trained agents: **Vanilla DQN**, **Double DQN**, **Double DQN + PER**,
**DRQN** (Double DQN + PER + LSTM) — an ablation ladder through
`docs/EXPERIMENTAL_DQN.md`'s three upgrades — plus **CQL**, **IQL**, and
**Discrete BCQ** from `docs/OFFLINE_RL.md`.

**Explicit caveat, printed by the script and repeated here:**
`avg_reward_logged_dataset` is the *same* observed reward for every row,
because none of these seven algorithms regenerates data (unlike Phase 2's
baseline policies) — they are all trained offline on one fixed logged
dataset. It is not a causal per-algorithm reward and must not be read as
one. What *is* real and per-algorithm: `loss` (each algorithm's own final
training loss) and `greedy_logged_agreement` (how often each algorithm's
learned greedy policy matches the logged rule-based decision — the
headline number `docs/OFFLINE_RL.md` reports in detail, including CQL and
BCQ's measured jump from ~3-15% to ~86-87% once their respective
conservatism mechanisms are engaged).

**Table shape**:

| Model | Avg reward (observed, non-causal) | Loss | Greedy/logged agreement | Training time |
|---|---|---|---|---|
| Rule-Based | (reference) | - | 1.0000 (reference) | - |
| Vanilla DQN | | | | |
| Double DQN | | | | |
| Double DQN + PER | | | | |
| DRQN (LSTM) | | | | |
| CQL | | | | |
| IQL | | | | |
| Discrete BCQ | | | | |

## Phase 4 — Deep Learning Comparison

See `docs/DEEP_LEARNING_COMPARISON.md` for the full writeup: Logistic
Regression, Random Forest, MLP (new), and LSTM (new) compared on identical
splits and metrics; Transformer documented as future work, not
implemented.

## Tests

- `tests/test_session_simulator.py` — the `intervention_policy` hook (4 new
  tests: default-matches-prior-behavior, always/never-intervene, and the
  callback receives only pre-interaction fields).
- `tests/test_baselines.py` — all three baseline policies actually control
  generation as claimed (5 tests).
- `tests/test_classifier.py` — MLP registration, training, and
  model-agnostic permutation importance (3 new tests).
- `tests/test_sequence_classifier.py` — the LSTM classifier's sequence
  windowing, convergence, and determinism (6 tests).
- `tests/test_experiment_reproducibility.py` — seeding and environment
  reporting (4 tests).
