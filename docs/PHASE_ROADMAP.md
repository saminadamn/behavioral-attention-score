# Phase Roadmap: 6, 10 (Not Yet Implemented)

Phases 1-4 (reproducibility, baseline comparison, RL evaluation, deep
learning comparison) are implemented and documented in
`docs/COMPARISON_STUDY.md` and `docs/DEEP_LEARNING_COMPARISON.md`. Phases
5, 7, 8, and 9 are implemented and documented in `docs/APPENDIX_ANALYSIS.md`
(`run_appendix_analysis.py`). Phase 6 (full hyperparameter grid) and
Phase 10 (Transformer and beyond) remain scoped but **not built** — each
is described here with what it would take and what already exists to
build it on.

## Phase 6 — Hyperparameter Study

**What exists to build on:** every hyperparameter listed (learning rate,
gamma, batch size, replay buffer size, target-sync frequency, hidden
dimension, LSTM hidden units, sequence length) is already a field on
`DQNConfig`/`CQLConfig`/`IQLConfig`/`BCQConfig`. A full grid across all
eight parameters is combinatorially large; a defensible first pass would
sweep 3-4 values of the two or three parameters likely to matter most
(learning rate, sequence length, batch size) via Module 13's existing
`SensitivityRunner` pattern (`docs/EVALUATION.md`), one-at-a-time, and
report the resulting curves — not a fabricated full-grid table.

## Phase 9 — Visualizations

**Mostly covered.** `run_experiment.py` produces confusion matrix, ROC
curve, precision-recall curve, calibration curve, feature importance, BAS
histogram, attention-state distribution, intervention distribution, and
transition heatmap as real matplotlib PNGs. `run_appendix_analysis.py`
adds the confusion matrices for the deliberately-restricted-feature error
case. Not yet built: training-loss/reward curves plotted per RL algorithm
(the data — `loss_per_epoch` — already exists on every trainer's artifact;
only the plotting call is missing).

## Phase 10 — Future Deep Learning

Explicitly future work, not approximated:

- **Transformer encoder** — see `docs/DEEP_LEARNING_COMPARISON.md`'s
  Future Work section; a from-scratch NumPy multi-head attention
  implementation deserves the same numerical-gradient verification the
  LSTM received, not a rushed version.
- **Knowledge Tracing** (e.g. Deep Knowledge Tracing / Bayesian Knowledge
  Tracing) — a different problem framing (modeling mastery of specific
  skills/concepts over time) than this project's attention-state
  classification; would need its own labeled concept-mastery signal, which
  doesn't currently exist in the generated dataset.
- **CQL, IQL, BCQ** — already implemented (`docs/OFFLINE_RL.md`); listed
  here in the original request as future work but actually complete.
- **LLM-based personalized tutor** — out of scope for this project's
  synthetic, privacy-safe design: an LLM tutor implies natural-language
  generation and real student interaction, which this project's
  non-clinical, non-deployed research scope (`README.md`'s disclaimer)
  deliberately does not attempt.
- **Multimodal attention detection** (video/audio/gaze) — would require
  real sensor data this project has no access to and no plan to collect
  (see the ethical-boundary discussion in `docs/DESIGN_DECISIONS.md`).
- **Real classroom dataset support** — the single largest prerequisite for
  every other future-work item in this project (`docs/EXPERIMENTAL_DQN.md`,
  `docs/OFFLINE_RL.md`) to become more than a research prototype: real,
  consented, ethically-governed data, which is an institutional and
  ethical undertaking, not an engineering task this repository can
  complete unilaterally.
