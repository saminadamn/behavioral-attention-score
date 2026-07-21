# Phase 4: Deep Learning Comparison (Attention-State Classifier)

`run_comparison_study.py` trains four attention-state classifiers on the
same dataset, split, and evaluation metrics, so their numbers are directly
comparable: **Logistic Regression**, **Random Forest** (both pre-existing,
Module 8), **MLP** (feedforward neural network, added this phase), and
**LSTM** (sequence model, added this phase). A fifth row, **Transformer**,
is explicitly future work — not implemented, not approximated, consistent
with the project's no-fabrication principle.

## What's new in this phase

**MLP** (`dataset_generator/classifier/models.py::MLPModel`) — scikit-learn's
`MLPClassifier` (already a core dependency; no new package), registered
with the same `ClassifierModelFactory` every other model uses. Two hidden
layers (64, 32) — deliberately modest, since this dataset is ~60 tabular
features per interaction, not images or raw sequences, so a large network
has no structural advantage. Runs through the *exact* same
`AttentionClassifierTrainer.train()` path as every other tabular model,
including permutation-based feature importance (verified to work without
any tree-specific attributes, since permutation importance is model-agnostic
by construction).

**LSTM** (`dataset_generator/classifier/sequence_model.py::train_lstm_classifier`)
— a genuinely different problem shape: instead of one interaction's
features in isolation, it sees the last `sequence_length` interactions and
predicts the *current* one's attention state. Reuses
`RecurrentQNetwork` — the same LSTM implementation already verified against
numerical gradients for the offline-RL package (`docs/RL_FORMALIZATION.md`
§1.5) — trained via ordinary softmax cross-entropy instead of a Bellman
target. No new backpropagation code: `apply_output_gradient` is the one
shared BPTT pass every algorithm in this project's LSTM family (DQN, CQL,
IQL, BCQ, and now this classifier) reuses.

Everything upstream of the model itself — `FeatureSelector`, `split_dataset`
(student-aware, same leakage-safety guarantee), `Preprocessor` — is the
exact same code the tabular classifiers use, so a difference in accuracy
between LSTM and the tabular models reflects the sequence-vs-snapshot
distinction, not a different feature set or split.

## Table shape

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Training time |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| MLP | | | | | |
| LSTM | | | | | |
| Transformer | not implemented — future work | - | - | - | - |

Populated by `run_comparison_study.py`'s `classifier_comparison_table.csv`
on every run — no numbers are hardcoded here, since they depend on dataset
scale and the run's seed.

## Reading the numbers honestly

As with every classifier result in this project (see `README.md`'s
research-software disclaimer and `docs/EVALUATION.md`), these metrics
describe how well each model recovers the *synthetic generator's own*
attention-state assignment rule — not real-world classification accuracy.
A model scoring near 100% on a small run reflects the known
feature-separability property of this dataset at small scale (documented
in `docs/DESIGN_DECISIONS.md`), not evidence the model would generalize to
real classroom data.

## Future work

- **Transformer encoder** — a self-attention sequence model over the same
  windowed features the LSTM sees, to test whether attention over the
  window outperforms the LSTM's recurrent summary. Not built here: a
  from-scratch NumPy multi-head attention + positional encoding
  implementation is a substantially larger addition than reusing the
  already-verified LSTM, and deserves its own dedicated pass rather than a
  rushed, unverified version. See `docs/PHASE_ROADMAP.md`.
