# Thesis Appendix: Phases 5-9

`run_appendix_analysis.py` is the entry point:

```bash
python run_appendix_analysis.py
python run_appendix_analysis.py --students 30 --sessions 3 --seeds 42 123 2025 2026 9999
```

writes `outputs_appendix/<timestamp>/{ablation_table.csv,
sensitivity_need_threshold.csv, sensitivity_reward_weight.csv,
seed_analysis.csv, significance_tests.csv, confusion_full_features.csv,
confusion_restricted_features.csv, error_analysis.md}`.

Two of the five phases here (5 and 6) are thin wrappers over Module 13's
already-tested `AblationRunner`/`SensitivityRunner` — nothing new was built
for those. What's new is `dataset_generator/experiment_appendix.py`: the
multi-seed statistics layer (Phase 7/8) and the confusion-matrix
error-ranking helper (Phase 9), neither of which existed before this pass.
10 new tests (`tests/test_experiment_appendix.py`) cover all three.

## Phase 5 — Ablation Study

A representative run (30 students × 3 sessions, seed 42):

| Configuration | BAS Mean | BAS Volatility | Reward Avg | Cooldown Activations |
|---|---|---|---|---|
| Full Model (baseline) | 0.7196 | 0.0306 | −0.0049 | 596 |
| Without BAS "learning_response" features | 0.6643 | 0.0277 | −0.0059 | 598 |
| Without BAS "behaviour" features | 0.7648 | 0.0396 | −0.0040 | 598 |
| Without EMA (temporal smoothing) | 0.7121 | **0.1075** | −0.0043 | 599 |
| Without Reward "performance" category | – | – | −0.0078 | 596 |
| Without Reward "behaviour" category | – | – | −0.0045 | 596 |
| Without Reward "cost" category | – | – | −0.0027 | 596 |
| Without Cooldown | – | – | – | 532 |

Removing EMA smoothing more than triples BAS volatility (0.031 → 0.108) —
direct, measured confirmation that temporal smoothing does real work.
Removing the "performance" reward category shifts average reward the most
(−0.0078, the most negative of any row) confirming it is the dominant
positive-pulling term in the baseline.

## Phase 6 — Sensitivity Analysis

**Reward weight (behaviour category multiplier):**

| Multiplier | Reward Avg | Positive Ratio |
|---|---|---|
| 0.0 (disabled) | −0.0045 | 41.60% |
| 0.5 | −0.0047 | 41.08% |
| 1.0 (default) | −0.0049 | 41.30% |
| 1.5 | −0.0050 | 41.26% |
| 2.0 | −0.0050 | 41.41% |

A real but small monotonic effect — doubling the weight shifts mean reward
by only ~0.0005. The reward function is not highly sensitive to this
weight at the ranges tested, which is itself a useful robustness finding,
not a null result to hide.

Need-threshold sensitivity (already reported in `docs/EVALUATION.md`):
0.05 → 0.15 → 0.25 drops the required-intervention count 251 → 29 → 0 — the
calibration cliff behind the `None`-not-zero execution-rate contract.

## Phase 7 — Multiple Seeds (5 runs, mean ± SD, 95% CI)

| Metric | Mean ± SD | 95% CI |
|---|---|---|
| DQN agreement | 0.0286 ± 0.0135 | (0.0119, 0.0454) |
| CQL agreement | 0.8815 ± 0.0079 | (0.8717, 0.8913) |
| BCQ agreement | 0.8704 ± 0.0019 | (0.8680, 0.8728) |
| Random Forest accuracy | 0.9993 ± 0.0009 | (0.9981, 1.0005) |

CQL/BCQ aren't just higher on average than vanilla DQN — their spread
(SD ≈ 0.002–0.008) is an order of magnitude tighter than DQN's (SD ≈
0.0135), meaning the conservative mechanisms are also far more
*consistent* across seeds, not just luckier on one run.

## Phase 8 — Statistical Significance

| Comparison | t | p | Cohen's d | Wilcoxon p |
|---|---|---|---|---|
| CQL vs. DQN agreement | 149.996 | <0.000001 | 77.30 | 0.0625 |
| BCQ vs. DQN agreement | 127.290 | <0.000001 | 87.49 | 0.0625 |

Both comparisons are significant at any conventional threshold, with
effect sizes far beyond the conventional "large" cutoff (Cohen's d > 0.8)
— the two distributions do not meaningfully overlap. Wilcoxon's p = 0.0625
is the minimum achievable value at n=5 paired samples (not a
contradiction — a nonparametric sign-based test has limited power at this
sample size; the parametric t-test is the more informative result here
given every seed shows the same large, same-signed difference).

## Phase 9 — Confusion Matrices and Error Analysis

**Full feature set** (Random Forest) — perfect diagonal, 100% accuracy;
consistent with the separability finding in
`docs/DEEP_LEARNING_COMPARISON.md`, restated here in matrix form rather
than re-argued.

**Restricted feature set** (`experiment_appendix.RESTRICTED_FEATURES` —
behavioural/contextual features only, deliberately excluding
`semantic_similarity`/`confidence`/`correctness`), Logistic Regression:
accuracy 0.8893, ROC-AUC (OvR) 0.9692, Expected Calibration Error 0.1044.

| True | Predicted | Count | Share of all errors |
|---|---|---|---|
| Distracted | Focused | 33 | 54.1% |
| Focused | Distracted | 19 | 31.1% |
| Impulsive | Focused | 5 | 8.2% |
| Focused | Impulsive | 4 | 6.6% |

Distracted↔Focused confusion accounts for 85% of all misclassifications.
Impulsive stays largely distinguishable (only 9 of 61 errors involve it)
despite having the smallest support — its behavioral signature (very fast
responses) survives even without the response-quality features. This is
the separability finding from the other direction: remove the features
that make classes trivially separable, and errors concentrate exactly
where domain intuition says two states should be genuinely hard to
distinguish (mild inattention vs. sustained focus), not scattered randomly.

## Tests

`tests/test_experiment_appendix.py` — 10 tests covering
`compute_seed_statistics` (basic correctness, minimum-sample-size
validation, narrower CI for tighter data), `paired_significance` (detects
a real large difference, doesn't falsely flag noise as significant,
input-validation errors), and `analyze_confusion_errors` (ranking order,
empty result for a perfect diagonal, percentages summing to 1.0).
