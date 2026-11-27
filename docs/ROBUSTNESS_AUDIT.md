# Robustness audit: latency without a question-complexity baseline

## How this was found

This repo's `BehaviouralAttentionScorer` (Module 9) is a mature, 11-feature
weighted engine with its own ablation/sensitivity tooling
(`docs/APPENDIX_ANALYSIS.md`, `docs/EVALUATION.md`). The audit here applies
the same adversarial-case methodology used to clinically stress-test the
companion live classifier in
[ADHD-behavioral-attention-score](https://github.com/saminadamn/ADHD-behavioral-attention-score)
(see that repo's `docs/model_limitations.md`): construct a scenario that
isolates one specific clinical pattern — here, an impulsive/premature
answer — and check whether the engine's score moves the way that pattern
implies, rather than relying only on the existing synthetic-dataset
statistics to surface a problem.

The constructed case: a response answered in 0.3 seconds to a long,
multi-clause question whose own `prompt_reading_time_seconds` is 14 —
i.e. an answer that arrived before there was plausibly time to finish
*reading* the question, independent of how fast that student normally
responds.

## What broke

Before the fix, this premature-answer case and a genuinely fast, fluent
answer from a quick-baseline student scored within noise of each other:

| Case | `prompt_reading_time_seconds` | `behaviour_response_latency` | Score (before fix) |
|---|---|---|---|
| Genuinely fast, fluent answer | 0.6 | 0.4 | 0.915 |
| Premature answer to a complex question | 14.0 | 0.3 | 0.912 |

A 0.003 gap is not a meaningful distinction — the engine could not tell
these two cases apart.

### Root cause

The only latency feature feeding the weighted score,
`abs_normalized_latency`, is built from `behaviour_normalized_latency`:
latency normalized against *that same student's own baseline and
variance* (`dataset_generator/bas/feature_extractor.py`). That is the
right signal for "unusually fast for this particular student," but it has
no notion of how long the *question itself* should plausibly take to read
and think about. A student with a fast natural baseline (0.5s) answering
in 0.3s reads as only mildly fast relative to their own history — the
exact same signal whether the question was "Six times seven?" or a
four-clause, multi-step word problem.

## Fix

Added `prompt_relative_latency` (`dataset_generator/bas/feature_extractor.py`):
response latency expressed as a multiple of the prompt's own
`prompt_reading_time_seconds`, guarded against a zero/near-zero reading
time with a small floor. This is deliberately a *separate* feature from
`abs_normalized_latency` rather than a replacement — the two capture
different things (student-relative vs. question-relative) and neither
subsumes the other.

Wired into `dataset_generator/bas/config.py`'s `default_bas_config()` as
an 11th weighted feature (positive polarity: a higher ratio is more
evidence of attention, since it means the student took at least as long
as the question warranted):

- `prompt_relative_latency`: weight `0.12`, min-max normalized `[0.0, 1.0]`
  (0x reading time → 0 evidence; ≥1x reading time → full evidence).
- `abs_normalized_latency`: weight reduced from `0.10` to `0.02` — it still
  contributes, but `prompt_relative_latency` is now the feature actually
  responsible for catching this pattern.
- `hesitation`: weight reduced from `0.10` to `0.06` to fund the above
  without changing the overall timing-signal budget relative to the
  learning-response features (`correctness`, `semantic_similarity`, etc.),
  since `hesitation` and the new feature both describe response timing.

No other feature, normalization, or weight changed. The scorer's
missing-value renormalization (`BehaviouralAttentionScorer.score`) and
every other package (classifier, reward, intervention, orchestration)
are unaffected — `prompt_relative_latency` is read straight off two
already-observable `DatasetRecord` fields (`behaviour_response_latency`,
`prompt_reading_time_seconds`), not new simulator state.

### Result after the fix

| Case | Score (before) | Score (after) |
|---|---|---|
| Genuinely fast, fluent answer | 0.915 | 0.899 |
| Premature answer to a complex question | 0.912 | **0.821** |

The gap widens from 0.003 to 0.078 — well past the 0.05 margin the
regression test asserts. The genuinely-fast case drops only slightly
(0.915 → 0.899, from the `hesitation`/`abs_normalized_latency` weight
reductions it also shares in), while the premature case drops sharply —
the fix adds a missing signal rather than uniformly penalizing speed.

## What's still a real limitation

- **`prompt_reading_time_seconds` is itself a generator-config-derived
  quantity**, not measured human reading time — see
  `dataset_generator/config/prompt_generation.py`. The ratio is only as
  good as that estimate.
- **This is a synthetic-simulator audit, not a clinical validation.** As
  with every other claim in this repository, `prompt_relative_latency`'s
  effect is demonstrated on constructed and generator-produced data, not
  real classroom interaction.
- **The weight split (0.12 new / 0.02 old / 0.06 hesitation) is a
  reasoned choice, not a fit to any ground truth** — `docs/EVALUATION.md`'s
  sensitivity-sweep tooling (`run_appendix_analysis.py`) can be re-run
  against this config to check how sensitive downstream metrics are to
  these particular numbers.

## Regression coverage

`tests/test_bas.py`:

- `test_feature_extractor_extracts_all_expected_features` — asserts
  `prompt_relative_latency` is present in `BASObservation.raw_values`.
- `test_prompt_relative_latency_low_for_premature_answer` — unit-checks
  the ratio computation.
- `test_prompt_relative_latency_guards_against_zero_reading_time` —
  checks the zero-reading-time floor.
- `test_premature_answer_scores_lower_than_fluent_fast_answer` —
  end-to-end: runs both constructed cases through `BASEngine` and asserts
  the premature case scores at least 0.05 below the fluent case. This is
  the test that fails on a revert of the fix (the gap collapses to ~0.01).

`dataset_generator/evaluation/ablation.py`'s `BAS_FEATURE_CATEGORIES["behaviour"]`
was also updated to include `prompt_relative_latency`, so existing
ablation/sensitivity sweeps (`docs/APPENDIX_ANALYSIS.md`) cover it
alongside the other behavioural features rather than silently ignoring it.
