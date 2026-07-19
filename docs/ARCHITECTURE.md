# Architecture

This document explains every module's purpose, inputs/outputs, internal
components, design decisions, tradeoffs, time complexity, extensibility, and
interfaces with its neighbors, in generation order.

```
Module 1 (Config)
    ↓
Module 2 (Student Profiles)
    ↓
Module 3 (Prompt Generator)
    ↓
Module 4 (Response Generator)
    ↓
Module 5 (Behaviour Generator)
    ↓
Module 6 (Session Simulator)
    ↓
Module 7 (Dataset Assembly)
    ↓
Module 8 (Attention Classifier) ─┐
    ↓                            │ (optional confidence, into 9/10/11)
Module 9 (BAS Engine)            │
    ↓                            │
Module 10 (Reward Model)  ←──────┤
    ↓                            │
Module 11 (Intervention Engine) ←┘
    ↓
Module 12 (LangGraph Orchestration)
```

---

## Module 1 — Configuration (`dataset_generator/config/`)

**Purpose:** the single source of truth for every generation parameter.
Nothing downstream hardcodes a probability, range, or distribution
parameter — everything traces back to one `GeneratorConfig` instance.

**Inputs:** none (it's the root of the dependency graph) — or, for a
non-default run, hand-constructed field overrides.

**Outputs:** a validated `GeneratorConfig`; `compute_fingerprint(config)` →
a deterministic SHA-256 hash (excluding only the free-form `experiment`
metadata field) used by every downstream artifact's manifest.

**Internal components:** `GeneratorConfig` (top-level model), plus per-
concern sub-models — `FeatureDistributionParams`/`StateDistributionConfig`/
`DistributionConfig` (per-attention-state statistical distributions),
`TransitionMatrixConfig` (base Markov matrix), `StudentProfileConfig`/
`ProfileMultipliers` (archetype definitions), `BaseRates`, `OutputConfig`,
`VersionMetadata`, `ExperimentMetadata`, `PromptGenerationConfig`,
`ResponseGenerationConfig`, `BehaviourGenerationConfig`,
`SessionSimulationConfig`, `CurriculumConfig`. `config/derive.py`'s
`resolve_profile_parameters` turns an archetype's multipliers into concrete
sampling ranges relative to the Focused baseline. `config/attention_state.py`
holds `combine_transition_matrix`/`reachability_violations`, extracted
specifically so both `GeneratorConfig`'s own validator and Module 6's
`TransitionEngine` can share the identical combination logic without a
circular import.

**Design decisions:** every model is a frozen Pydantic `BaseModel`, so an
invalid config (probabilities that don't sum to 1, an unreachable Markov
state, a profile referencing an undefined key) fails at construction time,
not mid-simulation. Archetype effects are multiplier-derived relative to a
Focused baseline (see [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)), not
hardcoded per-archetype absolutes.

**Tradeoffs:** the cross-field validator (`_check_cross_references`) is
non-trivial (checks Markov reachability for every profile's *effective*
matrix, not just the base one) — this makes config construction itself
O(states² × profiles), negligible in practice (single digits of each) but a
real cost paid once per `GeneratorConfig()` call, not amortized.

**Time complexity:** O(1) relative to dataset size — config validation cost
depends only on the number of attention states/profiles/subjects, never on
`students`/`sessions_per_student`.

**Extensibility:** adding a new attention state requires updating
`AttentionState`, every `TransitionMatrixConfig` row, every
`StateDistributionConfig`, and re-verifying reachability — a deliberately
non-trivial change, since attention states are a fixed taxonomy this
research project's contribution depends on, not meant to be casually
extended per-experiment. Adding a new *profile* (archetype) is much
cheaper: one new `StudentProfileConfig` entry plus a normalized
`profile_distribution`.

**Interfaces:** `GeneratorConfig` is passed by reference (never copied) into
every generator in Modules 2–7; `compute_fingerprint`/`config.version_metadata`
flow into Module 7's `DatasetManifest`.

**Interaction with neighbors:** every one of Modules 2–7 takes `config:
GeneratorConfig` as a constructor/function argument; none of them read
global state or environment variables for generation parameters.

---

## Module 2 — Student Profile Generator (`generators/student_profile_generator.py`, `generators/profiles.py`)

**Purpose:** assign each of `config.students` synthetic students one of five
behavioural archetypes, with individually-jittered parameters.

**Inputs:** `GeneratorConfig`, `RNGStreams` (uses `student_rng` for the
population-level archetype assignment draw, plus a per-student-index-derived
RNG for that student's own parameter sampling).

**Outputs:** `list[Student]` — frozen, session-independent identity records
(`baseline_latency`, `latency_variance`, `engagement_tendency`,
`fatigue_rate`, `intervention_sensitivity`, `transition_modifier`,
`profile_seed`).

**Internal components:** `ProfileFactory` (decorator-registry), `BaseProfile`
(ABC with a single shared `generate_student()` implementation — resolve
parameters → per-student RNG → sample within resolved ranges → jitter
`engagement_tendency`), five registered leaf classes (`FocusedProfile`,
`FatiguedProfile`, `DistractibleProfile`, `ImpulsiveProfile`,
`RecoveringProfile`) that only declare `profile_key`/`display_name`/
`description`.

**Design decisions:** the factory/registry pattern (shared with Modules 4
and 8) means adding a sixth archetype is "register a new class with three
class attributes," not editing a dispatch chain; `__init_subclass__`
enforces those attributes are non-empty at class-definition time.
Per-student parameter sampling uses `student_local_rng(seed, index)` (a
`SeedSequence([seed, index])`-derived generator) rather than drawing
sequentially from a shared stream, so a given student's parameters are
identical regardless of population size or generation order.

**Tradeoffs:** the population-level archetype *assignment* still comes from
one shared draw (`student_rng.choice(..., size=config.students)`) — so while
each *student's own* sampled parameters are order-independent, which
archetype gets assigned to student index `i` is not (changing
`config.students` reshuffles the whole assignment vector, since `choice`
draws all `size` values from one call). This is an accepted tradeoff:
per-student parameter reproducibility was judged more valuable than
archetype-assignment reproducibility across different population sizes.

**Time complexity:** O(students) — one archetype draw (vectorized) plus one
`generate_student()` call per student, each O(1).

**Extensibility:** new archetypes are cheap (see above); changing how
parameters are jittered only touches `BaseProfile.generate_student`, shared
by all five.

**Interfaces:** `generate_students(config, rng_streams) -> list[Student]` is
the sole public entry point; `Student` is consumed by Modules 4, 5, 6, and 7.

**Interaction with neighbors:** feeds `Student` objects into
`SessionSimulator` (Module 6), which threads them through Modules 4/5 for
every interaction; Module 7 copies `Student` fields directly onto
`DatasetRecord` (never re-deriving them).

---

## Module 3 — Prompt Generator (`generators/prompt_generator.py`)

**Purpose:** produce curriculum-driven, difficulty/Bloom's-taxonomy-
conditioned prompts, entirely independent of any student/attention-state
concept.

**Inputs:** `GeneratorConfig` (specifically `curriculum`, `prompt_generation`),
a dedicated `prompt_rng`.

**Outputs:** `Prompt` objects (`prompt_text`, `difficulty`,
`cognitive_level`, `keywords`, `learning_objective`, `metadata` — reading
time, token count, difficulty/cognitive-complexity scores, Flesch-Kincaid
readability grade).

**Internal components:** `PromptGenerator` (stateful only in an
ID counter and a duplicate-avoidance text set), `CurriculumConfig` (7
subjects × topics × learning-progression ordering), `Difficulty`/
`CognitiveLevel` enums with `BLOOM_ORDER` fixing ascending complexity.

**Design decisions:** the module's own docstring states it "knows nothing
about BAS, attention states, students, or reinforcement learning" — prompt
difficulty/content is generated independently of who will answer it or how,
which is what makes attention-state-conditioned response quality (Module 4)
a genuinely separate causal step rather than baked into prompt selection.

**Tradeoffs:** duplicate-text avoidance retries up to
`duplicate_retry_limit` (default 20) times before giving up and counting a
forced duplicate — an explicit, bounded tradeoff between guaranteed
uniqueness and unbounded retry cost at high generation volumes.

**Time complexity:** O(1) amortized per prompt (bounded retry loop);
O(n) for `generate_batch(n)`; O(topics) for `generate_curriculum`.

**Extensibility:** new subjects/topics are pure data (`CurriculumConfig`
additions); new difficulty/cognitive levels require touching the
`Difficulty`/`CognitiveLevel` enums and every template keyed by them.

**Interfaces:** `generate_prompt`/`generate_batch`/`generate_curriculum`.

**Interaction with neighbors:** `SessionSimulator` (Module 6) calls
`generate_prompt` once per interaction; the resulting `Prompt` is passed
into Module 4's `generate_response` and stored verbatim on
`InteractionRecord` (Module 6) so Module 7 never regenerates it.

---

## Module 4 — Response Generator (`generators/response_generator.py`, `response_strategies.py`, `response_scoring.py`)

**Purpose:** given a prompt, a student, and a **caller-supplied** attention
state, generate a synthetic response whose every feature (correctness,
semantic similarity, engagement, confidence, etc.) is computed from the
generated text and the attention-state/profile/session context — never a
label copied in from elsewhere.

**Inputs:** `Prompt`, `Student`, `AttentionState` (decided by Module 6, not
this module), `SessionContext`; `response_rng`.

**Outputs:** `Response` (`correctness_score`, `semantic_similarity`,
`lexical_diversity`, `sentiment`, `engagement_proxy`, `confidence`,
`hesitation_markers`, `features` (`token_count`/`repetition_ratio`/
`coherence_score`/`topic_shift`), `metadata` (`correctness_probability`,
`strategy_used`, etc.)).

**Internal components:** `ResponseStrategyFactory` (same registry pattern as
Module 2) with three registered strategies (`FocusedStrategy`,
`DistractedStrategy`, `ImpulsiveStrategy`), each a fixed trait vector
(`completeness`, `reasoning_depth`, `detail_level`, `topic_adherence`,
`error_tendency`, `length_multiplier`); `response_scoring.py`'s
`expected_correctness`/`sample_correctness_score`/`confidence_score`/
`coherence_score`/`engagement_proxy` — deterministic weighted formulas plus
one Gaussian noise draw for the final sampled score.

**Design decisions:** controlled overlap noise
(`semantic_similarity_noise_std`/`confidence_noise_std` per attention state)
was added specifically so Module 8's classifier faces a genuinely noisy,
non-trivially-separable problem rather than a shortcut feature (see
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)). `_build_response_id` includes
`session_id` (not just student + interaction number) since interaction
numbers restart at 1 every session — a fix for a real ID-collision bug
caught during Module 7 development.

**Tradeoffs:** `duplicate_retry_limit`-bounded retries again trade
guaranteed non-repetition for bounded cost, this time against the student's
*own* immediately-preceding response text.

**Time complexity:** O(1) per response (bounded retries, fixed-size feature
computation over the generated text).

**Extensibility:** a fourth attention state would require a fourth
registered `ResponseStrategy` with its own trait vector — the factory
pattern makes this additive, but calibrating a new trait vector against the
existing three is a modeling decision, not a mechanical one.

**Interfaces:** `ResponseGenerator.__init__(config, rng)`,
`.generate_response(*, prompt, student, attention_state, session_context)`.

**Interaction with neighbors:** consumes `Prompt` (Module 3), `Student`
(Module 2), attention state and `SessionContext` (Module 6); its output
feeds Module 5 (behaviour is derived partly from the response) and Module 7
(flattened onto `DatasetRecord`).

---

## Module 5 — Behaviour Generator (`generators/behaviour_generator.py`, `behaviour_scoring.py`)

**Purpose:** sample the behavioural signals that aren't already produced by
Module 4 (response latency, hesitation duration, interaction duration,
fatigue level, rolling latency/engagement), personalized to the student's
own baseline and evolved by session progress and intervention effects.

**Inputs:** `Student`, `Prompt`, `Response` (Module 4's output — read, never
resampled), `AttentionState`, `SessionContext`; the `noise_rng` stream.

**Outputs:** `BehaviourRecord` (`response_latency`, `interaction_duration`,
`hesitation_duration`, `fatigue_level`, `rolling_latency`,
`rolling_engagement`, `features.normalized_latency`,
`features.transition_occurred`, plus pass-through `engagement`/
`response_length`/`repetition_ratio`/`topic_shift` copied from `Response`).

**Internal components:** `fatigue_level()` (session-progress-driven, reduced
by recent interventions weighted by the student's own
`intervention_sensitivity`), `sample_response_latency()` (student-baseline-
relative, attention-state-multiplied, fatigue-inflated, drawn from a
truncated normal), `normalized_latency()` (z-score against the student's own
baseline — the same quantity Module 9's BAS features and Module 11's
observation extractor both reuse directly, rather than recomputing).

**Design decisions:** the module's docstring explicitly states it does
*not* resample `response_length`/`repetition_ratio`/`topic_shift`/
engagement — those are Module 4's outputs, copied through unchanged. This
"never regenerate behaviour" boundary is enforced at every downstream layer
too (Modules 7, 9, 10, 11 all read these fields from `DatasetRecord`/
`BehaviourRecord` rather than recomputing).

**Tradeoffs:** latency is clipped to `[latency_clip_min=0.3,
latency_clip_max=90.0]` seconds — a deliberate realism bound that could,
in principle, clip an extreme legitimate draw at very high fatigue/variance
combinations; accepted as a small realism cost against protecting
downstream normalization from unbounded outliers.

**Time complexity:** O(1) per interaction.

**Extensibility:** adding a new behavioural signal means adding a field to
`BehaviourRecord`/`BehaviourFeatures` and a corresponding scoring function —
isolated from Modules 4/6/7's own responsibilities.

**Interfaces:** `BehaviourGenerator.__init__(config, rng)`,
`.generate_behaviour(*, student, prompt, response, attention_state, session_context)`.

**Interaction with neighbors:** reads `Student` (2), `Prompt` (3), `Response`
(4); its `fatigue_level`/`rolling_engagement`/`normalized_latency` feed
directly into Module 9's BAS feature vector and Module 11's need-detection
signals.

---

## Module 6 — Temporal Session Simulator (`generators/session_simulator.py`, `transition_engine.py`, `session_batch.py`)

**Purpose:** the only module that *decides* attention state, via a per-
student, per-profile Markov chain, and drives one full session's worth of
prompt→response→behaviour→transition cycles.

**Inputs:** `Student`, `session_id`; internally, one shared
`PromptGenerator`/`ResponseGenerator`/`BehaviourGenerator`/`TransitionEngine`
(all dependency-injected via `SessionSimulator.__init__`, never constructed
inline).

**Outputs:** `SessionRecord` (`interactions: list[InteractionRecord]`,
`transition_history`, `intervention_history`, `statistics`, `summary`).

**Internal components:** `TransitionEngine.effective_matrix(profile_key)`
(base matrix + profile's `transition_modifiers`, via the same
`combine_transition_matrix` function `GeneratorConfig`'s own validator
already used — "guaranteed identical to the one already checked for
reachability"); `.sample_initial_state()` (drawn from `class_balance`, the
population marginal, since no prior state exists yet); `.sample_next_state()`
(drawn from the current state's effective-matrix row, boosted toward
`Focused` when an intervention just fired — reshaping the transition
probabilities, never overwriting the state directly). `SessionSimulator`
decides per-interaction whether an intervention fires
(`_decide_intervention`: never above an engagement threshold, otherwise at
`config.intervention_probability`), builds `SessionContext`, and calls
Modules 3/4/5 in sequence.

**Design decisions:** interventions only ever *reshape* the transition
probability row, never force a state — the causal story is "an intervention
makes recovery to Focused more likely," not "an intervention magically fixes
attention." `SessionStatistics` is built incrementally by a
`SessionStatisticsBuilder` (referenced from Module 6/7's shared pattern)
rather than recomputed from scratch at the end.

**Tradeoffs:** one shared `SessionSimulator` instance is reused across every
student/session in a `generate_sessions()` batch (not reconstructed per
session) — cheaper, but means its dependencies (the four generators) must
themselves be side-effect-free across calls, which they are by construction
(each only reads its own RNG stream and its call arguments).

**Time complexity:** O(session_length) per session — one prompt/response/
behaviour/transition cycle per interaction; O(students × sessions_per_student
× average_session_length) for a full `generate_sessions()` batch.

**Extensibility:** parallel session generation across processes is
explicitly not implemented — the module's own comments note it was
"deferred rather than half-built," since correct parallelization would
require partitioning RNG streams per process while preserving determinism,
a non-trivial addition rather than a quick win.

**Interfaces:** `build_session_simulator(config, rng_streams) ->
SessionSimulator`; `generate_sessions(config, students, sessions_per_student,
rng_streams) -> list[SessionRecord]`.

**Interaction with neighbors:** the only module that calls Modules 3, 4, and
5 directly; its `SessionRecord`s are the sole input to Module 7's
`DatasetBuilder`.

---

## Module 7 — Dataset Assembly (`pipeline/dataset_artifact.py`, `dataset_builder.py`, `dataset_statistics.py`, `dataset_export.py`)

**Purpose:** flatten every `SessionRecord` into one `DatasetRecord` per
interaction, validate the whole set, compute distributional statistics, and
stamp a versioned manifest — the single source of truth every downstream
module depends on.

**Inputs:** `GeneratorConfig`, `list[Student]`, `list[SessionRecord]` (all
dependency-injected — this module performs no simulation itself).

**Outputs:** `DatasetArtifact` (`records`, `statistics`, `validation`,
`metadata`, `manifest`, `exports`).

**Internal components:** `DatasetBuilder` (pure assembly — copies every
field from `Student`/`Prompt`/`Response`/`BehaviourRecord`/
`SessionRecord.summary` onto one `DatasetRecord`, never regenerating
anything); `validate_dataset()` (pandas-vectorized: missing values, duplicate
rows/IDs, range violations, invalid attention states, "impossible
transitions" — comparing each `behaviour_transition_occurred` claim against
the actually-observed state change within its session — orphan IDs, NaN/Inf
counts, schema consistency); `compute_dataset_statistics()` (per-feature
distributions, several distinct "balance" measurements at different
denominators — row-level `profile_balance` vs. session-level
`session_balance` are deliberately different, correlation matrix, missing-
value summary); `build_manifest()` (reuses `config.version_metadata` +
`compute_fingerprint` + `detect_git_commit()`, rather than introducing new
version-tracking fields).

**Design decisions:** validation and statistics are pandas-vectorized, not
per-row Python loops — necessary for the 100,000+-record stress-test scale
this module (and everything downstream) is designed against.
`export_dataset_artifact()` returns a **new** `DatasetArtifact` (via
`model_copy`) with `exports` populated, rather than mutating the artifact
passed in — consistent with the project's immutability convention.

**Tradeoffs:** `compute_dataset_statistics`'s correlation matrix forces
diagonal entries to `1.0` and falls back to `0.0` for undefined off-diagonal
pairs (e.g. constant columns) rather than `NaN` — a deliberate choice to
keep the matrix always fully populated for downstream consumers, at the
cost of `0.0` being ambiguous between "genuinely uncorrelated" and
"undefined."

**Time complexity:** O(n) for assembly (one `DatasetRecord` per interaction);
O(n) to O(n log n) for validation/statistics (vectorized pandas operations,
some involving per-column sorts).

**Extensibility:** new `DatasetRecord` fields require touching
`DatasetBuilder._build_record`, `FeatureRegistry` (`pipeline/
feature_registry.py`, used by Module 8's default feature selection), and any
validator that range-checks the new field.

**Interfaces:** `build_dataset_artifact(config, students, sessions) ->
DatasetArtifact`; `export_dataset_artifact(artifact, output_dir) ->
DatasetArtifact`.

**Interaction with neighbors:** the sole input to Modules 8, 9, 10, 11, and
(via `ObserverAgent`) Module 12's batch phase. Nothing downstream ever reads
a `SessionRecord`/`Student`/`Prompt`/`Response`/`BehaviourRecord` directly —
only `DatasetArtifact.records`.

---

## Module 8 — Attention Classifier (`classifier/`)

**Purpose:** an *optional* supervised classifier predicting `attention_state`
from a dataset's own feature columns, used only as an auxiliary confidence
signal for Modules 9/10/11 — never a required dependency, never a source of
the ground-truth attention-state label itself (that's Module 6's).

**Inputs:** `DatasetArtifact` (training), a `TrainingConfig` (model
name/split mode/calibration method/etc.); at inference time, a
`TrainingArtifact` plus `DatasetRecord`s.

**Outputs:** `TrainingArtifact` (fitted model + preprocessor,
`ClassificationMetrics`, optional `CalibrationResult`, optional
`FeatureImportanceReport`, `TrainingMetadata` including the exact
`feature_names` every inference call reindexes against); at inference,
`PredictionResult` (`predicted_state`, `probabilities`, `confidence`,
optional `explanation`).

**Internal components:** `ClassifierModelFactory` (same registry pattern as
Modules 2/4) with `LogisticRegressionModel`/`RandomForestModel`/
`GradientBoostingModel` always registered, plus optional `XGBoostModel`/
`LightGBMModel` registered only inside a `try/except ImportError` so a
missing optional dependency never breaks the factory; `Preprocessor`
(exactly one `.fit_transform()` call on train data, `.transform()` on
validation — enforced as a documented invariant); `calibrate_model()` (wraps
the already-fitted estimator in `sklearn.frozen.FrozenEstimator` before
`CalibratedClassifierCV`, fitting *only* the calibration mapping, never
refitting the base classifier); `compute_ece()` (hand-written Expected
Calibration Error + reliability bins).

**Design decisions:** `AttentionClassifierPredictor` is constructed *from* a
`TrainingArtifact` and holds no object with a `.fit()` method — structurally
incapable of accidentally retraining during inference.
`predict_with_explanation()` explicitly documents its explanation as the
training-time *global* top-10 permutation-importance features, not a true
per-instance attribution (SHAP is optional/separate, not computed by
default) — an implementation-honesty boundary consistent with the whole
project's stance.

**Tradeoffs:** target-leakage was caught and fixed during development —
`response_strategy_used` (bijective with the label by construction) and
`session_dominant_attention_state` (a session-level aggregate computed
*from* the target across the same rows) are both classified as
`FeatureCategory.TARGET` and excluded from default feature selection, not
silently left in as accidental shortcuts.

**Time complexity:** training is O(n) to O(n log n) depending on the chosen
model (random forest/gradient boosting scale roughly linearithmically in
practice); inference is O(1) per record after the model is fitted.

**Extensibility:** a new model type is one new `ClassifierModel` subclass
registered with the factory; feature selection is driven by
`FeatureRegistry`/`FeatureSelector`, so new dataset columns are automatically
excluded from default selection unless explicitly categorized as an input
feature.

**Interfaces:** `AttentionClassifierTrainer.train(dataset_artifact, config)
-> TrainingArtifact`; `AttentionClassifierPredictor(training_artifact)`,
`.predict`/`.predict_batch`/`.predict_proba`/`.predict_with_confidence`/
`.predict_with_explanation`.

**Interaction with neighbors:** entirely optional for Modules 9/10/11/12 —
each accepts `predictor: AttentionClassifierPredictor | None = None` and only
calls it once per batch (never per-record) when supplied, populating a
`classifier_confidence`-style field.

---

## Module 9 — BAS Engine (`bas/`)

**Purpose:** compute a Behavioural Attention Score per interaction:
`BAS_t = S(E(N(F(x_t))), BAS_{t-1})` — feature extraction, normalization,
evidence combination, and temporal smoothing as separate, inspectable
stages.

**Inputs:** `DatasetArtifact` (only — never regenerates behaviour, never
depends on Module 8's classifier except as an optional confidence input).

**Outputs:** `BASArtifact` (`records: list[BASRecord]` — `raw_score`,
`score`, `confidence`, per-feature `contributions`; `session_summaries`;
`statistics`).

**Internal components:** `feature_extractor.py` (`F`) — reads
`DatasetRecord` fields directly; `normalizer.py` (`N`) — per-feature
normalization strategies (`FeatureNormalizationConfig`); `evidence.py`/
`scorer.py` (`E`) — combines normalized features into a raw score, tracking
per-feature `contributions` for explainability; `smoother.py` (`S`) —
temporal smoothing against the previous interaction's BAS within the same
session; `confidence.py` — blends signal coverage/missingness into a
confidence score; `explanations.py`/`report.py` — human-readable
per-decision explanations and Markdown/JSON reports.

**Design decisions:** missing-feature handling renormalizes the weighted sum
over only the weight actually available, rather than treating a missing
signal as "zero evidence" — a missing signal contributes nothing, it
doesn't silently count as "no change." This same convention is reused
identically in Module 10's reward aggregation.

**Tradeoffs:** temporal smoothing means `BAS_t` depends on `BAS_{t-1}`
within a session — computation must proceed in interaction order per
session (cannot be trivially row-parallelized across a session, though
different sessions are independent).

**Time complexity:** O(n) — one pass per interaction, per session in order;
proven at 100,000+-record scale by its own stress test.

**Extensibility:** `FeatureNormalizationConfig`'s per-feature strategy
selection and `evidence.py`'s weighting are both config-driven — adding a
new feature to the BAS computation is a config change plus a feature
extractor addition, not a rewrite of the scoring formula.

**Interfaces:** `BASEngine().compute(dataset_artifact) -> BASArtifact`.

**Interaction with neighbors:** consumed by Modules 10, 11, and 12; Module
10 reuses `bas.config.FeatureNormalizationConfig`/`bas.normalizer.
normalize_value` directly (imported, not reimplemented) wherever reward
signals need the same normalization convention.

---

## Module 10 — Reward Model (`reward/`)

**Purpose:** compute a decomposed reward signal,
`R_t = R_performance + R_behaviour − R_cost`, from a dataset's BAS
trajectory — explicitly the credit-assignment preprocessing a future
intervention/RL layer would consume, not an RL algorithm itself.

**Inputs:** `DatasetArtifact`, `BASArtifact` (never recomputes BAS).

**Outputs:** `RewardArtifact` (`records: list[RewardRecord]` — `raw_reward`,
credited `reward`, `performance_reward`/`behaviour_reward`/`cost_reward`,
per-signal `contributions`, `confidence`; `session_summaries`; `statistics`).

**Internal components:** `signals.py` (`RewardSignalExtractor` — extracts
raw signals, e.g. `delta_bas`, `delta_engagement`, `delta_correctness`,
`intervention_cost`); `aggregator.py` (`RewardAggregator` — two-pass
aggregation producing an exact `raw_reward == performance + behaviour -
cost` identity, see [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md);
`RewardEngine`, the module's entry point); `temporal.py`
(`apply_temporal_credit_assignment` — immediate/discounted/moving-average
modes); `confidence.py` (`compute_reward_confidence` — blends missing-signal
ratio with BAS confidence).

**Design decisions:** the reward decomposition was refactored specifically
to be ablation-friendly — `RewardConfig.with_category_disabled(category)`
zero-weights a whole category (performance/behaviour/cost) via the real
constructor (never `model_copy`, which would skip validators), making "what
happens if we remove the behaviour term?" a one-line experiment.

**Tradeoffs:** like BAS, temporal credit assignment (discounted/moving-
average modes) requires session-ordered sequential computation — not
trivially parallelizable within a session.

**Time complexity:** O(n) — proven at 100,000+-record scale.

**Extensibility:** new reward signals are added to
`RewardSignalConfig`/`RewardCategory` and `RewardSignalExtractor`; the
aggregation/decomposition logic doesn't need to change since it operates
generically over whatever signals are configured.

**Interfaces:** `RewardEngine(config=None, predictor=None).compute(
dataset_artifact, bas_artifact) -> RewardArtifact`.

**Interaction with neighbors:** consumed by Module 11 and Module 12;
imports `bas.config.FeatureNormalizationConfig`/`bas.normalizer.
normalize_value` directly from Module 9 rather than reimplementing
normalization.

---

## Module 11 — Intervention Engine (`intervention/`)

**Purpose:** a deterministic (explicitly **not** RL) engine deciding
whether/when/which/why to intervene, from the already-computed
Dataset/BAS/Reward artifacts.

**Inputs:** `DatasetArtifact`, `BASArtifact`, `RewardArtifact`.

**Outputs:** `InterventionArtifact` (`decisions: list[InterventionDecision]`
— `need_score`, `trigger_reasons`, `severity`, `chosen_policy`,
`chosen_reason`, all evaluated `candidates`, `cooldown_suppressed`,
`confidence`; `session_summaries`; `statistics`).

**Internal components:** `observation.py`
(`InterventionObservationExtractor` — builds one `InterventionObservation`
per interaction, walking each session in order to track running state like
`consecutive_decline_count`); `detector.py` (`InterventionDetector` — 7
independently-computed need signals combined via
`config.need_signal_weights` into one `need_score`, with severity
thresholds); `policies.py` (`InterventionPolicy` ABC + 8 concrete policies —
`NoInterventionPolicy`, `HintPolicy`, `ConceptReviewPolicy`,
`DifficultyReductionPolicy`, `MotivationalPromptPolicy`,
`BreakRecommendationPolicy`, `EncouragementPolicy`, `QuestionReframingPolicy`
— each independently eligible/estimating gain-cost, sharing one `score()`
method); `scorer.py` (`PolicyScorer` — evaluates every eligible policy,
ranks candidates); `cooldown.py` (`CooldownManager` — per-session stateful:
minimum spacing, max interventions per session, duplicate-policy
prevention); `confidence.py` (blends policy agreement, signal coverage, and
BAS/reward confidence — reused directly, never recomputed); `planner.py`
(`InterventionPlanner`, the orchestrator wiring all of the above per
session, plus session-summary aggregation).

**Design decisions:** every policy is independent and "knows nothing about
the other policies" — `PolicyScorer` and `CooldownManager` are the only
places that reason across policies. `score()` is shared across all 8
policies rather than reimplemented per policy (see
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)).

**Tradeoffs:** gain/cost estimates per policy are explicitly heuristic
(scaled by how severe the triggering evidence is), not predictions from a
counterfactual simulation — because no such simulation exists. This is a
deliberate implementation-honesty tradeoff over a more "impressive" but
unfounded predictive claim.

**Time complexity:** O(n × p) where `p` is the number of registered policies
(currently 8, effectively a small constant) — proven at 100,000+-decision
scale.

**Extensibility:** `InterventionConfig.with_policy_disabled(name)` zero-
weights any policy for ablation studies via the real constructor. A new
policy is one new `InterventionPolicy` subclass registered with
`InterventionPolicyFactory` — the planner/scorer/cooldown machinery needs no
changes.

**Interfaces:** `InterventionPlanner(config=None,
predictor=None).plan(dataset_artifact, bas_artifact, reward_artifact) ->
InterventionArtifact`.

**Interaction with neighbors:** its components (`InterventionObservation`,
`InterventionDetector`, the 8 policies, `PolicyScorer`, `CooldownManager`)
are explicitly designed for Module 12 to reuse directly — and Module 12
does, via `InterventionAgent` wrapping `InterventionPlanner.plan` unchanged.

---

## Module 12 — LangGraph Orchestration (`orchestration/`)

**Purpose:** coordinate Modules 7/9/10/11 as a deterministic, checkpointable,
resumable, replayable multi-agent workflow — contributing zero domain logic
of its own.

**Inputs:** an optional pre-built `DatasetArtifact`, or generation
parameters (`student_count`, `sessions_per_student`) to build one fresh.

**Outputs:** a `WorkflowState` — the four batch artifacts plus
`tutor_actions`, `session_outputs`, `execution_metadata`, `errors`,
`execution_history`, `timing_stats`.

**Internal components:** see [ORCHESTRATION.md](ORCHESTRATION.md) for the
full breakdown — `state.py` (`WorkflowState`), `agents.py` (six thin
wrappers), `nodes.py` (six node factories + `_traced_node` + shared
helpers), `graph.py` (`route_next_step` + `build_graph`/`compile_graph`),
`memory.py` (`WorkflowMemory`), `checkpoint.py` (LangGraph-native
checkpointing + `recover_failed_session`), `serialization.py` (portable
JSON snapshots), `report.py` (execution/timing/failure reports).

**Design decisions:** `WorkflowState` is a `TypedDict`, not frozen Pydantic
(the one exception to the project-wide convention), because LangGraph's
partial-update merge model is fundamentally different — every artifact
*inside* it is still the real frozen Pydantic type. `route_next_step` is one
function reused at three wiring points rather than three near-duplicates.

**Tradeoffs:** the per-interaction walk costs one LangGraph "step" per
interaction, so very large batches require raising `recursion_limit`
explicitly — an operational cost of using LangGraph's own loop-safety
mechanism, documented directly in `graph.py`.

**Time complexity:** the batch phase is exactly as expensive as calling
`BASEngine.compute`/`RewardEngine.compute`/`InterventionPlanner.plan`
directly (O(n) each); the per-interaction walk adds O(n) LangGraph steps,
each O(1) node work — proven at 100,000+-interaction scale.

**Extensibility:** every agent/node accepts dependency-injected engine
instances, so a differently-configured `BASEngine`/`RewardEngine`/
`InterventionPlanner` (e.g. an ablation config) can be wired into the same
graph without touching `graph.py`. A future Module 13 (e.g. a dashboard)
would consume `WorkflowState`/`InterventionArtifact` reports rather than
re-deriving anything.

**Interfaces:** `build_graph(...) -> StateGraph`, `compile_graph(...) ->
CompiledStateGraph`, `compile_checkpointed_graph(...)`.

**Interaction with neighbors:** the only module that imports and calls all
four of Modules 7, 9, 10, and 11 in one place — every other module is
consumed independently by the tests/scripts that exercise it directly.
