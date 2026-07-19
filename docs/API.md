# API Reference

This document covers every module's primary public entry points in full
detail (signature, parameters, return value, example), with each module's
supporting data models summarized in reference tables. Every function/class
below is importable from its package's top-level `__init__.py` (e.g.
`from dataset_generator.bas import BASEngine`).

---

## Module 1 — `dataset_generator.config`

### `default_config() -> GeneratorConfig`
Returns the Stage-2-reference configuration: `seed=42, students=100,
sessions_per_student=5, interactions_per_session=(20,40)`, 5 student
profiles, a 3×3 attention-state transition matrix, and per-state
distributions.

```python
from dataset_generator.config import default_config
config = default_config()
```

### `GeneratorConfig`
The top-level, frozen config model. Key fields: `seed: int`, `students:
int`, `sessions_per_student: int`, `interactions_per_session: tuple[int,
int]`, `noise: float`, `class_balance: dict[AttentionState, float]`,
`profile_distribution: dict[str, float]`, `transition_matrix:
TransitionMatrixConfig`, `distributions: DistributionConfig`, `profiles:
dict[str, StudentProfileConfig]`, `curriculum: CurriculumConfig`,
`prompt_generation`/`response_generation`/`behaviour_generation`/
`session_simulation`: their respective config models, `version_metadata:
VersionMetadata`, `experiment: ExperimentMetadata`.

### `compute_fingerprint(config: GeneratorConfig) -> str`
A deterministic SHA-256 hash of `config` (excluding the free-form
`experiment` field), stored in every downstream artifact's manifest.

```python
from dataset_generator.config import compute_fingerprint
fingerprint = compute_fingerprint(config)
```

### `load_config(path) -> GeneratorConfig` / `save_config(config, path) -> None`
YAML round-trip for a `GeneratorConfig`.

**Supporting types:** `AttentionState`, `Difficulty`, `CognitiveLevel`,
`FeatureDistributionParams`, `StateDistributionConfig`, `DistributionConfig`,
`TransitionMatrixConfig`, `StudentProfileConfig`, `ProfileMultipliers`,
`BaseRates`, `OutputConfig`, `VersionMetadata`, `ExperimentMetadata`,
`PromptGenerationConfig`, `ResponseGenerationConfig`,
`BehaviourGenerationConfig`, `SessionSimulationConfig`, `CurriculumConfig`,
`SubjectDefinition`, `TopicDefinition`. Helpers: `combine_transition_matrix`,
`reachability_violations`, `resolve_profile_parameters`,
`ResolvedProfileParams`.

---

## Module 2 — `dataset_generator.generators` (Student Profiles)

### `generate_students(config: GeneratorConfig, rng_streams: RNGStreams) -> list[Student]`
Assigns each of `config.students` a profile archetype (drawn from
`config.profile_distribution` via `rng_streams.student_rng`) and samples that
student's individual parameters via a per-student-index-derived RNG.

```python
from dataset_generator.config import default_config
from dataset_generator.generators import generate_students
from dataset_generator.utils import build_rng_streams

config = default_config()
streams = build_rng_streams(config.seed)
students = generate_students(config, streams)
```

### `ProfileFactory`
Decorator-registry for student archetypes.

```python
from dataset_generator.generators import ProfileFactory
ProfileFactory.available_profiles()   # -> list[str] of registered profile_keys
profile = ProfileFactory.create("Consistently_Focused")
student = profile.generate_student(student_index=0, config=config)
```

**Registered profiles:** `FocusedProfile` (`Consistently_Focused`),
`FatiguedProfile` (`Gradually_Fatigued`), `DistractibleProfile`
(`Highly_Distractible`), `ImpulsiveProfile` (`Highly_Impulsive`),
`RecoveringProfile` (`Recovering_Learner`).

**`Student` model:** `student_id, profile_name, description,
baseline_latency, latency_variance, engagement_tendency, fatigue_rate,
intervention_sensitivity, transition_modifier, profile_seed`.

---

## Module 3 — `dataset_generator.generators` (Prompt Generator)

### `PromptGenerator(config: GeneratorConfig, rng: np.random.Generator)`

```python
def generate_prompt(self, *, subject=None, topic=None, difficulty=None,
                     cognitive_level=None) -> Prompt
def generate_batch(self, n: int, **kwargs) -> list[Prompt]
def generate_curriculum(self, subject: str, *, prompts_per_topic: int = 1,
                         cognitive_levels=None, difficulty=None) -> list[Prompt]
```

```python
from dataset_generator.generators import PromptGenerator
prompt_gen = PromptGenerator(config, streams.prompt_rng)
prompt = prompt_gen.generate_prompt(subject="Mathematics", difficulty="medium")
```

**`Prompt` model:** `prompt_id, subject, topic, difficulty, cognitive_level,
prompt_text, expected_answer_type, estimated_response_length, keywords,
learning_objective, metadata: PromptMetadata` (`estimated_reading_time_
seconds, token_count, concept_count, difficulty_score,
cognitive_complexity_score, readability_grade, subject_id, topic_id`).

---

## Module 4 — `dataset_generator.generators` (Response Generator)

### `ResponseGenerator(config: GeneratorConfig, rng: np.random.Generator)`

```python
def generate_response(self, *, prompt: Prompt, student: Student,
                       attention_state: AttentionState,
                       session_context: SessionContext) -> Response
```

```python
from dataset_generator.generators import ResponseGenerator
response_gen = ResponseGenerator(config, streams.response_rng)
response = response_gen.generate_response(
    prompt=prompt, student=student, attention_state="Focused",
    session_context=session_context,
)
```

### `ResponseStrategyFactory`
```python
from dataset_generator.generators import ResponseStrategyFactory
ResponseStrategyFactory.available_states()   # -> list[AttentionState]
strategy = ResponseStrategyFactory.for_state("Focused")
```
**Registered strategies:** `FocusedStrategy`, `DistractedStrategy`,
`ImpulsiveStrategy` — each declares `completeness, reasoning_depth,
detail_level, topic_adherence, error_tendency, length_multiplier`.

**`Response` model:** `response_id, student_id, prompt_id, response_text,
correctness_score, response_length, semantic_similarity, lexical_diversity,
sentiment, engagement_proxy, confidence, hesitation_markers, features:
ResponseFeatures (token_count, repetition_ratio, coherence_score,
topic_shift), metadata: ResponseMetadata (correctness_probability,
strategy_used, difficulty, cognitive_level, subject, topic, attention_state,
student_profile, intervention_applied, session_progress)`.

---

## Module 5 — `dataset_generator.generators` (Behaviour Generator)

### `BehaviourGenerator(config: GeneratorConfig, rng: np.random.Generator)`

```python
def generate_behaviour(self, *, student: Student, prompt: Prompt,
                        response: Response, attention_state: AttentionState,
                        session_context: SessionContext) -> BehaviourRecord
```

```python
from dataset_generator.generators import BehaviourGenerator
behaviour_gen = BehaviourGenerator(config, streams.noise_rng)
behaviour = behaviour_gen.generate_behaviour(
    student=student, prompt=prompt, response=response,
    attention_state="Focused", session_context=session_context,
)
```

**`BehaviourRecord` model:** `student_id, session_id, interaction_number,
attention_state, response_latency, interaction_duration,
hesitation_duration, response_length, engagement_score, repetition_ratio,
topic_shift, rolling_latency, rolling_engagement, fatigue_level,
intervention_applied, features: BehaviourFeatures (normalized_latency,
fatigue_progression, rolling_latency, rolling_engagement,
transition_occurred), metadata: BehaviourMetadata`.

---

## Module 6 — `dataset_generator.generators` (Session Simulator)

### `build_session_simulator(config: GeneratorConfig, rng_streams: RNGStreams) -> SessionSimulator`
### `generate_sessions(config: GeneratorConfig, students: list[Student], sessions_per_student: int, rng_streams: RNGStreams) -> list[SessionRecord]`

```python
from dataset_generator.generators import generate_sessions
from dataset_generator.utils import build_rng_streams

sessions = generate_sessions(config, students, sessions_per_student=2,
                              rng_streams=build_rng_streams(config.seed))
```

### `SessionSimulator.simulate_session(self, student: Student, session_id: str) -> SessionRecord`
### `TransitionEngine(config, rng)`
```python
def effective_matrix(self, profile_key: str) -> dict[AttentionState, dict[AttentionState, float]]
def sample_initial_state(self) -> AttentionState
def sample_next_state(self, current_state, profile_key, intervention_applied=False, intervention_sensitivity=0.0) -> AttentionState
```

**`SessionRecord` model:** `session_id, student_id, student_profile,
interactions: list[InteractionRecord], transition_history:
list[TransitionEvent], intervention_history: list[InterventionEvent],
statistics: SessionStatistics, summary: SessionSummary`.

---

## Module 7 — `dataset_generator.pipeline`

### `build_dataset_artifact(config: GeneratorConfig, students: list[Student], sessions: list[SessionRecord]) -> DatasetArtifact`

```python
from dataset_generator.pipeline import build_dataset_artifact
dataset_artifact = build_dataset_artifact(config, students, sessions)
```

### `export_dataset_artifact(artifact: DatasetArtifact, output_dir: str | Path) -> DatasetArtifact`
Writes CSV/Parquet/JSONL + metadata/manifest JSON; returns a **new**
artifact with `exports` populated.

```python
from dataset_generator.pipeline import export_dataset_artifact
exported = export_dataset_artifact(dataset_artifact, "output/")
```

### `DatasetBuilder(students: list[Student])`
```python
def build(self, sessions: list[SessionRecord]) -> list[DatasetRecord]
```

### `compute_dataset_statistics(records: list[DatasetRecord]) -> DatasetStatistics`

**`DatasetArtifact` model:** `records: list[DatasetRecord], statistics:
DatasetStatistics, validation: DatasetValidationReport, metadata:
DatasetMetadata, manifest: DatasetManifest, exports: dict[str, str]`.

**`DatasetRecord` model:** ~60 fields spanning identifiers
(`session_id`/`student_id`/`interaction_number`/`prompt_id`/`response_id`),
student fields (`student_profile`, `student_baseline_latency`, ...), prompt
fields (`prompt_subject`, `prompt_difficulty`, ...), response fields
(`response_correctness_score`, `response_semantic_similarity`, ...),
behaviour fields (`behaviour_response_latency`, `behaviour_fatigue_level`,
...), and session-level aggregates (`session_dominant_attention_state`,
`session_average_correctness`, ...). See `models/dataset.py` for the
authoritative full list.

---

## Module 8 — `dataset_generator.classifier`

### `TrainingConfig` (frozen dataclass)
```python
model_name: str = "random_forest"          # or "logistic_regression", "gradient_boosting", "xgboost", "lightgbm"
split_mode: SplitMode = "student_aware"    # or "random"
test_size: float = 0.2
random_state: int = 42
feature_names: list[str] | None = None
calibration_method: CalibrationMethod | None = None   # "platt" or "isotonic"
compute_feature_importance: bool = True
permutation_importance_repeats: int = 10
```

### `AttentionClassifierTrainer.train(dataset_artifact: DatasetArtifact, config: TrainingConfig) -> TrainingArtifact`

```python
from dataset_generator.classifier import AttentionClassifierTrainer, TrainingConfig

trainer = AttentionClassifierTrainer()
training_artifact = trainer.train(
    dataset_artifact, TrainingConfig(model_name="random_forest", split_mode="random"),
)
```

### `AttentionClassifierPredictor(training_artifact: TrainingArtifact)`
```python
def predict(self, record: DatasetRecord) -> str
def predict_batch(self, records: list[DatasetRecord]) -> list[str]
def predict_proba(self, records: list[DatasetRecord]) -> list[dict[str, float]]
def predict_with_confidence(self, records: list[DatasetRecord]) -> list[PredictionResult]
def predict_with_explanation(self, records: list[DatasetRecord]) -> list[PredictionResult]
```

```python
from dataset_generator.classifier import AttentionClassifierPredictor
predictor = AttentionClassifierPredictor(training_artifact)
results = predictor.predict_with_confidence(dataset_artifact.records)
```

### `save_training_artifact(artifact, directory) -> Path` / `load_training_artifact(directory) -> TrainingArtifact`
joblib (model/preprocessor) + JSON (metrics/metadata).

**`TrainingArtifact` model:** `model: Any, preprocessor: Any,
feature_selector_snapshot: list[str], metrics: ClassificationMetrics,
calibration: CalibrationResult | None, feature_importance:
FeatureImportanceReport | None, metadata: TrainingMetadata`.

---

## Module 9 — `dataset_generator.bas`

### `BASEngine().compute(dataset_artifact: DatasetArtifact) -> BASArtifact`

```python
from dataset_generator.bas import BASEngine
bas_artifact = BASEngine().compute(dataset_artifact)
```

### `default_bas_config() -> BASConfig`
### `save_bas_artifact(artifact, directory) -> Path` / `load_bas_artifact(directory) -> BASArtifact`
### `render_markdown_report(artifact: BASArtifact) -> str` / `build_json_report(artifact: BASArtifact) -> dict`

```python
from dataset_generator.bas import render_markdown_report
print(render_markdown_report(bas_artifact))
```

**`BASArtifact` model:** `records: list[BASRecord]` (`session_id,
interaction_number, raw_score, score, confidence, contributions:
list[BASContribution], metadata: BASRecordMetadata`), `session_summaries:
list[BASSessionSummary]`, `statistics: BASStatistics`,
`config_fingerprint, schema_version, generation_timestamp`.

**Formula:** `BAS_t = S(E(N(F(x_t))), BAS_{t-1})` — `F` = `BASFeatureExtractor`,
`N` = `normalize_value`/`Normalizer`, `E` = `map_to_evidence` +
`BehaviouralAttentionScorer`'s aggregation, `S` = `smooth`.

---

## Module 10 — `dataset_generator.reward`

### `RewardEngine(config: RewardConfig | None = None, predictor: AttentionClassifierPredictor | None = None)`
```python
def compute(self, dataset_artifact: DatasetArtifact, bas_artifact: BASArtifact) -> RewardArtifact
```

```python
from dataset_generator.reward import RewardEngine
reward_artifact = RewardEngine().compute(dataset_artifact, bas_artifact)
```

### `default_reward_config() -> RewardConfig`
### `RewardConfig.with_category_disabled(category: RewardCategory) -> RewardConfig`
Ablation helper — zero-weights a whole reward category via the real
constructor.

```python
from dataset_generator.reward import default_reward_config, RewardCategory
ablated = default_reward_config().with_category_disabled(RewardCategory.BEHAVIOUR)
```

### `decompose_reward(contributions: list[RewardContribution]) -> tuple[float, float, float]`
Returns `(performance_reward, behaviour_reward, cost_reward)` such that
`raw_reward == performance + behaviour - cost` exactly.

### `save_reward_artifact` / `load_reward_artifact` / `render_markdown_report` / `build_json_report`
Same pattern as Module 9.

**`RewardArtifact` model:** `records: list[RewardRecord]` (`raw_reward,
reward, performance_reward, behaviour_reward, cost_reward, contributions,
confidence, uncertainty, reliability, metadata`), `session_summaries,
statistics, config_fingerprint, schema_version, generation_timestamp`.

**Formula:** `R_t = R_performance + R_behaviour - R_cost`.

---

## Module 11 — `dataset_generator.intervention`

### `InterventionPlanner(config: InterventionConfig | None = None, predictor: AttentionClassifierPredictor | None = None)`
```python
def plan(self, dataset_artifact: DatasetArtifact, bas_artifact: BASArtifact,
         reward_artifact: RewardArtifact) -> InterventionArtifact
```

```python
from dataset_generator.intervention import InterventionPlanner
intervention_artifact = InterventionPlanner().plan(
    dataset_artifact, bas_artifact, reward_artifact,
)
```

### `default_intervention_config() -> InterventionConfig`
### `InterventionConfig.with_policy_disabled(policy_name: str) -> InterventionConfig`
Ablation helper.

```python
from dataset_generator.intervention import default_intervention_config
ablated = default_intervention_config().with_policy_disabled("HintPolicy")
```

### `InterventionPolicyFactory`
```python
InterventionPolicyFactory.names()          # -> list[str], the 8 registered policy names
InterventionPolicyFactory.create_all(config)  # -> list[InterventionPolicy]
```

**Registered policies:** `NoInterventionPolicy`, `HintPolicy`,
`ConceptReviewPolicy`, `DifficultyReductionPolicy`,
`MotivationalPromptPolicy`, `BreakRecommendationPolicy`,
`EncouragementPolicy`, `QuestionReframingPolicy` — each implements
`eligible(observation)`, `estimate_bas_gain(observation)`,
`estimate_reward_gain(observation)`, `estimated_cost(observation)`,
`generate_reason(observation)`, and shares one `score(observation,
need_result)`.

### `save_intervention_artifact` / `load_intervention_artifact` / `render_markdown_report` / `build_json_report`
Same pattern as Modules 9/10.

**`InterventionArtifact` model:** `decisions: list[InterventionDecision]`
(`need_score, trigger_reasons, severity, intervention_required,
chosen_policy, chosen_reason, candidates: list[InterventionCandidate],
cooldown_suppressed, confidence, uncertainty, reliability, metadata`),
`session_summaries: list[InterventionSessionSummary]`, `statistics:
InterventionStatistics`.

---

## Module 12 — `dataset_generator.orchestration`

### `build_graph(observer=None, bas_agent=None, reward_agent=None, intervention_agent=None, tutor_agent=None, session_agent=None, generator_config=None, student_count=None, sessions_per_student=2) -> StateGraph`
Builds (uncompiled) the 6-node orchestration graph. Every agent parameter is
independently dependency-injectable.

### `compile_graph(graph=None, checkpointer=None, **build_kwargs) -> CompiledStateGraph`

```python
from dataset_generator.orchestration import build_graph, compile_graph, new_workflow_state

graph = build_graph(student_count=5, sessions_per_student=2)
compiled = compile_graph(graph)
result = compiled.invoke(new_workflow_state())
```

### `new_workflow_state(max_interactions_per_session=None, execution_metadata=None) -> WorkflowState`
Constructs an empty, ready-to-run `WorkflowState`.

### `compile_checkpointed_graph(graph: StateGraph, checkpointer=None) -> CompiledStateGraph`
Compiles with `interrupt_after` set on every node, so execution pauses and
checkpoints at every node boundary.

```python
from dataset_generator.orchestration import compile_checkpointed_graph, run_to_completion, thread_config

compiled = compile_checkpointed_graph(build_graph(student_count=2))
result = run_to_completion(compiled, new_workflow_state(), thread_id="run-1")
```

### Checkpoint functions
```python
def run_to_completion(compiled, initial_state, thread_id) -> WorkflowState
def resume_execution(compiled, thread_id) -> WorkflowState
def is_complete(compiled, thread_id) -> bool
def checkpointed_state(compiled, thread_id) -> WorkflowState
def recover_failed_session(state: WorkflowState) -> WorkflowState
def thread_config(thread_id: str) -> RunnableConfig
def default_checkpointer() -> MemorySaver
```

```python
from dataset_generator.orchestration import recover_failed_session, checkpointed_state

mid_state = checkpointed_state(compiled, "run-1")
recovered = recover_failed_session(mid_state)   # clears current session's errors
result = run_to_completion(compiled, recovered, thread_id="run-1-retry")
```

### `WorkflowMemory(state: WorkflowState)`
```python
def previous_interventions(self, session_id=None, student_id=None, before_interaction=None) -> list[InterventionDecision]
def previous_tutor_actions(self, session_id=None, student_id=None) -> list[TutorAction]
def session_history(self, student_id=None) -> list[SessionOutput]
def reward_history(self, session_id=None) -> list[RewardRecord]
def bas_history(self, session_id=None) -> list[BASRecord]
```

```python
from dataset_generator.orchestration import WorkflowMemory
memory = WorkflowMemory(result)
past_actions = memory.previous_tutor_actions(session_id=result["session_ids"][0])
```

### Node factories
```python
make_load_dataset_node(observer=None, generator_config=None, student_count=None, sessions_per_student=2) -> NodeFn
make_compute_bas_node(agent: BASAgent | None = None) -> NodeFn
make_compute_reward_node(agent: RewardAgent | None = None) -> NodeFn
make_plan_intervention_node(agent: InterventionAgent | None = None) -> NodeFn
make_generate_tutor_action_node(agent: TutorAgent | None = None) -> NodeFn
make_finalize_session_node(agent: SessionAgent | None = None) -> NodeFn
```

### Agents
```python
ObserverAgent(config: GeneratorConfig | None = None)
  .generate(student_count=None, sessions_per_student=2) -> DatasetArtifact
  .observe(dataset_artifact: DatasetArtifact) -> DatasetArtifact
BASAgent(engine: BASEngine | None = None).compute(dataset_artifact) -> BASArtifact
RewardAgent(config=None, predictor=None, engine=None).compute(dataset_artifact, bas_artifact) -> RewardArtifact
InterventionAgent(config=None, predictor=None, planner=None).plan(dataset_artifact, bas_artifact, reward_artifact) -> InterventionArtifact
TutorAgent().generate_action(decision: InterventionDecision) -> TutorAction
SessionAgent().finalize(walk: SessionWalkResult) -> SessionOutput
```

### Serialization
```python
def save_workflow_state(state: WorkflowState, directory) -> Path
def load_workflow_state(directory) -> WorkflowState
def config_fingerprints(state: WorkflowState) -> dict[str, str | None]
```

```python
from dataset_generator.orchestration import save_workflow_state, load_workflow_state
save_workflow_state(result, "runs/run-1/")
loaded = load_workflow_state("runs/run-1/")
```

### Reporting
```python
def build_json_report(state: WorkflowState) -> dict[str, object]
def render_markdown_report(state: WorkflowState) -> str
def graph_statistics(state) -> dict
def decision_counts(state) -> dict
def intervention_frequencies(state) -> dict[str, float]
def node_timing_summary(state) -> dict
def failure_summary(state) -> FailureSummary
```

**`WorkflowState` (`TypedDict`):** `dataset_artifact, bas_artifact,
reward_artifact, intervention_artifact, session_ids,
current_session_index, current_session_id, current_student_id,
current_interaction_index, max_interactions_per_session, tutor_actions,
session_outputs, execution_metadata, errors, execution_history,
timing_stats`.

**`TutorAction` (`TypedDict`):** `student_id, session_id,
interaction_number, action_type, message, source_policy, confidence`.

**`SessionOutput` (`TypedDict`):** `student_id, session_id,
interactions_processed, interventions_triggered, tutor_actions,
terminated_early, termination_reason, final_bas, final_reward`.

---

## `dataset_generator.utils`

### `build_rng_streams(seed: int) -> RNGStreams`
Returns 5 independent `numpy.random.Generator`s (`student_rng, session_rng,
prompt_rng, response_rng, noise_rng`) via `SeedSequence(seed).spawn(5)`.

### `student_local_rng(master_seed: int, student_index: int) -> np.random.Generator`
### `student_local_seed(master_seed: int, student_index: int) -> int`
A per-student-index-derived RNG/seed, independent of population size or
generation order.

```python
from dataset_generator.utils import build_rng_streams, student_local_rng
streams = build_rng_streams(42)
one_students_rng = student_local_rng(42, student_index=7)
```

Text-metric helpers (used internally by Modules 3/4/5, also public):
`word_count`, `sentence_count`, `count_syllables`, `flesch_reading_ease`,
`flesch_kincaid_grade`, `estimate_reading_time_seconds`,
`token_jaccard_similarity`, `concept_coverage`, `repetition_ratio`,
`find_hesitation_markers`, `hesitation_marker_count`, `simple_sentiment`,
`detect_git_commit`.
