# Design Decisions

This document records the significant engineering decisions made across the
project, the alternatives considered, and why each alternative was rejected.
Organized roughly in the order these decisions were made.

## Implementation-first, not claims-first

**Decision:** every claim in the accompanying thesis is scoped to what is
actually implemented. The intervention engine is documented as "explicitly
not an RL policy"; the classifier's per-instance "explanation" is documented
as a global (training-time) permutation-importance ranking, not a true
per-instance attribution (no SHAP is computed unless a user adds it).

**Rejected alternative:** writing the research narrative first (multi-agent
RL, multimodal sensing, LLM-based tutoring) and building toward it.

**Why:** a thesis whose claims outrun its code is fragile under review.
Every "the system does X" statement in this project is true today, in this
repository, not aspirationally.

## Frozen Pydantic models everywhere

**Decision:** every domain model in Modules 1–11 is a frozen
(`ConfigDict(frozen=True)`) Pydantic `BaseModel`. Mutation happens only via
`model_copy(update=...)`, producing a new object.

**Rejected alternatives:** plain dataclasses (no runtime validation);
mutable Pydantic models (would allow a downstream consumer to silently
corrupt an artifact that's supposed to represent "what actually happened").

**Why:** validation at construction time catches malformed data immediately
at the boundary where it's generated, not three modules downstream. Freezing
makes "has this artifact been tampered with after generation?" a
non-question — it can't have been.

## RNG stream separation, not one global seed

**Decision:** `utils/rng.py`'s `RNGStreams` gives each concern (student
profile assignment, session/transition decisions, prompt generation,
response generation, behavioural noise) its own independent
`numpy.random.Generator`, derived via `SeedSequence(seed).spawn(5)`.

**Rejected alternative:** one global `random.seed()`/`np.random.seed()` call.

**Why:** with one shared stream, any module drawing an extra random number
(e.g. adding a new noise term to response generation) shifts every
subsequent draw for every other module — a silent, hard-to-diagnose
non-determinism bug whenever generation order or a module's internals
change. Independent streams mean changing the noise model never perturbs
student-profile sampling.

## Factory/registry pattern for archetypes, strategies, and models

**Decision:** `ProfileFactory` (Module 2), `ResponseStrategyFactory` (Module
4), and `ClassifierModelFactory` (Module 8) all use the same
decorator-registry shape: a `register()` classmethod decorator, `create()`/
`for_state()`, and `__init_subclass__` validation that catches a missing
required class attribute at class-definition time (import time), not at
first use.

**Rejected alternative:** an `if profile_key == "Consistently_Focused": ...
elif ...` dispatch chain.

**Why:** adding a sixth student archetype or a fourth response strategy
becomes "register a new class," not "find and edit a central dispatch
function" — and a missing/misspelled required attribute fails immediately
at import, not with a confusing `AttributeError` deep in a simulation run.

## Multiplier-derived parameters, not hardcoded absolutes

**Decision:** `StudentProfileConfig` (Module 2) and
`BehaviourGenerationConfig` (Module 5) express every archetype/attention-state
effect as a **multiplier relative to the Focused baseline**, resolved by
`config/derive.py`'s `resolve_profile_parameters`.

**Rejected alternative:** hardcoding each archetype's absolute latency mean/
variance/fatigue rate independently.

**Why:** retuning the Focused baseline (e.g. after calibrating against real
classroom timing data) automatically keeps every other archetype
proportionally consistent, rather than requiring five independent edits that
could drift out of relative alignment.

## Controlled overlap noise in Response generation (Module 4 revisit)

**Decision:** `semantic_similarity`/`confidence` receive small,
per-attention-state Gaussian noise (`semantic_similarity_noise_std`,
`confidence_noise_std`) rather than being deterministic functions of the
attention state alone.

**Rejected alternative (the original implementation):** exact,
noise-free values — e.g. Focused's semantic similarity was exactly `1.0`
with zero variance across ~1000 rows.

**Why:** a downstream attention classifier trained on exact, zero-variance
per-class values would trivially achieve ~100% accuracy by memorizing that
one feature — not because attention state is genuinely that separable, but
because the synthetic generator handed it a shortcut. Adding realistic
overlap noise (Focused → `Normal(0.88, 0.05)`, Impulsive → `Normal(0.57,
0.08)`) makes the classification problem genuinely representative of noisy
real-world signal, which is a precondition for the classifier's evaluation
metrics (Module 8) to mean anything.

## Reward decomposition as an exact identity, not an approximation

**Decision:** `R_t = R_performance + R_behaviour − R_cost` is enforced as an
**exact** equality (`raw_reward == performance + behaviour - cost`, tested
directly), achieved by a two-pass aggregation: first computing each
signal's signed evidence and the total weight actually available, then
building each signal's stored `contribution` using that signal's weight
**renormalized over the weight actually available** — not the raw configured
weight.

**Rejected alternative (the first implementation):** store each
contribution using the raw configured weight, and separately compute
`raw_reward` as a renormalized weighted average. This broke the identity
whenever a signal was "missing" (e.g. `intervention_cost` on any interaction
without an intervention) — which was almost every interaction — since
`weight_used < total_weight` made the two computations diverge
(`0.074 != 0.0666` in one caught case).

**Why:** the decomposition is only useful for ablation studies ("what if we
removed the behaviour term?") if `performance + behaviour − cost` is
*exactly* the reward that was used — an approximate decomposition would
silently misrepresent what the reward actually rewarded.

## The Intervention Engine is explicitly not an RL policy

**Decision:** `InterventionPlanner` is a deterministic pipeline —
`InterventionObservation → NeedDetector → EligiblePolicies → PolicyScorer →
CooldownManager → RankingEngine → InterventionDecision` — where scoring is a
weighted combination of heuristically-estimated gain/cost/confidence/
severity, not a learned value function.

**Rejected alternative:** train a policy (e.g. via PPO/DQN) against the
reward signal from Module 10.

**Why:** at this stage there is no real-world feedback loop to train
against, and a trained policy over synthetic data would only be as good as
the synthetic reward model — training one now would create the illusion of
a validated policy without validating anything. A deterministic, auditable
engine is honest about what's actually been built, and every one of its
components (`InterventionObservation`, `InterventionDetector`, the 8 policy
classes, `PolicyScorer`, `CooldownManager`) is designed so that **a future
RL or LangGraph-based orchestrator (Module 12, in fact) can reuse them
directly** rather than re-deriving the same observation/eligibility logic.

## Policy `score()` is shared, not reimplemented eight times

**Decision:** `InterventionPolicy.score()` is one concrete, non-abstract
method on the base class, combining gain/cost/confidence/severity via
`config.scoring_weights` — the 8 concrete policies only implement
`eligible()`/`estimate_bas_gain()`/`estimate_reward_gain()`/
`estimated_cost()`/`generate_reason()`.

**Rejected alternative (the user's own sketched `Protocol`):** each policy
independently implements its own `score()`.

**Why:** duplicating the same weighted-combination formula across 8 classes
risks them silently drifting apart (one policy's `score()` gets tweaked, the
other seven don't) and makes cross-policy ranking harder to reason about,
since scores wouldn't be computed the same way. A shared `score()` keeps
ranking comparable by construction, at the cost of a small, documented
deviation from the literal Protocol sketch — the eligibility/estimation
methods still keep each policy fully independent and extensible.

## LangGraph as a pure orchestration shell

**Decision:** Module 12 wires Modules 7/9/10/11 via LangGraph's `StateGraph`,
but every node is a thin wrapper (`agents.py`) around an existing engine's
public entry point — no BAS/Reward/Intervention logic is duplicated inside
any node.

**Rejected alternative:** embedding decision logic (e.g. the
intervention-needed/continue-session branch condition) directly into
LangGraph edge functions as ad hoc conditionals, rather than reading it off
an already-computed `InterventionDecision`.

**Why:** the intervention decision is already fully computed by
`InterventionPlanner` before the graph's per-interaction walk ever runs
(Phase 1 is a batch computation) — recomputing or approximating that
decision inside a routing function would be a second, parallel
implementation of intervention logic, exactly what the "never duplicate"
constraint across this whole project rules out.

## `WorkflowState` as `TypedDict`, not frozen Pydantic

**Decision:** unlike every other model in the project, `WorkflowState` is a
plain `TypedDict`.

**Rejected alternative:** a frozen Pydantic model, for consistency with
Modules 1–11.

**Why:** LangGraph's `StateGraph` expects nodes to return **partial** state
updates that it merges via reducers — a fundamentally different mutation
model than "construct one new frozen object per change." Fighting LangGraph
to accept a frozen Pydantic model would add friction for no benefit, since
every artifact *inside* `WorkflowState` is still the real, frozen Pydantic
type from its originating module — only the outer container adapts.

## Reusing `WorkflowMemory` from inside `FinalizeSessionNode`

**Decision:** `FinalizeSessionNode`'s BAS/reward record lookups call
`WorkflowMemory(state).bas_history(session_id)`/`.reward_history(session_id)`
rather than independently filtering `bas_artifact.records`/
`reward_artifact.records` by session and sorting by interaction number.

**Rejected alternative (the original implementation):** hand-roll the same
filter-and-sort logic inline inside the node.

**Why:** caught during a verification pass — the same filter+sort pattern
was written out twice (once in `memory.py`, once in `nodes.py`). Since
`WorkflowMemory`'s query already does exactly the filtering needed (before
`FinalizeSessionNode` additionally slices to the walked-count), reusing it
removes the duplication without changing behavior.

## Checkpointer serde: `allowed_msgpack_modules=True`

**Decision:** `checkpoint.py`'s `default_checkpointer()` explicitly
constructs `JsonPlusSerializer(allowed_msgpack_modules=True)`, with an
inline comment warning against "simplifying" it away.

**Rejected alternative:** LangGraph's own default serializer configuration.

**Why:** discovered directly during Module 12 verification — the default
serializer silently **drops** fields (`dataset_artifact`, `bas_artifact`)
it doesn't recognize on checkpoint reload, rather than raising an error.
`True` is safe here specifically because checkpoint state is exclusively
this project's own trusted, internally-defined types, never external data —
but the fix looks unnecessary at a glance, hence the explicit comment.

## Stress tests replicate a small real batch, not brute-force re-simulation

**Decision:** every 100,000+-scale stress test (Modules 7–12) replicates a
genuinely-simulated small batch with re-suffixed IDs, rather than
re-running the full generation/computation pipeline 100,000+ times.

**Rejected alternative:** literally generating 100,000+ students/sessions
from scratch, or (for Module 12) invoking a fresh graph 100,000 separate
times.

**Why:** both alternatives were measured as impractical — a first attempt at
Module 12's stress test using fresh per-run graph invocations was estimated
at 1.5–3 hours purely from per-run overhead. Replication produces
structurally identical, realistically-shaped data at target scale in a
fraction of the time, while still genuinely exercising the module under
test's own scaling behavior (the thing the stress test is actually meant to
verify).
