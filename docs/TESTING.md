# Testing

## Philosophy

Every package gets its own test module (`tests/test_<package>.py`), and
every test in this codebase follows a small set of rules, applied
consistently from Module 1 through Module 12:

1. **Integrate against the real upstream engine, never a mock.** A test for
   `RewardEngine` builds a real `DatasetArtifact` and a real `BASArtifact`
   via `BASEngine` — it does not stub either. This is a deliberate,
   memorialized choice after a past incident (unrelated project, but the
   principle is enforced here from the start): mocked integration tests can
   pass while the real integration is broken.
2. **Determinism is asserted, not assumed.** Every module with meaningful
   randomness has a dedicated `test_*_deterministic` test: run the same
   pipeline twice with the same seed, assert the outputs are equal (modulo
   wall-clock timestamp fields, which are metadata, not data).
3. **Edge cases are first-class**, not an afterthought: empty datasets,
   single-interaction sessions, all-missing optional signals, cooldown at
   the very first interaction, max-limit exactly at the boundary.
4. **Invariants are tested as invariants**, not just spot-checked outputs —
   e.g. Module 10's `raw_reward == performance + behaviour - cost` is
   asserted as an exact equality across every record, not sampled.
5. **Stress tests exercise realistic *shape* at scale**, not synthetic
   micro-benchmarks — see below.

## Test Suite Breakdown

| Module | Test file | Test count |
|---|---|---|
| 1 (Config) | `test_config.py` | 24 |
| 2 (Student Profiles) | `test_student_profile_generator.py` | 15 |
| 3 (Prompts) | `test_prompt_generator.py` | 22 |
| 4 (Responses) | `test_response_generator.py` | 31 |
| 5 (Behaviour) | `test_behaviour_generator.py` | 26 |
| 6 (Sessions) | `test_session_simulator.py` | 16 |
| 7 (Dataset Assembly) | `test_dataset_pipeline.py` | 27 |
| 8 (Classifier) | `test_classifier.py` | 43 |
| 9 (BAS) | `test_bas.py` | 51 |
| 10 (Reward) | `test_reward.py` | 42 |
| 11 (Intervention) | `test_intervention.py` | 48 |
| 12 (Orchestration) | `test_orchestration.py` | 40 |
| **Total** | | **385** |

Run everything:

```bash
pytest -q                       # full suite, including all 100k+ stress tests
pytest -q -k "not stress"       # fast path — skips every stress test
pytest tests/test_orchestration.py -v    # one module in isolation
```

## Stress Testing

Every module (from Module 7 onward) includes at least one
`test_stress_100000_*` test. All of them share one technique, established in
Module 7 and reused verbatim through Module 12: **replicate a small,
genuinely-simulated batch with re-suffixed IDs**, rather than re-running full
simulation 100,000+ times.

```python
base = ObserverAgent().generate(student_count=10, sessions_per_student=2)
replication_factor = 100_000 // len(base.records) + 1
records = []
for batch in range(replication_factor):
    for record in base.records:
        data = record.model_dump(mode="json")
        data["session_id"] = f"{record.session_id}_batch{batch}"
        data["student_id"] = f"{record.student_id}_batch{batch}"
        data["response_id"] = f"{record.response_id}_batch{batch}"
        records.append(DatasetRecord.model_validate(data))
```

This is deliberate: brute-force re-running the full generation pipeline
100,000+ times would take hours and mostly measure simulation cost, not the
module under test's own scaling behaviour. Replication instead produces a
dataset that is **structurally identical in shape** to a real one — same
field distributions, same per-session interaction counts — at the target
scale, cheaply.

Module 12's stress test additionally surfaced a real, non-obvious
requirement: LangGraph's Pregel scheduler caps total graph *steps* per
`invoke()` call via `recursion_limit` (unrelated to Python call-stack
recursion), and its default is far below 100,000+. Since the orchestration
graph's per-interaction walk takes one step per interaction, the stress test
passes an explicit, sufficiently high `recursion_limit` — this is documented
in `orchestration/graph.py`'s module docstring so future large-batch callers
aren't caught by it.

| Stress test | Scale exercised |
|---|---|
| Module 7 | 100,000+ dataset records assembled/validated |
| Module 8 | 100,000+ records trained/predicted |
| Module 9 | 100,000+ BAS computations |
| Module 10 | 100,000+ reward computations |
| Module 11 | 100,000+ intervention decisions |
| Module 12 | 100,000+ interactions walked through one compiled graph run |

## Determinism Testing

Each engine layer has at least one test asserting bit-identical output
across two independent runs with the same seed/config:

- `test_reward_engine_deterministic` (Module 10)
- `test_planner_deterministic` (Module 11)
- `test_planner_deterministic` / `test_full_run_deterministic` (Module 12) —
  the latter runs the *entire* graph twice and compares `tutor_actions`,
  `session_outputs`, and `bas_artifact.records`.

Determinism assertions always exclude `generation_timestamp`-style fields
(wall-clock metadata that legitimately differs between runs) and compare the
substantive data fields directly.

## Regression Testing

The full suite (`pytest -q`, all 385 tests) is run after every module's
completion to confirm zero regressions in upstream modules — this is a
standing verification step, not a one-off: Module 11's completion re-ran
all prior modules' tests (345 passed at that point), and Module 12's
completion re-ran the full 385-test suite again before any commit.

## Checkpoint Testing

Module 12's checkpoint tests verify, against the real `MemorySaver`-backed
checkpointer (not a mock):

- `test_checkpoint_every_node_pauses_between_calls` — a compiled,
  checkpointed graph genuinely stops after each node.
- `test_run_to_completion_matches_plain_invoke` — a fully-checkpointed run
  produces identical `session_outputs`/`tutor_actions` to a plain,
  non-checkpointed `invoke()`.
- `test_resume_execution_continues_interrupted_run` — an interrupted run
  (paused after one node) resumes to completion via `resume_execution`.
- `test_checkpointed_state_does_not_advance` — peeking at checkpointed state
  twice doesn't itself advance the graph.
- `test_recover_failed_session_clears_only_current_session_errors` and
  `test_recover_failed_session_end_to_end` — a simulated mid-run failure is
  recovered by clearing only the failing session's errors and replaying on
  a fresh thread, reaching completion with zero residual errors.

## Serialization Testing

Every module's `test_save_load_*_round_trip` test writes an artifact to a
`tmp_path`, reloads it, and asserts full equality against the original
(Modules 9–12). Module 12 additionally asserts `config_fingerprints()`
correctly surfaces every wrapped artifact's fingerprint, and is empty/`None`
for an empty `WorkflowState`.

## Memory Testing

Module 12's `test_memory_filters_by_session_and_student` and
`test_memory_empty_state_returns_empty_lists` verify `WorkflowMemory`'s five
query methods (`previous_interventions`, `previous_tutor_actions`,
`session_history`, `reward_history`, `bas_history`) correctly filter by
`session_id`/`student_id` against a real completed run, and degrade
gracefully (empty lists, not exceptions) against an empty state.

## Coverage

No formal coverage gate is enforced in CI (there is no CI configured yet —
see the Roadmap in the main README). `pytest-cov` can be run locally:

```bash
pip install pytest-cov
pytest --cov=dataset_generator --cov-report=term-missing -q
```

Practically, coverage of the "happy path" through every module is
effectively total, since integration tests (not just unit tests) exercise
every engine end-to-end; the main gaps by design are defensive branches only
reachable via direct, hand-crafted state rather than through the compiled
graph (e.g. `FinalizeSessionNode`'s `"early_termination"` reason, documented
inline as reachable only when a node is invoked directly, bypassing
`route_next_step`).
