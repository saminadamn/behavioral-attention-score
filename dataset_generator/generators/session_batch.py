"""Module 6, Step 10: Batch Session Generation.

Builds one shared `PromptGenerator`/`ResponseGenerator`/`BehaviourGenerator`/
`TransitionEngine`/`SessionSimulator` set (each already reusable across many
calls, by design since Modules 3-6) and reuses it across every student and
session in the batch — rather than constructing fresh instances per
session, which would reset nothing useful but would obscure that a single
seed determines the *entire* batch's RNG state, not just one session's.

Parallel generation (Step 10's "optional") is intentionally not
implemented: multiprocessing would need per-process RNG-stream partitioning
to preserve reproducibility, which is a meaningful design problem in its own
right, not a quick addition — deferred rather than half-built.
"""

from __future__ import annotations

from dataset_generator.config import GeneratorConfig
from dataset_generator.generators.behaviour_generator import BehaviourGenerator
from dataset_generator.generators.prompt_generator import PromptGenerator
from dataset_generator.generators.response_generator import ResponseGenerator
from dataset_generator.generators.session_simulator import InterventionPolicy, SessionSimulator
from dataset_generator.generators.transition_engine import TransitionEngine
from dataset_generator.models.session import SessionRecord
from dataset_generator.models.student import Student
from dataset_generator.utils.rng import RNGStreams


def build_session_simulator(
    config: GeneratorConfig, rng_streams: RNGStreams, intervention_policy: InterventionPolicy | None = None,
) -> SessionSimulator:
    """Construct one `SessionSimulator` wired to `rng_streams`'s dedicated streams.

    `rng_streams.session_rng` backs the simulator's own decisions (session
    length, intervention triggering) as well as the `TransitionEngine`'s
    state sampling — both are "which state/session-shape" decisions
    distinct from prompt/response/behaviour content, so sharing one stream
    between them is a deliberate scope choice, not an oversight.

    `intervention_policy` defaults to `None` (the simulator's own built-in
    heuristic, unchanged) — passing one substitutes a live, causal
    intervention-decision source, used only by baseline-policy comparisons
    (see `dataset_generator.rl_experimental.baselines`), never by the
    default pipeline.
    """

    prompt_generator = PromptGenerator(config, rng_streams.prompt_rng)
    response_generator = ResponseGenerator(config, rng_streams.response_rng)
    behaviour_generator = BehaviourGenerator(config, rng_streams.noise_rng)
    transition_engine = TransitionEngine(config, rng_streams.session_rng)
    return SessionSimulator(
        config, prompt_generator, response_generator, behaviour_generator,
        transition_engine, rng_streams.session_rng, intervention_policy=intervention_policy,
    )


def generate_sessions(
    config: GeneratorConfig,
    students: list[Student],
    sessions_per_student: int,
    rng_streams: RNGStreams,
    intervention_policy: InterventionPolicy | None = None,
) -> list[SessionRecord]:
    """Simulate `sessions_per_student` sessions for each of `students`.

    Session IDs are `f"{student_id}_SESS{session_index:02d}"`. Deterministic
    for a given `config.seed` (via `rng_streams`), since every generator and
    the simulator itself draw only from streams derived from that one seed.
    `intervention_policy` is documented on `build_session_simulator`.
    """

    simulator = build_session_simulator(config, rng_streams, intervention_policy=intervention_policy)

    sessions: list[SessionRecord] = []
    for student in students:
        for session_index in range(1, sessions_per_student + 1):
            session_id = f"{student.student_id}_SESS{session_index:02d}"
            sessions.append(simulator.simulate_session(student, session_id))
    return sessions
