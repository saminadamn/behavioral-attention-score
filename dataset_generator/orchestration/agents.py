"""Module 12, Step 2: Agent Definitions.

Every agent below wraps an existing module's public entry point. None of
them re-implement generation, scoring, credit assignment, or intervention
logic — they only adapt each engine's existing signature to the
orchestration layer, and (for `TutorAgent`/`SessionAgent`) translate or
aggregate results that are already fully computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_generator.bas.models import BASArtifact, BASRecord
from dataset_generator.bas.scorer import BASEngine
from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.config.defaults import default_config
from dataset_generator.config.schema import GeneratorConfig
from dataset_generator.generators.session_batch import generate_sessions
from dataset_generator.generators.student_profile_generator import generate_students
from dataset_generator.intervention.config import InterventionConfig
from dataset_generator.intervention.models import InterventionArtifact, InterventionDecision
from dataset_generator.intervention.planner import InterventionPlanner
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.pipeline.dataset_artifact import build_dataset_artifact
from dataset_generator.reward.aggregator import RewardEngine
from dataset_generator.reward.config import RewardConfig
from dataset_generator.reward.models import RewardArtifact, RewardRecord
from dataset_generator.utils.rng import build_rng_streams

from dataset_generator.orchestration.prompts import action_type_for_policy, format_tutor_message
from dataset_generator.orchestration.state import SessionOutput, TutorAction


class ObserverAgent:
    """Produces a `DatasetArtifact` — either by generating one deterministically
    from a `GeneratorConfig`, or by passing through one already provided.

    Never simulates anything itself: `generate()` simply calls
    `generate_students` -> `generate_sessions` -> `build_dataset_artifact`,
    exactly as every prior module's tests already do.
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self._config = config or default_config()

    def generate(
        self,
        student_count: int | None = None,
        sessions_per_student: int = 2,
    ) -> DatasetArtifact:
        """Deterministically generate a fresh `DatasetArtifact` from `self._config`."""

        seed = self._config.seed
        streams = build_rng_streams(seed)
        students = generate_students(self._config, streams)
        if student_count is not None:
            students = students[:student_count]
        sessions = generate_sessions(
            self._config, students, sessions_per_student=sessions_per_student,
            rng_streams=build_rng_streams(seed),
        )
        return build_dataset_artifact(self._config, students, sessions)

    def observe(self, dataset_artifact: DatasetArtifact) -> DatasetArtifact:
        """Pass through an already-built `DatasetArtifact` unchanged (dependency injection)."""

        return dataset_artifact


class BASAgent:
    """Wraps `BASEngine.compute` — no scoring logic lives here."""

    def __init__(self, engine: BASEngine | None = None) -> None:
        self._engine = engine or BASEngine()

    def compute(self, dataset_artifact: DatasetArtifact) -> BASArtifact:
        return self._engine.compute(dataset_artifact)


class RewardAgent:
    """Wraps `RewardEngine.compute` — no reward/credit-assignment logic lives here."""

    def __init__(
        self,
        config: RewardConfig | None = None,
        predictor: AttentionClassifierPredictor | None = None,
        engine: RewardEngine | None = None,
    ) -> None:
        self._engine = engine or RewardEngine(config=config, predictor=predictor)

    def compute(self, dataset_artifact: DatasetArtifact, bas_artifact: BASArtifact) -> RewardArtifact:
        return self._engine.compute(dataset_artifact, bas_artifact)


class InterventionAgent:
    """Wraps `InterventionPlanner.plan` — no detection/policy/cooldown logic lives here."""

    def __init__(
        self,
        config: InterventionConfig | None = None,
        predictor: AttentionClassifierPredictor | None = None,
        planner: InterventionPlanner | None = None,
    ) -> None:
        self._planner = planner or InterventionPlanner(config=config, predictor=predictor)

    def plan(
        self,
        dataset_artifact: DatasetArtifact,
        bas_artifact: BASArtifact,
        reward_artifact: RewardArtifact,
    ) -> InterventionArtifact:
        return self._planner.plan(dataset_artifact, bas_artifact, reward_artifact)


class TutorAgent:
    """Module 12, Step 6: translates one already-decided `InterventionDecision`
    into a `TutorAction`.

    This agent never calls `.eligible()`/`.score()`/`.estimate_*()` on any
    policy, and never touches `PolicyScorer`/`CooldownManager` — the
    decision (`chosen_policy`, `chosen_reason`, `confidence`) was already
    made by `InterventionPlanner` in Module 11. It only formats that
    decision for a tutor-facing action.
    """

    def generate_action(self, decision: InterventionDecision) -> TutorAction:
        return TutorAction(
            student_id=decision.student_id,
            session_id=decision.session_id,
            interaction_number=decision.interaction_number,
            action_type=action_type_for_policy(decision.chosen_policy),
            message=format_tutor_message(decision.chosen_policy, decision.chosen_reason),
            source_policy=decision.chosen_policy,
            confidence=decision.confidence,
        )


@dataclass
class SessionWalkResult:
    """What `FinalizeSessionNode` needs to build one `SessionOutput` — the
    interactions Phase 2 actually walked for one session, not necessarily
    every interaction in the dataset (early termination / interaction limit
    may have cut the walk short).
    """

    student_id: str
    session_id: str
    decisions: list[InterventionDecision] = field(default_factory=list)
    tutor_actions: list[TutorAction] = field(default_factory=list)
    bas_records: list[BASRecord] = field(default_factory=list)
    reward_records: list[RewardRecord] = field(default_factory=list)
    terminated_early: bool = False
    termination_reason: str | None = None


class SessionAgent:
    """Module 12: aggregates one session's walked interactions into a `SessionOutput`.

    Pure aggregation over already-computed decisions/actions/records —
    mirrors `InterventionSessionSummary`'s role in Module 11, but at the
    orchestration layer and scoped to whatever this run actually walked.
    """

    def finalize(self, walk: SessionWalkResult) -> SessionOutput:
        interventions_triggered = sum(
            1 for d in walk.decisions if d.intervention_required
        )
        final_bas = walk.bas_records[-1].score if walk.bas_records else None
        final_reward = walk.reward_records[-1].reward if walk.reward_records else None

        return SessionOutput(
            student_id=walk.student_id,
            session_id=walk.session_id,
            interactions_processed=len(walk.decisions),
            interventions_triggered=interventions_triggered,
            tutor_actions=list(walk.tutor_actions),
            terminated_early=walk.terminated_early,
            termination_reason=walk.termination_reason,
            final_bas=final_bas,
            final_reward=final_reward,
        )
