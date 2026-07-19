"""Module 11: Adaptive Intervention Engine.

Computes a deterministic intervention plan from `DatasetArtifact` +
`BASArtifact` + `RewardArtifact` trajectories — never regenerating
behaviour, never recomputing BAS or reward. This is *not* an
reinforcement-learning policy; it's a rule/heuristic-driven decision
engine whose components (`InterventionObservation`, `InterventionDetector`,
policies, `PolicyScorer`, `CooldownManager`) a future RL or LangGraph-based
orchestrator (Module 12) can reuse rather than reimplement.
`InterventionArtifact` is this package's single source of truth. Entry
point: `InterventionPlanner(config).plan(dataset_artifact, bas_artifact, reward_artifact)`.
"""

from dataset_generator.intervention.config import (
    InterventionConfig,
    NeedSignalWeights,
    RankingStrategy,
    ScoringWeights,
    default_intervention_config,
)
from dataset_generator.intervention.confidence import InterventionConfidenceEstimator
from dataset_generator.intervention.cooldown import NO_INTERVENTION_POLICY_NAME, CooldownManager
from dataset_generator.intervention.detector import (
    InterventionDetector,
    NeedDetectionResult,
    NeedSignalBreakdown,
)
from dataset_generator.intervention.models import (
    InterventionArtifact,
    InterventionCandidate,
    InterventionDecision,
    InterventionDecisionMetadata,
    InterventionObservation,
    InterventionSessionSummary,
    InterventionStatistics,
)
from dataset_generator.intervention.observation import InterventionObservationExtractor
from dataset_generator.intervention.planner import (
    InterventionPlanner,
    build_intervention_session_summary,
    compute_intervention_config_fingerprint,
)
from dataset_generator.intervention.policies import (
    BreakRecommendationPolicy,
    ConceptReviewPolicy,
    DifficultyReductionPolicy,
    EncouragementPolicy,
    HintPolicy,
    InterventionPolicy,
    InterventionPolicyFactory,
    MotivationalPromptPolicy,
    NoInterventionPolicy,
    QuestionReframingPolicy,
)
from dataset_generator.intervention.report import build_json_report, render_markdown_report
from dataset_generator.intervention.scorer import PolicyScorer
from dataset_generator.intervention.serialization import (
    load_intervention_artifact,
    save_intervention_artifact,
)

__all__ = [
    "NO_INTERVENTION_POLICY_NAME",
    "BreakRecommendationPolicy",
    "ConceptReviewPolicy",
    "CooldownManager",
    "DifficultyReductionPolicy",
    "EncouragementPolicy",
    "HintPolicy",
    "InterventionArtifact",
    "InterventionCandidate",
    "InterventionConfidenceEstimator",
    "InterventionConfig",
    "InterventionDecision",
    "InterventionDecisionMetadata",
    "InterventionDetector",
    "InterventionObservation",
    "InterventionObservationExtractor",
    "InterventionPlanner",
    "InterventionPolicy",
    "InterventionPolicyFactory",
    "InterventionSessionSummary",
    "InterventionStatistics",
    "MotivationalPromptPolicy",
    "NeedDetectionResult",
    "NeedSignalBreakdown",
    "NeedSignalWeights",
    "NoInterventionPolicy",
    "PolicyScorer",
    "QuestionReframingPolicy",
    "RankingStrategy",
    "ScoringWeights",
    "build_intervention_session_summary",
    "build_json_report",
    "compute_intervention_config_fingerprint",
    "default_intervention_config",
    "load_intervention_artifact",
    "render_markdown_report",
    "save_intervention_artifact",
]
