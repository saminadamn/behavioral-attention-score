"""Module 11, Step 7: Planning, and Step 10: Session Summary.

`InterventionPlanner` is this module's entry point (mirroring `BASEngine`/
`RewardEngine`'s placement in Modules 9-10), wiring the full pipeline per
the user's diagram:

    InterventionObservation -> NeedDetector -> EligiblePolicies ->
    PolicyScorer -> CooldownManager -> RankingEngine -> InterventionDecision

Session-summary aggregation (Step 10) is folded in here, since Module 11's
file list has no dedicated `session_summary.py` — the same choice Module 10
made by folding its session summary into `aggregator.py`.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

from dataset_generator.bas.models import BASArtifact
from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.reward.models import RewardArtifact

from dataset_generator.intervention.config import InterventionConfig, default_intervention_config
from dataset_generator.intervention.confidence import InterventionConfidenceEstimator
from dataset_generator.intervention.cooldown import NO_INTERVENTION_POLICY_NAME, CooldownManager
from dataset_generator.intervention.detector import InterventionDetector
from dataset_generator.intervention.models import (
    InterventionArtifact,
    InterventionDecision,
    InterventionDecisionMetadata,
    InterventionObservation,
    InterventionSessionSummary,
    InterventionStatistics,
)
from dataset_generator.intervention.observation import InterventionObservationExtractor
from dataset_generator.intervention.policies import InterventionPolicyFactory
from dataset_generator.intervention.scorer import PolicyScorer

SCHEMA_VERSION = "1.0"


def compute_intervention_config_fingerprint(config: InterventionConfig) -> str:
    """A deterministic SHA-256 fingerprint of `config`."""

    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


def build_intervention_session_summary(
    decisions: list[InterventionDecision], observations: list[InterventionObservation]
) -> InterventionSessionSummary:
    """Aggregate one session's `InterventionDecision`s (Module 11, Step 10)."""

    n = len(decisions)
    policy_frequencies: Counter[str] = Counter(d.chosen_policy for d in decisions)
    average_confidence = sum(d.confidence for d in decisions) / n
    severity_scores = {"low": 0.0, "medium": 0.5, "high": 1.0}
    average_severity_score = sum(severity_scores.get(d.severity, 0.0) for d in decisions) / n

    bas_by_interaction = {o.interaction_number: o.current_bas for o in observations}
    intervention_interactions = [
        d.interaction_number for d in decisions if d.chosen_policy != NO_INTERVENTION_POLICY_NAME
    ]

    bas_before = [
        bas_by_interaction[i] for i in intervention_interactions if i in bas_by_interaction
    ]
    bas_after = [
        bas_by_interaction[i + 1] for i in intervention_interactions if (i + 1) in bas_by_interaction
    ]

    average_bas_before = sum(bas_before) / len(bas_before) if bas_before else None
    average_bas_after = sum(bas_after) / len(bas_after) if bas_after else None

    estimated_cumulative_bas_gain = sum(
        max(c.estimated_bas_gain for c in d.candidates if c.policy_name == d.chosen_policy)
        for d in decisions
        if d.chosen_policy != NO_INTERVENTION_POLICY_NAME
    )
    estimated_cumulative_reward_gain = sum(
        max(c.estimated_reward_gain for c in d.candidates if c.policy_name == d.chosen_policy)
        for d in decisions
        if d.chosen_policy != NO_INTERVENTION_POLICY_NAME
    )

    return InterventionSessionSummary(
        student_id=decisions[0].student_id,
        session_id=decisions[0].session_id,
        interaction_count=n,
        intervention_count=len(intervention_interactions),
        policy_frequencies=dict(policy_frequencies),
        average_confidence=average_confidence,
        average_severity_score=average_severity_score,
        average_bas_before_intervention=average_bas_before,
        average_bas_after_intervention=average_bas_after,
        estimated_cumulative_bas_gain=estimated_cumulative_bas_gain,
        estimated_cumulative_reward_gain=estimated_cumulative_reward_gain,
        cooldown_suppressions=sum(1 for d in decisions if d.cooldown_suppressed),
    )


class InterventionPlanner:
    """Computes an `InterventionArtifact` from Dataset/BAS/Reward artifacts — the
    Module 11 entry point.
    """

    def __init__(
        self,
        config: InterventionConfig | None = None,
        predictor: AttentionClassifierPredictor | None = None,
    ) -> None:
        self._config = config or default_intervention_config()
        self._extractor = InterventionObservationExtractor(predictor)
        self._detector = InterventionDetector(self._config)
        self._scorer = PolicyScorer()
        self._confidence_estimator = InterventionConfidenceEstimator(self._config)
        self._policies = InterventionPolicyFactory.create_all(self._config)
        self._fingerprint = compute_intervention_config_fingerprint(self._config)

    def plan(
        self,
        dataset_artifact: DatasetArtifact,
        bas_artifact: BASArtifact,
        reward_artifact: RewardArtifact,
    ) -> InterventionArtifact:
        """Plan interventions for every interaction, grouped and cooled-down per session."""

        observations = self._extractor.extract_batch(dataset_artifact, bas_artifact, reward_artifact)

        by_session: dict[str, list[InterventionObservation]] = defaultdict(list)
        for observation in observations:
            by_session[observation.session_id].append(observation)

        all_decisions: list[InterventionDecision] = []
        session_summaries: list[InterventionSessionSummary] = []
        for session_id, session_observations in by_session.items():
            ordered = sorted(session_observations, key=lambda o: o.interaction_number)
            decisions = self._plan_session(ordered)
            all_decisions.extend(decisions)
            session_summaries.append(build_intervention_session_summary(decisions, ordered))

        statistics = self._compute_statistics(all_decisions, observations)

        return InterventionArtifact(
            decisions=all_decisions,
            session_summaries=session_summaries,
            statistics=statistics,
            config_fingerprint=self._fingerprint,
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _plan_session(
        self, observations: list[InterventionObservation]
    ) -> list[InterventionDecision]:
        cooldown = CooldownManager(self._config)
        decisions: list[InterventionDecision] = []

        for observation in observations:
            need_result = self._detector.detect(observation)
            candidates = self._scorer.evaluate_all(observation, self._policies, need_result)
            allowed_candidates, cooldown_suppressed = cooldown.filter_candidates(
                candidates, observation.interaction_number
            )

            chosen = allowed_candidates[0] if allowed_candidates else candidates[0]
            intervention_required = (
                need_result.need_score >= self._config.need_threshold
                and chosen.policy_name != NO_INTERVENTION_POLICY_NAME
            )

            confidence, uncertainty, reliability = self._confidence_estimator.estimate(
                observation, candidates
            )

            decisions.append(
                InterventionDecision(
                    student_id=observation.student_id,
                    session_id=observation.session_id,
                    interaction_number=observation.interaction_number,
                    need_score=need_result.need_score,
                    trigger_reasons=need_result.trigger_reasons,
                    severity=need_result.severity,
                    intervention_required=intervention_required,
                    chosen_policy=chosen.policy_name,
                    chosen_reason=chosen.reason,
                    candidates=candidates,
                    cooldown_suppressed=cooldown_suppressed,
                    confidence=confidence,
                    uncertainty=uncertainty,
                    reliability=reliability,
                    metadata=InterventionDecisionMetadata(
                        ranking_strategy=self._config.ranking_strategy,
                        config_fingerprint=self._fingerprint,
                        config_version=self._config.version,
                    ),
                )
            )
            cooldown.record_intervention(observation.interaction_number, chosen.policy_name)

        return decisions

    def _compute_statistics(
        self, decisions: list[InterventionDecision], observations: list[InterventionObservation]
    ) -> InterventionStatistics:
        if not decisions:
            return InterventionStatistics(
                record_count=0, intervention_rate=0.0, policy_distribution={},
                average_confidence=0.0, average_need_score=0.0,
                cooldown_suppression_rate=0.0, missing_value_summary={},
            )

        n = len(decisions)
        intervention_count = sum(
            1 for d in decisions if d.chosen_policy != NO_INTERVENTION_POLICY_NAME
        )
        policy_counts = Counter(d.chosen_policy for d in decisions)
        policy_distribution = {name: count / n for name, count in policy_counts.items()}

        missing_value_summary = {
            "previous_bas": sum(1 for o in observations if o.previous_bas is None),
            "bas_trend": sum(1 for o in observations if o.bas_trend is None),
            "reward_trend": sum(1 for o in observations if o.reward_trend is None),
            "classifier_confidence": sum(1 for o in observations if o.classifier_confidence is None),
        }

        return InterventionStatistics(
            record_count=n,
            intervention_rate=intervention_count / n,
            policy_distribution=policy_distribution,
            average_confidence=sum(d.confidence for d in decisions) / n,
            average_need_score=sum(d.need_score for d in decisions) / n,
            cooldown_suppression_rate=sum(1 for d in decisions if d.cooldown_suppressed) / n,
            missing_value_summary=missing_value_summary,
        )
