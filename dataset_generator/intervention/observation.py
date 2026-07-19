"""Module 11, Step 3: Observation Extraction.

Reads `DatasetRecord`/`BASRecord`/`RewardRecord` fields directly, matched by
`(session_id, interaction_number)` — never recomputes BAS, never recomputes
reward, never regenerates behaviour. `classifier_confidence` follows the
same optional-predictor pattern Modules 9-10 use (a separate, lightweight
inference call — not touching BAS's or Reward's already-computed values).

`consecutive_decline_count` and `previous_interventions_count` are the two
signals that need a running walk through the session rather than a single
lookup — computed here, once, alongside everything else.
"""

from __future__ import annotations

from collections import defaultdict

from dataset_generator.bas.models import BASArtifact, BASRecord
from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.models.dataset import DatasetArtifact, DatasetRecord
from dataset_generator.reward.models import RewardArtifact, RewardRecord

from dataset_generator.intervention.models import InterventionObservation


class InterventionObservationExtractor:
    """Extracts `InterventionObservation`s from matched Dataset/BAS/Reward records."""

    def __init__(self, predictor: AttentionClassifierPredictor | None = None) -> None:
        self._predictor = predictor

    def extract_batch(
        self,
        dataset_artifact: DatasetArtifact,
        bas_artifact: BASArtifact,
        reward_artifact: RewardArtifact,
    ) -> list[InterventionObservation]:
        """Extract observations for every interaction, grouped and walked per session."""

        bas_by_key = {(r.session_id, r.interaction_number): r for r in bas_artifact.records}
        reward_by_key = {(r.session_id, r.interaction_number): r for r in reward_artifact.records}

        by_session: dict[str, list[DatasetRecord]] = defaultdict(list)
        for record in dataset_artifact.records:
            by_session[record.session_id].append(record)

        classifier_confidences: dict[str, float] = {}
        if self._predictor is not None and dataset_artifact.records:
            predictions = self._predictor.predict_with_confidence(dataset_artifact.records)
            classifier_confidences = {
                record.response_id: prediction.confidence
                for record, prediction in zip(dataset_artifact.records, predictions)
            }

        observations: list[InterventionObservation] = []
        for session_id, records in by_session.items():
            ordered = sorted(records, key=lambda r: r.interaction_number)
            observations.extend(
                self._extract_session(ordered, bas_by_key, reward_by_key, classifier_confidences)
            )
        return observations

    def _extract_session(
        self,
        ordered_records: list[DatasetRecord],
        bas_by_key: dict[tuple[str, int], BASRecord],
        reward_by_key: dict[tuple[str, int], RewardRecord],
        classifier_confidences: dict[str, float],
    ) -> list[InterventionObservation]:
        observations: list[InterventionObservation] = []

        previous_bas: float | None = None
        previous_reward: float | None = None
        previous_interventions_count = 0
        consecutive_decline_count = 0

        for record in ordered_records:
            key = (record.session_id, record.interaction_number)
            bas_record = bas_by_key.get(key)
            reward_record = reward_by_key.get(key)

            current_bas = bas_record.score if bas_record is not None else 0.5
            current_reward = reward_record.reward if reward_record is not None else 0.0
            bas_confidence = bas_record.confidence if bas_record is not None else 0.0
            reward_confidence = reward_record.confidence if reward_record is not None else 0.0

            bas_trend = current_bas - previous_bas if previous_bas is not None else None
            reward_trend = current_reward - previous_reward if previous_reward is not None else None

            if bas_trend is not None and bas_trend < 0:
                consecutive_decline_count += 1
            else:
                consecutive_decline_count = 0

            observations.append(
                InterventionObservation(
                    student_id=record.student_id,
                    session_id=record.session_id,
                    interaction_number=record.interaction_number,
                    current_bas=current_bas,
                    previous_bas=previous_bas,
                    bas_trend=bas_trend,
                    current_reward=current_reward,
                    reward_trend=reward_trend,
                    fatigue=record.behaviour_fatigue_level,
                    engagement=record.response_engagement_proxy,
                    latency_deviation=abs(record.behaviour_normalized_latency),
                    correctness=record.response_correctness_score,
                    confidence=record.response_confidence,
                    semantic_similarity=record.response_semantic_similarity,
                    prompt_difficulty_score=record.prompt_difficulty_score,
                    classifier_confidence=classifier_confidences.get(record.response_id),
                    reward_confidence=reward_confidence,
                    bas_confidence=bas_confidence,
                    session_progress=record.session_progress,
                    previous_interventions_count=previous_interventions_count,
                    consecutive_decline_count=consecutive_decline_count,
                )
            )

            previous_bas = current_bas
            previous_reward = current_reward
            if record.intervention_applied:
                previous_interventions_count += 1

        return observations
