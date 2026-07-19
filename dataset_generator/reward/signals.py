"""Module 10, Step 3: Signal Extraction.

Reads `DatasetRecord`/`BASRecord` fields directly — never regenerates
behaviour, never recomputes BAS. `delta_bas` comes straight from
`BASRecord.score` (Module 9's already-computed output); every other delta
comes from consecutive `DatasetRecord`s within the same session.

`delta_latency_deviation` is computed from `behaviour_normalized_latency`
directly on `DatasetRecord` — it doesn't need `BASArtifact` at all, since
that field is already an observable quantity, not something BAS derived.

A session's first interaction has no previous interaction to diff against:
every delta is `None` there (not 0.0) — genuinely unobserved, not "no
change", the same distinction `BASObservation` draws for a missing feature.
"""

from __future__ import annotations

from collections import defaultdict

from dataset_generator.bas.models import BASRecord
from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.models.dataset import DatasetRecord

from dataset_generator.reward.models import RewardObservation


class RewardSignalExtractor:
    """Extracts `RewardObservation`s from matched `DatasetRecord`/`BASRecord` pairs."""

    def __init__(self, predictor: AttentionClassifierPredictor | None = None) -> None:
        self._predictor = predictor

    def extract_batch(
        self, dataset_records: list[DatasetRecord], bas_records: list[BASRecord]
    ) -> list[RewardObservation]:
        """Extract `RewardObservation`s for every interaction across all sessions present.

        `dataset_records` and `bas_records` need not be pre-sorted or
        pre-grouped — this groups by `session_id` and orders by
        `interaction_number` itself, then walks each session computing
        deltas against the previous interaction.
        """

        bas_by_key = {(r.session_id, r.interaction_number): r for r in bas_records}

        by_session: dict[str, list[DatasetRecord]] = defaultdict(list)
        for record in dataset_records:
            by_session[record.session_id].append(record)

        classifier_confidences: dict[str, float] = {}
        if self._predictor is not None and dataset_records:
            predictions = self._predictor.predict_with_confidence(dataset_records)
            classifier_confidences = {
                record.response_id: prediction.confidence
                for record, prediction in zip(dataset_records, predictions)
            }

        observations: list[RewardObservation] = []
        for session_id, records in by_session.items():
            ordered = sorted(records, key=lambda r: r.interaction_number)
            observations.extend(
                self._extract_session(ordered, bas_by_key, classifier_confidences)
            )
        return observations

    def _extract_session(
        self,
        ordered_records: list[DatasetRecord],
        bas_by_key: dict[tuple[str, int], BASRecord],
        classifier_confidences: dict[str, float],
    ) -> list[RewardObservation]:
        observations: list[RewardObservation] = []
        previous: DatasetRecord | None = None
        previous_bas_score: float | None = None
        previous_classifier_confidence: float | None = None

        for record in ordered_records:
            bas_record = bas_by_key.get((record.session_id, record.interaction_number))
            current_bas_score = bas_record.score if bas_record is not None else None
            current_classifier_confidence = classifier_confidences.get(record.response_id)

            if previous is None:
                raw_signals: dict[str, float | None] = {
                    "delta_bas": None,
                    "delta_engagement": None,
                    "delta_correctness": None,
                    "delta_confidence": None,
                    "delta_latency_deviation": None,
                    "delta_fatigue": None,
                    "delta_classifier_confidence": None,
                }
            else:
                raw_signals = {
                    "delta_bas": (
                        current_bas_score - previous_bas_score
                        if current_bas_score is not None and previous_bas_score is not None
                        else None
                    ),
                    "delta_engagement": record.response_engagement_proxy - previous.response_engagement_proxy,
                    "delta_correctness": record.response_correctness_score - previous.response_correctness_score,
                    "delta_confidence": record.response_confidence - previous.response_confidence,
                    # Positive = deviation from personal baseline GREW (like
                    # delta_fatigue: current - previous, positive = worse),
                    # paired with NEGATIVE polarity in config.py so growing
                    # deviation reduces reward, shrinking deviation increases it.
                    "delta_latency_deviation": (
                        abs(record.behaviour_normalized_latency) - abs(previous.behaviour_normalized_latency)
                    ),
                    "delta_fatigue": record.behaviour_fatigue_level - previous.behaviour_fatigue_level,
                    "delta_classifier_confidence": (
                        current_classifier_confidence - previous_classifier_confidence
                        if current_classifier_confidence is not None and previous_classifier_confidence is not None
                        else None
                    ),
                }

            raw_signals["intervention_cost"] = 1.0 if record.intervention_applied else None
            raw_signals["session_progress"] = record.session_progress

            observations.append(
                RewardObservation(
                    student_id=record.student_id,
                    session_id=record.session_id,
                    interaction_number=record.interaction_number,
                    raw_signals=raw_signals,
                )
            )

            previous = record
            previous_bas_score = current_bas_score
            previous_classifier_confidence = current_classifier_confidence

        return observations
