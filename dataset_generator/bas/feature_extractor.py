"""Module 9, Step 3: Observable Feature Extraction.

Reads fields directly off `DatasetRecord` (Module 7) — nothing here samples,
simulates, or touches a `SessionRecord`/`Response`/`BehaviourRecord`
directly. `abs_normalized_latency` is the one derived quantity computed
here, and it's arithmetic on an already-observable field
(`behaviour_normalized_latency`), not hidden simulator state.

`classifier_confidence` is the only feature requiring a dependency: an
`AttentionClassifierPredictor` (Module 8), injected optionally. Batched via
`extract_batch`'s single `predict_with_confidence` call across the whole
list — calling the predictor once per record would be needlessly slow at
the scale Step 13's stress test requires.
"""

from __future__ import annotations

from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.models.dataset import DatasetRecord

from dataset_generator.bas.models import BASObservation


class BASFeatureExtractor:
    """Extracts `BASObservation`s from `DatasetRecord`s."""

    def __init__(self, predictor: AttentionClassifierPredictor | None = None) -> None:
        self._predictor = predictor

    def extract(self, record: DatasetRecord) -> BASObservation:
        """Extract one `BASObservation`. Prefer `extract_batch` for many records."""

        return self.extract_batch([record])[0]

    def extract_batch(self, records: list[DatasetRecord]) -> list[BASObservation]:
        """Extract `BASObservation`s for all of `records` in one pass."""

        classifier_confidences: list[float | None]
        if self._predictor is not None and records:
            predictions = self._predictor.predict_with_confidence(records)
            classifier_confidences = [p.confidence for p in predictions]
        else:
            classifier_confidences = [None] * len(records)

        observations = []
        for record, classifier_confidence in zip(records, classifier_confidences):
            raw_values: dict[str, float | None] = {
                "response_latency": record.behaviour_response_latency,
                "abs_normalized_latency": abs(record.behaviour_normalized_latency),
                "rolling_latency": record.behaviour_rolling_latency,
                "rolling_engagement": record.behaviour_rolling_engagement,
                "fatigue": record.behaviour_fatigue_level,
                "correctness": record.response_correctness_score,
                "confidence": record.response_confidence,
                "semantic_similarity": record.response_semantic_similarity,
                "coherence": record.response_coherence_score,
                "lexical_diversity": record.response_lexical_diversity,
                "hesitation": record.behaviour_hesitation_duration,
                "topic_shift": record.response_topic_shift,
                "repetition_ratio": record.response_repetition_ratio,
                "session_progress": record.session_progress,
                "classifier_confidence": classifier_confidence,
            }
            observations.append(
                BASObservation(
                    student_id=record.student_id,
                    session_id=record.session_id,
                    interaction_number=record.interaction_number,
                    raw_values=raw_values,
                )
            )
        return observations
