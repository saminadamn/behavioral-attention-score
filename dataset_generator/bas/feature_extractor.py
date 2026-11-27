"""Module 9, Step 3: Observable Feature Extraction.

Reads fields directly off `DatasetRecord` (Module 7) — nothing here samples,
simulates, or touches a `SessionRecord`/`Response`/`BehaviourRecord`
directly. `abs_normalized_latency` and `prompt_relative_latency` are the
derived quantities computed here, and both are arithmetic on
already-observable fields, not hidden simulator state.

`classifier_confidence` is the only feature requiring a dependency: an
`AttentionClassifierPredictor` (Module 8), injected optionally. Batched via
`extract_batch`'s single `predict_with_confidence` call across the whole
list — calling the predictor once per record would be needlessly slow at
the scale Step 13's stress test requires.

`prompt_relative_latency` exists because `abs_normalized_latency` is
*student*-relative only (it's built from `behaviour_normalized_latency`,
which compares a response's latency to that same student's own baseline
and variance). That leaves a blind spot: a response that is answered far
too quickly for the *question just asked* — the hallmark of an impulsive,
premature answer — reads as merely "fast for this student" rather than
"implausibly fast for this question," and can score no worse than a
genuinely fast, attentive response from a quick-baseline student. See
`docs/ROBUSTNESS_AUDIT.md` for the adversarial case that exposed this and
confirmed the fix.
"""

from __future__ import annotations

from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.models.dataset import DatasetRecord

from dataset_generator.bas.models import BASObservation


_MIN_READING_TIME_SECONDS = 0.1


def _prompt_relative_latency(response_latency: float, prompt_reading_time_seconds: float) -> float:
    """Response latency as a multiple of the prompt's own reading time.

    A value near or below 1.0 means the response arrived in about the time
    it takes to *read* the question, before there was time to think about
    an answer — independent of how fast that particular student usually
    responds.
    """

    reading_time = max(prompt_reading_time_seconds, _MIN_READING_TIME_SECONDS)
    return response_latency / reading_time


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
                "prompt_relative_latency": _prompt_relative_latency(
                    record.behaviour_response_latency, record.prompt_reading_time_seconds
                ),
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
