"""Module 8, Step 10: Prediction API.

`AttentionClassifierPredictor` is constructed **from** a `TrainingArtifact`
and holds no reference to anything with a `fit()` method — structurally,
not just by convention, it cannot retrain. Every prediction path reindexes
its input to `artifact.metadata.feature_names` before calling
`preprocessor.transform`, per the "store feature names, never rely on
DataFrame column order" recommendation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataset_generator.models.dataset import DatasetRecord
from dataset_generator.models.training import PredictionResult, TrainingArtifact
from dataset_generator.validators.dataset_validator import records_to_frame


class AttentionClassifierPredictor:
    """Pure inference over a trained `TrainingArtifact`."""

    def __init__(self, artifact: TrainingArtifact) -> None:
        self._model = artifact.model
        self._preprocessor = artifact.preprocessor
        self._feature_names = list(artifact.metadata.feature_names)
        self._feature_importance = artifact.feature_importance

    def _transform(self, records: list[DatasetRecord]) -> pd.DataFrame:
        df = records_to_frame(records).reindex(columns=self._feature_names)
        return self._preprocessor.transform(df)

    def predict(self, record: DatasetRecord) -> str:
        """Predict the attention state for one record."""

        return self.predict_batch([record])[0]

    def predict_batch(self, records: list[DatasetRecord]) -> list[str]:
        """Predict attention states for many records."""

        X = self._transform(records)
        return list(self._model.predict(X))

    def predict_proba(self, records: list[DatasetRecord]) -> list[dict[str, float]]:
        """Predict per-class probabilities for many records."""

        X = self._transform(records)
        proba = self._model.predict_proba(X)
        classes = self._model.classes_.tolist()
        return [dict(zip(classes, row.tolist())) for row in proba]

    def predict_with_confidence(self, records: list[DatasetRecord]) -> list[PredictionResult]:
        """Predict with the top-class probability as an explicit confidence score."""

        X = self._transform(records)
        proba = self._model.predict_proba(X)
        classes = self._model.classes_.tolist()

        results = []
        for row in proba:
            top_index = int(np.argmax(row))
            results.append(
                PredictionResult(
                    predicted_state=classes[top_index],
                    probabilities=dict(zip(classes, row.tolist())),
                    confidence=float(row[top_index]),
                )
            )
        return results

    def predict_with_explanation(self, records: list[DatasetRecord]) -> list[PredictionResult]:
        """Predict with an attached feature-contribution explanation.

        The explanation is the model's *global* top feature importances
        (from training-time `permutation_importance_report`), not a true
        per-instance attribution — genuine per-instance explanation would
        need SHAP (optional; see `feature_importance.shap_importance_report`).
        Documented here rather than implied, to avoid overclaiming precision
        this method doesn't have.
        """

        base_results = self.predict_with_confidence(records)
        if self._feature_importance is None:
            explanation = None
        else:
            explanation = {e.feature: e.importance for e in self._feature_importance.top_k(10)}

        return [
            PredictionResult(
                predicted_state=r.predicted_state,
                probabilities=r.probabilities,
                confidence=r.confidence,
                explanation=explanation,
            )
            for r in base_results
        ]
