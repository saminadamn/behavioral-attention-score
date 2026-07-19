"""Training/inference domain models (Module 8).

`TrainingArtifact` mirrors Module 7's `DatasetArtifact`: it's the single
object downstream consumers (BAS, intervention policy, evaluation) should
depend on, rather than independently loading a pickled model file and
re-deriving feature order/preprocessing/metrics themselves.

`arbitrary_types_allowed` is enabled because `model` and `preprocessor` wrap
non-pydantic objects (a scikit-learn estimator, a fitted `ColumnTransformer`)
— the same accommodation `DatasetArtifact`/`SessionRecord` don't need only
because nothing in Modules 1-7 wraps a foreign, non-serializable object.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClassificationMetrics(BaseModel):
    """Evaluation metrics for one trained classifier (Module 8, Step 7)."""

    model_config = ConfigDict(frozen=True)

    accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    precision_macro: float = Field(ge=0.0, le=1.0)
    recall_macro: float = Field(ge=0.0, le=1.0)
    f1_macro: float = Field(ge=0.0, le=1.0)
    f1_weighted: float = Field(ge=0.0, le=1.0)
    roc_auc_ovr: float | None = Field(ge=0.0, le=1.0, default=None)
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    classification_report: str
    mean_prediction_confidence: float = Field(ge=0.0, le=1.0)


class ReliabilityBin(BaseModel):
    """One bin of a calibration reliability curve."""

    model_config = ConfigDict(frozen=True)

    bin_lower: float
    bin_upper: float
    mean_confidence: float
    mean_accuracy: float
    count: int = Field(ge=0)


class CalibrationResult(BaseModel):
    """Probability calibration outcome (Module 8, Step 8)."""

    model_config = ConfigDict(frozen=True)

    method: str
    expected_calibration_error: float = Field(ge=0.0)
    reliability_bins: list[ReliabilityBin]


class FeatureImportanceEntry(BaseModel):
    """One feature's importance score."""

    model_config = ConfigDict(frozen=True)

    feature: str
    importance: float
    std: float | None = None


class FeatureImportanceReport(BaseModel):
    """Ranked feature importance (Module 8, Step 9)."""

    model_config = ConfigDict(frozen=True)

    method: str
    ranked: list[FeatureImportanceEntry]

    def top_k(self, k: int) -> list[FeatureImportanceEntry]:
        """The `k` highest-ranked features."""

        return self.ranked[:k]


class TrainingMetadata(BaseModel):
    """Training provenance — including the exact, ordered feature list used.

    `feature_names` is the one persisted list every inference-time
    DataFrame gets reindexed to (see `Preprocessor.transform` and
    `AttentionClassifierPredictor`) — never inferred from a DataFrame's own
    column order at load time.
    """

    model_config = ConfigDict(frozen=True)

    trained_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_type: str
    target_column: str
    feature_names: list[str]
    split_mode: str
    test_size: float = Field(gt=0.0, lt=1.0)
    random_state: int
    train_record_count: int = Field(ge=0)
    validation_record_count: int = Field(ge=0)
    dataset_version: str
    config_fingerprint: str


class PredictionResult(BaseModel):
    """One structured prediction (Module 8, Step 10)."""

    model_config = ConfigDict(frozen=True)

    predicted_state: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: dict[str, float] | None = None


class TrainingArtifact(BaseModel):
    """The single source of truth for one trained classifier."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model: Any
    preprocessor: Any
    feature_selector_snapshot: list[str]
    metrics: ClassificationMetrics
    calibration: CalibrationResult | None
    feature_importance: FeatureImportanceReport | None
    metadata: TrainingMetadata
    version: str = "1.0.0"
