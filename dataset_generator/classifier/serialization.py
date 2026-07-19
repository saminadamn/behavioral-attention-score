"""Module 8, Step 11: Model Serialization.

Splits a `TrainingArtifact` into two files, matching this project's
established pattern (Module 7's manifest/metadata JSON alongside binary
exports): a `.joblib` bundle for the non-JSON-serializable pieces (the
fitted model and preprocessor), and a `.json` sidecar holding everything
else (metrics, calibration, feature importance, metadata, version,
config fingerprint) — human-readable and inspectable without unpickling
anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from dataset_generator.classifier.models import ClassifierModel
from dataset_generator.classifier.preprocessing import Preprocessor
from dataset_generator.models.training import (
    CalibrationResult,
    ClassificationMetrics,
    FeatureImportanceReport,
    TrainingArtifact,
    TrainingMetadata,
)

_BUNDLE_FILENAME = "model_bundle.joblib"
_METADATA_FILENAME = "training_metadata.json"


def save_training_artifact(artifact: TrainingArtifact, directory: str | Path) -> dict[str, Path]:
    """Persist `artifact` as a joblib model bundle + a JSON metadata sidecar."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    bundle_path = directory / _BUNDLE_FILENAME
    joblib.dump(
        {
            "model": artifact.model,
            "preprocessor": artifact.preprocessor,
            "feature_selector_snapshot": artifact.feature_selector_snapshot,
        },
        bundle_path,
    )

    metadata_path = directory / _METADATA_FILENAME
    payload = {
        "metrics": artifact.metrics.model_dump(mode="json"),
        "calibration": artifact.calibration.model_dump(mode="json") if artifact.calibration else None,
        "feature_importance": artifact.feature_importance.model_dump(mode="json") if artifact.feature_importance else None,
        "metadata": artifact.metadata.model_dump(mode="json"),
        "version": artifact.version,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {"bundle": bundle_path, "metadata": metadata_path}


def load_training_artifact(directory: str | Path) -> TrainingArtifact:
    """Load a `TrainingArtifact` previously saved by `save_training_artifact` — no retraining."""

    directory = Path(directory)
    bundle = joblib.load(directory / _BUNDLE_FILENAME)
    payload = json.loads((directory / _METADATA_FILENAME).read_text(encoding="utf-8"))

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    if not isinstance(model, ClassifierModel):
        raise TypeError(f"loaded model is not a ClassifierModel: {type(model)}")
    if not isinstance(preprocessor, Preprocessor):
        raise TypeError(f"loaded preprocessor is not a Preprocessor: {type(preprocessor)}")

    return TrainingArtifact(
        model=model,
        preprocessor=preprocessor,
        feature_selector_snapshot=bundle["feature_selector_snapshot"],
        metrics=ClassificationMetrics.model_validate(payload["metrics"]),
        calibration=CalibrationResult.model_validate(payload["calibration"]) if payload["calibration"] else None,
        feature_importance=(
            FeatureImportanceReport.model_validate(payload["feature_importance"])
            if payload["feature_importance"]
            else None
        ),
        metadata=TrainingMetadata.model_validate(payload["metadata"]),
        version=payload["version"],
    )
