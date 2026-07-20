"""Module 8, Step 6: Training Pipeline.

`AttentionClassifierTrainer.train()` is the **only** place `Preprocessor.fit`
is ever called in this module — every other component (metrics, calibration,
feature importance) receives already-transformed data. That's what "never
recompute preprocessing separately" means concretely: there is exactly one
fit, and everything downstream reuses its output.

`train()` consumes a `DatasetArtifact` directly (Module 7) via
`records_to_frame`, so no new data is generated or the underlying
`SessionRecord`s re-read — this module's whole job starts and ends with
already-assembled `DatasetRecord`s.
"""

from __future__ import annotations

from dataclasses import dataclass


from dataset_generator.classifier.calibration import CalibrationMethod, build_calibration_result, calibrate_model
from dataset_generator.classifier.feature_importance import permutation_importance_report
from dataset_generator.classifier.feature_selection import FeatureSelector
from dataset_generator.classifier.metrics import compute_classification_metrics
from dataset_generator.classifier.models import ClassifierModelFactory
from dataset_generator.classifier.preprocessing import Preprocessor
from dataset_generator.classifier.splitting import SplitMode, split_dataset
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.models.training import TrainingArtifact, TrainingMetadata
from dataset_generator.pipeline.feature_registry import FeatureRegistry
from dataset_generator.validators.dataset_validator import records_to_frame

TARGET_COLUMN = "attention_state"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for one `AttentionClassifierTrainer.train()` call."""

    model_name: str = "random_forest"
    split_mode: SplitMode = "student_aware"
    test_size: float = 0.2
    random_state: int = 42
    feature_names: list[str] | None = None  # None -> FeatureSelector defaults
    calibration_method: CalibrationMethod | None = None
    compute_feature_importance: bool = True
    permutation_importance_repeats: int = 10


class AttentionClassifierTrainer:
    """Trains one attention-state classifier from a `DatasetArtifact`.

    `feature_selector` and `registry` are dependency-injected — the trainer
    coordinates them, it does not construct its own defaults silently
    beyond what `FeatureSelector`/`FeatureRegistry`'s own constructors do.
    """

    def __init__(
        self,
        feature_selector: FeatureSelector | None = None,
        registry: FeatureRegistry | None = None,
    ) -> None:
        self._registry = registry or FeatureRegistry()
        self._feature_selector = feature_selector or FeatureSelector(self._registry)

    def train(self, dataset_artifact: DatasetArtifact, config: TrainingConfig) -> TrainingArtifact:
        """Train one classifier per `config`, returning a complete `TrainingArtifact`."""

        df = records_to_frame(dataset_artifact.records)

        feature_names = config.feature_names or self._feature_selector.select()

        train_df, val_df = split_dataset(
            df, config.split_mode, TARGET_COLUMN, config.test_size, config.random_state
        )

        preprocessor = Preprocessor(feature_names, self._registry)
        X_train = preprocessor.fit_transform(train_df)
        X_val = preprocessor.transform(val_df)
        y_train = train_df[TARGET_COLUMN]
        y_val = val_df[TARGET_COLUMN]

        model = ClassifierModelFactory.create(config.model_name, random_state=config.random_state)
        model.fit(X_train, y_train)

        class_labels = sorted(model.classes_.tolist())
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        # Reorder predict_proba's columns to match `class_labels`'s sort
        # order, since sklearn's `classes_` ordering and our sorted label
        # list aren't guaranteed to already match.
        proba_columns = model.classes_.tolist()
        reorder = [proba_columns.index(label) for label in class_labels]
        y_proba = y_proba[:, reorder]

        metrics = compute_classification_metrics(y_val.to_numpy(), y_pred, y_proba, class_labels)

        calibration = None
        if config.calibration_method is not None:
            calibrated = calibrate_model(model, X_val, y_val, config.calibration_method)
            calibrated_proba = calibrated.predict_proba(X_val)
            calibrated_columns = calibrated.classes_.tolist()
            calibrated_reorder = [calibrated_columns.index(label) for label in class_labels]
            calibration = build_calibration_result(
                y_val.to_numpy(),
                calibrated_proba[:, calibrated_reorder],
                class_labels,
                config.calibration_method,
            )

        feature_importance = None
        if config.compute_feature_importance:
            feature_importance = permutation_importance_report(
                model, X_val, y_val, preprocessor.feature_names_out_,
                n_repeats=config.permutation_importance_repeats, random_state=config.random_state,
            )

        metadata = TrainingMetadata(
            model_type=config.model_name,
            target_column=TARGET_COLUMN,
            feature_names=feature_names,
            split_mode=config.split_mode,
            test_size=config.test_size,
            random_state=config.random_state,
            train_record_count=len(train_df),
            validation_record_count=len(val_df),
            dataset_version=dataset_artifact.manifest.dataset_version,
            config_fingerprint=dataset_artifact.manifest.config_fingerprint,
        )

        return TrainingArtifact(
            model=model,
            preprocessor=preprocessor,
            feature_selector_snapshot=feature_names,
            metrics=metrics,
            calibration=calibration,
            feature_importance=feature_importance,
            metadata=metadata,
        )
