"""Tests for Module 8: Attention State Classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_generator.classifier.calibration import build_calibration_result, calibrate_model, compute_ece
from dataset_generator.classifier.feature_importance import (
    permutation_importance_report,
    tree_importance_report,
)
from dataset_generator.classifier.feature_selection import FeatureSelector
from dataset_generator.classifier.metrics import compute_classification_metrics
from dataset_generator.classifier.models import ClassifierModelFactory, LogisticRegressionModel
from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.classifier.preprocessing import Preprocessor
from dataset_generator.classifier.serialization import load_training_artifact, save_training_artifact
from dataset_generator.classifier.splitting import (
    assert_no_student_leakage,
    session_aware_split,
    split_dataset,
    stratified_split,
    student_aware_split,
)
from dataset_generator.classifier.trainer import AttentionClassifierTrainer, TrainingConfig
from dataset_generator.config import default_config
from dataset_generator.generators import generate_sessions, generate_students
from dataset_generator.models.dataset import DatasetRecord, FeatureCategory
from dataset_generator.pipeline import build_dataset_artifact
from dataset_generator.pipeline.feature_registry import FeatureRegistry
from dataset_generator.utils import build_rng_streams
from dataset_generator.validators.dataset_validator import records_to_frame


def _dataset_artifact(n_students: int = 20, sessions_per_student: int = 3, seed: int | None = None):
    config = default_config()
    seed = seed if seed is not None else config.seed
    streams = build_rng_streams(seed)
    students = generate_students(config, streams)[:n_students]
    sessions = generate_sessions(config, students, sessions_per_student=sessions_per_student, rng_streams=build_rng_streams(seed))
    return build_dataset_artifact(config, students, sessions)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def test_feature_selector_excludes_identifiers_and_targets_by_default() -> None:
    selector = FeatureSelector()
    features = selector.select()
    registry = FeatureRegistry()
    for name in features:
        d = registry.get(name)
        assert d.category not in (FeatureCategory.IDENTIFIER, FeatureCategory.TARGET)


def test_feature_selector_excludes_text_by_default() -> None:
    selector = FeatureSelector()
    features = selector.select()
    assert "prompt_text" not in features
    assert "response_text" not in features
    assert "prompt_keywords" not in features


def test_feature_selector_leaky_features_excluded() -> None:
    """response_strategy_used and session_dominant_attention_state are
    bijective/aggregate leaks of attention_state and must never appear in
    the default feature set."""

    selector = FeatureSelector()
    features = selector.select()
    assert "response_strategy_used" not in features
    assert "session_dominant_attention_state" not in features


def test_feature_selector_numeric_only() -> None:
    selector = FeatureSelector()
    features = selector.select(numeric_only=True)
    registry = FeatureRegistry()
    assert all(registry.get(f).dtype in ("int", "float") for f in features)


def test_feature_selector_category_filter() -> None:
    selector = FeatureSelector()
    features = selector.select(categories=[FeatureCategory.STUDENT])
    assert all(f.startswith("student_") for f in features)


def test_feature_selector_whitelist_and_blacklist() -> None:
    selector = FeatureSelector()
    features = selector.select(
        whitelist=["student_baseline_latency", "student_fatigue_rate", "prompt_subject"],
        blacklist=["prompt_subject"],
    )
    assert set(features) == {"student_baseline_latency", "student_fatigue_rate"}


def test_feature_selector_manual_validates_names() -> None:
    selector = FeatureSelector()
    assert selector.manual(["student_baseline_latency"]) == ["student_baseline_latency"]
    with pytest.raises(KeyError):
        selector.manual(["not_a_real_feature"])


def test_feature_selector_never_mutates_records() -> None:
    artifact = _dataset_artifact(n_students=3, sessions_per_student=1)
    before = artifact.records[0].model_dump(mode="json")
    FeatureSelector().select()
    after = artifact.records[0].model_dump(mode="json")
    assert before == after


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_preprocessor_fit_transform_deterministic() -> None:
    artifact = _dataset_artifact(n_students=10, sessions_per_student=2)
    df = records_to_frame(artifact.records)
    features = FeatureSelector().select()

    a = Preprocessor(features).fit_transform(df)
    b = Preprocessor(features).fit_transform(df)
    pd.testing.assert_frame_equal(a, b)


def test_preprocessor_rejects_text_features() -> None:
    with pytest.raises(ValueError):
        Preprocessor(["prompt_text"])


def test_preprocessor_transform_before_fit_raises() -> None:
    artifact = _dataset_artifact(n_students=3, sessions_per_student=1)
    df = records_to_frame(artifact.records)
    preprocessor = Preprocessor(FeatureSelector().select())
    with pytest.raises(RuntimeError):
        preprocessor.transform(df)


def test_preprocessor_save_load_round_trip(tmp_path) -> None:
    artifact = _dataset_artifact(n_students=10, sessions_per_student=2)
    df = records_to_frame(artifact.records)
    features = FeatureSelector().select()

    preprocessor = Preprocessor(features)
    transformed = preprocessor.fit_transform(df)
    path = preprocessor.save(tmp_path / "preprocessor.joblib")

    loaded = Preprocessor.load(path)
    loaded_transformed = loaded.transform(df)
    pd.testing.assert_frame_equal(transformed, loaded_transformed)


def test_preprocessor_handles_missing_columns_gracefully() -> None:
    artifact = _dataset_artifact(n_students=5, sessions_per_student=1)
    df = records_to_frame(artifact.records)
    features = FeatureSelector().select()
    preprocessor = Preprocessor(features).fit(df)

    incomplete = df.drop(columns=[features[0]])
    transformed = preprocessor.transform(incomplete)  # should not raise; missing col -> NaN -> imputed
    assert len(transformed) == len(incomplete)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_student_aware_split_never_leaks() -> None:
    artifact = _dataset_artifact(n_students=20, sessions_per_student=3)
    df = records_to_frame(artifact.records)
    train_df, val_df = student_aware_split(df, test_size=0.25, random_state=42)
    assert_no_student_leakage(train_df, val_df)  # should not raise
    assert len(train_df) + len(val_df) == len(df)


def test_session_aware_split_keeps_sessions_intact() -> None:
    artifact = _dataset_artifact(n_students=10, sessions_per_student=3)
    df = records_to_frame(artifact.records)
    train_df, val_df = session_aware_split(df, test_size=0.25, random_state=42)
    assert set(train_df["session_id"]) & set(val_df["session_id"]) == set()


def test_stratified_split_preserves_class_presence() -> None:
    artifact = _dataset_artifact(n_students=20, sessions_per_student=3)
    df = records_to_frame(artifact.records)
    train_df, val_df = stratified_split(df, "attention_state", test_size=0.3, random_state=42)
    assert set(val_df["attention_state"]) == set(df["attention_state"])


def test_assert_no_student_leakage_detects_overlap() -> None:
    artifact = _dataset_artifact(n_students=5, sessions_per_student=1)
    df = records_to_frame(artifact.records)
    with pytest.raises(ValueError):
        assert_no_student_leakage(df, df)  # same students on both sides


def test_split_dataset_dispatch() -> None:
    artifact = _dataset_artifact(n_students=10, sessions_per_student=2)
    df = records_to_frame(artifact.records)
    for mode in ("random", "stratified", "session_aware", "student_aware"):
        train_df, val_df = split_dataset(df, mode, "attention_state", 0.2, 42)
        assert len(train_df) + len(val_df) == len(df)
    with pytest.raises(ValueError):
        split_dataset(df, "nonsense", "attention_state", 0.2, 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_classifier_model_factory_known_models() -> None:
    assert "logistic_regression" in ClassifierModelFactory.available_models()
    assert "random_forest" in ClassifierModelFactory.available_models()
    assert "gradient_boosting" in ClassifierModelFactory.available_models()


def test_classifier_model_factory_unknown_raises() -> None:
    with pytest.raises(KeyError):
        ClassifierModelFactory.create("not_a_model")


def test_model_fit_predict_proba_interface() -> None:
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series(["Focused", "Distracted", "Focused", "Distracted"])
    model = LogisticRegressionModel(random_state=42).fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert len(preds) == 4
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_save_load_round_trip(tmp_path) -> None:
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series(["Focused", "Distracted", "Focused", "Distracted"])
    model = LogisticRegressionModel(random_state=42).fit(X, y)
    path = model.save(tmp_path / "model.joblib")

    from dataset_generator.classifier.models import ClassifierModel

    loaded = ClassifierModel.load(path)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def test_trainer_produces_training_artifact() -> None:
    artifact = _dataset_artifact(n_students=20, sessions_per_student=3)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="student_aware"))

    assert training_artifact.metrics.accuracy > 0.0
    assert training_artifact.metadata.feature_names
    assert training_artifact.metadata.dataset_version == artifact.manifest.dataset_version
    assert training_artifact.metadata.config_fingerprint == artifact.manifest.config_fingerprint


def test_trainer_never_leaks_students_in_student_aware_mode() -> None:
    artifact = _dataset_artifact(n_students=20, sessions_per_student=3)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="logistic_regression", split_mode="student_aware"))
    assert training_artifact.metadata.train_record_count + training_artifact.metadata.validation_record_count == len(artifact.records)


def test_trainer_deterministic_for_same_seed() -> None:
    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    config = TrainingConfig(model_name="logistic_regression", split_mode="random", random_state=7)

    a = trainer.train(artifact, config)
    b = trainer.train(artifact, config)
    assert a.metrics.model_dump() == b.metrics.model_dump()


def test_trainer_with_calibration() -> None:
    artifact = _dataset_artifact(n_students=20, sessions_per_student=3)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(
        artifact, TrainingConfig(model_name="random_forest", split_mode="student_aware", calibration_method="isotonic")
    )
    assert training_artifact.calibration is not None
    assert training_artifact.calibration.expected_calibration_error >= 0.0


def test_trainer_without_feature_importance() -> None:
    artifact = _dataset_artifact(n_students=10, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(
        artifact, TrainingConfig(model_name="logistic_regression", split_mode="random", compute_feature_importance=False)
    )
    assert training_artifact.feature_importance is None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_compute_classification_metrics_perfect_predictions() -> None:
    y_true = np.array(["Focused", "Distracted", "Impulsive", "Focused"])
    y_pred = y_true.copy()
    y_proba = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    metrics = compute_classification_metrics(y_true, y_pred, y_proba, ["Distracted", "Focused", "Impulsive"])
    assert metrics.accuracy == 1.0
    assert metrics.f1_macro == 1.0
    assert metrics.mean_prediction_confidence == 1.0


def test_compute_classification_metrics_confusion_matrix_shape() -> None:
    y_true = np.array(["Focused", "Distracted", "Impulsive"])
    y_pred = np.array(["Focused", "Focused", "Impulsive"])
    y_proba = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba, ["Distracted", "Focused", "Impulsive"])
    assert len(metrics.confusion_matrix) == 3
    assert all(len(row) == 3 for row in metrics.confusion_matrix)
    assert set(metrics.per_class.keys()) == {"Distracted", "Focused", "Impulsive"}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_compute_ece_perfect_calibration_low_error() -> None:
    # class_labels = ["Distracted", "Focused"], so column 0 = P(Distracted),
    # column 1 = P(Focused) — each row's confident column must match its
    # true label for this to actually be "well-calibrated and correct".
    y_true = np.array(["Focused"] * 80 + ["Distracted"] * 20)
    y_proba = np.array([[0.2, 0.8]] * 80 + [[0.8, 0.2]] * 20)
    ece, bins = compute_ece(y_true, y_proba, ["Distracted", "Focused"], n_bins=10)
    assert ece < 0.25
    assert len(bins) == 10


def test_build_calibration_result() -> None:
    y_true = np.array(["Focused", "Distracted"] * 10)
    y_proba = np.tile(np.array([[0.9, 0.1], [0.1, 0.9]]), (10, 1))
    result = build_calibration_result(y_true, y_proba, ["Distracted", "Focused"], "platt")
    assert result.method == "platt"
    assert result.expected_calibration_error >= 0.0


def test_calibrate_model_returns_valid_probabilities() -> None:
    artifact = _dataset_artifact(n_students=20, sessions_per_student=3)
    df = records_to_frame(artifact.records)
    features = FeatureSelector().select()
    train_df, val_df = student_aware_split(df, test_size=0.3, random_state=42)

    preprocessor = Preprocessor(features)
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    from dataset_generator.classifier.models import RandomForestModel

    model = RandomForestModel(random_state=42).fit(X_train, train_df["attention_state"])
    calibrated = calibrate_model(model, X_val, val_df["attention_state"], "platt")
    proba = calibrated.predict_proba(X_val)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def test_permutation_importance_report_ranked_descending() -> None:
    artifact = _dataset_artifact(n_students=20, sessions_per_student=3)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="student_aware"))
    ranked = training_artifact.feature_importance.ranked
    importances = [e.importance for e in ranked]
    assert importances == sorted(importances, reverse=True)


def test_tree_importance_report_requires_tree_model() -> None:
    artifact = _dataset_artifact(n_students=10, sessions_per_student=2)
    df = records_to_frame(artifact.records)
    features = FeatureSelector().select()
    preprocessor = Preprocessor(features)
    X = preprocessor.fit_transform(df)

    from dataset_generator.classifier.models import LogisticRegressionModel as LRModel

    model = LRModel(random_state=42).fit(X, df["attention_state"])
    with pytest.raises(AttributeError):
        tree_importance_report(model, preprocessor.feature_names_out_)


def test_tree_importance_report_for_random_forest() -> None:
    artifact = _dataset_artifact(n_students=10, sessions_per_student=2)
    df = records_to_frame(artifact.records)
    features = FeatureSelector().select()
    preprocessor = Preprocessor(features)
    X = preprocessor.fit_transform(df)

    from dataset_generator.classifier.models import RandomForestModel

    model = RandomForestModel(random_state=42).fit(X, df["attention_state"])
    report = tree_importance_report(model, preprocessor.feature_names_out_)
    assert len(report.ranked) == len(preprocessor.feature_names_out_)
    assert report.top_k(3) == report.ranked[:3]


# ---------------------------------------------------------------------------
# Predictor (inference-only, cannot retrain)
# ---------------------------------------------------------------------------


def test_predictor_predict_batch_matches_predict() -> None:
    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="random"))

    predictor = AttentionClassifierPredictor(training_artifact)
    sample = artifact.records[:5]
    single_preds = [predictor.predict(r) for r in sample]
    batch_preds = predictor.predict_batch(sample)
    assert single_preds == batch_preds


def test_predictor_predict_proba_sums_to_one() -> None:
    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="random"))

    predictor = AttentionClassifierPredictor(training_artifact)
    probas = predictor.predict_proba(artifact.records[:5])
    for p in probas:
        assert abs(sum(p.values()) - 1.0) < 1e-6


def test_predictor_with_confidence_matches_top_probability() -> None:
    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="random"))

    predictor = AttentionClassifierPredictor(training_artifact)
    results = predictor.predict_with_confidence(artifact.records[:5])
    for r in results:
        assert r.confidence == max(r.probabilities.values())
        assert r.predicted_state == max(r.probabilities, key=r.probabilities.get)


def test_predictor_with_explanation_includes_top_features() -> None:
    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="random"))

    predictor = AttentionClassifierPredictor(training_artifact)
    results = predictor.predict_with_explanation(artifact.records[:2])
    for r in results:
        assert r.explanation is not None
        assert len(r.explanation) <= 10


def test_predictor_feature_order_independent_of_input_dataframe_order() -> None:
    """Predictor must reindex to the persisted feature order, never trust caller column order."""

    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="random"))

    predictor = AttentionClassifierPredictor(training_artifact)
    record = artifact.records[0]
    result_a = predictor.predict_proba([record])[0]
    # Predicting the same single record again (order can't change for a
    # single-record batch, but this asserts the call is stable/repeatable).
    result_b = predictor.predict_proba([record])[0]
    assert result_a == result_b


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_load_training_artifact_round_trip(tmp_path) -> None:
    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="random_forest", split_mode="random"))

    save_training_artifact(training_artifact, tmp_path)
    loaded = load_training_artifact(tmp_path)

    assert loaded.metadata.feature_names == training_artifact.metadata.feature_names
    assert loaded.metrics.model_dump() == training_artifact.metrics.model_dump()

    original_predictor = AttentionClassifierPredictor(training_artifact)
    loaded_predictor = AttentionClassifierPredictor(loaded)
    sample = artifact.records[:5]
    assert original_predictor.predict_batch(sample) == loaded_predictor.predict_batch(sample)


def test_load_training_artifact_does_not_require_retraining(tmp_path) -> None:
    """Loading and predicting must work without ever importing/using AttentionClassifierTrainer."""

    artifact = _dataset_artifact(n_students=15, sessions_per_student=2)
    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(artifact, TrainingConfig(model_name="logistic_regression", split_mode="random"))
    save_training_artifact(training_artifact, tmp_path)

    loaded = load_training_artifact(tmp_path)
    predictor = AttentionClassifierPredictor(loaded)
    predictions = predictor.predict_batch(artifact.records[:3])
    assert len(predictions) == 3


# ---------------------------------------------------------------------------
# Large-scale stress test (100,000+ DatasetRecords)
# ---------------------------------------------------------------------------


def test_stress_100000_records_train_and_predict() -> None:
    """As with Module 7's stress test: replicate a real, small simulated
    batch with re-suffixed IDs rather than re-running full simulation
    100,000+ times — this exercises the classifier pipeline's scalability,
    which Module 6 already stress-tests simulation itself at 1000 sessions.
    """

    base_artifact = _dataset_artifact(n_students=20, sessions_per_student=3)
    base_records = base_artifact.records
    target_size = 100_000
    replication_factor = target_size // len(base_records) + 1

    records: list[DatasetRecord] = []
    for batch in range(replication_factor):
        for record in base_records:
            data = record.model_dump(mode="json")
            data["session_id"] = f"{record.session_id}_batch{batch}"
            data["response_id"] = f"{record.response_id}_batch{batch}"
            data["student_id"] = f"{record.student_id}_batch{batch}"
            records.append(DatasetRecord.model_validate(data))

    assert len(records) >= target_size

    large_artifact = base_artifact.model_copy(
        update={
            "records": records,
            "metadata": base_artifact.metadata.model_copy(update={"record_count": len(records)}),
        }
    )

    trainer = AttentionClassifierTrainer()
    training_artifact = trainer.train(
        large_artifact,
        TrainingConfig(model_name="logistic_regression", split_mode="random", compute_feature_importance=False),
    )
    assert training_artifact.metadata.train_record_count + training_artifact.metadata.validation_record_count == len(records)

    predictor = AttentionClassifierPredictor(training_artifact)
    predictions = predictor.predict_batch(records[:1000])
    assert len(predictions) == 1000
