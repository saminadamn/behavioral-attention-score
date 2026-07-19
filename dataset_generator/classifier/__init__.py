"""Module 8: Attention State Classifier.

Trains and evaluates classifiers predicting `attention_state` from Module
7's `DatasetArtifact` — no synthetic data is generated here. `TrainingArtifact`
(see `models/training.py`) is this module's single source of truth, the way
`DatasetArtifact` is Module 7's.
"""

from dataset_generator.classifier.calibration import (
    build_calibration_result,
    calibrate_model,
    compute_ece,
)
from dataset_generator.classifier.feature_importance import (
    permutation_importance_report,
    shap_importance_report,
    tree_importance_report,
)
from dataset_generator.classifier.feature_selection import FeatureSelector
from dataset_generator.classifier.metrics import compute_classification_metrics
from dataset_generator.classifier.models import (
    ClassifierModel,
    ClassifierModelFactory,
    GradientBoostingModel,
    LogisticRegressionModel,
    RandomForestModel,
)
from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.classifier.preprocessing import Preprocessor
from dataset_generator.classifier.serialization import load_training_artifact, save_training_artifact
from dataset_generator.classifier.splitting import (
    assert_no_student_leakage,
    cv_groups,
    make_cv_splitter,
    split_dataset,
)
from dataset_generator.classifier.trainer import AttentionClassifierTrainer, TrainingConfig

__all__ = [
    "AttentionClassifierPredictor",
    "AttentionClassifierTrainer",
    "ClassifierModel",
    "ClassifierModelFactory",
    "FeatureSelector",
    "GradientBoostingModel",
    "LogisticRegressionModel",
    "Preprocessor",
    "RandomForestModel",
    "TrainingConfig",
    "assert_no_student_leakage",
    "build_calibration_result",
    "calibrate_model",
    "compute_classification_metrics",
    "compute_ece",
    "cv_groups",
    "load_training_artifact",
    "make_cv_splitter",
    "permutation_importance_report",
    "save_training_artifact",
    "shap_importance_report",
    "split_dataset",
    "tree_importance_report",
]
