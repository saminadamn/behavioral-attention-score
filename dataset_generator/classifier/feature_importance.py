"""Module 8, Step 9: Feature Importance.

`permutation_importance_report` uses `sklearn.inspection.permutation_importance`
directly (works for any model). `tree_importance_report` reads a tree-based
estimator's own `feature_importances_` (raises clearly if the model doesn't
have one, rather than silently returning garbage). SHAP is optional — only
imported if installed, and its absence is reported as `None`, never a hard
failure, matching the brief's "SHAP (optional)".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from dataset_generator.classifier.models import ClassifierModel
from dataset_generator.models.training import FeatureImportanceEntry, FeatureImportanceReport


def permutation_importance_report(
    model: ClassifierModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> FeatureImportanceReport:
    """Rank `feature_names` by permutation importance on `(X_val, y_val)`."""

    result = permutation_importance(
        model.underlying_estimator, X_val, y_val, n_repeats=n_repeats, random_state=random_state
    )
    entries = [
        FeatureImportanceEntry(feature=name, importance=float(mean), std=float(std))
        for name, mean, std in zip(feature_names, result.importances_mean, result.importances_std)
    ]
    entries.sort(key=lambda e: e.importance, reverse=True)
    return FeatureImportanceReport(method="permutation", ranked=entries)


def tree_importance_report(model: ClassifierModel, feature_names: list[str]) -> FeatureImportanceReport:
    """Rank `feature_names` by a tree-based model's built-in `feature_importances_`."""

    estimator = model.underlying_estimator
    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(
            f"{type(estimator).__name__} has no feature_importances_ "
            "(tree importance only applies to tree-based models)"
        )
    importances = estimator.feature_importances_
    entries = [
        FeatureImportanceEntry(feature=name, importance=float(value))
        for name, value in zip(feature_names, importances)
    ]
    entries.sort(key=lambda e: e.importance, reverse=True)
    return FeatureImportanceReport(method="tree", ranked=entries)


def shap_importance_report(
    model: ClassifierModel, X_val: pd.DataFrame, feature_names: list[str]
) -> FeatureImportanceReport | None:
    """Rank `feature_names` by mean absolute SHAP value, if the `shap` package is installed.

    Returns `None` (not an error) if `shap` isn't installed — it's an
    optional dependency this project never requires.
    """

    try:
        import shap
    except ImportError:
        return None

    explainer = shap.Explainer(model.underlying_estimator, X_val)
    shap_values = explainer(X_val)
    values = shap_values.values
    if values.ndim == 3:  # (n_samples, n_features, n_classes) for multi-class
        values = np.abs(values).mean(axis=2)
    mean_abs = np.abs(values).mean(axis=0)

    entries = [
        FeatureImportanceEntry(feature=name, importance=float(value))
        for name, value in zip(feature_names, mean_abs)
    ]
    entries.sort(key=lambda e: e.importance, reverse=True)
    return FeatureImportanceReport(method="shap", ranked=entries)
