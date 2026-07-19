"""Module 8, Step 7: Metrics.

Every number here comes from `sklearn.metrics` directly — this module's
only job is assembling them into one `ClassificationMetrics` object, not
computing anything scikit-learn doesn't already provide.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from dataset_generator.models.training import ClassificationMetrics


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_labels: list[str]
) -> ClassificationMetrics:
    """Compute a full `ClassificationMetrics` snapshot for one evaluation set.

    `class_labels` must match the column order of `y_proba` (i.e. the
    model's `classes_`), since ROC-AUC (one-vs-rest) and per-class
    precision/recall/F1 both depend on that alignment.
    """

    accuracy = float((y_true == y_pred).mean())
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(y_true, y_pred, labels=class_labels, zero_division=0)
    )
    per_class = {
        label: {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": float(per_class_support[i]),
        }
        for i, label in enumerate(class_labels)
    }

    roc_auc_ovr: float | None
    try:
        y_true_binarized = label_binarize(y_true, classes=class_labels)
        if y_true_binarized.shape[1] > 1:
            roc_auc_ovr = float(
                roc_auc_score(y_true_binarized, y_proba, multi_class="ovr", average="macro")
            )
        else:
            roc_auc_ovr = None  # only one class present in y_true; AUC undefined
    except ValueError:
        roc_auc_ovr = None

    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    report_text = classification_report(y_true, y_pred, labels=class_labels, zero_division=0)

    mean_confidence = float(np.max(y_proba, axis=1).mean()) if len(y_proba) else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        f1_weighted=f1_weighted,
        roc_auc_ovr=roc_auc_ovr,
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        class_labels=list(class_labels),
        classification_report=report_text,
        mean_prediction_confidence=mean_confidence,
    )
