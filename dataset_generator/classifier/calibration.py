"""Module 8, Step 8: Calibration.

`calibrate_model` wraps the already-fitted estimator with
`sklearn.frozen.FrozenEstimator` before handing it to
`CalibratedClassifierCV` — this fits *only* the calibration mapping (Platt/
sigmoid or isotonic) on validation data, never re-fitting the base
classifier itself (the older `cv="prefit"` API this replaces was removed in
recent scikit-learn versions). Expected Calibration Error and the
reliability curve are plain numpy binning over `(confidence, correctness)`
pairs — no scikit-learn primitive for those exists, so that part is
hand-written, but deliberately just a bin-and-average.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from dataset_generator.classifier.models import ClassifierModel
from dataset_generator.models.training import CalibrationResult, ReliabilityBin

CalibrationMethod = Literal["platt", "isotonic"]

_METHOD_TO_SKLEARN = {"platt": "sigmoid", "isotonic": "isotonic"}


def calibrate_model(
    model: ClassifierModel, X_val: pd.DataFrame, y_val: pd.Series, method: CalibrationMethod
) -> CalibratedClassifierCV:
    """Fit a calibration mapping for `model` on `(X_val, y_val)`.

    Returns a fitted `CalibratedClassifierCV` — callers use its own
    `predict_proba` for calibrated probabilities; `model` itself is
    untouched.
    """

    if method not in _METHOD_TO_SKLEARN:
        raise ValueError(f"unknown calibration method {method!r}; use 'platt' or 'isotonic'")

    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(model.underlying_estimator), method=_METHOD_TO_SKLEARN[method]
    )
    calibrator.fit(X_val, y_val)
    return calibrator


def compute_ece(
    y_true: np.ndarray, y_proba: np.ndarray, class_labels: list[str], n_bins: int = 10
) -> tuple[float, list[ReliabilityBin]]:
    """Expected Calibration Error + reliability bins, from max-probability confidence.

    For each prediction, "confidence" is the model's top class probability
    and "correct" is whether that top class matches the true label. Rows
    are bucketed into `n_bins` equal-width confidence bins; ECE is the
    support-weighted mean absolute gap between each bin's average
    confidence and its actual accuracy.
    """

    predicted_indices = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    predicted_labels = np.array(class_labels)[predicted_indices]
    correct = (predicted_labels == np.asarray(y_true)).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append(ReliabilityBin(bin_lower=lo, bin_upper=hi, mean_confidence=0.0, mean_accuracy=0.0, count=0))
            continue
        mean_confidence = float(confidences[in_bin].mean())
        mean_accuracy = float(correct[in_bin].mean())
        bins.append(
            ReliabilityBin(bin_lower=lo, bin_upper=hi, mean_confidence=mean_confidence, mean_accuracy=mean_accuracy, count=count)
        )
        ece += (count / total) * abs(mean_confidence - mean_accuracy)

    return ece, bins


def build_calibration_result(
    y_true: np.ndarray, y_proba: np.ndarray, class_labels: list[str], method: str, n_bins: int = 10
) -> CalibrationResult:
    """Build a `CalibrationResult` (ECE + reliability bins) for `y_proba` against `y_true`."""

    ece, bins = compute_ece(y_true, y_proba, class_labels, n_bins)
    return CalibrationResult(method=method, expected_calibration_error=ece, reliability_bins=bins)
