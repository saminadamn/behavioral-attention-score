"""An LSTM attention-state classifier — Phase 4's "deep learning
comparison" (`docs/DEEP_LEARNING_COMPARISON.md`) needs a genuine sequence
model alongside the tabular classifiers already in `models.py`
(logistic regression, random forest, gradient boosting, MLP).

Reuses `RecurrentQNetwork` (`dataset_generator.rl_experimental.network`) —
the same LSTM implementation already verified against numerical gradients
for the offline-RL package — trained here via ordinary softmax
cross-entropy instead of a Bellman target, exactly the same "swap the
output-gradient formula, reuse the one shared BPTT pass" pattern
`bcq.py`'s behavior-cloning model already uses. No new backpropagation
code was written for this: `apply_output_gradient` is the same method,
`network.py`'s BPTT the same implementation.

This is a genuinely different problem shape from the tabular classifiers:
Preprocessor + LogisticRegression/RandomForest/MLP see one interaction's
features in isolation; this model sees the last `sequence_length`
interactions' transformed features and predicts the *current* one's
attention state — the same "trend, not snapshot" argument
`rl_experimental/network.py`'s docstring makes for the DRQN intervention
agent, applied here to classification instead of control.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset_generator.classifier.feature_selection import FeatureSelector
from dataset_generator.classifier.preprocessing import Preprocessor
from dataset_generator.classifier.splitting import SplitMode, split_dataset
from dataset_generator.classifier.trainer import TARGET_COLUMN
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.pipeline.feature_registry import FeatureRegistry
from dataset_generator.rl_experimental.network import RecurrentQNetwork
from dataset_generator.validators.dataset_validator import records_to_frame

CLASS_LABELS = ("Distracted", "Focused", "Impulsive")
CLASS_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _build_sequences(
    df: pd.DataFrame, features: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """One (sequence, label) pair per row, grouped by `session_id` and
    ordered by `interaction_number` — never a window crossing a session
    boundary, matching `rl_experimental.environment._windowed_sequence`'s
    convention exactly (independently reimplemented at tabular-feature
    granularity rather than imported, since that helper is keyed on the
    12-dim intervention state vector, not this module's preprocessed
    feature matrix).
    """

    sequences: list[np.ndarray] = []
    labels: list[int] = []

    working = df[["session_id", "interaction_number", TARGET_COLUMN]].copy()
    working["_row"] = np.arange(len(df))

    for _, group in working.groupby("session_id"):
        ordered = group.sort_values("interaction_number")
        row_positions = ordered["_row"].to_numpy()
        class_labels = ordered[TARGET_COLUMN].to_numpy()

        for i, row_index in enumerate(row_positions):
            start = i - sequence_length + 1
            if start >= 0:
                window_positions = row_positions[start : i + 1]
            else:
                pad = [row_positions[0]] * (-start)
                window_positions = np.array(pad + list(row_positions[: i + 1]))
            sequences.append(features[window_positions])
            labels.append(CLASS_INDEX[class_labels[i]])

    return np.stack(sequences), np.array(labels)


@dataclass(frozen=True)
class LSTMClassifierResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    training_seconds: float
    loss_per_epoch: list[float]


def train_lstm_classifier(
    dataset_artifact: DatasetArtifact,
    *,
    sequence_length: int = 5,
    hidden_dim: int = 16,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    split_mode: SplitMode = "student_aware",
    test_size: float = 0.2,
    random_state: int = 42,
) -> LSTMClassifierResult:
    """Trains and evaluates the LSTM classifier, mirroring
    `AttentionClassifierTrainer.train`'s split/preprocess steps exactly
    (same `FeatureSelector`, same `split_dataset`, same `Preprocessor`) so
    its accuracy is comparable to the tabular classifiers' — the only
    difference is what happens after preprocessing.
    """

    from sklearn.metrics import f1_score, precision_score, recall_score

    df = records_to_frame(dataset_artifact.records)
    registry = FeatureRegistry()
    feature_names = FeatureSelector(registry).select()

    train_df, val_df = split_dataset(df, split_mode, TARGET_COLUMN, test_size, random_state)

    preprocessor = Preprocessor(feature_names, registry)
    X_train = preprocessor.fit_transform(train_df).to_numpy()
    X_val = preprocessor.transform(val_df).to_numpy()

    train_sequences, train_labels = _build_sequences(train_df.reset_index(drop=True), X_train, sequence_length)
    val_sequences, val_labels = _build_sequences(val_df.reset_index(drop=True), X_val, sequence_length)

    network = RecurrentQNetwork(
        state_dim=X_train.shape[1], action_dim=len(CLASS_LABELS), hidden_dim=hidden_dim, seed=random_state
    )
    rng = np.random.default_rng(random_state)

    start_time = time.perf_counter()
    loss_per_epoch: list[float] = []
    n_train = len(train_sequences)
    steps_per_epoch = max(1, n_train // batch_size)

    for _ in range(epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            batch_idx = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
            batch_seq = train_sequences[batch_idx]
            batch_labels = train_labels[batch_idx]

            logits, cache = network.forward(batch_seq)
            probs = _softmax(logits)
            loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(batch_idx)), batch_labels], 1e-12, 1.0))))
            epoch_losses.append(loss)

            grad_logits = probs.copy()
            grad_logits[np.arange(len(batch_idx)), batch_labels] -= 1.0
            grad_logits /= len(batch_idx)
            network.apply_output_gradient(cache, grad_logits, learning_rate)
        loss_per_epoch.append(float(np.mean(epoch_losses)))
    training_seconds = time.perf_counter() - start_time

    val_logits = network.predict(val_sequences)
    val_predictions = np.argmax(val_logits, axis=1)

    accuracy = float(np.mean(val_predictions == val_labels))
    precision_macro = float(precision_score(val_labels, val_predictions, average="macro", zero_division=0))
    recall_macro = float(recall_score(val_labels, val_predictions, average="macro", zero_division=0))
    f1_macro = float(f1_score(val_labels, val_predictions, average="macro", zero_division=0))

    return LSTMClassifierResult(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        training_seconds=training_seconds,
        loss_per_epoch=loss_per_epoch,
    )
