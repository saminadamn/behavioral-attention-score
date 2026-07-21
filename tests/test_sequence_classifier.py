"""Tests for dataset_generator.classifier.sequence_model — the LSTM
attention-state classifier built for Phase 4's deep-learning comparison.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.classifier.sequence_model import CLASS_LABELS, _build_sequences, train_lstm_classifier
from dataset_generator.config import default_config
from dataset_generator.generators import generate_sessions, generate_students
from dataset_generator.pipeline import build_dataset_artifact
from dataset_generator.utils import build_rng_streams


def _dataset(n_students: int = 15, sessions_per_student: int = 2):
    config = default_config()
    streams = build_rng_streams(config.seed)
    students = generate_students(config, streams)[:n_students]
    sessions = generate_sessions(config, students, sessions_per_student, build_rng_streams(config.seed))
    return build_dataset_artifact(config, students, sessions)


def test_build_sequences_shapes_match_row_count():
    import pandas as pd

    df = pd.DataFrame({
        "session_id": ["s1", "s1", "s1", "s2"],
        "interaction_number": [1, 2, 3, 1],
        "attention_state": ["Focused", "Distracted", "Focused", "Impulsive"],
    })
    features = np.arange(4 * 3).reshape(4, 3).astype(float)
    sequences, labels = _build_sequences(df, features, sequence_length=2)
    assert sequences.shape == (4, 2, 3)
    assert labels.shape == (4,)


def test_build_sequences_pads_short_sessions_by_repeating_first_frame():
    import pandas as pd

    df = pd.DataFrame({
        "session_id": ["s1", "s1"],
        "interaction_number": [1, 2],
        "attention_state": ["Focused", "Distracted"],
    })
    features = np.array([[1.0, 1.0], [2.0, 2.0]])
    sequences, _ = _build_sequences(df, features, sequence_length=3)
    # First row (only one real frame available): padded with two copies of frame 0.
    assert np.array_equal(sequences[0], np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))
    # Second row: one pad frame + the two real frames.
    assert np.array_equal(sequences[1], np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]))


def test_train_lstm_classifier_returns_well_formed_result():
    dataset = _dataset()
    result = train_lstm_classifier(dataset, epochs=4, sequence_length=4, hidden_dim=8)

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision_macro <= 1.0
    assert 0.0 <= result.recall_macro <= 1.0
    assert 0.0 <= result.f1_macro <= 1.0
    assert result.training_seconds > 0.0
    assert len(result.loss_per_epoch) == 4


def test_train_lstm_classifier_loss_decreases():
    dataset = _dataset()
    result = train_lstm_classifier(dataset, epochs=8, sequence_length=4, hidden_dim=8)
    assert result.loss_per_epoch[-1] < result.loss_per_epoch[0]


def test_train_lstm_classifier_deterministic():
    dataset = _dataset()
    result_a = train_lstm_classifier(dataset, epochs=4, sequence_length=4, hidden_dim=8, random_state=7)
    result_b = train_lstm_classifier(dataset, epochs=4, sequence_length=4, hidden_dim=8, random_state=7)
    assert result_a.loss_per_epoch == result_b.loss_per_epoch
    assert result_a.accuracy == result_b.accuracy


def test_class_labels_match_attention_states():
    assert set(CLASS_LABELS) == {"Focused", "Distracted", "Impulsive"}
