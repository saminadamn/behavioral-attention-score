"""Tests for Module 7: Dataset Assembly and Verification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset_generator.config import default_config
from dataset_generator.generators import generate_sessions, generate_students
from dataset_generator.models.dataset import DatasetRecord, FeatureCategory
from dataset_generator.pipeline.dataset_artifact import (
    build_dataset_artifact,
    build_manifest,
    export_dataset_artifact,
)
from dataset_generator.pipeline.dataset_builder import DatasetBuilder
from dataset_generator.pipeline.dataset_export import (
    export_csv,
    export_jsonl,
    export_manifest_json,
    export_metadata_json,
    export_parquet,
)
from dataset_generator.pipeline.dataset_statistics import compute_dataset_statistics
from dataset_generator.pipeline.feature_registry import FeatureRegistry
from dataset_generator.utils import build_rng_streams
from dataset_generator.validators.dataset_validator import validate_dataset


def _small_dataset(config=None, seed: int | None = None, n_students: int = 5, sessions_per_student: int = 2):
    config = config or default_config()
    streams = build_rng_streams(seed if seed is not None else config.seed)
    students = generate_students(config, streams)[:n_students]
    sessions = generate_sessions(
        config, students, sessions_per_student=sessions_per_student, rng_streams=build_rng_streams(config.seed)
    )
    return config, students, sessions


# ---------------------------------------------------------------------------
# Dataset building / flattening
# ---------------------------------------------------------------------------


def test_dataset_builder_produces_one_record_per_interaction() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    expected = sum(len(s.interactions) for s in sessions)
    assert len(records) == expected


def test_dataset_record_has_no_nested_objects() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    for field_name in DatasetRecord.model_fields:
        value = getattr(records[0], field_name)
        assert isinstance(value, (str, int, float, bool)), f"{field_name} is not a flat scalar: {type(value)}"


def test_dataset_builder_raises_for_unknown_student() -> None:
    config, students, sessions = _small_dataset()
    builder = DatasetBuilder(students[:1])  # deliberately missing some students
    with pytest.raises(KeyError):
        builder.build(sessions)


def test_response_ids_unique_across_multiple_sessions_per_student() -> None:
    """Regression test for the response_id collision Module 7 uncovered:
    interaction_number restarts at 1 every session, so student_id +
    interaction_number alone collided across a student's sessions."""

    config, students, sessions = _small_dataset(sessions_per_student=3)
    records = DatasetBuilder(students).build(sessions)
    response_ids = [r.response_id for r in records]
    assert len(response_ids) == len(set(response_ids))


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


def test_feature_registry_covers_every_dataset_record_field() -> None:
    registry = FeatureRegistry()
    registered_names = {d.name for d in registry.all_features()}
    assert registered_names == set(DatasetRecord.model_fields)


def test_feature_registry_lookup_and_categories() -> None:
    registry = FeatureRegistry()
    definition = registry.get("attention_state")
    assert definition.category == FeatureCategory.TARGET

    student_features = registry.by_category(FeatureCategory.STUDENT)
    assert all(d.name.startswith("student_") for d in student_features)
    assert len(student_features) > 0


def test_feature_registry_counts_sum_to_total() -> None:
    registry = FeatureRegistry()
    counts = registry.feature_counts()
    assert sum(counts.values()) == len(registry.all_features())


def test_feature_registry_unknown_feature_raises() -> None:
    with pytest.raises(KeyError):
        FeatureRegistry().get("not_a_real_feature")


def test_feature_registry_schema_version_present() -> None:
    assert FeatureRegistry().schema_version


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_dataset_clean_generator_output() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    report = validate_dataset(
        records,
        known_student_ids={s.student_id for s in students},
        known_session_ids={s.session_id for s in sessions},
    )
    assert report.is_valid
    assert report.duplicate_id_count == 0
    assert report.impossible_transition_count == 0
    assert report.invalid_attention_state_count == 0


def test_validate_dataset_detects_duplicate_ids() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    duplicated = records + [records[0]]
    report = validate_dataset(duplicated)
    assert report.duplicate_id_count >= 1
    assert not report.is_valid


def test_validate_dataset_detects_out_of_range_values() -> None:
    # `DatasetRecord`'s own Field(ge=,le=) constraints already reject this via
    # the normal constructor/model_validate — dataset_validator's range check
    # exists for data read from an untrusted source (e.g. a hand-edited CSV
    # reloaded without going through Pydantic), so model_construct bypasses
    # validation here to simulate exactly that.
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    data = records[0].model_dump(mode="json")
    data["response_sentiment"] = 5.0  # out of [-1, 1]
    tampered = DatasetRecord.model_construct(**data)
    report = validate_dataset([tampered])
    assert "response_sentiment" in report.invalid_range_issues


def test_validate_dataset_detects_invalid_attention_state() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    data = records[0].model_dump(mode="json")
    data["attention_state"] = "Bored"
    tampered = DatasetRecord.model_validate(data)
    report = validate_dataset([tampered])
    assert report.invalid_attention_state_count == 1


def test_validate_dataset_detects_orphan_ids() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    report = validate_dataset(records, known_student_ids=set(), known_session_ids=set())
    assert len(report.orphan_student_ids) > 0
    assert len(report.orphan_session_ids) > 0


def test_validate_dataset_empty_input() -> None:
    report = validate_dataset([])
    assert report.record_count == 0
    assert report.is_valid


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_dataset_statistics_balances_sum_to_one() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    stats = compute_dataset_statistics(records)
    assert abs(sum(stats.attention_balance.values()) - 1.0) < 1e-9
    assert abs(sum(stats.profile_balance.values()) - 1.0) < 1e-9
    assert abs(sum(stats.subject_balance.values()) - 1.0) < 1e-9
    assert abs(sum(stats.difficulty_balance.values()) - 1.0) < 1e-9


def test_dataset_statistics_class_balance_equals_attention_balance() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    stats = compute_dataset_statistics(records)
    assert stats.class_balance == stats.attention_balance


def test_dataset_statistics_correlation_matrix_diagonal_is_one() -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)
    stats = compute_dataset_statistics(records)
    for feature in stats.correlation_matrix:
        assert abs(stats.correlation_matrix[feature][feature] - 1.0) < 1e-6


def test_dataset_statistics_session_balance_counts_sessions_not_rows() -> None:
    config, students, sessions = _small_dataset(sessions_per_student=2)
    records = DatasetBuilder(students).build(sessions)
    stats = compute_dataset_statistics(records)
    assert sum(stats.session_balance.values()) == len(sessions)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_export_csv_jsonl_parquet_roundtrip(tmp_path) -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)

    csv_path = export_csv(records, tmp_path / "dataset.csv")
    jsonl_path = export_jsonl(records, tmp_path / "dataset.jsonl")
    parquet_path = export_parquet(records, tmp_path / "dataset.parquet")

    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == len(records)

    with jsonl_path.open(encoding="utf-8") as f:
        jsonl_rows = [json.loads(line) for line in f]
    assert len(jsonl_rows) == len(records)

    parquet_df = pd.read_parquet(parquet_path)
    assert len(parquet_df) == len(records)
    assert list(parquet_df.columns) == list(DatasetRecord.model_fields)


def test_export_is_deterministic_regardless_of_input_order(tmp_path) -> None:
    config, students, sessions = _small_dataset()
    records = DatasetBuilder(students).build(sessions)

    path_a = export_csv(records, tmp_path / "a.csv")
    path_b = export_csv(list(reversed(records)), tmp_path / "b.csv")

    assert path_a.read_bytes() == path_b.read_bytes()


def test_export_metadata_and_manifest_json(tmp_path) -> None:
    config, students, sessions = _small_dataset()
    artifact = build_dataset_artifact(config, students, sessions)

    metadata_path = export_metadata_json(artifact.metadata, tmp_path / "metadata.json")
    manifest_path = export_manifest_json(artifact.manifest, tmp_path / "manifest.json")

    metadata_data = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metadata_data["record_count"] == artifact.metadata.record_count
    assert manifest_data["seed"] == config.seed


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def test_manifest_reuses_config_version_metadata() -> None:
    config = default_config()
    manifest = build_manifest(config)
    assert manifest.dataset_version == config.version_metadata.dataset_version
    assert manifest.schema_version == config.version_metadata.schema_version
    assert manifest.generator_version == config.version_metadata.generator_version
    assert manifest.seed == config.seed
    assert manifest.config_fingerprint


def test_manifest_fingerprint_changes_with_seed() -> None:
    config = default_config()
    data = config.model_dump(mode="json")
    data["seed"] = config.seed + 1
    from dataset_generator.config import GeneratorConfig

    other_config = GeneratorConfig.model_validate(data)
    assert build_manifest(config).config_fingerprint != build_manifest(other_config).config_fingerprint


# ---------------------------------------------------------------------------
# Deterministic generation / DatasetArtifact
# ---------------------------------------------------------------------------


def test_dataset_artifact_deterministic_for_same_seed() -> None:
    config = default_config()

    def build():
        streams = build_rng_streams(config.seed)
        students = generate_students(config, streams)[:3]
        sessions = generate_sessions(config, students, sessions_per_student=2, rng_streams=build_rng_streams(config.seed))
        return build_dataset_artifact(config, students, sessions)

    a, b = build(), build()
    assert [r.model_dump(mode="json") for r in a.records] == [r.model_dump(mode="json") for r in b.records]


def test_export_dataset_artifact_populates_exports(tmp_path) -> None:
    config, students, sessions = _small_dataset()
    artifact = build_dataset_artifact(config, students, sessions)
    assert artifact.exports == {}

    exported = export_dataset_artifact(artifact, tmp_path)
    assert set(exported.exports) == {"csv", "parquet", "jsonl", "metadata_json", "manifest_json"}
    for path in exported.exports.values():
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Large dataset stress test (100,000+ DatasetRecords)
# ---------------------------------------------------------------------------


def test_stress_100000_dataset_records(tmp_path) -> None:
    """Exercises validate_dataset/compute_dataset_statistics/export at 100k+ scale.

    Building 100,000 records via full session simulation would take far
    longer than this pipeline stage actually needs to be tested at — Module
    6 already stress-tests simulation itself at 1000 sessions. Here, a
    real, small simulated batch is replicated with re-suffixed IDs
    (session_id/response_id) to produce a realistic-shaped 100k+ row
    dataset that genuinely exercises the pipeline's scalability.
    """

    config, students, sessions = _small_dataset(n_students=5, sessions_per_student=2)
    base_records = DatasetBuilder(students).build(sessions)
    assert len(base_records) > 0

    target_size = 100_000
    replication_factor = target_size // len(base_records) + 1

    records: list[DatasetRecord] = []
    for batch in range(replication_factor):
        for record in base_records:
            data = record.model_dump(mode="json")
            data["session_id"] = f"{record.session_id}_batch{batch}"
            data["response_id"] = f"{record.response_id}_batch{batch}"
            records.append(DatasetRecord.model_validate(data))

    assert len(records) >= target_size

    report = validate_dataset(records)
    assert report.record_count == len(records)
    assert report.duplicate_id_count == 0
    assert report.invalid_attention_state_count == 0

    stats = compute_dataset_statistics(records)
    assert stats.record_count == len(records)
    assert abs(sum(stats.attention_balance.values()) - 1.0) < 1e-9

    csv_path = export_csv(records, tmp_path / "large.csv")
    parquet_path = export_parquet(records, tmp_path / "large.parquet")
    assert csv_path.exists()
    assert parquet_path.exists()

    reloaded = pd.read_parquet(parquet_path)
    assert len(reloaded) == len(records)
