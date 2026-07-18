"""Module 7: `DatasetArtifact` assembly — the single source of truth.

Ties together `DatasetBuilder` (Step 2), `validate_dataset` (Step 4),
`compute_dataset_statistics` (Step 5), and manifest versioning (Step 7) into
one `DatasetArtifact`. Downstream modules (an attention classifier, BAS, the
evaluation pipeline, paper figures) should depend on this object — or on
files exported from it — rather than re-deriving records from
`SessionRecord`s themselves or re-implementing any of the validation/
statistics logic already here.
"""

from __future__ import annotations

from pathlib import Path

from dataset_generator.config import GeneratorConfig, compute_fingerprint
from dataset_generator.models.dataset import DatasetArtifact, DatasetManifest, DatasetMetadata
from dataset_generator.models.session import SessionRecord
from dataset_generator.models.student import Student
from dataset_generator.pipeline.dataset_builder import DatasetBuilder
from dataset_generator.pipeline.dataset_export import (
    export_csv,
    export_jsonl,
    export_manifest_json,
    export_metadata_json,
    export_parquet,
)
from dataset_generator.pipeline.dataset_statistics import compute_dataset_statistics
from dataset_generator.utils.git_info import detect_git_commit
from dataset_generator.validators.dataset_validator import validate_dataset


def build_manifest(config: GeneratorConfig) -> DatasetManifest:
    """Build a `DatasetManifest` from `config`'s existing version metadata.

    Reuses `config.version_metadata` (Module 1) and `compute_fingerprint`
    (Module 1) rather than introducing new, separately-tracked version
    fields — this manifest is a *view* over provenance that already exists,
    plus a fresh generation timestamp and a best-effort git commit hash.
    """

    return DatasetManifest(
        dataset_version=config.version_metadata.dataset_version,
        schema_version=config.version_metadata.schema_version,
        generator_version=config.version_metadata.generator_version,
        generation_timestamp=DatasetManifest.now_timestamp(),
        seed=config.seed,
        config_fingerprint=compute_fingerprint(config),
        git_commit_hash=detect_git_commit(),
    )


def build_dataset_artifact(
    config: GeneratorConfig,
    students: list[Student],
    sessions: list[SessionRecord],
) -> DatasetArtifact:
    """Assemble a complete `DatasetArtifact` from simulated sessions.

    `students` and `sessions` are dependency-injected (already produced by
    earlier modules) — this function performs no simulation itself.
    """

    builder = DatasetBuilder(students)
    records = builder.build(sessions)

    known_student_ids = {s.student_id for s in students}
    known_session_ids = {s.session_id for s in sessions}
    validation = validate_dataset(records, known_student_ids, known_session_ids)
    statistics = compute_dataset_statistics(records)

    metadata = DatasetMetadata(
        student_count=len(students),
        session_count=len(sessions),
        record_count=len(records),
        subjects_covered=sorted({r.prompt_subject for r in records}),
        profiles_covered=sorted({r.student_profile for r in records}),
    )

    manifest = build_manifest(config)

    return DatasetArtifact(
        records=records,
        statistics=statistics,
        validation=validation,
        metadata=metadata,
        manifest=manifest,
        exports={},
    )


def export_dataset_artifact(artifact: DatasetArtifact, output_dir: str | Path) -> DatasetArtifact:
    """Write `artifact` to disk (CSV, Parquet, JSONL, metadata, manifest).

    Returns a **new** `DatasetArtifact` (immutable, like every model in this
    project) whose `exports` field records the written file paths — the
    artifact returned here is what a caller should hold onto afterward, not
    the one passed in, which still has an empty `exports` dict.
    """

    output_dir = Path(output_dir)
    exported_files = {
        "csv": str(export_csv(artifact.records, output_dir / "dataset.csv")),
        "parquet": str(export_parquet(artifact.records, output_dir / "dataset.parquet")),
        "jsonl": str(export_jsonl(artifact.records, output_dir / "dataset.jsonl")),
        "metadata_json": str(export_metadata_json(artifact.metadata, output_dir / "metadata.json")),
        "manifest_json": str(export_manifest_json(artifact.manifest, output_dir / "manifest.json")),
    }
    return artifact.model_copy(update={"exports": exported_files})
