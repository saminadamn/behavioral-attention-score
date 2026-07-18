"""Module 7, Step 6: Export Engine.

Every export writes rows in the same deterministic order — sorted by
`(session_id, interaction_number)`, with columns in `DatasetRecord`'s field
definition order — so re-exporting the same records always produces
byte-identical CSV/JSONL and value-identical Parquet, regardless of the
order `records` happened to be passed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataset_generator.models.dataset import DatasetManifest, DatasetMetadata, DatasetRecord
from dataset_generator.validators.dataset_validator import records_to_frame


def _deterministic_frame(records: list[DatasetRecord]) -> pd.DataFrame:
    df = records_to_frame(records)
    if df.empty:
        return df
    df = df.sort_values(["session_id", "interaction_number"]).reset_index(drop=True)
    return df[list(DatasetRecord.model_fields)]


def export_csv(records: list[DatasetRecord], path: str | Path) -> Path:
    """Export `records` as CSV, deterministically ordered."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _deterministic_frame(records).to_csv(file_path, index=False)
    return file_path


def export_parquet(records: list[DatasetRecord], path: str | Path) -> Path:
    """Export `records` as Parquet, deterministically ordered."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _deterministic_frame(records).to_parquet(file_path, index=False)
    return file_path


def export_jsonl(records: list[DatasetRecord], path: str | Path) -> Path:
    """Export `records` as newline-delimited JSON, deterministically ordered."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df = _deterministic_frame(records)
    with file_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    return file_path


def export_metadata_json(metadata: DatasetMetadata, path: str | Path) -> Path:
    """Export `DatasetMetadata` as JSON."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return file_path


def export_manifest_json(manifest: DatasetManifest, path: str | Path) -> Path:
    """Export `DatasetManifest` as JSON."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    return file_path
