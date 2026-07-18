"""Module 7: dataset assembly, validation, statistics, export, and reporting.

`DatasetArtifact` (see `models/dataset.py`) is the single source of truth
this package produces — build one with `build_dataset_artifact`, export it
with `export_dataset_artifact`, and inspect it with `render_dataset_report`.
"""

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
from dataset_generator.pipeline.dataset_report import render_dataset_report
from dataset_generator.pipeline.dataset_statistics import compute_dataset_statistics
from dataset_generator.pipeline.feature_registry import FEATURE_DEFINITIONS, FeatureRegistry

__all__ = [
    "FEATURE_DEFINITIONS",
    "DatasetBuilder",
    "FeatureRegistry",
    "build_dataset_artifact",
    "build_manifest",
    "compute_dataset_statistics",
    "export_csv",
    "export_dataset_artifact",
    "export_jsonl",
    "export_manifest_json",
    "export_metadata_json",
    "export_parquet",
    "render_dataset_report",
]
