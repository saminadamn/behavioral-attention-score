"""Module 9, Step 11: Serialization.

`BASArtifact` is pure Pydantic (no non-serializable payloads, unlike Module
8's `TrainingArtifact`), so it round-trips through plain JSON directly —
no joblib bundle needed here.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset_generator.bas.models import BASArtifact

_ARTIFACT_FILENAME = "bas_artifact.json"


def save_bas_artifact(artifact: BASArtifact, directory: str | Path) -> Path:
    """Persist `artifact` as a single JSON file."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / _ARTIFACT_FILENAME
    file_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return file_path


def load_bas_artifact(directory: str | Path) -> BASArtifact:
    """Load a `BASArtifact` previously saved by `save_bas_artifact`."""

    directory = Path(directory)
    file_path = directory / _ARTIFACT_FILENAME
    return BASArtifact.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
