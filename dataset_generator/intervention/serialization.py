"""Module 11, Step 11: Serialization.

`InterventionArtifact` is pure Pydantic — round-trips through plain JSON
directly, same as `BASArtifact`/`RewardArtifact`, no joblib bundle needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset_generator.intervention.models import InterventionArtifact

_ARTIFACT_FILENAME = "intervention_artifact.json"


def save_intervention_artifact(artifact: InterventionArtifact, directory: str | Path) -> Path:
    """Persist `artifact` as a single JSON file."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / _ARTIFACT_FILENAME
    file_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return file_path


def load_intervention_artifact(directory: str | Path) -> InterventionArtifact:
    """Load an `InterventionArtifact` previously saved by `save_intervention_artifact`."""

    directory = Path(directory)
    file_path = directory / _ARTIFACT_FILENAME
    return InterventionArtifact.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
