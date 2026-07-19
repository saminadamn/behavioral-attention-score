"""Module 10, Step 9: Serialization.

`RewardArtifact` is pure Pydantic — round-trips through plain JSON directly,
same as `BASArtifact` (Module 9), no joblib bundle needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset_generator.reward.models import RewardArtifact

_ARTIFACT_FILENAME = "reward_artifact.json"


def save_reward_artifact(artifact: RewardArtifact, directory: str | Path) -> Path:
    """Persist `artifact` as a single JSON file."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / _ARTIFACT_FILENAME
    file_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return file_path


def load_reward_artifact(directory: str | Path) -> RewardArtifact:
    """Load a `RewardArtifact` previously saved by `save_reward_artifact`."""

    directory = Path(directory)
    file_path = directory / _ARTIFACT_FILENAME
    return RewardArtifact.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
