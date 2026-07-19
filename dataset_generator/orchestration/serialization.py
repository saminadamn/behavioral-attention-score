"""Module 12, Step 9: Serialization.

Distinct from `checkpoint.py` (Step 8): the checkpointer persists state for
LangGraph's own pause/resume machinery, keyed by `thread_id`, generally
in-process. This module saves/loads a `WorkflowState` as one portable JSON
file — for archival, cross-process transfer, or handing a run's final
state to another tool — independent of any checkpointer or thread.

Each wrapped artifact (`DatasetArtifact`/`BASArtifact`/`RewardArtifact`/
`InterventionArtifact`) already carries its own `config_fingerprint`
(computed by Modules 7/9/10/11) — this module surfaces those fingerprints
in one place rather than recomputing anything.

Caveat: a loaded `WorkflowState` is a plain value, not a resumable graph
position. Passing it straight to `compiled.invoke(loaded_state)` on a
non-checkpointed graph restarts from `START` (`LoadDatasetNode` always
resets the interaction/session cursors as the entry point), which would
re-walk Phase 2 and double the `_append`-reducer fields on top of what was
loaded. To resume mid-run, use `checkpoint.py`'s thread-based flow (or
`recover_failed_session` + a fresh `thread_id`) instead — this module is
for archival/inspection/cross-process handoff of the value itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_generator.bas.models import BASArtifact
from dataset_generator.intervention.models import InterventionArtifact
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.reward.models import RewardArtifact

from dataset_generator.orchestration.state import WorkflowState

ORCHESTRATION_SCHEMA_VERSION = "1.0"
_STATE_FILENAME = "workflow_state.json"

_PASSTHROUGH_FIELDS = (
    "session_ids",
    "current_session_index",
    "current_session_id",
    "current_student_id",
    "current_interaction_index",
    "max_interactions_per_session",
    "tutor_actions",
    "session_outputs",
    "execution_metadata",
    "errors",
    "execution_history",
    "timing_stats",
)


def config_fingerprints(state: WorkflowState) -> dict[str, str | None]:
    """The config fingerprint each wrapped artifact already carries, in one place."""

    dataset_artifact = state.get("dataset_artifact")
    bas_artifact = state.get("bas_artifact")
    reward_artifact = state.get("reward_artifact")
    intervention_artifact = state.get("intervention_artifact")

    return {
        "dataset": dataset_artifact.manifest.config_fingerprint if dataset_artifact else None,
        "bas": bas_artifact.config_fingerprint if bas_artifact else None,
        "reward": reward_artifact.config_fingerprint if reward_artifact else None,
        "intervention": intervention_artifact.config_fingerprint if intervention_artifact else None,
    }


def save_workflow_state(state: WorkflowState, directory: str | Path) -> Path:
    """Persist `state` as one portable JSON file under `directory`."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": ORCHESTRATION_SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprints": config_fingerprints(state),
        "dataset_artifact": _dump(state.get("dataset_artifact")),
        "bas_artifact": _dump(state.get("bas_artifact")),
        "reward_artifact": _dump(state.get("reward_artifact")),
        "intervention_artifact": _dump(state.get("intervention_artifact")),
    }
    for field in _PASSTHROUGH_FIELDS:
        payload[field] = state.get(field)

    file_path = directory / _STATE_FILENAME
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return file_path


def load_workflow_state(directory: str | Path) -> WorkflowState:
    """Load a `WorkflowState` previously saved by `save_workflow_state`."""

    directory = Path(directory)
    payload = json.loads((directory / _STATE_FILENAME).read_text(encoding="utf-8"))

    state: dict[str, Any] = {}
    if payload.get("dataset_artifact") is not None:
        state["dataset_artifact"] = DatasetArtifact.model_validate(payload["dataset_artifact"])
    if payload.get("bas_artifact") is not None:
        state["bas_artifact"] = BASArtifact.model_validate(payload["bas_artifact"])
    if payload.get("reward_artifact") is not None:
        state["reward_artifact"] = RewardArtifact.model_validate(payload["reward_artifact"])
    if payload.get("intervention_artifact") is not None:
        state["intervention_artifact"] = InterventionArtifact.model_validate(
            payload["intervention_artifact"]
        )

    for field in _PASSTHROUGH_FIELDS:
        if field in payload:
            state[field] = payload[field]

    return state  # type: ignore[return-value]


def _dump(artifact: Any) -> dict[str, Any] | None:
    return artifact.model_dump(mode="json") if artifact is not None else None
