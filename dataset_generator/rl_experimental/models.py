"""Output artifact for the experimental DQN — same immutable, fingerprinted
artifact convention as every other module, so it can be compared and
reported on identically. Field names avoid implying validated performance:
`greedy_policy_agreement_rate` describes overlap with the logged policy,
not correctness.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DQNTrainingArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str

    transitions_used: int = Field(ge=0)
    epochs_run: int = Field(ge=0)
    batches_trained: int = Field(ge=0)
    sequence_length: int = Field(gt=0)

    loss_per_epoch: list[float]

    mean_q_value: float
    mean_abs_td_error: float = Field(ge=0.0)
    action_distribution: dict[str, float]

    greedy_policy_agreement_rate: float = Field(ge=0.0, le=1.0)

    config_fingerprint: str
    schema_version: str
    generation_timestamp: str

    disclaimer: str = (
        "Trained offline on logged transitions from the rule-based "
        "InterventionPlanner and a synthetic reward. Not validated against "
        "any real outcome. Not used by the default pipeline."
    )
