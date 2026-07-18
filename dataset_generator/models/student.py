"""The `Student` domain model: a persistent behavioural identity.

Per Module 2's design principle, a `Student` holds only long-term,
session-independent characteristics. It must never carry session-specific
state (current BAS, current attention state, current latency, rolling
engagement) — that belongs to the session simulator (Step 6), which reads a
`Student`'s static parameters but does not mutate them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.config.schema import AttentionState


class Student(BaseModel):
    """A synthetic student's persistent behavioural identity.

    `transition_modifier` stores only the *deltas* this student's profile
    applies to the base transition matrix (Stage 2 profile design), not a
    complete matrix — the session simulator is responsible for combining
    `base_matrix + transition_modifier -> effective_matrix` per interaction.
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    profile_name: str
    description: str

    baseline_latency: float = Field(gt=0.0)
    latency_variance: float = Field(gt=0.0)

    engagement_tendency: float = Field(ge=0.0, le=1.0)

    fatigue_rate: float = Field(ge=0.0, le=1.0)

    intervention_sensitivity: float = Field(ge=0.0, le=2.0)

    transition_modifier: dict[AttentionState, dict[AttentionState, float]]

    profile_seed: int

    def descriptor(self) -> dict[str, str]:
        """A small, dataset-exportable summary (id / profile / description).

        Not consumed by any model — intended for debugging, visualization,
        and explaining the simulator's population in the paper.
        """

        return {
            "student_id": self.student_id,
            "profile": self.profile_name,
            "description": self.description,
        }
