"""Session-simulation settings (Module 6).

Deliberately minimal: session length range already lives in
`GeneratorConfig.interactions_per_session`, the rolling-window size in
`GeneratorConfig.rolling_window`, and the intervention-boost strength is
reused from `BehaviourGenerationConfig.intervention_recovery_weight` (see
Module 6's design-decision summary — one "how strongly does this student
respond to interventions" coefficient, applied in two domains, rather than
two near-duplicate config knobs). The only genuinely new policy this module
needs is *when* to trigger an intervention in the first place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SessionSimulationConfig(BaseModel):
    """Intervention-triggering policy for `SessionSimulator` (Module 6, Step 6)."""

    model_config = ConfigDict(frozen=True)

    intervention_engagement_threshold: float = Field(ge=0.0, le=1.0, default=0.7)


def default_session_simulation_config() -> SessionSimulationConfig:
    return SessionSimulationConfig(intervention_engagement_threshold=0.7)
