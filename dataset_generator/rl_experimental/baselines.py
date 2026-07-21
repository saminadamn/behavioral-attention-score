"""Baseline intervention-decision policies for a **live, causal** comparison
— these actually drive `SessionSimulator` (via the `intervention_policy` hook
added to `generate_sessions`/`build_session_simulator`/`SessionSimulator`),
so the reward/BAS each one produces reflects a real regenerated trajectory,
not a static number reused across "policies."

This distinction matters: the synthetic generator's own ground-truth
`intervention_applied` flag (Module 6's `_decide_intervention`) is entirely
separate from `InterventionPlanner` (Module 11) — Module 11's decisions are
computed *after* a dataset already exists and never feed back into it. That
means a report claiming "reward under Random Policy" without actually
re-simulating under that policy would be comparing apples that were never
picked. The three policies below make that comparison honest by actually
controlling generation:

- `rule_based_policy` = `None` — the simulator's own built-in heuristic
  (rolling-engagement threshold + configured probability), i.e. "no policy
  override" — included as the reference row, not a new mechanism.
- `no_intervention_policy` — always `False`.
- `make_random_policy(rng)` — a Bernoulli draw at `intervention_probability`
  each decision (same base rate as the reference policy, so the comparison
  isolates *when* interventions happen, not *how often*).

The trained offline-RL agents (DQN/CQL/IQL/BCQ) are deliberately NOT wired
in here: they were trained on `InterventionObservation` features computed
*after* an interaction (current BAS, current reward), which aren't
available at this pre-interaction decision point without redefining their
state space — live rollout of the trained agents is listed as future work
in `docs/OFFLINE_RL.md`, not approximated here.
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config import GeneratorConfig
from dataset_generator.generators.session_simulator import InterventionDecisionInput, InterventionPolicy

# `None` means "use SessionSimulator's own built-in heuristic" — the
# reference policy every comparison in docs/BASELINE_COMPARISON.md calls
# "Rule-Based (generator heuristic)". Named here so call sites read clearly
# instead of passing a bare `None`.
rule_based_policy: InterventionPolicy | None = None


def no_intervention_policy(_: InterventionDecisionInput) -> bool:
    """Never intervenes — the "do nothing" floor every comparison needs."""

    return False


def make_random_policy(config: GeneratorConfig, seed: int) -> InterventionPolicy:
    """A Bernoulli policy at `config.intervention_probability` — the same
    base rate the reference heuristic uses, so this isolates *timing*
    (random vs. engagement-triggered) rather than *frequency*.
    """

    rng = np.random.default_rng(seed)
    probability = config.intervention_probability

    def _policy(_: InterventionDecisionInput) -> bool:
        return bool(rng.random() < probability)

    return _policy
