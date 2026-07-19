"""Module 11, Step 6: Policy Scoring.

Evaluates every eligible policy for one observation and returns ranked
`InterventionCandidate`s. Only eligible policies become candidates (ineligible
policies are simply skipped, not represented as zero-score candidates),
except `NoInterventionPolicy`, which is always eligible by construction and
so always appears as the baseline.
"""

from __future__ import annotations

from dataset_generator.intervention.detector import NeedDetectionResult
from dataset_generator.intervention.models import InterventionCandidate, InterventionObservation
from dataset_generator.intervention.policies import InterventionPolicy


class PolicyScorer:
    """Evaluates every eligible policy and returns candidates ranked by score."""

    def evaluate_all(
        self,
        observation: InterventionObservation,
        policies: list[InterventionPolicy],
        need_result: NeedDetectionResult,
    ) -> list[InterventionCandidate]:
        """Return eligible policies as `InterventionCandidate`s, sorted by score descending."""

        candidates: list[InterventionCandidate] = []
        for policy in policies:
            if not policy.eligible(observation):
                continue

            bas_gain = policy.estimate_bas_gain(observation)
            reward_gain = policy.estimate_reward_gain(observation)
            cost = policy.estimated_cost(observation) * policy.config.policy_cost_multiplier(
                policy.name
            )
            score = policy.score(observation, need_result)

            candidates.append(
                InterventionCandidate(
                    policy_name=policy.name,
                    eligible=True,
                    estimated_bas_gain=bas_gain,
                    estimated_reward_gain=reward_gain,
                    estimated_cost=cost,
                    score=score,
                    reason=policy.generate_reason(observation),
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
