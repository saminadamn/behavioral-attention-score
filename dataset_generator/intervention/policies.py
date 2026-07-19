"""Module 11, Step 5: Intervention Policies.

Each policy is independent: it knows nothing about the other policies, and
nothing about cooldown, ranking, or planning — those concerns live in
`scorer.py`/`cooldown.py`/`planner.py`. Gain/cost estimates are explicitly
heuristic (scaled by how severe the triggering evidence is), not predictions
from a counterfactual simulation, since no such simulation exists.

`score()` is shared (not reimplemented per policy) so the same
gain/cost/confidence/severity formula, driven by `config.scoring_weights`,
is used everywhere — avoiding eight copies of the same arithmetic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dataset_generator.intervention.config import InterventionConfig
from dataset_generator.intervention.detector import NeedDetectionResult
from dataset_generator.intervention.models import InterventionObservation


class InterventionPolicy(ABC):
    """Common interface every intervention strategy implements.

    Mirrors the `ResponseStrategy`/`ClassifierModel` factory-registered
    pattern used elsewhere in this project. Policies never reference each
    other or the planner.
    """

    name: str = "InterventionPolicy"

    def __init__(self, config: InterventionConfig) -> None:
        self._config = config

    @property
    def config(self) -> InterventionConfig:
        return self._config

    @abstractmethod
    def eligible(self, observation: InterventionObservation) -> bool:
        """Whether this policy is a candidate at all for this observation."""

    @abstractmethod
    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        """Heuristic expected BAS improvement in [0, 1] if this policy is applied."""

    @abstractmethod
    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        """Heuristic expected reward improvement in [0, 1] if this policy is applied."""

    @abstractmethod
    def estimated_cost(self, observation: InterventionObservation) -> float:
        """Heuristic cost in [0, 1] (time/engagement/momentum spent applying this policy)."""

    @abstractmethod
    def generate_reason(self, observation: InterventionObservation) -> str:
        """A short, human-readable justification for recommending this policy."""

    def score(
        self, observation: InterventionObservation, need_result: NeedDetectionResult
    ) -> float:
        """Combine gain/cost/confidence/severity via `config.scoring_weights`.

        Shared across all policies so ranking is comparable and consistent;
        only the four estimate_* methods vary per policy.
        """

        weights = self._config.scoring_weights
        bas_gain = self.estimate_bas_gain(observation)
        reward_gain = self.estimate_reward_gain(observation)
        cost = self.estimated_cost(observation)
        expected_gain = (bas_gain + reward_gain) / 2.0
        confidence = (observation.bas_confidence + observation.reward_confidence) / 2.0

        raw_score = (
            weights.expected_gain_weight * expected_gain
            - weights.cost_weight * cost
            + weights.confidence_weight * confidence
            + weights.severity_weight * need_result.need_score
        )
        policy_weight = self._config.policy_weight(self.name)
        cost_multiplier = self._config.policy_cost_multiplier(self.name)
        return policy_weight * (raw_score - weights.cost_weight * cost * (cost_multiplier - 1.0))


class InterventionPolicyFactory:
    """Decorator-registry factory for `InterventionPolicy` subclasses."""

    _registry: dict[str, type[InterventionPolicy]] = {}

    @classmethod
    def register(cls, policy_cls: type[InterventionPolicy]) -> type[InterventionPolicy]:
        cls._registry[policy_cls.name] = policy_cls
        return policy_cls

    @classmethod
    def create_all(cls, config: InterventionConfig) -> list[InterventionPolicy]:
        """Instantiate every registered policy, wired with `config` (dependency injection)."""

        return [policy_cls(config) for policy_cls in cls._registry.values()]

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._registry.keys())


@InterventionPolicyFactory.register
class NoInterventionPolicy(InterventionPolicy):
    """Always eligible; the zero-gain, zero-cost baseline fallback."""

    name = "NoInterventionPolicy"

    def eligible(self, observation: InterventionObservation) -> bool:
        return True

    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        return 0.0

    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        return 0.0

    def estimated_cost(self, observation: InterventionObservation) -> float:
        return 0.0

    def generate_reason(self, observation: InterventionObservation) -> str:
        return "No signal warrants an intervention at this interaction."


@InterventionPolicyFactory.register
class HintPolicy(InterventionPolicy):
    """Struggling but still trying: low correctness, adequate engagement."""

    name = "HintPolicy"

    def eligible(self, observation: InterventionObservation) -> bool:
        return (
            observation.correctness < self._config.min_correctness
            and observation.engagement >= self._config.min_engagement
        )

    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        severity = self._config.min_correctness - observation.correctness
        return min(1.0, max(0.0, severity) * 1.5)

    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        return self.estimate_bas_gain(observation) * 0.8

    def estimated_cost(self, observation: InterventionObservation) -> float:
        return 0.15

    def generate_reason(self, observation: InterventionObservation) -> str:
        return (
            f"Correctness ({observation.correctness:.2f}) is below the "
            f"({self._config.min_correctness:.2f}) threshold while engagement remains "
            "adequate — a hint is likely to help without disrupting flow."
        )


@InterventionPolicyFactory.register
class ConceptReviewPolicy(InterventionPolicy):
    """Persistent, deeper misunderstanding: repeated declines plus low correctness."""

    name = "ConceptReviewPolicy"

    def eligible(self, observation: InterventionObservation) -> bool:
        return (
            observation.consecutive_decline_count >= self._config.consecutive_decline_threshold
            and observation.correctness < self._config.min_correctness
        )

    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        decline_factor = min(
            1.0, observation.consecutive_decline_count / (2 * self._config.consecutive_decline_threshold)
        )
        return min(1.0, 0.5 + 0.5 * decline_factor)

    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        return self.estimate_bas_gain(observation) * 0.9

    def estimated_cost(self, observation: InterventionObservation) -> float:
        return 0.4

    def generate_reason(self, observation: InterventionObservation) -> str:
        return (
            f"{observation.consecutive_decline_count} consecutive BAS declines with low "
            "correctness suggest a deeper conceptual gap — a full concept review is warranted."
        )


@InterventionPolicyFactory.register
class DifficultyReductionPolicy(InterventionPolicy):
    """Content is too hard: low correctness on a high-difficulty prompt."""

    name = "DifficultyReductionPolicy"

    def eligible(self, observation: InterventionObservation) -> bool:
        return (
            observation.correctness < self._config.min_correctness
            and observation.prompt_difficulty_score > self._config.min_difficulty_for_reduction
        )

    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        excess = observation.prompt_difficulty_score - self._config.min_difficulty_for_reduction
        return min(1.0, 0.4 + excess)

    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        return self.estimate_bas_gain(observation) * 0.7

    def estimated_cost(self, observation: InterventionObservation) -> float:
        return 0.25

    def generate_reason(self, observation: InterventionObservation) -> str:
        return (
            f"Prompt difficulty ({observation.prompt_difficulty_score:.2f}) exceeds the "
            f"({self._config.min_difficulty_for_reduction:.2f}) threshold alongside low correctness "
            "— reducing difficulty should restore traction."
        )


@InterventionPolicyFactory.register
class MotivationalPromptPolicy(InterventionPolicy):
    """Disengaged but capable: low engagement, correctness still fine."""

    name = "MotivationalPromptPolicy"

    def eligible(self, observation: InterventionObservation) -> bool:
        return (
            observation.engagement < self._config.min_engagement
            and observation.correctness >= self._config.min_correctness
        )

    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        gap = self._config.min_engagement - observation.engagement
        return min(1.0, max(0.0, gap) * 1.2)

    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        return self.estimate_bas_gain(observation) * 0.6

    def estimated_cost(self, observation: InterventionObservation) -> float:
        return 0.1

    def generate_reason(self, observation: InterventionObservation) -> str:
        return (
            f"Engagement ({observation.engagement:.2f}) is low despite adequate correctness — "
            "a motivational prompt should re-engage without addressing content the student "
            "already understands."
        )


@InterventionPolicyFactory.register
class BreakRecommendationPolicy(InterventionPolicy):
    """Fatigue is the dominant signal; highest cost since it removes learning time."""

    name = "BreakRecommendationPolicy"

    def eligible(self, observation: InterventionObservation) -> bool:
        return observation.fatigue > self._config.max_fatigue

    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        excess = observation.fatigue - self._config.max_fatigue
        return min(1.0, 0.3 + excess)

    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        return self.estimate_bas_gain(observation) * 0.5

    def estimated_cost(self, observation: InterventionObservation) -> float:
        return 0.6

    def generate_reason(self, observation: InterventionObservation) -> str:
        return (
            f"Fatigue ({observation.fatigue:.2f}) exceeds the ({self._config.max_fatigue:.2f}) "
            "threshold — a short break is likely to be more effective than continued content."
        )


@InterventionPolicyFactory.register
class EncouragementPolicy(InterventionPolicy):
    """Unsure but doing fine: low self-reported confidence, correctness fine. Lowest cost."""

    name = "EncouragementPolicy"

    def eligible(self, observation: InterventionObservation) -> bool:
        return (
            observation.confidence < self._config.min_confidence
            and observation.correctness >= self._config.min_correctness
        )

    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        gap = self._config.min_confidence - observation.confidence
        return min(1.0, max(0.0, gap) * 1.0)

    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        return self.estimate_bas_gain(observation) * 0.4

    def estimated_cost(self, observation: InterventionObservation) -> float:
        return 0.05

    def generate_reason(self, observation: InterventionObservation) -> str:
        return (
            f"Confidence ({observation.confidence:.2f}) is low despite correctness being "
            "adequate — brief encouragement should close the confidence gap at minimal cost."
        )


@InterventionPolicyFactory.register
class QuestionReframingPolicy(InterventionPolicy):
    """Off-topic or misunderstanding the question itself: low semantic similarity."""

    name = "QuestionReframingPolicy"

    def eligible(self, observation: InterventionObservation) -> bool:
        return observation.semantic_similarity < self._config.min_semantic_similarity

    def estimate_bas_gain(self, observation: InterventionObservation) -> float:
        gap = self._config.min_semantic_similarity - observation.semantic_similarity
        return min(1.0, max(0.0, gap) * 1.3)

    def estimate_reward_gain(self, observation: InterventionObservation) -> float:
        return self.estimate_bas_gain(observation) * 0.75

    def estimated_cost(self, observation: InterventionObservation) -> float:
        return 0.2

    def generate_reason(self, observation: InterventionObservation) -> str:
        return (
            f"Semantic similarity ({observation.semantic_similarity:.2f}) is below the "
            f"({self._config.min_semantic_similarity:.2f}) threshold — the response suggests "
            "the question itself was misunderstood, so reframing it should help."
        )
