"""Module 4, Step 3: Response Strategies.

One class per attention state, registered with `ResponseStrategyFactory` —
the same decorator-registry pattern as `profiles.ProfileFactory` — instead
of an `if attention_state == ...` chain in the generator. Each strategy is a
pure data-plus-lookup object: it declares the behavioural traits Step 3
asks for (completeness, reasoning depth, detail level, topic adherence,
error tendency, length multiplier) and delegates actual text assembly to
`response_templates.compose` for its own attention state. `ResponseGenerator`
supplies the RNG and the keyword/topic to substitute — it is the only place
randomness enters, and the only place that decides *which* keyword a given
interaction uses (needed downstream for semantic-similarity scoring).
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

import numpy as np

from dataset_generator.config.attention_state import AttentionState
from dataset_generator.generators.response_templates import compose


class ResponseStrategy(ABC):
    """Behavioural traits for one attention state's response style.

    Traits are all in [0, 1] except `length_multiplier` (a scale factor on
    the prompt's expected response length). `error_tendency` is an
    additional nudge on top of `ResponseGenerationConfig`'s
    `attention_state_correctness_modifier` — the config holds the primary,
    tunable population-level effect; the strategy's `error_tendency`
    captures the qualitative trait ("this style tends to err") the config
    modifier doesn't distinguish from a strategy's other characteristics.
    """

    attention_state: ClassVar[AttentionState]
    completeness: ClassVar[float]
    reasoning_depth: ClassVar[float]
    detail_level: ClassVar[float]
    topic_adherence: ClassVar[float]
    error_tendency: ClassVar[float]
    length_multiplier: ClassVar[float]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        required = (
            "attention_state", "completeness", "reasoning_depth", "detail_level",
            "topic_adherence", "error_tendency", "length_multiplier",
        )
        missing = [name for name in required if not hasattr(cls, name)]
        if missing:
            raise TypeError(f"{cls.__name__} must define: {missing}")
        for trait in ("completeness", "reasoning_depth", "detail_level", "topic_adherence", "error_tendency"):
            value = getattr(cls, trait)
            if not (0.0 <= value <= 1.0):
                raise TypeError(f"{cls.__name__}.{trait} must be in [0, 1] (got {value})")
        if cls.length_multiplier <= 0:
            raise TypeError(f"{cls.__name__}.length_multiplier must be > 0")

    def generate_text(
        self, rng: np.random.Generator, correct: bool, keyword: str, topic: str
    ) -> str:
        """Compose one response string for this strategy's attention state."""

        return compose(rng, self.attention_state, correct, keyword, topic)

    def target_length(self, expected_response_length: int) -> int:
        """Scale the prompt's expected response length by this strategy's trait."""

        return max(1, round(expected_response_length * self.length_multiplier))


class ResponseStrategyFactory:
    """Registry mapping an `AttentionState` to its `ResponseStrategy` implementation."""

    _registry: ClassVar[dict[AttentionState, type[ResponseStrategy]]] = {}

    @classmethod
    def register(cls, strategy_cls: type[ResponseStrategy]) -> type[ResponseStrategy]:
        if strategy_cls.attention_state in cls._registry:
            raise ValueError(f"strategy for {strategy_cls.attention_state!r} already registered")
        cls._registry[strategy_cls.attention_state] = strategy_cls
        return strategy_cls

    @classmethod
    def for_state(cls, attention_state: AttentionState) -> ResponseStrategy:
        if attention_state not in cls._registry:
            raise KeyError(f"no strategy registered for {attention_state!r}")
        return cls._registry[attention_state]()

    @classmethod
    def available_states(cls) -> list[AttentionState]:
        return sorted(cls._registry, key=lambda s: s.value)


@ResponseStrategyFactory.register
class FocusedStrategy(ResponseStrategy):
    """Complete, reasoned, on-topic answers; rarely errs."""

    attention_state = AttentionState.FOCUSED
    completeness = 0.90
    reasoning_depth = 0.90
    detail_level = 0.85
    topic_adherence = 0.90
    error_tendency = 0.05
    length_multiplier = 1.20


@ResponseStrategyFactory.register
class DistractedStrategy(ResponseStrategy):
    """Incomplete, vague, prone to drifting off-topic."""

    attention_state = AttentionState.DISTRACTED
    completeness = 0.40
    reasoning_depth = 0.30
    detail_level = 0.30
    topic_adherence = 0.40
    error_tendency = 0.35
    length_multiplier = 0.60


@ResponseStrategyFactory.register
class ImpulsiveStrategy(ResponseStrategy):
    """Very short, rushed, keyword-matched with little reasoning."""

    attention_state = AttentionState.IMPULSIVE
    completeness = 0.30
    reasoning_depth = 0.15
    detail_level = 0.20
    topic_adherence = 0.60
    error_tendency = 0.20
    length_multiplier = 0.35
