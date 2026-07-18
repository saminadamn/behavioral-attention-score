"""Student archetypes as independent classes, assembled via `ProfileFactory`.

Design rationale (per Module 2 spec): a chain of `if profile_key == ...`
does not scale as new archetypes are added and is easy to get subtly wrong
(a forgotten branch silently falls through). Instead each archetype is a
small class registered with `ProfileFactory`; adding a sixth profile means
adding one `@ProfileFactory.register`-decorated class, nothing else.

All five classes share one `BaseProfile.generate_student` implementation —
the only thing that differs between archetypes is *which* config key they
resolve (`profile_key`) and their descriptive metadata. The actual numeric
behaviour (multipliers, ranges) lives entirely in `GeneratorConfig`
(Step 1) via `resolve_profile_parameters`; nothing here hardcodes a
behavioural value.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from dataset_generator.config import GeneratorConfig, resolve_profile_parameters
from dataset_generator.models.student import Student
from dataset_generator.utils.rng import student_local_rng, student_local_seed


class BaseProfile(ABC):
    """Shared student-generation logic for one archetype.

    Subclasses declare only `profile_key` (must match a key in
    `GeneratorConfig.profiles`), `display_name`, and `description`
    (exported via `Student.descriptor()` for debugging/visualization/paper
    use). `__init_subclass__` fails fast if any of these is left unset.
    """

    profile_key: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.profile_key or not cls.display_name or not cls.description:
            raise TypeError(
                f"{cls.__name__} must define non-empty profile_key, display_name, "
                "and description"
            )

    def generate_student(self, student_index: int, config: GeneratorConfig) -> Student:
        """Sample one unique `Student` of this archetype.

        Uses a per-student-index RNG (`student_local_rng`) rather than a
        shared advancing stream, so this student's parameters are
        reproducible in isolation (Step 1's determinism guarantee) and never
        collide with another student's draws regardless of generation
        order — "never generate identical students" holds by construction:
        each index gets its own independent random stream.
        """

        if self.profile_key not in config.profiles:
            raise KeyError(
                f"{self.profile_key!r} is not defined in config.profiles "
                f"(known: {sorted(config.profiles)})"
            )

        resolved = resolve_profile_parameters(config, self.profile_key)
        profile_cfg = config.profiles[self.profile_key]
        rng = student_local_rng(config.seed, student_index)

        baseline_latency = float(rng.uniform(*resolved.baseline_latency_mean_range))
        latency_variance = float(rng.uniform(*resolved.baseline_latency_std_range))
        fatigue_rate = float(rng.uniform(*resolved.fatigue_rate_range))
        intervention_sensitivity = float(rng.uniform(*resolved.intervention_sensitivity_range))

        # Personality noise: engagement_tendency has no pre-derived range
        # (Step 1's resolver returns a single archetype-level value), so a
        # small per-student multiplicative jitter — bounded by the same
        # `param_spread` used for every other resolved range — gives each
        # student individuality without a second hardcoded constant.
        jitter = rng.uniform(1.0 - profile_cfg.param_spread, 1.0 + profile_cfg.param_spread)
        engagement_tendency = min(1.0, max(0.0, resolved.engagement_tendency * jitter))

        return Student(
            student_id=f"S{student_index + 1:05d}",
            profile_name=self.profile_key,
            description=self.description,
            baseline_latency=baseline_latency,
            latency_variance=latency_variance,
            engagement_tendency=engagement_tendency,
            fatigue_rate=fatigue_rate,
            intervention_sensitivity=intervention_sensitivity,
            transition_modifier=profile_cfg.transition_modifiers,
            profile_seed=student_local_seed(config.seed, student_index),
        )


class ProfileFactory:
    """Registry mapping a config profile key to its `BaseProfile` implementation."""

    _registry: ClassVar[dict[str, type[BaseProfile]]] = {}

    @classmethod
    def register(cls, profile_cls: type[BaseProfile]) -> type[BaseProfile]:
        """Class decorator: register `profile_cls` under its `profile_key`."""

        if profile_cls.profile_key in cls._registry:
            raise ValueError(f"profile_key {profile_cls.profile_key!r} already registered")
        cls._registry[profile_cls.profile_key] = profile_cls
        return profile_cls

    @classmethod
    def create(cls, profile_key: str) -> BaseProfile:
        """Instantiate the profile class registered for `profile_key`."""

        if profile_key not in cls._registry:
            raise KeyError(
                f"no profile registered for {profile_key!r}; known: {sorted(cls._registry)}"
            )
        return cls._registry[profile_key]()

    @classmethod
    def available_profiles(cls) -> list[str]:
        return sorted(cls._registry)


@ProfileFactory.register
class FocusedProfile(BaseProfile):
    profile_key = "Consistently_Focused"
    display_name = "Consistently Focused"
    description = (
        "High engagement and stable latency with low fatigue; intervention is "
        "rarely needed."
    )


@ProfileFactory.register
class FatiguedProfile(BaseProfile):
    profile_key = "Gradually_Fatigued"
    display_name = "Gradually Fatigued"
    description = (
        "Starts near Focused-baseline behaviour. This profile only supplies a "
        "fatigue_rate; the session simulator applies it over session duration, "
        "increasing latency and lowering engagement as the session progresses."
    )


@ProfileFactory.register
class DistractibleProfile(BaseProfile):
    profile_key = "Highly_Distractible"
    display_name = "Highly Distractible"
    description = (
        "High topic shift and inconsistent latency with weak semantic similarity; "
        "tends to remain in the Distracted state."
    )


@ProfileFactory.register
class ImpulsiveProfile(BaseProfile):
    profile_key = "Highly_Impulsive"
    display_name = "Highly Impulsive"
    description = (
        "Extremely low latency and short responses with low hesitation; "
        "frequently answers before full deliberation."
    )


@ProfileFactory.register
class RecoveringProfile(BaseProfile):
    profile_key = "Recovering_Learner"
    display_name = "Recovering Learner"
    description = (
        "Average baseline behaviour; engagement improves and distraction "
        "decreases markedly following interventions."
    )
