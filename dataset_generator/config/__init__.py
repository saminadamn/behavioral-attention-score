"""Configuration system: typed, validated settings for the dataset generator.

Everything the generator does — student count, session structure, class
balance, per-state distribution parameters, transition dynamics, curriculum
content, response-generation scoring, and output format — is driven by a
single `GeneratorConfig` object. No module outside this package should
hardcode a generation parameter. Student-profile parameters are *derived*
from multipliers via `config.derive`, not hardcoded (see
`resolve_profile_parameters`).
"""

from dataset_generator.config.attention_state import (
    BEHAVIOURAL_FEATURES,
    AttentionState,
    combine_transition_matrix,
    reachability_violations,
)
from dataset_generator.config.behaviour_generation import (
    BehaviourGenerationConfig,
    default_behaviour_generation_config,
)
from dataset_generator.config.curriculum import (
    CurriculumConfig,
    SubjectDefinition,
    TopicDefinition,
    default_curriculum,
)
from dataset_generator.config.prompt_generation import (
    BLOOM_ORDER,
    CognitiveLevel,
    Difficulty,
    PromptGenerationConfig,
    default_prompt_generation_config,
)
from dataset_generator.config.response_generation import (
    ResponseGenerationConfig,
    default_response_generation_config,
)
from dataset_generator.config.session_simulation import (
    SessionSimulationConfig,
    default_session_simulation_config,
)
from dataset_generator.config.schema import (
    BaseRates,
    DistributionConfig,
    ExperimentMetadata,
    FeatureDistributionParams,
    GeneratorConfig,
    OutputConfig,
    ProfileMultipliers,
    StateDistributionConfig,
    StudentProfileConfig,
    TransitionMatrixConfig,
    VersionMetadata,
)
from dataset_generator.config.defaults import default_config
from dataset_generator.config.derive import ResolvedProfileParams, resolve_profile_parameters
from dataset_generator.config.fingerprint import compute_fingerprint
from dataset_generator.config.loader import load_config, save_config

__all__ = [
    "AttentionState",
    "BEHAVIOURAL_FEATURES",
    "BLOOM_ORDER",
    "BaseRates",
    "BehaviourGenerationConfig",
    "CognitiveLevel",
    "CurriculumConfig",
    "Difficulty",
    "DistributionConfig",
    "ExperimentMetadata",
    "FeatureDistributionParams",
    "GeneratorConfig",
    "OutputConfig",
    "ProfileMultipliers",
    "PromptGenerationConfig",
    "ResolvedProfileParams",
    "ResponseGenerationConfig",
    "SessionSimulationConfig",
    "StateDistributionConfig",
    "StudentProfileConfig",
    "SubjectDefinition",
    "TopicDefinition",
    "TransitionMatrixConfig",
    "VersionMetadata",
    "combine_transition_matrix",
    "compute_fingerprint",
    "default_behaviour_generation_config",
    "default_config",
    "default_curriculum",
    "default_prompt_generation_config",
    "default_response_generation_config",
    "default_session_simulation_config",
    "load_config",
    "reachability_violations",
    "resolve_profile_parameters",
    "save_config",
]
