"""Module 10: Reward Model.

Computes a deterministic, signed reward signal from `DatasetArtifact` +
`BASArtifact` trajectories — never regenerating behaviour, never
recomputing BAS. This is *not* a reinforcement-learning algorithm; it's the
credit-assignment preprocessing a future RL-based intervention engine would
consume. `RewardArtifact` is this package's single source of truth. Entry
point: `RewardEngine(config).compute(dataset_artifact, bas_artifact)`.
"""

from dataset_generator.reward.aggregator import (
    RewardAggregator,
    RewardEngine,
    build_reward_session_summary,
    compute_reward_config_fingerprint,
    decompose_reward,
)
from dataset_generator.reward.config import (
    RewardCategory,
    RewardConfig,
    RewardSignalConfig,
    RewardSignalPolarity,
    TemporalMode,
    default_reward_config,
)
from dataset_generator.reward.confidence import compute_reward_confidence
from dataset_generator.reward.models import (
    RewardArtifact,
    RewardContribution,
    RewardObservation,
    RewardRecord,
    RewardRecordMetadata,
    RewardSessionSummary,
    RewardStatistics,
)
from dataset_generator.reward.report import build_json_report, render_markdown_report, reward_trend_metadata
from dataset_generator.reward.serialization import load_reward_artifact, save_reward_artifact
from dataset_generator.reward.signals import RewardSignalExtractor
from dataset_generator.reward.temporal import apply_temporal_credit_assignment

__all__ = [
    "RewardAggregator",
    "RewardArtifact",
    "RewardCategory",
    "RewardConfig",
    "RewardContribution",
    "RewardEngine",
    "RewardObservation",
    "RewardRecord",
    "RewardRecordMetadata",
    "RewardSessionSummary",
    "RewardSignalConfig",
    "RewardSignalExtractor",
    "RewardSignalPolarity",
    "RewardStatistics",
    "TemporalMode",
    "apply_temporal_credit_assignment",
    "build_json_report",
    "build_reward_session_summary",
    "compute_reward_confidence",
    "compute_reward_config_fingerprint",
    "decompose_reward",
    "default_reward_config",
    "load_reward_artifact",
    "render_markdown_report",
    "reward_trend_metadata",
    "save_reward_artifact",
]
