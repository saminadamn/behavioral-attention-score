"""Domain models shared across generator modules."""

from dataset_generator.models.behaviour import BehaviourFeatures, BehaviourMetadata, BehaviourRecord
from dataset_generator.models.dataset import (
    DatasetArtifact,
    DatasetManifest,
    DatasetMetadata,
    DatasetRecord,
    DatasetStatistics,
    DatasetValidationReport,
    FeatureCategory,
    FeatureDefinition,
    FeatureDistributionSummary,
)
from dataset_generator.models.prompt import Prompt, PromptMetadata
from dataset_generator.models.response import Response, ResponseFeatures, ResponseMetadata
from dataset_generator.models.session import (
    InteractionRecord,
    InterventionEvent,
    SessionRecord,
    SessionStatistics,
    SessionSummary,
    TransitionEvent,
)
from dataset_generator.models.session_context import SessionContext
from dataset_generator.models.student import Student
from dataset_generator.models.training import (
    CalibrationResult,
    ClassificationMetrics,
    FeatureImportanceEntry,
    FeatureImportanceReport,
    PredictionResult,
    ReliabilityBin,
    TrainingArtifact,
    TrainingMetadata,
)

__all__ = [
    "BehaviourFeatures",
    "BehaviourMetadata",
    "BehaviourRecord",
    "CalibrationResult",
    "ClassificationMetrics",
    "DatasetArtifact",
    "DatasetManifest",
    "DatasetMetadata",
    "DatasetRecord",
    "DatasetStatistics",
    "DatasetValidationReport",
    "FeatureCategory",
    "FeatureDefinition",
    "FeatureDistributionSummary",
    "FeatureImportanceEntry",
    "FeatureImportanceReport",
    "InteractionRecord",
    "InterventionEvent",
    "PredictionResult",
    "Prompt",
    "PromptMetadata",
    "ReliabilityBin",
    "Response",
    "ResponseFeatures",
    "ResponseMetadata",
    "SessionContext",
    "SessionRecord",
    "SessionStatistics",
    "SessionSummary",
    "Student",
    "TrainingArtifact",
    "TrainingMetadata",
    "TransitionEvent",
]
