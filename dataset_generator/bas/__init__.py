"""Module 9: Behavioural Attention Score (BAS).

Computes a continuous, interpretable, explainable BAS from observable
behavioural evidence in a `DatasetArtifact` (Module 7) — never regenerating
behaviour, never touching hidden simulator variables. `BASArtifact` is this
package's single source of truth, the way `DatasetArtifact`/`TrainingArtifact`
are for Modules 7/8. Entry point: `BASEngine(config).compute(dataset_artifact)`.
"""

from dataset_generator.bas.config import (
    BASConfig,
    FeatureBASConfig,
    FeatureNormalizationConfig,
    FeaturePolarity,
    NormalizationStrategy,
    SmoothingStrategy,
    default_bas_config,
)
from dataset_generator.bas.confidence import compute_confidence
from dataset_generator.bas.evidence import map_to_evidence
from dataset_generator.bas.explanations import generate_explanation, top_contributors
from dataset_generator.bas.feature_extractor import BASFeatureExtractor
from dataset_generator.bas.models import (
    BASArtifact,
    BASContribution,
    BASEvidence,
    BASObservation,
    BASRecord,
    BASRecordMetadata,
    BASSessionSummary,
    BASStatistics,
)
from dataset_generator.bas.normalizer import Normalizer, normalize_value
from dataset_generator.bas.report import build_json_report, render_markdown_report, session_plot_metadata
from dataset_generator.bas.scorer import BASEngine, BehaviouralAttentionScorer, compute_bas_config_fingerprint
from dataset_generator.bas.serialization import load_bas_artifact, save_bas_artifact
from dataset_generator.bas.session_summary import build_session_summary
from dataset_generator.bas.smoother import smooth

__all__ = [
    "BASArtifact",
    "BASConfig",
    "BASContribution",
    "BASEngine",
    "BASEvidence",
    "BASFeatureExtractor",
    "BASObservation",
    "BASRecord",
    "BASRecordMetadata",
    "BASSessionSummary",
    "BASStatistics",
    "BehaviouralAttentionScorer",
    "FeatureBASConfig",
    "FeatureNormalizationConfig",
    "FeaturePolarity",
    "NormalizationStrategy",
    "Normalizer",
    "SmoothingStrategy",
    "build_json_report",
    "build_session_summary",
    "compute_bas_config_fingerprint",
    "compute_confidence",
    "default_bas_config",
    "generate_explanation",
    "load_bas_artifact",
    "map_to_evidence",
    "normalize_value",
    "render_markdown_report",
    "save_bas_artifact",
    "session_plot_metadata",
    "smooth",
    "top_contributors",
]
