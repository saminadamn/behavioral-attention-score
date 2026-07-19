"""Module 9, Step 6: Score Aggregation, plus the top-level `BASEngine`.

`BehaviouralAttentionScorer` is the pure weighted-sum aggregator. `BASEngine`
is the orchestrator this package's file list doesn't name explicitly (see
this module's design-decision summary) — it's the one place
`feature_extractor` -> `normalizer` -> `evidence` -> `BehaviouralAttentionScorer`
-> `smoother` -> `confidence` -> `explanations` are wired together per
interaction, threading smoothing/explanation state across each session, and
producing the final `BASArtifact` from a `DatasetArtifact` (Module 7).
Every dependency is injected, per the brief's "support dependency injection".
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

from dataset_generator.bas.config import BASConfig, default_bas_config
from dataset_generator.bas.confidence import compute_confidence
from dataset_generator.bas.evidence import map_to_evidence
from dataset_generator.bas.explanations import generate_explanation, top_contributors
from dataset_generator.bas.feature_extractor import BASFeatureExtractor
from dataset_generator.bas.models import (
    BASArtifact,
    BASContribution,
    BASEvidence,
    BASRecord,
    BASRecordMetadata,
    BASStatistics,
)
from dataset_generator.bas.normalizer import Normalizer
from dataset_generator.bas.session_summary import build_session_summary
from dataset_generator.bas.smoother import smooth
from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.models.dataset import DatasetArtifact, DatasetRecord

SCHEMA_VERSION = "1.0"


def compute_bas_config_fingerprint(config: BASConfig) -> str:
    """A deterministic SHA-256 fingerprint of `config`, for `BASRecordMetadata`/`BASArtifact`."""

    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


class BehaviouralAttentionScorer:
    """Aggregates `BASEvidence` into a raw score + ranked per-feature contributions.

    Missing features are handled by renormalizing over the weight that was
    actually available (`raw_score = weighted_sum / weight_used`) rather
    than treating a missing feature as zero evidence, which would bias the
    score toward "distracted" purely because of missing data, not observed
    behaviour.
    """

    def __init__(self, config: BASConfig) -> None:
        self._config = config

    def score(self, evidence: BASEvidence) -> tuple[float, list[BASContribution]]:
        weighted_sum = 0.0
        weight_used = 0.0
        contributions: list[BASContribution] = []

        for feature, evidence_value in evidence.values.items():
            weight = self._config.feature_configs[feature].weight
            if weight <= 0:
                continue
            contribution_value = weight * evidence_value
            weighted_sum += contribution_value
            weight_used += weight
            contributions.append(
                BASContribution(
                    feature=feature, weight=weight, evidence_value=evidence_value, contribution=contribution_value
                )
            )

        raw_score = weighted_sum / weight_used if weight_used > 0 else 0.5
        raw_score = min(self._config.score_clip_max, max(self._config.score_clip_min, raw_score))
        contributions.sort(key=lambda c: c.contribution, reverse=True)
        return raw_score, contributions


class BASEngine:
    """Computes a `BASArtifact` from a `DatasetArtifact` — the Module 9 entry point."""

    def __init__(
        self,
        config: BASConfig | None = None,
        predictor: AttentionClassifierPredictor | None = None,
    ) -> None:
        self._config = config or default_bas_config()
        self._extractor = BASFeatureExtractor(predictor)
        self._normalizer = Normalizer(self._config)
        self._scorer = BehaviouralAttentionScorer(self._config)
        self._config_fingerprint = compute_bas_config_fingerprint(self._config)

    def compute(self, dataset_artifact: DatasetArtifact) -> BASArtifact:
        """Compute BAS for every record in `dataset_artifact`, grouped and smoothed per session."""

        by_session: dict[str, list[DatasetRecord]] = defaultdict(list)
        for record in dataset_artifact.records:
            by_session[record.session_id].append(record)

        all_records: list[BASRecord] = []
        session_summaries = []
        for session_id, session_records in by_session.items():
            ordered = sorted(session_records, key=lambda r: r.interaction_number)
            bas_records = self._compute_session(ordered)
            all_records.extend(bas_records)
            session_summaries.append(build_session_summary(bas_records, self._config))

        statistics = self._compute_statistics(all_records)

        return BASArtifact(
            records=all_records,
            session_summaries=session_summaries,
            statistics=statistics,
            config_fingerprint=self._config_fingerprint,
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_session(self, records: list[DatasetRecord]) -> list[BASRecord]:
        observations = self._extractor.extract_batch(records)
        normalization_metadata = self._normalizer.metadata()

        bas_records: list[BASRecord] = []
        previous_smoothed: float | None = None
        previous_contributions: list[BASContribution] | None = None
        raw_history: list[float] = []

        for observation in observations:
            normalized = self._normalizer.normalize(observation)
            evidence = map_to_evidence(normalized, self._config)
            raw_score, contributions = self._scorer.score(evidence)

            raw_history.append(raw_score)
            score = smooth(raw_score, previous_smoothed, self._config, raw_history)

            classifier_confidence = observation.raw_values.get("classifier_confidence")
            confidence, uncertainty, reliability = compute_confidence(
                evidence, self._config, classifier_confidence
            )

            missing_ratio = (
                len(evidence.missing_features) / (len(evidence.values) + len(evidence.missing_features))
                if (len(evidence.values) + len(evidence.missing_features)) > 0
                else 0.0
            )

            explanation = generate_explanation(contributions, previous_contributions)
            top_positive = [c.feature for c in top_contributors(contributions, 3, positive=True)]
            top_negative = [c.feature for c in top_contributors(contributions, 3, positive=False)]

            bas_records.append(
                BASRecord(
                    student_id=observation.student_id,
                    session_id=observation.session_id,
                    interaction_number=observation.interaction_number,
                    raw_score=raw_score,
                    score=score,
                    contributions=contributions,
                    confidence=confidence,
                    uncertainty=uncertainty,
                    reliability=reliability,
                    missing_feature_ratio=missing_ratio,
                    explanation=explanation,
                    top_positive=top_positive,
                    top_negative=top_negative,
                    metadata=BASRecordMetadata(
                        normalization_strategy=normalization_metadata,
                        smoothing_strategy=self._config.smoothing_strategy.value,
                        config_fingerprint=self._config_fingerprint,
                        config_version=self._config.version,
                    ),
                )
            )

            previous_smoothed = score
            previous_contributions = contributions

        return bas_records

    def _compute_statistics(self, records: list[BASRecord]) -> BASStatistics:
        if not records:
            return BASStatistics(
                record_count=0, average_score=0.0, score_distribution={}, average_confidence=0.0,
                feature_contribution_summary={}, missing_value_summary={},
            )

        scores = [r.score for r in records]
        average_score = sum(scores) / len(scores)
        average_confidence = sum(r.confidence for r in records) / len(records)

        bins = ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00")
        bin_counts: Counter[str] = Counter()
        for score in scores:
            index = min(3, int(score * 4))
            bin_counts[bins[index]] += 1
        score_distribution = {label: count / len(records) for label, count in bin_counts.items()}

        contribution_totals: dict[str, float] = defaultdict(float)
        contribution_counts: dict[str, int] = defaultdict(int)
        missing_counts: dict[str, int] = defaultdict(int)
        for record in records:
            seen_features = {c.feature for c in record.contributions}
            for contribution in record.contributions:
                contribution_totals[contribution.feature] += contribution.contribution
                contribution_counts[contribution.feature] += 1
            for feature in self._config.feature_configs:
                if self._config.feature_configs[feature].weight > 0 and feature not in seen_features:
                    missing_counts[feature] += 1

        feature_contribution_summary = {
            feature: contribution_totals[feature] / contribution_counts[feature]
            for feature in contribution_totals
        }

        return BASStatistics(
            record_count=len(records),
            average_score=average_score,
            score_distribution=score_distribution,
            average_confidence=average_confidence,
            feature_contribution_summary=feature_contribution_summary,
            missing_value_summary=dict(missing_counts),
        )
