"""Module 10, Step 5: Reward Aggregation, Step 8: Session Summary, and the
top-level `RewardEngine`.

Your file list for this module has no dedicated `session_summary.py`
(unlike Module 9's), so session-level aggregation lives here alongside
per-interaction aggregation — the natural home given Step 5 already owns
"aggregation" as a concept. `RewardEngine` (this module's orchestrator, not
explicitly named in your file list, mirroring `BASEngine`'s placement in
Module 9's `scorer.py`) wires `RewardSignalExtractor` ->
`RewardAggregator` -> `apply_temporal_credit_assignment` ->
`compute_reward_confidence` together per session.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

from dataset_generator.bas.models import BASArtifact
from dataset_generator.bas.normalizer import normalize_value
from dataset_generator.classifier.predictor import AttentionClassifierPredictor
from dataset_generator.models.dataset import DatasetArtifact

from dataset_generator.reward.config import (
    RewardCategory,
    RewardConfig,
    RewardSignalPolarity,
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
from dataset_generator.reward.signals import RewardSignalExtractor
from dataset_generator.reward.temporal import apply_temporal_credit_assignment

SCHEMA_VERSION = "1.0"


def compute_reward_config_fingerprint(config: RewardConfig) -> str:
    """A deterministic SHA-256 fingerprint of `config`."""

    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


class RewardAggregator:
    """Aggregates one `RewardObservation` into a signed raw reward + ranked contributions.

    Missing signals are excluded from the weighted sum and its
    normalization denominator (renormalizing over the weight actually
    available), the same missing-value handling `BehaviouralAttentionScorer`
    uses — a missing signal contributes nothing, it doesn't silently count
    as "no change" or "zero evidence".
    """

    def __init__(self, config: RewardConfig) -> None:
        self._config = config

    def aggregate(self, observation: RewardObservation) -> tuple[float, list[RewardContribution], float]:
        """Return `(raw_reward, contributions, missing_signal_ratio)`.

        Two passes: the first computes each observed signal's signed
        evidence and totals the weight actually available
        (`weight_used`); the second builds each `RewardContribution` using
        that signal's weight *renormalized over `weight_used`*, not its raw
        configured weight. This is what makes
        `raw_reward == sum(c.contribution for c in contributions)` an exact
        identity regardless of which signals were missing — and therefore
        what makes `decompose_reward`'s category partition
        (`performance_reward + behaviour_reward - cost_reward`) reconstruct
        `raw_reward` exactly. Storing the *un*-renormalized weight here
        instead would silently break that identity on every interaction
        with at least one missing signal (in practice: almost every
        interaction, since `intervention_cost` is "missing" whenever no
        intervention fired).
        """

        observed: list[tuple[str, RewardSignalPolarity, RewardCategory, float, float]] = []
        weight_used = 0.0
        missing_count = 0

        for signal, signal_config in self._config.signal_configs.items():
            if signal_config.weight <= 0 or signal_config.polarity == RewardSignalPolarity.NEUTRAL:
                continue

            raw_value = observation.raw_signals.get(signal)
            if raw_value is None:
                missing_count += 1
                continue

            if signal_config.polarity == RewardSignalPolarity.PENALTY:
                evidence_value = 0.0  # triggered penalty = worst-case evidence
            else:
                normalized = normalize_value(raw_value, signal_config.normalization)
                evidence_value = normalized if signal_config.polarity == RewardSignalPolarity.POSITIVE else 1.0 - normalized

            signed_evidence = evidence_value * 2.0 - 1.0
            observed.append((signal, signal_config.weight, signal_config.category, evidence_value, signed_evidence))
            weight_used += signal_config.weight

        contributions: list[RewardContribution] = []
        for signal, original_weight, category, evidence_value, signed_evidence in observed:
            effective_weight = original_weight / weight_used if weight_used > 0 else 0.0
            contributions.append(
                RewardContribution(
                    signal=signal, weight=original_weight, category=category,
                    evidence_value=evidence_value, signed_evidence=signed_evidence,
                    contribution=effective_weight * signed_evidence,
                )
            )

        raw_reward = sum(c.contribution for c in contributions) if contributions else 0.0
        raw_reward = min(self._config.reward_clip_max, max(self._config.reward_clip_min, raw_reward))
        contributions.sort(key=lambda c: c.contribution, reverse=True)

        total = missing_count + len(contributions)
        missing_ratio = missing_count / total if total > 0 else 0.0

        return raw_reward, contributions, missing_ratio


def decompose_reward(contributions: list[RewardContribution]) -> tuple[float, float, float]:
    """Split `contributions` into `(performance_reward, behaviour_reward, cost_reward)`.

    `performance_reward`/`behaviour_reward` are each the sum of that
    category's `contribution` values (already signed, already weighted).
    `cost_reward` is the *negated* sum of the `COST` category's
    contributions — cost contributions are always <= 0 by construction
    (see `RewardAggregator.aggregate`'s `PENALTY` handling), so negating
    turns it into the non-negative magnitude the
    `total = performance + behaviour - cost` framing subtracts.

    The invariant `sum(c.contribution for c in contributions) ==
    performance_reward + behaviour_reward - cost_reward` holds by
    construction — there is no renormalization or double-counting between
    categories, just a partition of the same contributions already summed
    into `raw_reward`.
    """

    performance_reward = sum(
        c.contribution for c in contributions if c.category == RewardCategory.PERFORMANCE
    )
    behaviour_reward = sum(
        c.contribution for c in contributions if c.category == RewardCategory.BEHAVIOUR
    )
    cost_reward = -sum(c.contribution for c in contributions if c.category == RewardCategory.COST)
    return performance_reward, behaviour_reward, cost_reward


def build_reward_session_summary(
    records: list[RewardRecord], observations: list[RewardObservation], config: RewardConfig
) -> RewardSessionSummary:
    """Build a `RewardSessionSummary` from one session's ordered records + observations.

    `observations` (not just `records`) is needed to detect *when* an
    intervention was applied, for `recovery_count` — that fact lives on
    `RewardObservation.raw_signals["intervention_cost"]`, not on the
    aggregated `RewardRecord` itself.
    """

    if not records:
        raise ValueError("cannot build a session summary from an empty record list")

    rewards = [r.reward for r in records]
    n = len(rewards)

    average = sum(rewards) / n
    maximum = max(rewards)
    minimum = min(rewards)
    variance = 0.0 if n < 2 else sum((r - average) ** 2 for r in rewards) / n

    half = max(1, n // 2)
    first_half_avg = sum(rewards[:half]) / half
    second_half_avg = sum(rewards[-half:]) / half
    delta = second_half_avg - first_half_avg
    if delta > config.trend_tolerance:
        trend = "improving"
    elif delta < -config.trend_tolerance:
        trend = "declining"
    else:
        trend = "stable"

    deltas = [rewards[i + 1] - rewards[i] for i in range(n - 1)]
    largest_improvement = max((d for d in deltas if d > 0), default=0.0)
    largest_deterioration = max((-d for d in deltas if d < 0), default=0.0)

    cumulative_reward = sum(rewards)

    recovery_count = 0
    for i in range(1, n):
        intervened_previously = observations[i - 1].raw_signals.get("intervention_cost") is not None
        if intervened_previously and rewards[i] >= config.recovery_threshold:
            recovery_count += 1

    return RewardSessionSummary(
        student_id=records[0].student_id,
        session_id=records[0].session_id,
        interaction_count=n,
        average_reward=average,
        maximum_reward=maximum,
        minimum_reward=minimum,
        variance_reward=variance,
        reward_trend=trend,
        largest_improvement=largest_improvement,
        largest_deterioration=largest_deterioration,
        cumulative_reward=cumulative_reward,
        recovery_count=recovery_count,
    )


class RewardEngine:
    """Computes a `RewardArtifact` from a `DatasetArtifact` + `BASArtifact` — the
    Module 10 entry point.
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        predictor: AttentionClassifierPredictor | None = None,
    ) -> None:
        self._config = config or default_reward_config()
        self._extractor = RewardSignalExtractor(predictor)
        self._aggregator = RewardAggregator(self._config)
        self._fingerprint = compute_reward_config_fingerprint(self._config)

    def compute(self, dataset_artifact: DatasetArtifact, bas_artifact: BASArtifact) -> RewardArtifact:
        """Compute reward for every interaction, grouped and credit-assigned per session."""

        observations = self._extractor.extract_batch(dataset_artifact.records, bas_artifact.records)
        bas_confidence_by_key = {
            (r.session_id, r.interaction_number): r.confidence for r in bas_artifact.records
        }

        by_session: dict[str, list[RewardObservation]] = defaultdict(list)
        for observation in observations:
            by_session[observation.session_id].append(observation)

        all_records: list[RewardRecord] = []
        session_summaries: list[RewardSessionSummary] = []
        for session_id, session_observations in by_session.items():
            ordered = sorted(session_observations, key=lambda o: o.interaction_number)
            records = self._compute_session(ordered, bas_confidence_by_key)
            all_records.extend(records)
            session_summaries.append(build_reward_session_summary(records, ordered, self._config))

        statistics = self._compute_statistics(all_records)

        return RewardArtifact(
            records=all_records,
            session_summaries=session_summaries,
            statistics=statistics,
            config_fingerprint=self._fingerprint,
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_session(
        self, observations: list[RewardObservation], bas_confidence_by_key: dict[tuple[str, int], float]
    ) -> list[RewardRecord]:
        raw_rewards: list[float] = []
        contributions_per_observation = []
        missing_ratios = []

        for observation in observations:
            raw_reward, contributions, missing_ratio = self._aggregator.aggregate(observation)
            raw_rewards.append(raw_reward)
            contributions_per_observation.append(contributions)
            missing_ratios.append(missing_ratio)

        credited_rewards = apply_temporal_credit_assignment(raw_rewards, self._config)

        records = []
        for observation, raw_reward, credited_reward, contributions, missing_ratio in zip(
            observations, raw_rewards, credited_rewards, contributions_per_observation, missing_ratios
        ):
            bas_confidence = bas_confidence_by_key.get(
                (observation.session_id, observation.interaction_number)
            )
            confidence, uncertainty, reliability = compute_reward_confidence(
                contributions, missing_ratio, self._config, bas_confidence
            )
            clipped_reward = min(
                self._config.reward_clip_max, max(self._config.reward_clip_min, credited_reward)
            )
            performance_reward, behaviour_reward, cost_reward = decompose_reward(contributions)

            records.append(
                RewardRecord(
                    student_id=observation.student_id,
                    session_id=observation.session_id,
                    interaction_number=observation.interaction_number,
                    raw_reward=raw_reward,
                    reward=clipped_reward,
                    performance_reward=performance_reward,
                    behaviour_reward=behaviour_reward,
                    cost_reward=cost_reward,
                    contributions=contributions,
                    confidence=confidence,
                    uncertainty=uncertainty,
                    reliability=reliability,
                    missing_signal_ratio=missing_ratio,
                    metadata=RewardRecordMetadata(
                        temporal_mode=self._config.temporal_mode.value,
                        discount_factor=self._config.discount_factor,
                        config_fingerprint=self._fingerprint,
                        config_version=self._config.version,
                    ),
                )
            )
        return records

    def _compute_statistics(self, records: list[RewardRecord]) -> RewardStatistics:
        if not records:
            return RewardStatistics(
                record_count=0, average_reward=0.0, reward_distribution={}, average_confidence=0.0,
                average_performance_reward=0.0, average_behaviour_reward=0.0, average_cost_reward=0.0,
                contribution_summary={}, missing_value_summary={},
            )

        rewards = [r.reward for r in records]
        average_reward = sum(rewards) / len(rewards)
        average_confidence = sum(r.confidence for r in records) / len(records)
        average_performance_reward = sum(r.performance_reward for r in records) / len(records)
        average_behaviour_reward = sum(r.behaviour_reward for r in records) / len(records)
        average_cost_reward = sum(r.cost_reward for r in records) / len(records)

        bins = ("-1.00--0.50", "-0.50-0.00", "0.00-0.50", "0.50-1.00")
        bin_counts: Counter[str] = Counter()
        for reward in rewards:
            index = min(3, int((reward + 1.0) * 2))
            bin_counts[bins[index]] += 1
        reward_distribution = {label: count / len(records) for label, count in bin_counts.items()}

        contribution_totals: dict[str, float] = defaultdict(float)
        contribution_counts: dict[str, int] = defaultdict(int)
        missing_counts: dict[str, int] = defaultdict(int)
        for record in records:
            seen_signals = {c.signal for c in record.contributions}
            for contribution in record.contributions:
                contribution_totals[contribution.signal] += contribution.contribution
                contribution_counts[contribution.signal] += 1
            for signal, signal_config in self._config.signal_configs.items():
                if signal_config.weight > 0 and signal_config.polarity != RewardSignalPolarity.NEUTRAL:
                    if signal not in seen_signals:
                        missing_counts[signal] += 1

        contribution_summary = {
            signal: contribution_totals[signal] / contribution_counts[signal]
            for signal in contribution_totals
        }

        return RewardStatistics(
            record_count=len(records),
            average_reward=average_reward,
            reward_distribution=reward_distribution,
            average_confidence=average_confidence,
            average_performance_reward=average_performance_reward,
            average_behaviour_reward=average_behaviour_reward,
            average_cost_reward=average_cost_reward,
            contribution_summary=contribution_summary,
            missing_value_summary=dict(missing_counts),
        )
