"""Module 13, Step 2: Evaluation Metrics.

Every function here computes a metric *from* an already-produced artifact
(`DatasetArtifact`/`BASArtifact`/`RewardArtifact`/`InterventionArtifact`)
or already-gathered measurements — nothing here recomputes BAS, reward, or
intervention decisions, and nothing here is a new predictive model.

Where an artifact's own `.statistics` already carries a value (dataset
class/profile balance, BAS average score, reward average/decomposition,
intervention rate/policy distribution), these functions reuse it directly
rather than re-deriving it. New quantities introduced here (BAS variance,
session trend, recovery rate, volatility; reward positive-ratio and
temporal consistency; intervention execution rate, spacing, diversity) are
plain descriptive statistics over already-generated data, not inference.

Intervention Execution Rate (not "need detection precision"): this project
has no externally-labeled "should intervene" ground truth, so no
precision/recall/F1/AUROC is reported for intervention need detection.
Execution rate — how often the planner actually executed an intervention
once one was required, after cooldown and policy constraints — is an
operational description of planner behavior, not a classification metric
against a label this repository does not have. If labeled intervention
data becomes available, conventional classification metrics can be added
without changing this framework.
"""

from __future__ import annotations

from collections import Counter
from math import log2

from pydantic import BaseModel, ConfigDict

from dataset_generator.bas.models import BASArtifact
from dataset_generator.intervention.models import InterventionArtifact
from dataset_generator.models.dataset import DatasetArtifact, FeatureDistributionSummary
from dataset_generator.orchestration.state import WorkflowState
from dataset_generator.reward.models import RewardArtifact

NO_INTERVENTION_POLICY_NAME = "NoInterventionPolicy"


# ---------------------------------------------------------------------------
# Dataset metrics
# ---------------------------------------------------------------------------


class DatasetMetrics(BaseModel):
    """Descriptive metrics over a `DatasetArtifact`."""

    model_config = ConfigDict(frozen=True)

    class_balance: dict[str, float]
    profile_balance: dict[str, float]
    transition_balance: dict[str, dict[str, float]]
    missing_value_summary: dict[str, int]
    feature_distributions: dict[str, FeatureDistributionSummary]


def _empirical_transition_balance(dataset_artifact: DatasetArtifact) -> dict[str, dict[str, float]]:
    """Empirical from-state -> to-state transition frequencies.

    Walks each session's records in interaction order and tabulates
    consecutive `attention_state` pairs — purely descriptive over the
    already-generated ground-truth attention states (Module 6), not a new
    prediction of what the transition "should" be.
    """

    by_session: dict[str, list] = {}
    for record in dataset_artifact.records:
        by_session.setdefault(record.session_id, []).append(record)

    counts: dict[str, Counter[str]] = {}
    for records in by_session.values():
        ordered = sorted(records, key=lambda r: r.interaction_number)
        for prev, curr in zip(ordered, ordered[1:]):
            counts.setdefault(prev.attention_state, Counter())[curr.attention_state] += 1

    balance: dict[str, dict[str, float]] = {}
    for from_state, to_counts in counts.items():
        total = sum(to_counts.values())
        balance[from_state] = {to_state: count / total for to_state, count in to_counts.items()}
    return balance


def compute_dataset_metrics(dataset_artifact: DatasetArtifact) -> DatasetMetrics:
    """Compute `DatasetMetrics`, reusing `dataset_artifact.statistics` wherever possible."""

    stats = dataset_artifact.statistics
    return DatasetMetrics(
        class_balance=stats.class_balance,
        profile_balance=stats.profile_balance,
        transition_balance=_empirical_transition_balance(dataset_artifact),
        missing_value_summary=stats.missing_value_summary,
        feature_distributions=stats.feature_distributions,
    )


# ---------------------------------------------------------------------------
# BAS metrics
# ---------------------------------------------------------------------------


class BASMetrics(BaseModel):
    """Descriptive metrics over a `BASArtifact`, beyond what `BASStatistics` carries.

    `average_confidence` is reused directly from `bas_artifact.statistics`
    (never recomputed) specifically so that ablations affecting BAS
    confidence weighting but not the score itself (e.g. Module 13's
    `disable_confidence_weighting`) are still visible somewhere in this
    metric set — without it, such an ablation would show identical
    `mean`/`variance`/etc. to the baseline and look like it had no effect,
    when the effect is real but only shows up in confidence, not score.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    variance: float
    average_session_trend: float
    recovery_rate: float
    volatility: float
    average_confidence: float


def compute_bas_metrics(bas_artifact: BASArtifact, recovery_window: int = 3) -> BASMetrics:
    """Compute `BASMetrics` from `bas_artifact.records`.

    `average_session_trend` is the mean, per session, of (last score - first
    score). `recovery_rate` is the fraction of declines (score_t < score_{t-1})
    followed by a recovery to at least the pre-decline level within
    `recovery_window` interactions. `volatility` is the mean absolute
    successive difference in score.
    """

    scores = [r.score for r in bas_artifact.records]
    if not scores:
        return BASMetrics(
            mean=0.0, variance=0.0, average_session_trend=0.0, recovery_rate=0.0, volatility=0.0,
            average_confidence=bas_artifact.statistics.average_confidence,
        )

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)

    by_session: dict[str, list] = {}
    for record in bas_artifact.records:
        by_session.setdefault(record.session_id, []).append(record)

    session_trends = []
    declines_seen = 0
    recoveries = 0
    volatility_terms = []

    for records in by_session.values():
        ordered = sorted(records, key=lambda r: r.interaction_number)
        ordered_scores = [r.score for r in ordered]
        if len(ordered_scores) >= 2:
            session_trends.append(ordered_scores[-1] - ordered_scores[0])
        for i in range(1, len(ordered_scores)):
            volatility_terms.append(abs(ordered_scores[i] - ordered_scores[i - 1]))
            if ordered_scores[i] < ordered_scores[i - 1]:
                declines_seen += 1
                pre_decline_level = ordered_scores[i - 1]
                window_end = min(len(ordered_scores), i + 1 + recovery_window)
                if any(ordered_scores[j] >= pre_decline_level for j in range(i + 1, window_end)):
                    recoveries += 1

    average_session_trend = sum(session_trends) / len(session_trends) if session_trends else 0.0
    recovery_rate = recoveries / declines_seen if declines_seen else 0.0
    volatility = sum(volatility_terms) / len(volatility_terms) if volatility_terms else 0.0

    return BASMetrics(
        mean=mean, variance=variance, average_session_trend=average_session_trend,
        recovery_rate=recovery_rate, volatility=volatility,
        average_confidence=bas_artifact.statistics.average_confidence,
    )


# ---------------------------------------------------------------------------
# Reward metrics
# ---------------------------------------------------------------------------


class RewardMetrics(BaseModel):
    """Descriptive metrics over a `RewardArtifact`, reusing `RewardStatistics` where possible."""

    model_config = ConfigDict(frozen=True)

    average_reward: float
    positive_ratio: float
    reward_decomposition: dict[str, float]
    temporal_consistency: float


def compute_reward_metrics(reward_artifact: RewardArtifact) -> RewardMetrics:
    """Compute `RewardMetrics`.

    `temporal_consistency` is the Pearson correlation between a reward and
    the immediately preceding reward within the same session, averaged
    across sessions with at least two interactions — a high value means
    reward moves smoothly rather than erratically from one interaction to
    the next.
    """

    stats = reward_artifact.statistics
    records = reward_artifact.records
    positive_ratio = (
        sum(1 for r in records if r.reward > 0) / len(records) if records else 0.0
    )

    by_session: dict[str, list] = {}
    for record in records:
        by_session.setdefault(record.session_id, []).append(record)

    correlations = []
    for session_records in by_session.values():
        ordered = sorted(session_records, key=lambda r: r.interaction_number)
        if len(ordered) < 3:
            continue
        current = [r.reward for r in ordered[1:]]
        previous = [r.reward for r in ordered[:-1]]
        correlations.append(_pearson_correlation(previous, current))

    temporal_consistency = sum(correlations) / len(correlations) if correlations else 0.0

    return RewardMetrics(
        average_reward=stats.average_reward,
        positive_ratio=positive_ratio,
        reward_decomposition={
            "performance": stats.average_performance_reward,
            "behaviour": stats.average_behaviour_reward,
            "cost": stats.average_cost_reward,
        },
        temporal_consistency=temporal_consistency,
    )


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    std_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return cov / (std_x * std_y)


# ---------------------------------------------------------------------------
# Intervention metrics
# ---------------------------------------------------------------------------


class InterventionMetrics(BaseModel):
    """Operational metrics over an `InterventionArtifact` — see module
    docstring for why these are operational, not classification, metrics.

    `intervention_execution_rate` is `None`, not `0.0`, when
    `intervention_required` never fires (see `compute_intervention_metrics`
    for why the two are not interchangeable).
    """

    model_config = ConfigDict(frozen=True)

    intervention_execution_rate: float | None
    intervention_required_count: int
    executed_intervention_count: int
    policy_frequencies: dict[str, float]
    cooldown_activations: int
    average_intervention_spacing: float
    policy_diversity: float


def compute_intervention_metrics(intervention_artifact: InterventionArtifact) -> InterventionMetrics:
    """Compute `InterventionMetrics`.

    `intervention_execution_rate = executed_interventions / intervention_required`,
    where `executed_interventions` is the count of decisions with
    `chosen_policy != "NoInterventionPolicy"` and `cooldown_suppressed is False`.

    `intervention_required` (Module 11) and `chosen_policy != "NoInterventionPolicy"`
    are two independent signals, not the same thing measured twice:
    `intervention_required` additionally requires the *aggregate* need score
    to clear `config.need_threshold`, while a policy can be selected and
    applied purely on its own eligibility rule (e.g. `HintPolicy` firing on
    low correctness alone) regardless of the aggregate need score. Under
    `InterventionConfig`'s default `need_threshold=0.35`, empirically
    observed average need scores are far lower than that, so
    `intervention_required` can legitimately be `0` for an entire run even
    while real policies are still being chosen and applied. Reporting `0.0`
    for a `0/0` case here would misrepresent that as "the planner never
    executes when an intervention is needed", which is false — it is "the
    aggregate threshold rarely or never triggers under this config". `None`
    is returned instead, and `intervention_required_count`/
    `executed_intervention_count` are exposed alongside it so this
    threshold-calibration question (Step 5's sensitivity analysis territory)
    is visible rather than silently flattened to zero.

    `policy_diversity` is the normalized Shannon entropy of the policy
    distribution (1.0 = every registered policy used equally often, 0.0 =
    only one policy ever chosen).
    """

    decisions = intervention_artifact.decisions
    intervention_required_count = sum(1 for d in decisions if d.intervention_required)
    executed_count = sum(
        1 for d in decisions if d.chosen_policy != NO_INTERVENTION_POLICY_NAME and not d.cooldown_suppressed
    )
    intervention_execution_rate = (
        executed_count / intervention_required_count if intervention_required_count else None
    )

    cooldown_activations = sum(1 for d in decisions if d.cooldown_suppressed)

    by_session: dict[str, list] = {}
    for decision in decisions:
        by_session.setdefault(decision.session_id, []).append(decision)

    spacings = []
    for session_decisions in by_session.values():
        ordered = sorted(session_decisions, key=lambda d: d.interaction_number)
        executed_interactions = [
            d.interaction_number for d in ordered
            if d.chosen_policy != NO_INTERVENTION_POLICY_NAME and not d.cooldown_suppressed
        ]
        for prev, curr in zip(executed_interactions, executed_interactions[1:]):
            spacings.append(curr - prev)
    average_intervention_spacing = sum(spacings) / len(spacings) if spacings else 0.0

    policy_frequencies = dict(intervention_artifact.statistics.policy_distribution)
    policy_diversity = _normalized_entropy(policy_frequencies)

    return InterventionMetrics(
        intervention_execution_rate=intervention_execution_rate,
        intervention_required_count=intervention_required_count,
        executed_intervention_count=executed_count,
        policy_frequencies=policy_frequencies,
        cooldown_activations=cooldown_activations,
        average_intervention_spacing=average_intervention_spacing,
        policy_diversity=policy_diversity,
    )


def _normalized_entropy(distribution: dict[str, float]) -> float:
    probabilities = [p for p in distribution.values() if p > 0.0]
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(p * log2(p) for p in probabilities)
    max_entropy = log2(len(distribution)) if len(distribution) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Workflow metrics
# ---------------------------------------------------------------------------


class WorkflowMetrics(BaseModel):
    """Metrics describing one orchestration run.

    Timing figures are reused directly from Module 12's own
    `orchestration.report.node_timing_summary`/`execution_history` rather
    than re-measured here. `peak_memory_bytes`, `checkpoint_latency_seconds`,
    and `serialization_latency_seconds` are *received*, not measured, by
    this function — actual instrumentation of a live run is `benchmark.py`'s
    job (Step 3); this function only assembles already-gathered numbers.
    """

    model_config = ConfigDict(frozen=True)

    total_execution_seconds: float
    node_timings: dict[str, dict[str, float | int]]
    peak_memory_bytes: int | None = None
    checkpoint_latency_seconds: float | None = None
    serialization_latency_seconds: float | None = None
    replay_correct: bool | None = None


def compute_workflow_metrics(
    state: WorkflowState,
    peak_memory_bytes: int | None = None,
    checkpoint_latency_seconds: float | None = None,
    serialization_latency_seconds: float | None = None,
    replay_correct: bool | None = None,
) -> WorkflowMetrics:
    """Assemble `WorkflowMetrics` from a completed `WorkflowState` plus
    externally-measured resource figures.
    """

    from dataset_generator.orchestration.report import node_timing_summary

    timings = node_timing_summary(state)
    total_execution_seconds = sum(entry["total_seconds"] for entry in timings.values())

    return WorkflowMetrics(
        total_execution_seconds=total_execution_seconds,
        node_timings=timings,
        peak_memory_bytes=peak_memory_bytes,
        checkpoint_latency_seconds=checkpoint_latency_seconds,
        serialization_latency_seconds=serialization_latency_seconds,
        replay_correct=replay_correct,
    )


def check_replay_correctness(first_state: WorkflowState, second_state: WorkflowState) -> bool:
    """Whether two `WorkflowState`s from the same input are behaviourally
    identical — compares `tutor_actions`/`session_outputs`, not wall-clock
    metadata such as `generation_timestamp`, which legitimately differs run
    to run.
    """

    return (
        first_state.get("tutor_actions") == second_state.get("tutor_actions")
        and first_state.get("session_outputs") == second_state.get("session_outputs")
    )
