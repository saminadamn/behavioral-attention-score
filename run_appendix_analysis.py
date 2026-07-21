#!/usr/bin/env python
"""Thesis appendix: Phases 5-9, in one repeatable command.

    python run_appendix_analysis.py
    python run_appendix_analysis.py --students 30 --sessions 3 --seeds 42 123 2025 2026 9999

Writes `outputs_appendix/<timestamp>/`:

    ablation_table.csv            Phase 5 — BAS/reward/cooldown ablations
                                   (Module 13's AblationRunner, unmodified)
    sensitivity_need_threshold.csv     Phase 6 — need-threshold calibration cliff
    sensitivity_reward_weight.csv      Phase 6 — behaviour-category weight sweep
                                   (Module 13's SensitivityRunner, unmodified)
    seed_analysis.csv              Phase 7 — mean/std/95%-CI over N seeds for
                                    DQN/CQL/BCQ agreement and classifier accuracy
    significance_tests.csv         Phase 8 — paired t-test, Wilcoxon, Cohen's d
                                    (CQL vs. DQN, BCQ vs. DQN)
    confusion_full_features.csv    Phase 9 — perfect-separation case
    confusion_restricted_features.csv  Phase 9 — genuine-error case
    error_analysis.md              Phase 9 — ranked confusion-matrix errors,
                                    generated from the restricted-feature run

Nothing here recomputes ablation/sensitivity logic — those two phases are
thin wrappers over Module 13's already-tested `AblationRunner` /
`SensitivityRunner`. What's new is the multi-seed statistics layer
(`dataset_generator.experiment_appendix`) and the confusion-matrix error
ranking, neither of which existed before this script.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dataset_generator.classifier import AttentionClassifierTrainer, TrainingConfig
from dataset_generator.evaluation import AblationOptions, ExperimentConfig, AblationRunner, SensitivityRunner, build_comparison_table, build_sweep_table
from dataset_generator.experiment_appendix import RESTRICTED_FEATURES, analyze_confusion_errors, compute_seed_statistics, paired_significance
from dataset_generator.experiment_reproducibility import set_global_seed
from dataset_generator.orchestration import BASAgent, InterventionAgent, ObserverAgent, RewardAgent
from dataset_generator.reward.config import RewardCategory
from dataset_generator.rl_experimental import DQNConfig, DQNTrainer
from dataset_generator.rl_experimental.offline import BCQConfig, BCQTrainer, CQLConfig, CQLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--students", type=int, default=30)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025, 2026, 9999])
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[run_appendix_analysis] {message}", flush=True)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Phase 5 — Ablation (Module 13's AblationRunner, unmodified)
# --------------------------------------------------------------------------- #

def run_ablation(dataset, output_dir: Path) -> None:
    _log("Phase 5: ablation study (BAS features, EMA smoothing, reward categories, cooldown)")
    config = ExperimentConfig(ablation_options=AblationOptions(
        disable_bas_feature_categories=True,
        disable_temporal_smoothing=True,
        disable_reward_categories=True,
        disable_cooldown=True,
    ))
    ablations = AblationRunner(config=config).run(dataset)
    rows = build_comparison_table(ablations)
    _write_csv(rows, output_dir / "ablation_table.csv")
    for row in rows:
        _log(f"  {row['ablation']}")


# --------------------------------------------------------------------------- #
# Phase 6 — Sensitivity (Module 13's SensitivityRunner, unmodified)
# --------------------------------------------------------------------------- #

def run_sensitivity(dataset, output_dir: Path) -> None:
    _log("Phase 6: sensitivity sweeps (need_threshold, reward-weight)")
    runner = SensitivityRunner()

    need_sweep = runner.sweep_need_threshold(dataset)
    _write_csv(build_sweep_table(need_sweep), output_dir / "sensitivity_need_threshold.csv")

    weight_sweep = runner.sweep_reward_category_weight(dataset, category=RewardCategory.BEHAVIOUR)
    _write_csv(build_sweep_table(weight_sweep), output_dir / "sensitivity_reward_weight.csv")


# --------------------------------------------------------------------------- #
# Phase 7 & 8 — Multi-seed analysis + statistical significance
# --------------------------------------------------------------------------- #

def run_seed_analysis(args, output_dir: Path) -> None:
    _log(f"Phase 7: multi-seed analysis ({len(args.seeds)} seeds: {args.seeds})")

    dqn_agreement, cql_agreement, bcq_agreement, rf_accuracy = [], [], [], []

    for seed in args.seeds:
        dataset = ObserverAgent().generate(student_count=args.students, sessions_per_student=args.sessions)
        bas = BASAgent().compute(dataset)
        reward = RewardAgent().compute(dataset, bas)
        intervention = InterventionAgent().plan(dataset, bas, reward)

        dqn_config = DQNConfig(seed=seed, epochs=6, batch_size=32, min_replay_size=32,
                                use_double_dqn=True, use_prioritized_replay=True, sequence_length=1)
        dqn_artifact = DQNTrainer(dqn_config).train(dataset, bas, reward, intervention)
        dqn_agreement.append(dqn_artifact.greedy_policy_agreement_rate)

        cql_artifact = CQLTrainer(CQLConfig(seed=seed, epochs=6, batch_size=32, min_replay_size=32, cql_alpha=1.0)).train(
            dataset, bas, reward, intervention
        )
        cql_agreement.append(cql_artifact.greedy_policy_agreement_rate)

        bcq_artifact = BCQTrainer(
            BCQConfig(seed=seed, epochs=6, behavior_epochs=5, batch_size=32, min_replay_size=32)
        ).train(dataset, bas, reward, intervention)
        bcq_agreement.append(bcq_artifact.constrained_greedy_agreement_rate)

        classifier_artifact = AttentionClassifierTrainer().train(
            dataset, TrainingConfig(model_name="random_forest", random_state=seed, compute_feature_importance=False)
        )
        rf_accuracy.append(classifier_artifact.metrics.accuracy)

        _log(f"  seed={seed}: dqn={dqn_agreement[-1]:.4f} cql={cql_agreement[-1]:.4f} "
             f"bcq={bcq_agreement[-1]:.4f} rf_acc={rf_accuracy[-1]:.4f}")

    seed_rows = []
    for name, values in [("dqn_agreement", dqn_agreement), ("cql_agreement", cql_agreement),
                          ("bcq_agreement", bcq_agreement), ("rf_accuracy", rf_accuracy)]:
        stats = compute_seed_statistics(values)
        seed_rows.append({
            "metric": name, "seeds": args.seeds, "values": [round(v, 4) for v in values],
            "mean": round(stats.mean, 4), "std": round(stats.std, 4),
            "ci_95_lower": round(stats.ci_lower, 4), "ci_95_upper": round(stats.ci_upper, 4),
        })
    _write_csv(seed_rows, output_dir / "seed_analysis.csv")

    _log("Phase 8: statistical significance (CQL vs. DQN, BCQ vs. DQN)")
    significance_rows = []
    for name, treatment in [("CQL_vs_DQN", cql_agreement), ("BCQ_vs_DQN", bcq_agreement)]:
        result = paired_significance(treatment, dqn_agreement)
        significance_rows.append({"comparison": name, **{k: round(v, 6) for k, v in asdict(result).items()}})
        _log(f"  {name}: t={result.t_statistic:.4f} p={result.t_test_p_value:.6f} "
             f"cohens_d={result.cohens_d:.4f}")
    _write_csv(significance_rows, output_dir / "significance_tests.csv")


# --------------------------------------------------------------------------- #
# Phase 9 — Confusion matrices and error analysis
# --------------------------------------------------------------------------- #

def run_confusion_and_error_analysis(dataset, args, output_dir: Path) -> None:
    _log("Phase 9: confusion matrices (full features vs. restricted features) + error analysis")

    full_artifact = AttentionClassifierTrainer().train(
        dataset, TrainingConfig(model_name="random_forest", split_mode="student_aware", compute_feature_importance=False)
    )
    labels = full_artifact.metrics.class_labels
    _write_csv(
        [{"true_label": labels[i], **{f"predicted_{lbl}": row[j] for j, lbl in enumerate(labels)}}
         for i, row in enumerate(full_artifact.metrics.confusion_matrix)],
        output_dir / "confusion_full_features.csv",
    )

    restricted_artifact = AttentionClassifierTrainer().train(
        dataset, TrainingConfig(model_name="logistic_regression", split_mode="student_aware",
                                 feature_names=RESTRICTED_FEATURES, calibration_method="isotonic",
                                 compute_feature_importance=False),
    )
    restricted_labels = restricted_artifact.metrics.class_labels
    _write_csv(
        [{"true_label": restricted_labels[i], **{f"predicted_{lbl}": row[j] for j, lbl in enumerate(restricted_labels)}}
         for i, row in enumerate(restricted_artifact.metrics.confusion_matrix)],
        output_dir / "confusion_restricted_features.csv",
    )

    errors = analyze_confusion_errors(restricted_artifact.metrics.confusion_matrix, restricted_labels)
    lines = [
        "# Error Analysis (Phase 9)",
        "",
        f"Full-feature-set accuracy: {full_artifact.metrics.accuracy:.4f} "
        f"(perfect or near-perfect separation — see docs/DEEP_LEARNING_COMPARISON.md)",
        "",
        f"Restricted-feature-set accuracy: {restricted_artifact.metrics.accuracy:.4f}, "
        f"ROC-AUC (OvR): {restricted_artifact.metrics.roc_auc_ovr:.4f}",
        f"Expected Calibration Error: "
        f"{restricted_artifact.calibration.expected_calibration_error:.4f}" if restricted_artifact.calibration else "",
        "",
        "## Ranked confusion errors (restricted feature set)",
        "",
        "| True | Predicted | Count | Share of all errors |",
        "|---|---|---|---|",
    ]
    for e in errors:
        lines.append(f"| {e.true_label} | {e.predicted_label} | {e.count} | {e.share_of_all_errors:.1%} |")
    lines.append("")
    if errors:
        top = errors[0]
        lines.append(
            f"The dominant error is **{top.true_label} -> {top.predicted_label}** ({top.count} cases, "
            f"{top.share_of_all_errors:.0%} of all misclassifications), consistent with those two "
            f"states' behavioral signatures genuinely overlapping once the near-perfectly-separating "
            f"response-quality features are removed."
        )
    (output_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_global_seed(args.seeds[0])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs_appendix") / timestamp
    _log(f"Output directory: {output_dir}")

    start = time.perf_counter()
    dataset = ObserverAgent().generate(student_count=args.students, sessions_per_student=args.sessions)
    _log(f"Reference dataset: {len(dataset.records)} records ({time.perf_counter() - start:.1f}s)")

    run_ablation(dataset, output_dir)
    run_sensitivity(dataset, output_dir)
    run_seed_analysis(args, output_dir)
    run_confusion_and_error_analysis(dataset, args, output_dir)

    _log("Done.")
    _log(f"Results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
