"""One-command experiment runner for the Behavioral Attention Score project.

Runs the full pipeline (dataset generation -> BAS -> reward -> intervention
-> classifier training -> evaluation) using the project's own real APIs
(`ObserverAgent`, `BASAgent`, `RewardAgent`, `InterventionAgent`,
`AttentionClassifierTrainer`) and writes every artifact — figures, metrics,
the trained model, and the generated dataset — to `outputs/`.

Usage:
    python run_experiment.py
    python run_experiment.py --students 60 --sessions 4 --model random_forest
    python run_experiment.py --config configs/thesis.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score, log_loss, matthews_corrcoef, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from dataset_generator.classifier import AttentionClassifierTrainer, TrainingConfig
from dataset_generator.classifier.splitting import split_dataset
from dataset_generator.config import default_config, load_config
from dataset_generator.orchestration import BASAgent, InterventionAgent, ObserverAgent, RewardAgent
from dataset_generator.pipeline import export_csv, export_metadata_json
from dataset_generator.validators.dataset_validator import records_to_frame

TARGET_COLUMN = "attention_state"

OUT = Path("outputs")
FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"
MODEL_DIR = OUT / "models"
DATA_DIR = OUT / "datasets"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=60, help="number of synthetic students")
    parser.add_argument("--sessions", type=int, default=4, help="sessions per student")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["logistic_regression", "random_forest", "gradient_boosting"],
        help="classifier to train",
    )
    parser.add_argument(
        "--split-mode",
        default="student_aware",
        choices=["random", "stratified", "session_aware", "student_aware"],
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--calibration",
        default="platt",
        choices=["platt", "isotonic", "none"],
        help="probability calibration method ('none' to skip)",
    )
    parser.add_argument("--config", default=None, help="optional path to a saved GeneratorConfig YAML/JSON")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    cm = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_val: np.ndarray, y_proba: np.ndarray, labels: list[str], path: Path) -> None:
    y_bin = label_binarize(y_val, classes=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    if y_bin.shape[1] == 1:  # binary edge case
        fpr, tpr, _ = roc_curve(y_val, y_proba[:, 1])
        ax.plot(fpr, tpr, label=labels[1])
    else:
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_val: np.ndarray, y_proba: np.ndarray, labels: list[str], path: Path) -> None:
    y_bin = label_binarize(y_val, classes=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    if y_bin.shape[1] == 1:
        precision, recall, _ = precision_recall_curve(y_val, y_proba[:, 1])
        ax.plot(recall, precision, label=labels[1])
    else:
        for i, label in enumerate(labels):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (One-vs-Rest)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(ranked, path: Path, top_k: int = 15) -> None:
    top = ranked[:top_k][::-1]
    names = [e.feature for e in top]
    values = [e.importance for e in top]
    errs = [e.std or 0.0 for e in top]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names, values, xerr=errs, color="#4C72B0")
    ax.set_xlabel("Permutation importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(reliability_bins, path: Path) -> None:
    conf = [b.mean_confidence for b in reliability_bins if b.count > 0]
    acc = [b.mean_accuracy for b in reliability_bins if b.count > 0]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.plot(conf, acc, marker="o", label="Model")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Mean observed accuracy")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bas_distribution(scores: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(scores, bins=30, color="#55A868", edgecolor="white")
    ax.set_xlabel("BAS score")
    ax.set_ylabel("Count")
    ax.set_title("BAS Score Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_category_distribution(counts: dict, title: str, xlabel: str, path: Path) -> None:
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(names, values, color="#C44E52")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transition_heatmap(dataset_df, labels: list[str], path: Path) -> None:
    """Attention-state transition frequency, one interaction to the next, per session."""

    index = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)))
    for _, group in dataset_df.sort_values("interaction_number").groupby("session_id"):
        states = group[TARGET_COLUMN].tolist()
        for a, b in zip(states, states[1:]):
            counts[index[a], index[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(probs, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title("Attention State Transition Heatmap")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center",
                     color="white" if probs[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    for d in (FIG_DIR, REPORT_DIR, MODEL_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config) if args.config else default_config()

    print(f"[1/7] Generating dataset  (students={args.students}, sessions/student={args.sessions})")
    dataset = ObserverAgent(config).generate(student_count=args.students, sessions_per_student=args.sessions)
    print(f"      -> {len(dataset.records)} interaction records")

    print("[2/7] Computing BAS, reward, and intervention plan")
    bas = BASAgent().compute(dataset)
    reward = RewardAgent().compute(dataset, bas)
    intervention = InterventionAgent().plan(dataset, bas, reward)

    print(f"[3/7] Training classifier ({args.model}, split={args.split_mode})")
    trainer = AttentionClassifierTrainer()
    training_config = TrainingConfig(
        model_name=args.model,
        split_mode=args.split_mode,
        test_size=args.test_size,
        random_state=args.random_state,
        calibration_method=None if args.calibration == "none" else args.calibration,
        compute_feature_importance=True,
    )
    artifact = trainer.train(dataset, training_config)
    metrics = artifact.metrics
    print(f"      -> accuracy={metrics.accuracy:.4f}  f1_macro={metrics.f1_macro:.4f}  "
          f"roc_auc_ovr={metrics.roc_auc_ovr}")

    print("[4/7] Reconstructing validation predictions for curve plots")
    df = records_to_frame(dataset.records)
    _, val_df = split_dataset(df, training_config.split_mode, TARGET_COLUMN,
                               training_config.test_size, training_config.random_state)
    X_val = artifact.preprocessor.transform(val_df)
    y_val = val_df[TARGET_COLUMN].to_numpy()
    class_labels = metrics.class_labels
    proba_columns = artifact.model.classes_.tolist()
    reorder = [proba_columns.index(label) for label in class_labels]
    y_proba = artifact.model.predict_proba(X_val)[:, reorder]

    # Statistical metrics `ClassificationMetrics` doesn't carry fields for —
    # computed once here directly from the validation split via scikit-learn,
    # not duplicated from anywhere in the classifier module.
    extra_metrics = {
        "log_loss": float(log_loss(y_val, y_proba, labels=class_labels)),
        "cohens_kappa": float(cohen_kappa_score(y_val, artifact.model.predict(X_val))),
        "matthews_corrcoef": float(matthews_corrcoef(y_val, artifact.model.predict(X_val))),
    }
    print(f"      -> log_loss={extra_metrics['log_loss']:.4f}  "
          f"cohens_kappa={extra_metrics['cohens_kappa']:.4f}  "
          f"mcc={extra_metrics['matthews_corrcoef']:.4f}")

    print("[5/7] Generating graphs")
    plot_confusion_matrix(metrics.confusion_matrix, class_labels, FIG_DIR / "confusion_matrix.png")
    plot_roc_curve(y_val, y_proba, class_labels, FIG_DIR / "roc_curve.png")
    plot_pr_curve(y_val, y_proba, class_labels, FIG_DIR / "pr_curve.png")
    if artifact.feature_importance is not None:
        plot_feature_importance(artifact.feature_importance.ranked, FIG_DIR / "feature_importance.png")
    if artifact.calibration is not None:
        plot_calibration_curve(artifact.calibration.reliability_bins, FIG_DIR / "calibration_curve.png")
    plot_bas_distribution([r.score for r in bas.records], FIG_DIR / "bas_distribution.png")
    plot_category_distribution(
        {label: count for label, count in
         zip(*np.unique([r.attention_state for r in dataset.records], return_counts=True))},
        "Attention State Distribution", "Count", FIG_DIR / "attention_state_distribution.png",
    )
    plot_category_distribution(
        dict(intervention.statistics.policy_distribution),
        "Intervention Policy Distribution", "Proportion", FIG_DIR / "intervention_distribution.png",
    )
    plot_category_distribution(
        {profile: count for profile, count in
         zip(*np.unique([s for s in df["student_profile"]], return_counts=True))},
        "Student Profile Distribution", "Count", FIG_DIR / "student_profile_distribution.png",
    )
    plot_transition_heatmap(df, class_labels, FIG_DIR / "transition_heatmap.png")

    print("[6/7] Saving model, dataset, and metrics")
    artifact.model.save(MODEL_DIR / "classifier.joblib")
    joblib.dump(artifact.preprocessor, MODEL_DIR / "preprocessor.joblib")
    export_csv(dataset.records, DATA_DIR / "synthetic_dataset.csv")
    export_metadata_json(dataset.metadata, DATA_DIR / "metadata.json")

    with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(metrics.classification_report)

    metrics_rows = [
        ("accuracy", metrics.accuracy),
        ("balanced_accuracy", metrics.balanced_accuracy),
        ("precision_macro", metrics.precision_macro),
        ("recall_macro", metrics.recall_macro),
        ("f1_macro", metrics.f1_macro),
        ("f1_weighted", metrics.f1_weighted),
        ("roc_auc_ovr", metrics.roc_auc_ovr),
        ("mean_prediction_confidence", metrics.mean_prediction_confidence),
    ]
    metrics_rows.append(("log_loss", extra_metrics["log_loss"]))
    metrics_rows.append(("cohens_kappa", extra_metrics["cohens_kappa"]))
    metrics_rows.append(("matthews_corrcoef", extra_metrics["matthews_corrcoef"]))
    if artifact.calibration is not None:
        metrics_rows.append(("expected_calibration_error", artifact.calibration.expected_calibration_error))
    metrics_rows.append(("intervention_rate", intervention.statistics.intervention_rate))
    metrics_rows.append(("bas_average_score", bas.statistics.average_score))
    metrics_rows.append(("bas_score_std", float(np.std([r.score for r in bas.records]))))
    metrics_rows.append(("reward_average", sum(r.reward for r in reward.records) / len(reward.records)))

    with open(REPORT_DIR / "metrics.csv", "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for name, value in metrics_rows:
            f.write(f"{name},{value}\n")

    per_class_path = REPORT_DIR / "per_class_metrics.csv"
    with open(per_class_path, "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for label, stats in metrics.per_class.items():
            f.write(f"{label},{stats['precision']},{stats['recall']},{stats['f1']},{stats['support']}\n")

    summary = {
        "students": args.students,
        "sessions_per_student": args.sessions,
        "record_count": len(dataset.records),
        "model": args.model,
        "split_mode": args.split_mode,
        "dataset_fingerprint": dataset.manifest.config_fingerprint,
        "bas_fingerprint": bas.config_fingerprint,
        "reward_fingerprint": reward.config_fingerprint,
        "intervention_fingerprint": intervention.config_fingerprint,
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.f1_macro,
        "roc_auc_ovr": metrics.roc_auc_ovr,
        "log_loss": extra_metrics["log_loss"],
        "cohens_kappa": extra_metrics["cohens_kappa"],
        "matthews_corrcoef": extra_metrics["matthews_corrcoef"],
        "intervention_rate": intervention.statistics.intervention_rate,
        "bas_average_score": bas.statistics.average_score,
    }
    with open(REPORT_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_md = [
        "# BAS Experiment Summary",
        "",
        "Generated by `run_experiment.py`. Every figure and metric below is computed "
        "from this run's own artifacts and is reproducible from the recorded seed/config "
        "fingerprints — see `docs/RL_FORMALIZATION.md` and `docs/EVALUATION.md` for the "
        "underlying formulas.",
        "",
        "## Run configuration",
        "",
        f"- **students**: {args.students}",
        f"- **sessions_per_student**: {args.sessions}",
        f"- **record_count**: {len(dataset.records)}",
        f"- **model**: {args.model}",
        f"- **split_mode**: {args.split_mode}",
        f"- **dataset_fingerprint**: `{dataset.manifest.config_fingerprint}`",
        f"- **bas_fingerprint**: `{bas.config_fingerprint}`",
        f"- **reward_fingerprint**: `{reward.config_fingerprint}`",
        f"- **intervention_fingerprint**: `{intervention.config_fingerprint}`",
        "",
        "## Classifier performance",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name, value in metrics_rows:
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        summary_md.append(f"| {name} | {formatted} |")
    average_reward = sum(r.reward for r in reward.records) / len(reward.records)
    summary_md += [
        "",
        "**Note:** the classifier is trained on synthetic, simulator-generated labels. "
        "These metrics describe how well the model recovers the generator's own "
        "attention-state assignment rule, not real-world diagnostic accuracy — see "
        "`README.md`'s research-software disclaimer.",
        "",
        "## Behavioral Attention Score",
        "",
        f"- **mean**: {bas.statistics.average_score:.4f}",
        f"- **average_confidence**: {bas.statistics.average_confidence:.4f}",
        "",
        "## Reward",
        "",
        f"- **average_reward**: {average_reward:.4f}",
        "",
        "## Intervention",
        "",
        f"- **intervention_rate**: {intervention.statistics.intervention_rate:.4f}",
        f"- **policy_distribution**: {dict(intervention.statistics.policy_distribution)}",
        "",
        "## Figures",
        "",
    ]
    for fig_name in sorted(p.name for p in FIG_DIR.glob("*.png")):
        summary_md.append(f"- `figures/{fig_name}`")
    summary_md.append("")

    with open(REPORT_DIR / "experiment_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md))

    print("[7/7] Done.")
    print(f"\nResults written under {OUT.resolve()}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()