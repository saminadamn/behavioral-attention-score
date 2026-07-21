#!/usr/bin/env python
"""Phases 1-4 of the thesis comparison study.

    python run_comparison_study.py
    python run_comparison_study.py --students 40 --sessions 3

Writes `outputs_comparison/<timestamp>/`:

    reproducibility.json         Phase 1: seed, config, package versions, timestamp
    baseline_policy_table.csv    Phase 2: Rule-Based vs No-Intervention vs Random
                                  (a REAL causal comparison — each policy actually
                                  drives session generation via the intervention_policy
                                  hook on SessionSimulator, not a reused static reward)
    rl_evaluation_table.csv      Phase 3: Rule-Based reference + Vanilla/Double/
                                  Double+PER/DRQN DQN variants + CQL/IQL/BCQ
                                  (offline metrics: loss, greedy/logged agreement,
                                  training time — NOT causal reward, see the caveat
                                  printed with the table and docs/OFFLINE_RL.md)
    classifier_comparison_table.csv   Phase 4: LogReg/RandomForest/MLP/LSTM
                                  (Transformer is future work, not implemented —
                                  see docs/PHASE_ROADMAP.md)
    models/                      every trained artifact, saved (Phase 1's
                                  "save trained models and experiment artifacts")

Everything here is orchestration over already-tested modules — no new
inference logic, no fabricated numbers. Where a requested metric doesn't
apply (e.g. "accuracy" for a policy that isn't a classifier), the table
says so explicitly rather than filling in a placeholder.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dataset_generator.bas import BASEngine
from dataset_generator.classifier import AttentionClassifierTrainer, TrainingConfig
from dataset_generator.classifier.sequence_model import train_lstm_classifier
from dataset_generator.classifier.serialization import save_training_artifact
from dataset_generator.config import default_config
from dataset_generator.experiment_reproducibility import build_reproducibility_report, set_global_seed
from dataset_generator.generators import generate_sessions, generate_students
from dataset_generator.intervention import InterventionPlanner
from dataset_generator.pipeline import build_dataset_artifact
from dataset_generator.reward import RewardEngine
from dataset_generator.rl_experimental import DQNConfig, DQNTrainer
from dataset_generator.rl_experimental.baselines import make_random_policy, no_intervention_policy, rule_based_policy
from dataset_generator.rl_experimental.offline import BCQConfig, BCQTrainer, CQLConfig, CQLTrainer, IQLConfig, IQLTrainer
from dataset_generator.utils import build_rng_streams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--students", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[run_comparison_study] {message}", flush=True)


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Phase 2 — Baseline Evaluation (a real causal comparison)
# --------------------------------------------------------------------------- #

def run_baseline_policy_comparison(config, students, args) -> list[dict]:
    _log("Phase 2: Rule-Based vs No-Intervention vs Random (live, causal comparison)")
    rows = []
    policies = [
        ("Rule-Based (generator heuristic)", rule_based_policy),
        ("No Intervention", no_intervention_policy),
        ("Random Policy", make_random_policy(config, seed=args.seed)),
    ]
    for name, policy in policies:
        start = time.perf_counter()
        sessions = generate_sessions(
            config, students, sessions_per_student=args.sessions,
            rng_streams=build_rng_streams(config.seed), intervention_policy=policy,
        )
        dataset = build_dataset_artifact(config, students, sessions)
        bas = BASEngine().compute(dataset)
        reward = RewardEngine().compute(dataset, bas)
        elapsed = time.perf_counter() - start

        n_interventions = sum(1 for r in dataset.records if r.intervention_applied)
        rows.append({
            "policy": name,
            "record_count": len(dataset.records),
            "interventions_applied": n_interventions,
            "intervention_rate": round(n_interventions / len(dataset.records), 4),
            "bas_mean": round(bas.statistics.average_score, 4),
            "reward_mean": round(reward.statistics.average_reward, 4),
            "accuracy": "N/A (not a classifier)",
            "precision": "N/A (not a classifier)",
            "recall": "N/A (not a classifier)",
            "f1_score": "N/A (not a classifier)",
            "generation_seconds": round(elapsed, 3),
        })
        _log(f"  {name}: interventions={n_interventions}/{len(dataset.records)}  "
             f"bas_mean={bas.statistics.average_score:.4f}  reward_mean={reward.statistics.average_reward:.4f}")
    return rows


# --------------------------------------------------------------------------- #
# Phase 3 — RL Evaluation (offline metrics; NOT causal reward — see caveat)
# --------------------------------------------------------------------------- #

def run_rl_evaluation(dataset, bas, reward, intervention, args, models_dir: Path) -> list[dict]:
    _log("Phase 3: Rule-Based reference + DQN family (Vanilla/Double/Double+PER/DRQN) + CQL/IQL/BCQ")
    rows: list[dict] = [{
        "model": "Rule-Based (InterventionPlanner)",
        "avg_reward_logged_dataset": round(reward.statistics.average_reward, 4),
        "loss": "-",
        "greedy_logged_agreement": "1.0000 (reference)",
        "training_seconds": "-",
    }]

    dqn_configs: list[tuple[str, DQNConfig]] = [
        ("Vanilla DQN", DQNConfig(seed=args.seed, epochs=8, batch_size=32, min_replay_size=32,
                                   use_double_dqn=False, use_prioritized_replay=False, sequence_length=1)),
        ("Double DQN", DQNConfig(seed=args.seed, epochs=8, batch_size=32, min_replay_size=32,
                                  use_double_dqn=True, use_prioritized_replay=False, sequence_length=1)),
        ("Double DQN + PER", DQNConfig(seed=args.seed, epochs=8, batch_size=32, min_replay_size=32,
                                        use_double_dqn=True, use_prioritized_replay=True, sequence_length=1)),
        ("DRQN (Double DQN + PER + LSTM)", DQNConfig(seed=args.seed, epochs=8, batch_size=32, min_replay_size=32,
                                                       use_double_dqn=True, use_prioritized_replay=True, sequence_length=5)),
    ]
    for name, config in dqn_configs:
        start = time.perf_counter()
        artifact = DQNTrainer(config).train(dataset, bas, reward, intervention)
        elapsed = time.perf_counter() - start
        rows.append({
            "model": name,
            "avg_reward_logged_dataset": round(reward.statistics.average_reward, 4),
            "loss": round(artifact.loss_per_epoch[-1], 4),
            "greedy_logged_agreement": round(artifact.greedy_policy_agreement_rate, 4),
            "training_seconds": round(elapsed, 3),
        })
        _log(f"  {name}: final_loss={artifact.loss_per_epoch[-1]:.4f}  "
             f"agreement={artifact.greedy_policy_agreement_rate:.4f}  time={elapsed:.2f}s")

    offline_runs = [
        ("CQL (alpha=1.0)", CQLTrainer, CQLConfig(seed=args.seed, epochs=8, batch_size=32, min_replay_size=32)),
        ("IQL (tau=0.7)", IQLTrainer, IQLConfig(seed=args.seed, epochs=8, batch_size=32, min_replay_size=32)),
        ("Discrete BCQ (threshold=0.3)", BCQTrainer, BCQConfig(seed=args.seed, epochs=8, behavior_epochs=5, batch_size=32, min_replay_size=32)),
    ]
    for name, trainer_cls, offline_config in offline_runs:
        start = time.perf_counter()
        artifact = trainer_cls(offline_config).train(dataset, bas, reward, intervention)
        elapsed = time.perf_counter() - start
        agreement = getattr(artifact, "greedy_policy_agreement_rate", None)
        if agreement is None:
            agreement = artifact.constrained_greedy_agreement_rate
        loss = artifact.td_loss_per_epoch[-1] if hasattr(artifact, "td_loss_per_epoch") else (
            artifact.q_loss_per_epoch[-1]
        )
        rows.append({
            "model": name,
            "avg_reward_logged_dataset": round(reward.statistics.average_reward, 4),
            "loss": round(loss, 4),
            "greedy_logged_agreement": round(agreement, 4),
            "training_seconds": round(elapsed, 3),
        })
        _log(f"  {name}: final_loss={loss:.4f}  agreement={agreement:.4f}  time={elapsed:.2f}s")
        (models_dir / f"{trainer_cls.__name__}_artifact.json").write_text(
            artifact.model_dump_json(indent=2), encoding="utf-8"
        )

    return rows


# --------------------------------------------------------------------------- #
# Phase 4 — Deep Learning Comparison (classifiers)
# --------------------------------------------------------------------------- #

def run_classifier_comparison(dataset, args, models_dir: Path) -> list[dict]:
    _log("Phase 4: Logistic Regression / Random Forest / MLP / LSTM classifier comparison")
    rows = []

    for model_name in ("logistic_regression", "random_forest", "mlp"):
        config = TrainingConfig(model_name=model_name, random_state=args.seed, compute_feature_importance=False)
        start = time.perf_counter()
        artifact = AttentionClassifierTrainer().train(dataset, config)
        elapsed = time.perf_counter() - start
        metrics = artifact.metrics
        rows.append({
            "model": model_name,
            "accuracy": round(metrics.accuracy, 4),
            "precision_macro": round(metrics.precision_macro, 4),
            "recall_macro": round(metrics.recall_macro, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "training_seconds": round(elapsed, 3),
        })
        save_training_artifact(artifact, models_dir / f"classifier_{model_name}")
        _log(f"  {model_name}: accuracy={metrics.accuracy:.4f}  f1_macro={metrics.f1_macro:.4f}  time={elapsed:.2f}s")

    start = time.perf_counter()
    lstm_result = train_lstm_classifier(dataset, random_state=args.seed)
    elapsed = time.perf_counter() - start
    rows.append({
        "model": "lstm",
        "accuracy": round(lstm_result.accuracy, 4),
        "precision_macro": round(lstm_result.precision_macro, 4),
        "recall_macro": round(lstm_result.recall_macro, 4),
        "f1_macro": round(lstm_result.f1_macro, 4),
        "training_seconds": round(elapsed, 3),
    })
    _log(f"  lstm: accuracy={lstm_result.accuracy:.4f}  f1_macro={lstm_result.f1_macro:.4f}  time={elapsed:.2f}s")

    rows.append({
        "model": "transformer",
        "accuracy": "not implemented (future work — docs/PHASE_ROADMAP.md)",
        "precision_macro": "-", "recall_macro": "-", "f1_macro": "-", "training_seconds": "-",
    })
    return rows


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs_comparison") / timestamp
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Output directory: {output_dir}")

    config = default_config()
    students = generate_students(config, build_rng_streams(config.seed))[: args.students]

    # -- Phase 1: reproducibility report --
    config_summary = {
        "students": args.students, "sessions_per_student": args.sessions,
        "seed": args.seed, "dataset_seed": config.seed,
    }
    reproducibility = build_reproducibility_report(args.seed, config_summary)
    (output_dir / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    _log("Phase 1: reproducibility.json written (seed, config, package versions, timestamp)")

    # -- Phase 2 --
    baseline_rows = run_baseline_policy_comparison(config, students, args)
    _write_csv(baseline_rows, output_dir / "baseline_policy_table.csv")

    # -- Build the reference (rule-based) dataset once, reused for Phases 3 & 4 --
    sessions = generate_sessions(
        config, students, sessions_per_student=args.sessions, rng_streams=build_rng_streams(config.seed),
    )
    dataset = build_dataset_artifact(config, students, sessions)
    bas = BASEngine().compute(dataset)
    reward = RewardEngine().compute(dataset, bas)
    intervention = InterventionPlanner().plan(dataset, bas, reward)

    # -- Phase 3 --
    rl_rows = run_rl_evaluation(dataset, bas, reward, intervention, args, models_dir)
    _write_csv(rl_rows, output_dir / "rl_evaluation_table.csv")
    _log("Phase 3 caveat: avg_reward_logged_dataset is the SAME observed reward for every "
         "row (none of these algorithms regenerates data) - it is not a causal per-policy "
         "reward. loss and greedy_logged_agreement are the real, per-algorithm metrics; see "
         "docs/OFFLINE_RL.md.")

    # -- Phase 4 --
    classifier_rows = run_classifier_comparison(dataset, args, models_dir)
    _write_csv(classifier_rows, output_dir / "classifier_comparison_table.csv")

    _log("Done.")
    _log(f"Results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
