"""Export real pipeline artifacts as one compact JSON for the web dashboard.

Runs the actual engines (ObserverAgent -> BASEngine -> RewardEngine ->
InterventionPlanner) at a small, fixed scale and writes
`web/data/dashboard.json` — the dashboard page renders ONLY what this
script computed; nothing on that page is simulated in the browser.

Re-run after any engine change, commit the JSON, and the deployed
dashboard updates with the next push.

Usage:
    python scripts/export_dashboard_data.py [--students 8] [--sessions 2]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dataset_generator.config import default_config
from dataset_generator.evaluation.metrics import (
    compute_bas_metrics,
    compute_intervention_metrics,
    compute_reward_metrics,
)
from dataset_generator.orchestration import BASAgent, InterventionAgent, ObserverAgent, RewardAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "web" / "data" / "dashboard.json"

MAX_SESSIONS_CHARTED = 6  # keep the page light; all sessions still summarized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=2)
    args = parser.parse_args()

    config = default_config()
    dataset = ObserverAgent().generate(student_count=args.students, sessions_per_student=args.sessions)
    bas = BASAgent().compute(dataset)
    reward = RewardAgent().compute(dataset, bas)
    intervention = InterventionAgent().plan(dataset, bas, reward)

    bas_metrics = compute_bas_metrics(bas)
    reward_metrics = compute_reward_metrics(reward)
    intervention_metrics = compute_intervention_metrics(intervention)

    session_ids = sorted({r.session_id for r in dataset.records})
    charted = session_ids[:MAX_SESSIONS_CHARTED]

    bas_by_session = {}
    for sid in charted:
        records = sorted(
            (r for r in bas.records if r.session_id == sid), key=lambda r: r.interaction_number
        )
        bas_by_session[sid] = {
            "interactions": [r.interaction_number for r in records],
            "scores": [round(r.score, 4) for r in records],
            "confidence": [round(r.confidence, 4) for r in records],
        }

    # Reward decomposition for the first charted session, per interaction.
    first = charted[0]
    reward_records = sorted(
        (r for r in reward.records if r.session_id == first), key=lambda r: r.interaction_number
    )
    reward_series = {
        "session_id": first,
        "interactions": [r.interaction_number for r in reward_records],
        "performance": [round(r.performance_reward, 4) for r in reward_records],
        "behaviour": [round(r.behaviour_reward, 4) for r in reward_records],
        "cost": [round(r.cost_reward, 4) for r in reward_records],
        "total": [round(r.reward, 4) for r in reward_records],
    }

    executed_decisions = [
        {
            "session_id": d.session_id,
            "interaction": d.interaction_number,
            "policy": d.chosen_policy,
            "reason": d.chosen_reason,
        }
        for d in intervention.decisions
        if d.chosen_policy != "NoInterventionPolicy" and not d.cooldown_suppressed
    ][:12]

    session_table = [
        {
            "session_id": s.session_id,
            "student_id": s.student_id,
            "interactions": s.interaction_count,
            "interventions": s.intervention_count,
            "avg_confidence": round(s.average_confidence, 3),
            "cooldown_suppressions": s.cooldown_suppressions,
        }
        for s in intervention.session_summaries
    ]

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": config.seed,
            "students": args.students,
            "sessions_per_student": args.sessions,
            "record_count": len(dataset.records),
            "dataset_fingerprint": dataset.manifest.config_fingerprint,
            "bas_fingerprint": bas.config_fingerprint,
            "reward_fingerprint": reward.config_fingerprint,
            "intervention_fingerprint": intervention.config_fingerprint,
        },
        "headline": {
            "bas_mean": round(bas_metrics.mean, 4),
            "bas_volatility": round(bas_metrics.volatility, 4),
            "bas_recovery_rate": round(bas_metrics.recovery_rate, 4),
            "reward_average": round(reward_metrics.average_reward, 4),
            "reward_positive_ratio": round(reward_metrics.positive_ratio, 4),
            "executed_interventions": intervention_metrics.executed_intervention_count,
            "cooldown_activations": intervention_metrics.cooldown_activations,
            "avg_intervention_spacing": round(intervention_metrics.average_intervention_spacing, 2),
            "policy_diversity": round(intervention_metrics.policy_diversity, 4),
        },
        "bas_by_session": bas_by_session,
        "reward_series": reward_series,
        "reward_decomposition_avg": {
            k: round(v, 4) for k, v in reward_metrics.reward_decomposition.items()
        },
        "policy_distribution": {
            k: round(v, 4) for k, v in intervention_metrics.policy_frequencies.items()
        },
        "executed_decisions": executed_decisions,
        "session_table": session_table,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  records={len(dataset.records)} sessions={len(session_ids)} charted={len(charted)}")


if __name__ == "__main__":
    main()
