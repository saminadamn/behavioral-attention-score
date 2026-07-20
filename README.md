# Behavioral Attention Score (BAS)

A synthetic classroom simulation for evaluating attention scoring and tutoring interventions.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/saminadamn/behavioral-attention-score)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/saminadamn/behavioral-attention-score)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](https://github.com/saminadamn/behavioral-attention-score)

> **Research software.** This project works entirely with synthetic classroom data. It does not model or diagnose any real individual or clinical condition.

---

## Overview

This project simulates classroom learning sessions using synthetic students with different attention patterns.

For every interaction, it computes a Behavioral Attention Score (BAS), estimates a reward, and decides whether a tutoring intervention should be given. The pipeline is deterministic and reproducible: a given seed always produces the same dataset, scores, and decisions.

It was built as the codebase behind a B.Tech thesis on adaptive tutoring for ADHD-aware education.

## Motivation

Real classroom data is hard to collect because of privacy and ethical constraints. This project generates synthetic classroom interactions so attention-scoring and intervention strategies can be developed and evaluated before real data is available. Every generative assumption lives in a versioned config, so any dataset can be regenerated exactly from its seed.

## Key Features

- **Reproducible pipeline** — a single seed regenerates an identical dataset, BAS trajectory, reward trajectory, and intervention plan.
- **Five student archetypes** — Consistently Focused, Gradually Fatigued, Highly Distractible, Highly Impulsive, and Recovering Learner, each implemented as its own strategy class.
- **Attention score with temporal smoothing** — feature extraction, normalization, and evidence combination are separate, inspectable stages rather than one opaque formula.
- **Decomposed reward model** — performance, behavior, and intervention cost are tracked separately so each can be analyzed or ablated independently.
- **Rule-based intervention engine** — eight policies (hints, concept review, difficulty reduction, motivational prompts, breaks, encouragement, question reframing, no intervention), each with a traceable trigger reason. Not a trained RL policy.
- **LangGraph orchestration** — the pipeline runs as a checkpointable, resumable workflow.
- **Stress-tested** with datasets of 100,000+ interactions, alongside a full unit and integration test suite.

## What's here

- Synthetic classroom simulator
- Attention scoring engine (BAS)
- Rule-based intervention engine
- Reproducible evaluation pipeline
- LangGraph-based orchestration layer

## Architecture

```
Config → Students → Prompts → Responses → Behavior → Sessions → Dataset
                                                                    ↓
                                              Classifier ←→ BAS → Reward → Intervention → LangGraph
```

Each step is a deterministic function call over the dataset. LangGraph only adds control flow (routing, looping, checkpointing) on top — it doesn't compute anything itself. Full diagrams are in `docs/ARCHITECTURE.md`.

## Repository Structure

```
dataset_generator/
    config/          # GeneratorConfig, defaults, fingerprinting
    models/          # Student, Prompt, Response, Session, Dataset schemas
    generators/       # Student, prompt, response, behavior, session generation
    classifier/       # Attention-state classifier
    bas/              # Behavioral Attention Score engine
    reward/           # Reward model
    intervention/      # Intervention engine
    orchestration/     # LangGraph workflow

tests/    # Unit, integration, and stress tests
docs/     # Architecture, pipeline, orchestration, API, testing, design notes
```

## Installation

```bash
git clone https://github.com/saminadamn/behavioral-attention-score.git
cd behavioral-attention-score
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. Core dependencies: `pydantic`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `langgraph`.

## Quick Start

```bash
pytest -q                   # run the full test suite
pytest -q -k "not stress"   # skip the 100k+ stress tests
```

## Example Usage

Run the full pipeline through the orchestration layer:

```python
from dataset_generator.orchestration import build_graph, compile_graph, new_workflow_state

graph = build_graph(student_count=5, sessions_per_student=2)
compiled = compile_graph(graph)
result = compiled.invoke(new_workflow_state())

print(f"Sessions processed: {len(result['session_outputs'])}")
print(f"Tutor actions generated: {len(result['tutor_actions'])}")
```

Or call each engine directly:

```python
from dataset_generator.config import default_config
from dataset_generator.generators import generate_students, generate_sessions
from dataset_generator.pipeline import build_dataset_artifact
from dataset_generator.bas import BASEngine
from dataset_generator.reward import RewardEngine
from dataset_generator.intervention import InterventionPlanner
from dataset_generator.utils import build_rng_streams

config = default_config()
streams = build_rng_streams(config.seed)
students = generate_students(config, streams)[:5]
sessions = generate_sessions(config, students, sessions_per_student=2,
                              rng_streams=build_rng_streams(config.seed))

dataset = build_dataset_artifact(config, students, sessions)
bas = BASEngine().compute(dataset)
reward = RewardEngine().compute(dataset, bas)
intervention = InterventionPlanner().plan(dataset, bas, reward)

print(intervention.statistics.intervention_rate)
```

## Configuration

Every parameter lives in a single typed `GeneratorConfig` (`dataset_generator/config/schema.py`). `default_config()` returns a ready-to-use instance, and every section — student profiles, prompt curriculum, response generation, behavior distributions, attention-state transitions — can be overridden independently. Each config has a fingerprint (SHA-256 hash) stored in downstream artifacts so a dataset's provenance is always traceable.

## Testing

One test module per package, covering unit behavior, end-to-end integration, determinism checks, and 100,000+-scale stress tests.

```bash
pytest -q                              # everything
pytest -q -k "not stress"              # fast path
pytest tests/test_orchestration.py -v  # orchestration only
```

See `docs/TESTING.md` for full coverage details and current runtimes.

## Documentation

- `docs/ARCHITECTURE.md` — per-module design and tradeoffs
- `docs/PIPELINE.md` — end-to-end data flow
- `docs/ORCHESTRATION.md` — LangGraph workflow, checkpointing, and recovery
- `docs/API.md` — public API reference
- `docs/TESTING.md` — test philosophy and coverage
- `docs/DESIGN_DECISIONS.md` — engineering decisions and alternatives considered

## Roadmap

- Public dataset release (CSV/JSONL/Parquet + manifest)
- Persistent (SQLite-backed) checkpointing for cross-process resume
- Lightweight dashboard over intervention and workflow reports
- Expanded ablation-study tooling

## License

TBD — to be finalized before public release.

## Acknowledgements

Built as part of a B.Tech thesis project. Orchestration built on [LangGraph](https://github.com/langchain-ai/langgraph); statistical modeling via `numpy`/`scipy`; classification via `scikit-learn`; data modeling via `pydantic`.
