# Behavioral Attention Score (BAS) — Synthetic Dataset Generator

Research software for a B.Tech thesis project: *"Behavioral Attention Score (BAS):
An Explainable Multi-Agent Reinforcement Learning Framework for Adaptive
Educational Support in ADHD."*

This repository contains the **synthetic dataset generator** only — a modular,
reproducible, configuration-driven simulator that produces synthetic classroom
interaction data (prompts, student responses, and behavioural signals) for
evaluating a Behavioural Attention Score pipeline.

**This is synthetic research data.** It does not represent, model, or
diagnose any real individual or clinical condition, and must not be used or
interpreted as a diagnostic tool.

## Status

Actively under development. Implemented so far:

- **Configuration system** (`dataset_generator/config/`) — a single typed,
  validated `GeneratorConfig` drives every other module. No generation
  parameter is hardcoded outside this package.
- **Student Profile Generator** — five behavioural archetypes
  (Consistently Focused, Gradually Fatigued, Highly Distractible, Highly
  Impulsive, Recovering Learner), each a registered `ResponseStrategy`/profile
  class rather than an `if`/`else` chain.
- **Prompt Generator** — a 7-subject curriculum with Bloom's-taxonomy /
  difficulty-conditioned templated prompts, plus a validation report.
- **Response Generator** — attention-state-conditioned synthetic student
  responses, with every feature (lexical diversity, sentiment, semantic
  similarity, engagement, etc.) computed from the generated text itself.
- **Behaviour Generator** — samples response latency, hesitation, and
  interaction duration from configurable statistical distributions
  (Normal/Gamma/Beta/Poisson/Truncated-Normal), personalized to each
  student's profile and evolved through session fatigue/intervention effects.

Not yet implemented: temporal/session simulation (Markov attention-state
transitions), the Behavioural Attention Score itself, the attention
classifier, reinforcement-learning intervention policy, LangGraph
orchestration, and the dashboard.

## Running the tests

```bash
python -m venv .venv
.venv/Scripts/activate  # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pytest -q
```
