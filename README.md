# KBS AI Recommender for Entrepreneurial SMEs

A compact, reproducible proof of concept for the short communication:
**Knowledge-Based AI Recommender System for Data-Driven Digital Transformation and Innovation in Entrepreneurial SMEs**.

The system combines auditable knowledge rules with a learned implementation-success model to rank six digital-transformation actions. It creates a controlled synthetic benchmark solely to demonstrate reproducibility and evaluation. It is **not** evidence from real SMEs; replace the simulator with ethically governed, shareable SME outcome data before making field-effectiveness claims.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 run_experiment.py
```

Generated outputs are written to `results/`:

- `summary.json` - aggregate Precision@3, NDCG@3, and catalogue coverage
- `benchmark_runs.csv` - five deterministic simulation runs
- `figures/` - architecture and benchmark figures used by the manuscript

## Design

1. **Knowledge layer** encodes capability-action relationships and explanation traces.
2. **Data layer** learns pairwise implementation-success probabilities from prior outcomes.
3. **Hybrid ranker** combines both scores: `0.55 × knowledge fit + 0.45 × learned probability`.

The dependency-free model deliberately favors auditability over scale. In a field deployment, learn rule weights with domain experts, version the ontology, apply privacy controls, and evaluate on a held-out real-world cohort.
