# Reproducibility Materials for TORS-2026-0062

Manuscript: `Popularity Calibration of Counterfactual Explanations in Recommender Systems: A Shared-Backbone Benchmark and Group-wise Evaluation Study`

This package is prepared for ACM TORS reviewer access. It supports auditing the reported aggregate tables, regenerating the manuscript figures, and inspecting the benchmark/metric scripts used in the study.

## Contents

- `results/`: aggregate CSV/JSON summaries, per-seed summaries, group-wise tables, repair-effect tables, M20/S10 candidate-pool sensitivity outputs, and raw-popularity Wasserstein/mean-shift diagnostic outputs.
- `figure_inputs/`: lightweight CSV inputs used to regenerate the three manuscript figures.
- `scripts/`: aggregation scripts, figure-generation script, raw-popularity diagnostic script, candidate-table script, and benchmark execution scripts used for the shared-backbone experiments.
- `docs/data_and_code_availability.txt`: data/code availability statement and dataset-access notes.

## What Can Be Reproduced Immediately

From the package root, the manuscript figures can be regenerated with:

```bash
python scripts/generate_paper_figures.py --input-dir figure_inputs --output-dir generated_figures
```

The aggregate tables can be audited directly from the CSV/Markdown files under `results/`. For example:

- Main benchmark tables: `results/main_mlp_*`
- VAPC repair tables: `results/vapc_*`
- MC-Shapley breadth checks: `results/breadth_*_mc_shapley_*`
- M20/S10 protocol sensitivity: `results/protocol_sensitivity_*`
- Raw popularity diagnostic: `results/raw_popularity_*`

## Full Benchmark Reruns

Full end-to-end reruns require the public source datasets and local preprocessing paths described in the manuscript. The raw ML1M, Amazon Books, and Yelp assets are not redistributed here. The benchmark scripts are included for transparency:

- `scripts/run_ml1m_mlp_cross_explainer_smoke.py`
- `scripts/run_ml1m_neumf_cross_explainer_gate.py`
- `scripts/run_protocol_sensitivity_m20_s10.sh`
- `scripts/prepare_ml1m_shared_domain.py`
- `scripts/prepare_amazon_books_dense_subset.py`
- `scripts/prepare_yelp_dense_subset.py`

The scripts use explicit command-line arguments for data directories, seeds, candidate-pool size, and output locations. Some default paths reflect the authors' local execution environment and should be replaced by the reviewer's local dataset paths when rerunning.

## Minimal Python Environment

For figure regeneration and aggregate-table inspection:

```bash
python>=3.10
numpy
pandas
matplotlib
```

For full benchmark reruns:

```bash
torch
numpy
pandas
matplotlib
```

GPU acceleration is optional but recommended for full reruns.

## Scope Note

This package is intended to support review-time reproducibility for a benchmark/evaluation paper. It contains the aggregate evidence used in the manuscript and scripts for auditing metrics, figures, repair summaries, sensitivity checks, and raw-popularity diagnostics. It does not redistribute third-party raw datasets.
