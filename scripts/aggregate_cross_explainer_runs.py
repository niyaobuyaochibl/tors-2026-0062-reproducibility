#!/usr/bin/env python3
"""Aggregate saved cross-explainer smoke summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in args.run_dirs:
        summary_path = run_dir / "summary.json"
        payload = json.loads(summary_path.read_text())
        seed = payload["config"]["seed"]
        dataset_label = payload.get("dataset_label", payload["config"].get("dataset_label"))
        backbone_label = payload.get("backbone_label", "mlp")
        recommender_metrics = payload["recommender_metrics"]
        for explainer, stats in payload["explainer_summaries"].items():
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "seed": seed,
                    "dataset_label": dataset_label,
                    "backbone_label": backbone_label,
                    "explainer": explainer,
                    "hr10": recommender_metrics["hr10"],
                    "hr50": recommender_metrics["hr50"],
                    "hr100": recommender_metrics["hr100"],
                    "mrr": recommender_metrics["mrr"],
                    "mpr": recommender_metrics["mpr"],
                    "eval_users": stats["eval_users"],
                    "validity": stats["validity"],
                    "no_cf_found_rate": stats["no_cf_found_rate"],
                    "mean_explanation_size": stats["mean_explanation_size"],
                    "mean_score_drop": stats["mean_score_drop"],
                    "mean_cf_pce": stats["mean_cf_pce"],
                    "mean_sps": stats["mean_sps"],
                    "mean_minimality_gap": stats.get("mean_minimality_gap"),
                    "median_minimality_gap": stats.get("median_minimality_gap"),
                    "repair_changed_rate": stats.get("repair_changed_rate"),
                    "mean_per_user_elapsed_seconds": stats.get("mean_per_user_elapsed_seconds"),
                    "median_per_user_elapsed_seconds": stats.get("median_per_user_elapsed_seconds"),
                    "mean_ranking_seconds": stats.get("mean_ranking_seconds"),
                    "mean_candidate_eval_seconds": stats.get("mean_candidate_eval_seconds"),
                    "mean_vapc_scoring_seconds": stats.get("mean_vapc_scoring_seconds"),
                }
            )

    per_seed = (
        pd.DataFrame(rows)
        .sort_values(["dataset_label", "backbone_label", "explainer", "seed"])
        .reset_index(drop=True)
    )
    agg = (
        per_seed.groupby(["dataset_label", "backbone_label", "explainer"], as_index=False)
        .agg(
            runs=("seed", "count"),
            hr10_mean=("hr10", "mean"),
            hr10_std=("hr10", "std"),
            hr50_mean=("hr50", "mean"),
            hr50_std=("hr50", "std"),
            hr100_mean=("hr100", "mean"),
            hr100_std=("hr100", "std"),
            mrr_mean=("mrr", "mean"),
            mrr_std=("mrr", "std"),
            validity_mean=("validity", "mean"),
            validity_std=("validity", "std"),
            no_cf_found_mean=("no_cf_found_rate", "mean"),
            no_cf_found_std=("no_cf_found_rate", "std"),
            mean_explanation_size_mean=("mean_explanation_size", "mean"),
            mean_explanation_size_std=("mean_explanation_size", "std"),
            mean_score_drop_mean=("mean_score_drop", "mean"),
            mean_score_drop_std=("mean_score_drop", "std"),
            mean_cf_pce_mean=("mean_cf_pce", "mean"),
            mean_cf_pce_std=("mean_cf_pce", "std"),
            mean_sps_mean=("mean_sps", "mean"),
            mean_sps_std=("mean_sps", "std"),
            mean_minimality_gap_mean=("mean_minimality_gap", "mean"),
            mean_minimality_gap_std=("mean_minimality_gap", "std"),
            repair_changed_rate_mean=("repair_changed_rate", "mean"),
            repair_changed_rate_std=("repair_changed_rate", "std"),
            mean_per_user_elapsed_seconds_mean=("mean_per_user_elapsed_seconds", "mean"),
            mean_per_user_elapsed_seconds_std=("mean_per_user_elapsed_seconds", "std"),
            mean_ranking_seconds_mean=("mean_ranking_seconds", "mean"),
            mean_ranking_seconds_std=("mean_ranking_seconds", "std"),
            mean_candidate_eval_seconds_mean=("mean_candidate_eval_seconds", "mean"),
            mean_candidate_eval_seconds_std=("mean_candidate_eval_seconds", "std"),
            mean_vapc_scoring_seconds_mean=("mean_vapc_scoring_seconds", "mean"),
            mean_vapc_scoring_seconds_std=("mean_vapc_scoring_seconds", "std"),
        )
    )

    per_seed.to_csv(args.output_dir / "per_seed.csv", index=False)
    agg.to_csv(args.output_dir / "aggregate.csv", index=False)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(agg.to_dict(orient="records"), indent=2) + "\n"
    )

    print(per_seed.to_string(index=False))
    print()
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
