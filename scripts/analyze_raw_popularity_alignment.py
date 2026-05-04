#!/usr/bin/env python3
"""Compute raw-popularity alignment diagnostics from saved per-user outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_ml1m_mlp_lxr_smoke import load_dense_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-filename", type=str, required=True)
    parser.add_argument("--static-test-filename", type=str, required=True)
    parser.add_argument("--dataset-label", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Exact W1 distance between two 1D empirical distributions."""
    left = np.sort(np.asarray(left, dtype=np.float64))
    right = np.sort(np.asarray(right, dtype=np.float64))
    if left.size == 0 or right.size == 0:
        return float("nan")

    all_values = np.sort(np.concatenate([left, right]))
    if all_values.size <= 1:
        return 0.0

    deltas = np.diff(all_values)
    grid = all_values[:-1]
    left_cdf = np.searchsorted(left, grid, side="right") / left.size
    right_cdf = np.searchsorted(right, grid, side="right") / right.size
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def parse_items(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        items = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [int(item) for item in items]


def format_mean_std(mean: float, std: float | None, digits: int = 4) -> str:
    if pd.isna(mean):
        return "NA"
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dense_dataset(
        data_dir=args.data_dir,
        train_filename=args.train_filename,
        static_test_filename=args.static_test_filename,
        train_user_limit=0,
        test_user_limit=0,
    )
    history_by_user = {
        int(user_id): np.flatnonzero(row > 0)
        for user_id, row in zip(data.test_user_ids, data.test_features)
    }

    rows: list[dict[str, object]] = []
    for run_dir in args.run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        seed = int(summary["config"]["seed"])
        per_user = pd.read_csv(run_dir / "per_user_all.csv")

        for record in per_user[per_user["valid_cf_found"]].to_dict(orient="records"):
            user_id = int(record["user_id"])
            history_items = history_by_user.get(user_id)
            explanation_items = parse_items(record.get("explanation_items"))
            if history_items is None or len(history_items) == 0 or not explanation_items:
                continue

            history_pop = data.item_pop_percentiles[history_items]
            explanation_pop = data.item_pop_percentiles[np.asarray(explanation_items, dtype=np.int64)]
            raw_shift = float(np.mean(explanation_pop) - np.mean(history_pop))
            rows.append(
                {
                    "dataset": args.dataset_label,
                    "seed": seed,
                    "explainer": record["explainer"],
                    "base_explainer": record.get("base_explainer"),
                    "repair_stage": record.get("repair_stage"),
                    "segment": record.get("segment"),
                    "user_id": user_id,
                    "history_size": int(len(history_items)),
                    "explanation_size": int(record["explanation_size"]),
                    "history_pop_mean": float(np.mean(history_pop)),
                    "explanation_pop_mean": float(np.mean(explanation_pop)),
                    "raw_pop_shift": raw_shift,
                    "abs_raw_pop_shift": abs(raw_shift),
                    "raw_wasserstein": wasserstein_1d(history_pop, explanation_pop),
                    "cf_pce": float(record.get("cf_pce", np.nan)),
                    "sps": float(record.get("sps", np.nan)),
                }
            )

    per_user_raw = pd.DataFrame(rows)
    if per_user_raw.empty:
        raise RuntimeError("No valid explanations found for raw-popularity analysis")

    per_seed = (
        per_user_raw.groupby(["dataset", "seed", "explainer"], as_index=False)
        .agg(
            users=("user_id", "count"),
            mean_abs_raw_pop_shift=("abs_raw_pop_shift", "mean"),
            mean_raw_pop_shift=("raw_pop_shift", "mean"),
            mean_raw_wasserstein=("raw_wasserstein", "mean"),
            mean_cf_pce=("cf_pce", "mean"),
            mean_sps=("sps", "mean"),
        )
        .sort_values(["dataset", "explainer", "seed"])
        .reset_index(drop=True)
    )
    aggregate = (
        per_seed.groupby(["dataset", "explainer"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            users_mean=("users", "mean"),
            raw_wasserstein_mean=("mean_raw_wasserstein", "mean"),
            raw_wasserstein_std=("mean_raw_wasserstein", "std"),
            abs_raw_shift_mean=("mean_abs_raw_pop_shift", "mean"),
            abs_raw_shift_std=("mean_abs_raw_pop_shift", "std"),
            raw_shift_mean=("mean_raw_pop_shift", "mean"),
            raw_shift_std=("mean_raw_pop_shift", "std"),
            cf_pce_mean=("mean_cf_pce", "mean"),
            cf_pce_std=("mean_cf_pce", "std"),
            sps_mean=("mean_sps", "mean"),
            sps_std=("mean_sps", "std"),
        )
        .sort_values(["dataset", "explainer"])
        .reset_index(drop=True)
    )

    aggregate_fmt = pd.DataFrame(
        {
            "dataset": aggregate["dataset"],
            "explainer": aggregate["explainer"],
            "seeds": aggregate["seeds"],
            "users/seed": aggregate["users_mean"].round(1),
            "Raw W1": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["raw_wasserstein_mean"], aggregate["raw_wasserstein_std"])
            ],
            "Abs raw shift": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["abs_raw_shift_mean"], aggregate["abs_raw_shift_std"])
            ],
            "Signed raw shift": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["raw_shift_mean"], aggregate["raw_shift_std"])
            ],
            "CF-PCE": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["cf_pce_mean"], aggregate["cf_pce_std"])
            ],
            "SPS": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["sps_mean"], aggregate["sps_std"])
            ],
        }
    )

    per_seed.to_csv(args.output_dir / "per_seed_raw_popularity.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_raw_popularity_numeric.csv", index=False)
    aggregate_fmt.to_csv(args.output_dir / "aggregate_raw_popularity.csv", index=False)
    (args.output_dir / "aggregate_raw_popularity.md").write_text(markdown_table(aggregate_fmt))
    (args.output_dir / "manifest_raw_popularity.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset_label,
                "diagnostic": "raw_popularity_alignment",
                "inputs": [str(path) for path in args.run_dirs],
                "outputs": [
                    "per_seed_raw_popularity.csv",
                    "aggregate_raw_popularity_numeric.csv",
                    "aggregate_raw_popularity.csv",
                    "aggregate_raw_popularity.md",
                ],
            },
            indent=2,
        )
        + "\n"
    )

    print(aggregate_fmt.to_string(index=False))


if __name__ == "__main__":
    main()
