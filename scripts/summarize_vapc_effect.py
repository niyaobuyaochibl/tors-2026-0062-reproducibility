#!/usr/bin/env python3
"""Summarize base-vs-VAPC repair effects from saved cross-explainer runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser.parse_args()


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def format_mean_std(mean: float, std: float | None, digits: int = 4) -> str:
    if pd.isna(mean):
        return "NA"
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, object]] = []
    for run_dir in args.run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        per_user = pd.read_csv(run_dir / "per_user_all.csv")
        if "repair_stage" not in per_user.columns or "base_explainer" not in per_user.columns:
            continue

        seed = int(summary["config"]["seed"])
        dataset = str(summary.get("dataset_label", "unknown"))

        for base_explainer in sorted(per_user["base_explainer"].dropna().unique()):
            base = per_user[
                (per_user["base_explainer"] == base_explainer) & (per_user["repair_stage"] == "base")
            ].copy()
            vapc = per_user[
                (per_user["base_explainer"] == base_explainer) & (per_user["repair_stage"] == "vapc")
            ].copy()
            if base.empty or vapc.empty:
                continue

            merged = base.merge(
                vapc,
                on="user_id",
                suffixes=("_base", "_vapc"),
            )
            both_valid = merged[merged["valid_cf_found_base"] & merged["valid_cf_found_vapc"]].copy()

            if both_valid.empty:
                cf_pce_improvement = None
                abs_sps_improvement = None
                win = 0
                tie = 0
                loss = 0
            else:
                cf_pce_delta = both_valid["cf_pce_base"] - both_valid["cf_pce_vapc"]
                abs_sps_delta = both_valid["sps_base"].abs() - both_valid["sps_vapc"].abs()
                cf_pce_improvement = float(cf_pce_delta.mean())
                abs_sps_improvement = float(abs_sps_delta.mean())
                tolerance = 1e-12
                win = int((cf_pce_delta > tolerance).sum())
                tie = int((cf_pce_delta.abs() <= tolerance).sum())
                loss = int((cf_pce_delta < -tolerance).sum())

            per_seed_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "base_explainer": base_explainer,
                    "users": int(len(merged)),
                    "paired_valid_users": int(len(both_valid)),
                    "base_validity": float(base["valid_cf_found"].mean()),
                    "vapc_validity": float(vapc["valid_cf_found"].mean()),
                    "validity_delta": float(vapc["valid_cf_found"].mean() - base["valid_cf_found"].mean()),
                    "base_no_cf_found": float(base["no_cf_found"].mean()),
                    "vapc_no_cf_found": float(vapc["no_cf_found"].mean()),
                    "no_cf_found_delta": float(vapc["no_cf_found"].mean() - base["no_cf_found"].mean()),
                    "cf_pce_improvement": cf_pce_improvement,
                    "abs_sps_improvement": abs_sps_improvement,
                    "mean_minimality_gap": float(vapc["minimality_gap"].dropna().mean())
                    if "minimality_gap" in vapc
                    else None,
                    "median_minimality_gap": float(vapc["minimality_gap"].dropna().median())
                    if "minimality_gap" in vapc
                    else None,
                    "repair_changed_rate": float(vapc["repair_changed"].mean())
                    if "repair_changed" in vapc
                    else None,
                    "win": win,
                    "tie": tie,
                    "loss": loss,
                }
            )

    per_seed = pd.DataFrame(per_seed_rows).sort_values(["dataset", "base_explainer", "seed"]).reset_index(drop=True)
    if per_seed.empty:
        raise SystemExit("No VAPC runs with base/vapc rows were found.")

    aggregate = (
        per_seed.groupby(["dataset", "base_explainer"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            users_mean=("users", "mean"),
            paired_valid_users_mean=("paired_valid_users", "mean"),
            validity_delta_mean=("validity_delta", "mean"),
            validity_delta_std=("validity_delta", "std"),
            no_cf_found_delta_mean=("no_cf_found_delta", "mean"),
            no_cf_found_delta_std=("no_cf_found_delta", "std"),
            cf_pce_improvement_mean=("cf_pce_improvement", "mean"),
            cf_pce_improvement_std=("cf_pce_improvement", "std"),
            abs_sps_improvement_mean=("abs_sps_improvement", "mean"),
            abs_sps_improvement_std=("abs_sps_improvement", "std"),
            minimality_gap_mean=("mean_minimality_gap", "mean"),
            minimality_gap_std=("mean_minimality_gap", "std"),
            repair_changed_rate_mean=("repair_changed_rate", "mean"),
            repair_changed_rate_std=("repair_changed_rate", "std"),
            win_mean=("win", "mean"),
            tie_mean=("tie", "mean"),
            loss_mean=("loss", "mean"),
        )
    )

    table3 = pd.DataFrame(
        {
            "dataset": aggregate["dataset"],
            "explainer": aggregate["base_explainer"],
            "seeds": aggregate["seeds"],
            "users/seed": aggregate["users_mean"].round(1),
            "paired valid/seed": aggregate["paired_valid_users_mean"].round(1),
            "CF-PCE improvement": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["cf_pce_improvement_mean"], aggregate["cf_pce_improvement_std"])
            ],
            "Abs SPS improvement": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["abs_sps_improvement_mean"], aggregate["abs_sps_improvement_std"])
            ],
            "Validity delta": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["validity_delta_mean"], aggregate["validity_delta_std"])
            ],
            "No-CF delta": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["no_cf_found_delta_mean"], aggregate["no_cf_found_delta_std"])
            ],
            "Minimality gap": [
                format_mean_std(m, s, digits=3)
                for m, s in zip(aggregate["minimality_gap_mean"], aggregate["minimality_gap_std"])
            ],
            "Repair changed": [
                format_mean_std(m, s)
                for m, s in zip(aggregate["repair_changed_rate_mean"], aggregate["repair_changed_rate_std"])
            ],
            "Win/Tie/Loss": [
                f"{w:.1f}/{t:.1f}/{l:.1f}"
                for w, t, l in zip(aggregate["win_mean"], aggregate["tie_mean"], aggregate["loss_mean"])
            ],
        }
    )

    per_seed.to_csv(args.output_dir / "per_seed_vapc_effect.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_vapc_effect_numeric.csv", index=False)
    table3.to_csv(args.output_dir / "table3_vapc_effect.csv", index=False)
    (args.output_dir / "table3_vapc_effect.md").write_text(markdown_table(table3))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_dirs": [str(path) for path in args.run_dirs],
                "outputs": {
                    "per_seed_csv": str(args.output_dir / "per_seed_vapc_effect.csv"),
                    "aggregate_numeric_csv": str(args.output_dir / "aggregate_vapc_effect_numeric.csv"),
                    "table3_csv": str(args.output_dir / "table3_vapc_effect.csv"),
                    "table3_md": str(args.output_dir / "table3_vapc_effect.md"),
                },
            },
            indent=2,
        )
        + "\n"
    )

    print(per_seed.to_string(index=False))
    print()
    print(table3.to_string(index=False))


if __name__ == "__main__":
    main()
