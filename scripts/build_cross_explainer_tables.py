#!/usr/bin/env python3
"""Build draft-ready tables from saved cross-explainer runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ML1M")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser.parse_args()


def format_mean_std(mean: float, std: float | None, digits: int = 4) -> str:
    if pd.isna(mean):
        return "NA"
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def load_per_user(run_dir: Path) -> pd.DataFrame:
    combined = run_dir / "per_user_all.csv"
    if combined.exists():
        return pd.read_csv(combined)

    frames = []
    for name in ["per_user_lxr.csv", "per_user_loo.csv"]:
        path = run_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No per-user files found under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    group_rows = []

    for run_dir in args.run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        seed = int(summary["config"]["seed"])
        recommender = summary["recommender_metrics"]
        per_user = load_per_user(run_dir)

        for explainer, stats in summary["explainer_summaries"].items():
            overall_rows.append(
                {
                    "dataset": args.dataset,
                    "seed": seed,
                    "explainer": explainer,
                    "hr10": recommender["hr10"],
                    "hr50": recommender["hr50"],
                    "hr100": recommender["hr100"],
                    "mrr": recommender["mrr"],
                    "mpr": recommender["mpr"],
                    "validity": stats["validity"],
                    "no_cf_found_rate": stats["no_cf_found_rate"],
                    "mean_explanation_size": stats["mean_explanation_size"],
                    "mean_score_drop": stats["mean_score_drop"],
                    "mean_cf_pce": stats["mean_cf_pce"],
                    "mean_sps": stats["mean_sps"],
                }
            )

        for (explainer, segment), frame in per_user.groupby(["explainer", "segment"]):
            valid = frame[frame["valid_cf_found"]].copy()
            group_rows.append(
                {
                    "dataset": args.dataset,
                    "seed": seed,
                    "explainer": explainer,
                    "segment": segment,
                    "users": int(len(frame)),
                    "validity": float(frame["valid_cf_found"].mean()),
                    "no_cf_found_rate": float(frame["no_cf_found"].mean()),
                    "mean_cf_pce": float(valid["cf_pce"].mean()) if not valid.empty else None,
                    "mean_sps": float(valid["sps"].mean()) if not valid.empty else None,
                    "history_pop_median": float(frame["history_pop_median"].mean()),
                }
            )

    per_seed_overall = pd.DataFrame(overall_rows).sort_values(["explainer", "seed"]).reset_index(drop=True)
    per_seed_group = pd.DataFrame(group_rows).sort_values(["explainer", "segment", "seed"]).reset_index(drop=True)

    table1 = (
        per_seed_overall.groupby(["dataset", "explainer"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            hr10_mean=("hr10", "mean"),
            hr10_std=("hr10", "std"),
            validity_mean=("validity", "mean"),
            validity_std=("validity", "std"),
            no_cf_found_mean=("no_cf_found_rate", "mean"),
            no_cf_found_std=("no_cf_found_rate", "std"),
            explanation_size_mean=("mean_explanation_size", "mean"),
            explanation_size_std=("mean_explanation_size", "std"),
            score_drop_mean=("mean_score_drop", "mean"),
            score_drop_std=("mean_score_drop", "std"),
            cf_pce_mean=("mean_cf_pce", "mean"),
            cf_pce_std=("mean_cf_pce", "std"),
            sps_mean=("mean_sps", "mean"),
            sps_std=("mean_sps", "std"),
        )
    )

    table1_fmt = pd.DataFrame(
        {
            "dataset": table1["dataset"],
            "explainer": table1["explainer"],
            "seeds": table1["seeds"],
            "HR@10": [format_mean_std(m, s) for m, s in zip(table1["hr10_mean"], table1["hr10_std"])],
            "Validity": [format_mean_std(m, s) for m, s in zip(table1["validity_mean"], table1["validity_std"])],
            "No-CF-found": [
                format_mean_std(m, s) for m, s in zip(table1["no_cf_found_mean"], table1["no_cf_found_std"])
            ],
            "Mean Expl Size": [
                format_mean_std(m, s, digits=3)
                for m, s in zip(table1["explanation_size_mean"], table1["explanation_size_std"])
            ],
            "Score Drop": [format_mean_std(m, s) for m, s in zip(table1["score_drop_mean"], table1["score_drop_std"])],
            "CF-PCE": [format_mean_std(m, s) for m, s in zip(table1["cf_pce_mean"], table1["cf_pce_std"])],
            "SPS": [format_mean_std(m, s) for m, s in zip(table1["sps_mean"], table1["sps_std"])],
        }
    )

    table2 = (
        per_seed_group.groupby(["dataset", "explainer", "segment"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            users_mean=("users", "mean"),
            validity_mean=("validity", "mean"),
            validity_std=("validity", "std"),
            no_cf_found_mean=("no_cf_found_rate", "mean"),
            no_cf_found_std=("no_cf_found_rate", "std"),
            cf_pce_mean=("mean_cf_pce", "mean"),
            cf_pce_std=("mean_cf_pce", "std"),
            sps_mean=("mean_sps", "mean"),
            sps_std=("mean_sps", "std"),
            history_pop_median_mean=("history_pop_median", "mean"),
        )
    )

    table2_fmt = pd.DataFrame(
        {
            "dataset": table2["dataset"],
            "explainer": table2["explainer"],
            "segment": table2["segment"],
            "users/seed": table2["users_mean"].round(1),
            "Validity": [format_mean_std(m, s) for m, s in zip(table2["validity_mean"], table2["validity_std"])],
            "No-CF-found": [
                format_mean_std(m, s) for m, s in zip(table2["no_cf_found_mean"], table2["no_cf_found_std"])
            ],
            "CF-PCE": [format_mean_std(m, s) for m, s in zip(table2["cf_pce_mean"], table2["cf_pce_std"])],
            "SPS": [format_mean_std(m, s) for m, s in zip(table2["sps_mean"], table2["sps_std"])],
            "Hist Pop Median": table2["history_pop_median_mean"].round(4),
        }
    )

    niche = table2[table2["segment"] == "niche"].copy()
    mainstream = table2[table2["segment"] == "mainstream"].copy()
    gap = niche.merge(
        mainstream,
        on=["dataset", "explainer"],
        suffixes=("_niche", "_mainstream"),
    )
    table_gap = pd.DataFrame(
        {
            "dataset": gap["dataset"],
            "explainer": gap["explainer"],
            "Gap Validity (niche-mainstream)": (gap["validity_mean_niche"] - gap["validity_mean_mainstream"]).round(4),
            "Gap CF-PCE (niche-mainstream)": (gap["cf_pce_mean_niche"] - gap["cf_pce_mean_mainstream"]).round(4),
            "Gap SPS (niche-mainstream)": (gap["sps_mean_niche"] - gap["sps_mean_mainstream"]).round(4),
            "Gap No-CF-found (niche-mainstream)": (
                gap["no_cf_found_mean_niche"] - gap["no_cf_found_mean_mainstream"]
            ).round(4),
        }
    )

    per_seed_overall.to_csv(args.output_dir / "per_seed_overall.csv", index=False)
    per_seed_group.to_csv(args.output_dir / "per_seed_group.csv", index=False)
    table1.to_csv(args.output_dir / "table1_overall_numeric.csv", index=False)
    table1_fmt.to_csv(args.output_dir / "table1_overall.csv", index=False)
    table2.to_csv(args.output_dir / "table2_groupwise_numeric.csv", index=False)
    table2_fmt.to_csv(args.output_dir / "table2_groupwise.csv", index=False)
    table_gap.to_csv(args.output_dir / "table2_group_gap.csv", index=False)

    (args.output_dir / "table1_overall.md").write_text(markdown_table(table1_fmt))
    (args.output_dir / "table2_groupwise.md").write_text(markdown_table(table2_fmt))
    (args.output_dir / "table2_group_gap.md").write_text(markdown_table(table_gap))

    payload = {
        "dataset": args.dataset,
        "run_dirs": [str(path) for path in args.run_dirs],
        "outputs": {
            "per_seed_overall_csv": str(args.output_dir / "per_seed_overall.csv"),
            "per_seed_group_csv": str(args.output_dir / "per_seed_group.csv"),
            "table1_overall_csv": str(args.output_dir / "table1_overall.csv"),
            "table2_groupwise_csv": str(args.output_dir / "table2_groupwise.csv"),
            "table2_group_gap_csv": str(args.output_dir / "table2_group_gap.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(table1_fmt.to_string(index=False))
    print()
    print(table2_fmt.to_string(index=False))
    print()
    print(table_gap.to_string(index=False))


if __name__ == "__main__":
    main()
