#!/usr/bin/env python3
"""Combine per-dataset VAPC effect packages into a single snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("table_dirs", nargs="+", type=Path)
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_frames = []
    table_frames = []
    for table_dir in args.table_dirs:
        per_seed_frames.append(pd.read_csv(table_dir / "per_seed_vapc_effect.csv"))
        table_frames.append(pd.read_csv(table_dir / "table3_vapc_effect.csv"))

    per_seed = pd.concat(per_seed_frames, ignore_index=True).sort_values(
        ["dataset", "base_explainer", "seed"]
    ).reset_index(drop=True)
    table3 = pd.concat(table_frames, ignore_index=True).sort_values(
        ["dataset", "explainer"]
    ).reset_index(drop=True)

    per_seed.to_csv(args.output_dir / "per_seed_vapc_effect_combined.csv", index=False)
    table3.to_csv(args.output_dir / "table3_vapc_effect_combined.csv", index=False)
    (args.output_dir / "table3_vapc_effect_combined.md").write_text(markdown_table(table3))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "table_dirs": [str(path) for path in args.table_dirs],
                "outputs": {
                    "per_seed_csv": str(args.output_dir / "per_seed_vapc_effect_combined.csv"),
                    "table3_csv": str(args.output_dir / "table3_vapc_effect_combined.csv"),
                    "table3_md": str(args.output_dir / "table3_vapc_effect_combined.md"),
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
