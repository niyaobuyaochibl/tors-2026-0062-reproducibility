#!/usr/bin/env python3
"""Generate manuscript figures for the CFE popularity-calibration paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DATASET_LABELS = {
    "ML1M": "ML1M",
    "AmazonBooksDense_u3000_i1500": "Amazon",
    "YelpDense_u2000_i1000": "Yelp",
}

DATASET_COLORS = {
    "ML1M": "#1b9e77",
    "Amazon": "#d95f02",
    "Yelp": "#7570b3",
}
EXPLAINER_COLORS = {
    "loo": "#d95f02",
    "lxr": "#1f78b4",
}
SEGMENT_MARKERS = {
    "mainstream": "o",
    "niche": "^",
}
SEGMENT_COLORS = {
    "mainstream": "#4c78a8",
    "niche": "#e45756",
}


def load_benchmark_user_rows(input_dir: Path | None = None) -> pd.DataFrame:
    packaged_input = None if input_dir is None else input_dir / "benchmark_user_rows.csv"
    if packaged_input is None or not packaged_input.exists():
        raise FileNotFoundError(
            "Missing packaged figure input: benchmark_user_rows.csv. "
            "Use --input-dir to point to replication/figure_inputs."
        )
    data = pd.read_csv(packaged_input)
    data["valid_cf_found"] = data["valid_cf_found"].astype(bool)
    data["history_pop_median"] = pd.to_numeric(data["history_pop_median"], errors="coerce")
    data["sps"] = pd.to_numeric(data["sps"], errors="coerce")
    data["cf_pce"] = pd.to_numeric(data["cf_pce"], errors="coerce")
    data["explanation_mean_quantile"] = 0.5 + data["sps"]
    return data


def plot_profile_scatter(data: pd.DataFrame, output_path: Path) -> None:
    valid = data.loc[
        data["valid_cf_found"] & data["history_pop_median"].notna() & data["sps"].notna()
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

    for ax, dataset in zip(axes, DATASET_LABELS.keys()):
        subset = valid.loc[valid["dataset"] == dataset]
        for explainer in ["loo", "lxr"]:
            for segment in ["mainstream", "niche"]:
                part = subset.loc[
                    (subset["explainer"] == explainer) & (subset["segment"] == segment)
                ]
                if part.empty:
                    continue
                ax.scatter(
                    part["history_pop_median"],
                    part["explanation_mean_quantile"],
                    s=18,
                    alpha=0.45,
                    c=EXPLAINER_COLORS[explainer],
                    marker=SEGMENT_MARKERS[segment],
                    linewidths=0,
                )
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("History popularity median percentile")
        ax.grid(alpha=0.18, linewidth=0.6)

    axes[0].set_ylabel("Explanation mean quantile (0.5 + SPS)")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="LOO", markerfacecolor=EXPLAINER_COLORS["loo"], markersize=7),
        Line2D([0], [0], marker="o", color="w", label="LXR", markerfacecolor=EXPLAINER_COLORS["lxr"], markersize=7),
        Line2D([0], [0], marker="o", color="#444444", label="Mainstream", linestyle="None", markersize=6),
        Line2D([0], [0], marker="^", color="#444444", label="Niche", linestyle="None", markersize=6),
        Line2D([0], [0], color="#666666", linestyle="--", label="Neutral explanation baseline"),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_sps_distribution(data: pd.DataFrame, output_path: Path) -> None:
    valid = data.loc[data["valid_cf_found"] & data["sps"].notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    width = 0.28
    centers = [0, 1]

    for ax, dataset in zip(axes, DATASET_LABELS.keys()):
        subset = valid.loc[valid["dataset"] == dataset]
        positions = []
        box_data = []
        colors = []
        labels = []
        for idx, explainer in enumerate(["loo", "lxr"]):
            for offset, segment in [(-width / 2, "mainstream"), (width / 2, "niche")]:
                part = subset.loc[
                    (subset["explainer"] == explainer) & (subset["segment"] == segment),
                    "sps",
                ].dropna()
                positions.append(centers[idx] + offset)
                box_data.append(part.values)
                colors.append(SEGMENT_COLORS[segment])
                labels.append(segment)

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.1},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(centers)
        ax.set_xticklabels(["LOO", "LXR"])
        ax.set_xlabel("Explainer")
        ax.grid(alpha=0.18, linewidth=0.6, axis="y")

    axes[0].set_ylabel("Signed popularity shift (SPS)")
    legend_items = [
        Line2D([0], [0], color=SEGMENT_COLORS["mainstream"], linewidth=8, alpha=0.55, label="Mainstream"),
        Line2D([0], [0], color=SEGMENT_COLORS["niche"], linewidth=8, alpha=0.55, label="Niche"),
        Line2D([0], [0], color="#666666", linestyle="--", label="Neutral shift"),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_repair_tradeoff(bundle_dir: Path, output_path: Path) -> None:
    data = pd.read_csv(bundle_dir / "paper_table3_per_seed.csv")
    data["dataset_label"] = data["dataset"].map(
        {
            "ML1M": "ML1M",
            "AmazonBooksDense_u3000_i1500": "Amazon",
            "YelpDense_u2000_i1000": "Yelp",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    y_specs = [
        ("cf_pce_improvement", "CF-PCE improvement"),
        ("abs_sps_improvement", "Absolute SPS improvement"),
    ]

    for ax, (y_col, y_label) in zip(axes, y_specs):
        for dataset in DATASET_LABELS.keys():
            subset = data.loc[data["dataset"] == dataset]
            for explainer in ["loo", "lxr"]:
                part = subset.loc[subset["base_explainer"] == explainer]
                ax.scatter(
                    part["mean_minimality_gap"],
                    part[y_col],
                    s=55,
                    alpha=0.9,
                    c=DATASET_COLORS[part["dataset_label"].iloc[0]],
                    marker="o" if explainer == "loo" else "s",
                    edgecolors="black",
                    linewidths=0.4,
                )
        ax.set_xlabel("Mean minimality gap")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2, linewidth=0.6)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="ML1M", markerfacecolor=DATASET_COLORS["ML1M"], markeredgecolor="black", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Amazon", markerfacecolor=DATASET_COLORS["Amazon"], markeredgecolor="black", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Yelp", markerfacecolor=DATASET_COLORS["Yelp"], markeredgecolor="black", markersize=7),
        Line2D([0], [0], marker="o", color="#333333", label="LOO", linestyle="None", markersize=6),
        Line2D([0], [0], marker="s", color="#333333", label="LXR", linestyle="None", markersize=6),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(default_root / "figure_inputs"),
        help="Directory containing benchmark_user_rows.csv and paper_table3_per_seed.csv.",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Optional legacy bundle directory containing paper_table3_per_seed.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_root / "generated_figures"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else input_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_data = load_benchmark_user_rows(input_dir)
    plot_profile_scatter(user_data, output_dir / "fig1_history_vs_explanation_profile.pdf")
    plot_sps_distribution(user_data, output_dir / "fig2_sps_distribution.pdf")
    plot_repair_tradeoff(bundle_dir, output_dir / "fig3_repair_tradeoff.pdf")

    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
