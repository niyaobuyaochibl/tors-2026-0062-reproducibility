#!/usr/bin/env python3
"""Prepare a compact dense benchmark slice from Yelp processed data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_INPUT_DIR = Path(os.environ.get("CFE_POPCAL_YELP_INPUT_DIR", "data/raw/yelp-processed"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("CFE_POPCAL_YELP_OUTPUT_DIR", "data/processed/yelp-dense-v1-u3000-i1500"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positive-threshold", type=float, default=3.5)
    parser.add_argument("--top-users", type=int, default=3000)
    parser.add_argument("--top-items", type=int, default=1500)
    parser.add_argument("--min-train-history", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_frames(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_pickle(input_dir / "train.pkl")
    test = pd.read_pickle(input_dir / "test.pkl")
    for frame in (train, test):
        frame["user_idx"] = frame["user_idx"].astype(int)
        frame["item_idx"] = frame["item_idx"].astype(int)
    return train, test


def filter_positive(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df["rating"] > threshold].copy()


def choose_candidate_slice(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_users: int,
    top_items: int,
    min_train_history: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_counts = train.groupby("user_idx").size().sort_values(ascending=False)
    selected_users = set(user_counts.head(top_users).index)

    train = train[train["user_idx"].isin(selected_users)].copy()
    test = test[test["user_idx"].isin(selected_users)].copy()

    item_counts = train.groupby("item_idx").size().sort_values(ascending=False)
    selected_items = set(item_counts.head(top_items).index)

    train = train[train["item_idx"].isin(selected_items)].copy()
    test = test[test["item_idx"].isin(selected_items)].copy()

    train_counts = train.groupby("user_idx").size()
    test_counts = test.groupby("user_idx").size()
    eligible_users = set(train_counts[train_counts >= min_train_history].index) & set(test_counts[test_counts >= 1].index)

    train = train[train["user_idx"].isin(eligible_users)].copy()
    test = test[test["user_idx"].isin(eligible_users)].copy()
    return train, test


def remap_ids(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_users = sorted(set(train["user_idx"].unique()) | set(test["user_idx"].unique()))
    final_items = sorted(set(train["item_idx"].unique()) | set(test["item_idx"].unique()))

    user_map = {source_id: idx for idx, source_id in enumerate(final_users)}
    item_map = {source_id: idx for idx, source_id in enumerate(final_items)}

    train = train.copy()
    test = test.copy()
    train["user_id_new"] = train["user_idx"].map(user_map)
    train["item_id_new"] = train["item_idx"].map(item_map)
    test["user_id_new"] = test["user_idx"].map(user_map)
    test["item_id_new"] = test["item_idx"].map(item_map)

    user_map_df = pd.DataFrame({"user_id_new": range(len(final_users)), "source_user_idx": final_users})
    item_map_df = pd.DataFrame({"item_id_new": range(len(final_items)), "source_item_idx": final_items})
    return train, test, user_map_df, item_map_df


def build_histories(train: pd.DataFrame, num_items: int) -> tuple[pd.DataFrame, pd.Series]:
    histories = train.groupby("user_id_new")["item_id_new"].apply(list)
    matrix = np.zeros((len(histories), num_items), dtype=np.int8)
    ordered_users = histories.index.to_numpy(dtype=np.int64)
    for row_idx, items in enumerate(histories):
        matrix[row_idx, items] = 1
    history_df = pd.DataFrame(matrix, index=ordered_users, columns=[str(i) for i in range(num_items)])
    return history_df, histories


def choose_test_positive(test: pd.DataFrame) -> pd.DataFrame:
    ordered = test.sort_values(["user_id_new", "item_id_new"], ascending=[True, True])
    chosen = ordered.groupby("user_id_new", as_index=False).first()
    return chosen[["user_id_new", "item_id_new"]].rename(columns={"item_id_new": "pos"})


def sample_negative_items(
    history_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    pop_array: np.ndarray,
    rng: np.random.Generator,
) -> pd.Series:
    neg_items = {}
    all_items = np.arange(history_df.shape[1], dtype=np.int64)

    for user_id, pos_item in pos_df.set_index("user_id_new")["pos"].items():
        row = history_df.loc[user_id].to_numpy(dtype=np.int8)
        available = all_items[(row == 0) & (all_items != pos_item)]
        weights = pop_array[available]
        if float(weights.sum()) > 0:
            weights = weights / weights.sum()
            neg_item = int(rng.choice(available, p=weights))
        else:
            neg_item = int(rng.choice(available))
        neg_items[user_id] = neg_item

    return pd.Series(neg_items, name="neg")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    train, test = load_frames(args.input_dir)
    train = filter_positive(train, args.positive_threshold)
    test = filter_positive(test, args.positive_threshold)

    train, test = choose_candidate_slice(
        train,
        test,
        top_users=args.top_users,
        top_items=args.top_items,
        min_train_history=args.min_train_history,
    )

    train, test, user_map_df, item_map_df = remap_ids(train, test)

    num_items = item_map_df.shape[0]
    history_df, histories = build_histories(train, num_items)
    pos_df = choose_test_positive(test)

    pop_counts = train.groupby("item_id_new").size().reindex(range(num_items), fill_value=0).to_numpy(dtype=np.float32)
    pop_array = pop_counts / max(pop_counts.max(), 1.0)

    eligible_users = sorted(set(history_df.index) & set(pos_df["user_id_new"]))
    history_df = history_df.loc[eligible_users].copy()
    pos_df = pos_df[pos_df["user_id_new"].isin(eligible_users)].copy()

    train_users, test_users = train_test_split(
        np.array(eligible_users, dtype=np.int64),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_users = np.sort(train_users)
    test_users = np.sort(test_users)

    train_data = history_df.loc[train_users].copy()
    test_history = history_df.loc[test_users].copy()
    pos_test = pos_df[pos_df["user_id_new"].isin(test_users)].set_index("user_id_new").loc[test_users]
    neg_test = sample_negative_items(test_history, pos_test.reset_index(), pop_array, rng)

    static_test = test_history.copy()
    static_test["pos"] = pos_test["pos"].astype(np.int64)
    static_test["neg"] = neg_test.astype(np.int64)
    test_data = static_test.copy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(output_dir / "train_data.csv")
    test_data.to_csv(output_dir / "test_data.csv")
    static_test.to_csv(output_dir / "static_test_data.csv")
    user_map_df.to_csv(output_dir / "user_id_map.csv", index=False)
    item_map_df.to_csv(output_dir / "item_id_map.csv", index=False)

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "positive_threshold": args.positive_threshold,
        "top_users": args.top_users,
        "top_items": args.top_items,
        "min_train_history": args.min_train_history,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "num_users": int(len(eligible_users)),
        "num_items": int(num_items),
        "train_users": int(len(train_users)),
        "test_users": int(len(test_users)),
        "train_interactions": int(train.shape[0]),
        "test_positive_pairs": int(len(pos_test)),
        "history_size_median": float(history_df.sum(axis=1).median()),
        "history_size_mean": float(history_df.sum(axis=1).mean()),
        "artifacts": {
            "train_data_csv": str(output_dir / "train_data.csv"),
            "test_data_csv": str(output_dir / "test_data.csv"),
            "static_test_data_csv": str(output_dir / "static_test_data.csv"),
            "user_id_map_csv": str(output_dir / "user_id_map.csv"),
            "item_id_map_csv": str(output_dir / "item_id_map.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
