#!/usr/bin/env python3
"""Reproduce the LXR ML-1M domain and export ACCENT-friendly artifacts.

This script mirrors the official LXR ML1M preprocessing choices that matter for
cross-explainer alignment work:

- filter users with at least `min_items_per_user` interacted items
- filter items with at least `min_users_per_item` interacting users
- label-encode users/items into contiguous 0-based ids
- split users with the same `train_test_split(..., random_state=42)` protocol

It then exports lightweight artifacts that help later Gate-1 compatibility work:

- `accent_movielens_train.tsv`: ACCENT-compatible tab-separated interactions
- `movielens_train.tsv`: symlink alias for ACCENT's native loader path
- `user_id_map.csv`: encoded -> original user ids
- `item_id_map.csv`: encoded -> original item ids
- `train_users.csv`, `test_users.csv`: user split lists in both id spaces
- `manifest.json`: counts and path metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


DEFAULT_INPUT = Path("/root/autodl-tmp/projects_archive/MGPrompt-CDR/data/raw/ml-1m/ratings.dat")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/shared-domain/ml1m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-items-per-user", type=int, default=2)
    parser.add_argument("--min-users-per-item", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_ratings(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id_original", "item_id_original", "rating", "timestamp"],
    )


def filter_domain(
    data: pd.DataFrame,
    min_items_per_user: int,
    min_users_per_item: int,
) -> pd.DataFrame:
    # Match the official LXR ML1M preprocessing: keep only positive feedback.
    data = data.copy()
    data["rating"] = data["rating"].apply(lambda x: 1 if x > 3.5 else 0)
    data = data[data["rating"] == 1].reset_index(drop=True)

    user_counts = (
        data.groupby("user_id_original")["item_id_original"]
        .nunique()
        .reset_index(name="item_count")
    )
    filtered_users = user_counts.loc[
        user_counts["item_count"] >= min_items_per_user, "user_id_original"
    ]
    data = data[data["user_id_original"].isin(filtered_users)].reset_index(drop=True)

    item_counts = (
        data.groupby("item_id_original")["user_id_original"]
        .nunique()
        .reset_index(name="user_count")
    )
    filtered_items = item_counts.loc[
        item_counts["user_count"] >= min_users_per_item, "item_id_original"
    ]
    data = data[data["item_id_original"].isin(filtered_items)].reset_index(drop=True)
    return data


def encode_ids(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_encoder.fit(data["user_id_original"])
    item_encoder.fit(data["item_id_original"])

    encoded = data.copy()
    encoded["user_id"] = user_encoder.transform(encoded["user_id_original"])
    encoded["item_id"] = item_encoder.transform(encoded["item_id_original"])

    user_map = pd.DataFrame(
        {
            "user_id": range(len(user_encoder.classes_)),
            "user_id_original": user_encoder.classes_,
        }
    )
    item_map = pd.DataFrame(
        {
            "item_id": range(len(item_encoder.classes_)),
            "item_id_original": item_encoder.classes_,
        }
    )
    return encoded, user_map, item_map


def build_user_split(
    encoded: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_group = encoded[["user_id", "item_id"]].groupby(encoded["user_id"])
    users_data = pd.DataFrame(
        {
            "user_id": list(user_group.groups.keys()),
            "item_ids": list(user_group.item_id.apply(list)),
        }
    )

    mlb = MultiLabelBinarizer()
    user_one_hot = pd.DataFrame(
        mlb.fit_transform(users_data["item_ids"]),
        columns=mlb.classes_,
        index=users_data["item_ids"].index,
    )
    user_one_hot["user_id"] = users_data["user_id"]

    x_train, x_test, y_train, y_test = train_test_split(
        user_one_hot.iloc[:, :-1],
        user_one_hot.iloc[:, -1],
        test_size=test_size,
        random_state=random_state,
    )

    train_users = pd.DataFrame({"user_id": y_train.to_numpy()}).sort_values("user_id").reset_index(drop=True)
    test_users = pd.DataFrame({"user_id": y_test.to_numpy()}).sort_values("user_id").reset_index(drop=True)
    return train_users, test_users


def write_outputs(
    output_dir: Path,
    encoded: pd.DataFrame,
    user_map: pd.DataFrame,
    item_map: pd.DataFrame,
    train_users: pd.DataFrame,
    test_users: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    accent_tsv = output_dir / "accent_movielens_train.tsv"
    accent_loader_tsv = output_dir / "movielens_train.tsv"
    user_map_path = output_dir / "user_id_map.csv"
    item_map_path = output_dir / "item_id_map.csv"
    train_users_path = output_dir / "train_users.csv"
    test_users_path = output_dir / "test_users.csv"
    manifest_path = output_dir / "manifest.json"

    encoded.loc[:, ["user_id", "item_id", "rating", "timestamp"]].to_csv(
        accent_tsv,
        sep="\t",
        index=False,
        header=False,
    )
    if accent_loader_tsv.exists() or accent_loader_tsv.is_symlink():
        accent_loader_tsv.unlink()
    accent_loader_tsv.symlink_to(accent_tsv.name)
    user_map.to_csv(user_map_path, index=False)
    item_map.to_csv(item_map_path, index=False)

    train_users = train_users.merge(user_map, on="user_id", how="left")
    test_users = test_users.merge(user_map, on="user_id", how="left")
    train_users.to_csv(train_users_path, index=False)
    test_users.to_csv(test_users_path, index=False)

    manifest = {
        "input": str(args.input),
        "output_dir": str(output_dir),
        "min_items_per_user": args.min_items_per_user,
        "min_users_per_item": args.min_users_per_item,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "num_interactions": int(encoded.shape[0]),
        "num_users": int(user_map.shape[0]),
        "num_items": int(item_map.shape[0]),
        "train_users": int(train_users.shape[0]),
        "test_users": int(test_users.shape[0]),
        "artifacts": {
            "accent_movielens_train_tsv": str(accent_tsv),
            "movielens_train_tsv": str(accent_loader_tsv),
            "user_id_map_csv": str(user_map_path),
            "item_id_map_csv": str(item_map_path),
            "train_users_csv": str(train_users_path),
            "test_users_csv": str(test_users_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    ratings = load_ratings(args.input)
    filtered = filter_domain(
        ratings,
        min_items_per_user=args.min_items_per_user,
        min_users_per_item=args.min_users_per_item,
    )
    encoded, user_map, item_map = encode_ids(filtered)
    train_users, test_users = build_user_split(
        encoded,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    write_outputs(
        output_dir=args.output_dir,
        encoded=encoded,
        user_map=user_map,
        item_map=item_map,
        train_users=train_users,
        test_users=test_users,
        args=args,
    )

    print(f"input={args.input}")
    print(f"output_dir={args.output_dir}")
    print(f"num_interactions={encoded.shape[0]}")
    print(f"num_users={user_map.shape[0]}")
    print(f"num_items={item_map.shape[0]}")
    print(f"train_users={train_users.shape[0]}")
    print(f"test_users={test_users.shape[0]}")


if __name__ == "__main__":
    main()
