#!/usr/bin/env python3
"""Run a minimal ML1M + MLP + LXR smoke experiment.

This script rewrites the relevant LXR notebook logic into a stable Python path
for Week-1 benchmark work:

- train a lightweight MLP recommender on ML1M one-hot user histories
- train an LXR explainer against the recommender's top-1 predictions
- convert the explainer's ranking into prefix counterfactual subsets
- report simple popularity-alignment diagnostics on valid explanations

The popularity diagnostics are intentionally lightweight smoke metrics:

- SPS: mean user-history empirical-CDF position of explanation items minus 0.5
- CF-PCE: average absolute deviation between sorted explanation quantiles and a
  uniform target grid
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_DATA_DIR = Path("/root/autodl-tmp/lxr_processed_data/ML1M")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/cfe-popcal-runs/ml1m_mlp_lxr_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-user-limit", type=int, default=0)
    parser.add_argument("--test-user-limit", type=int, default=0)
    parser.add_argument("--recommender-epochs", type=int, default=2)
    parser.add_argument("--explainer-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recommender-hidden-dim", type=int, default=64)
    parser.add_argument("--explainer-hidden-dim", type=int, default=64)
    parser.add_argument("--recommender-lr", type=float, default=0.005)
    parser.add_argument("--explainer-lr", type=float, default=0.003)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lambda-pos", type=float, default=10.0)
    parser.add_argument("--lambda-neg", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--monitor-users", type=int, default=30)
    parser.add_argument("--eval-users", type=int, default=30)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--max-expl-size", type=int, default=5)
    parser.add_argument("--activity-buckets", type=int, default=3)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        return df.drop(columns=unnamed)
    return df


@dataclass
class ML1MData:
    train_features: np.ndarray
    train_user_ids: np.ndarray
    test_features: np.ndarray
    test_user_ids: np.ndarray
    static_test_features: np.ndarray
    static_test_pos: np.ndarray
    static_test_neg: np.ndarray
    num_items: int
    pop_array: np.ndarray
    item_pop_percentiles: np.ndarray


def load_dense_dataset(
    data_dir: Path,
    train_filename: str,
    static_test_filename: str,
    train_user_limit: int,
    test_user_limit: int,
) -> ML1MData:
    train_df = pd.read_csv(data_dir / train_filename, index_col=0)
    static_test_df = pd.read_csv(data_dir / static_test_filename, index_col=0)

    train_df = drop_unnamed(train_df)
    static_test_df = drop_unnamed(static_test_df)

    num_items = static_test_df.shape[1] - 2
    train_features_df = train_df.iloc[:, :num_items].copy()
    test_features_df = static_test_df.iloc[:, :num_items].copy()

    train_user_ids = train_features_df.index.to_numpy(dtype=np.int64)
    test_user_ids = test_features_df.index.to_numpy(dtype=np.int64)

    if train_user_limit > 0:
        train_features_df = train_features_df.iloc[:train_user_limit].copy()
        train_user_ids = train_user_ids[:train_user_limit]
    if test_user_limit > 0:
        test_features_df = test_features_df.iloc[:test_user_limit].copy()
        static_test_df = static_test_df.iloc[:test_user_limit].copy()
        test_user_ids = test_user_ids[:test_user_limit]

    train_features = train_features_df.to_numpy(dtype=np.float32)
    test_features = test_features_df.to_numpy(dtype=np.float32)
    static_test_features = static_test_df.iloc[:, :num_items].to_numpy(dtype=np.float32)
    static_test_pos = static_test_df["pos"].to_numpy(dtype=np.int64)
    static_test_neg = static_test_df["neg"].to_numpy(dtype=np.int64)

    item_pop = train_features.sum(axis=0).astype(np.float32)
    pop_array = item_pop / max(item_pop.max(), 1.0)
    item_pop_percentiles = (
        pd.Series(item_pop).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    )

    return ML1MData(
        train_features=train_features,
        train_user_ids=train_user_ids,
        test_features=test_features,
        test_user_ids=test_user_ids,
        static_test_features=static_test_features,
        static_test_pos=static_test_pos,
        static_test_neg=static_test_neg,
        num_items=num_items,
        pop_array=pop_array,
        item_pop_percentiles=item_pop_percentiles,
    )


def load_ml1m(data_dir: Path, train_user_limit: int, test_user_limit: int) -> ML1MData:
    return load_dense_dataset(
        data_dir=data_dir,
        train_filename="train_data_ML1M.csv",
        static_test_filename="static_test_data_ML1M.csv",
        train_user_limit=train_user_limit,
        test_user_limit=test_user_limit,
    )


class MLP(nn.Module):
    def __init__(self, hidden_size: int, num_items: int, device: torch.device):
        super().__init__()
        self.device = device
        self.users_fc = nn.Linear(num_items, hidden_size, bias=True).to(device)
        self.items_fc = nn.Linear(num_items, hidden_size, bias=True).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        if user_tensor.dim() == 1:
            user_tensor = user_tensor.unsqueeze(0)
        if item_tensor.dim() == 1:
            item_tensor = item_tensor.unsqueeze(0)
        user_vec = self.users_fc(user_tensor.float().to(self.device))
        item_vec = self.items_fc(item_tensor.float().to(self.device))
        return self.sigmoid(torch.matmul(user_vec, item_vec.T))


class Explainer(nn.Module):
    def __init__(self, num_items: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.device = device
        self.users_fc = nn.Linear(num_items, hidden_size).to(device)
        self.items_fc = nn.Linear(num_items, hidden_size).to(device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size).to(device),
            nn.Tanh(),
            nn.Linear(hidden_size, num_items).to(device),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        if user_tensor.dim() == 1:
            user_tensor = user_tensor.unsqueeze(0)
        if item_tensor.dim() == 1:
            item_tensor = item_tensor.unsqueeze(0)
        user_output = self.users_fc(user_tensor.float().to(self.device))
        item_output = self.items_fc(item_tensor.float().to(self.device))
        combined = torch.cat((user_output, item_output), dim=-1)
        return self.bottleneck(combined)


def recommender_run(
    user_tensor: torch.Tensor,
    recommender: MLP,
    item_tensor: torch.Tensor,
    wanted_output: str = "single",
) -> torch.Tensor:
    if wanted_output == "single":
        return recommender(user_tensor, item_tensor)
    return recommender(user_tensor, item_tensor).squeeze()


def get_scores_for_user(
    user_tensor: torch.Tensor,
    recommender: MLP,
    all_items_tensor: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return recommender_run(user_tensor, recommender, all_items_tensor, wanted_output="vector")


def get_masked_scores(
    user_tensor: torch.Tensor,
    original_user_tensor: torch.Tensor,
    recommender: MLP,
    all_items_tensor: torch.Tensor,
) -> torch.Tensor:
    scores = get_scores_for_user(user_tensor, recommender, all_items_tensor)
    original = original_user_tensor[: all_items_tensor.shape[0]]
    catalog = torch.ones_like(original) - original
    masked_scores = scores.clone()
    masked_scores[catalog <= 0] = float("-inf")
    return masked_scores


def get_top_k(
    user_tensor: torch.Tensor,
    original_user_tensor: torch.Tensor,
    recommender: MLP,
    all_items_tensor: torch.Tensor,
    k: int | None = None,
) -> list[tuple[int, float]]:
    masked_scores = get_masked_scores(user_tensor, original_user_tensor, recommender, all_items_tensor)
    valid_count = int(torch.isfinite(masked_scores).sum().item())
    if valid_count == 0:
        return []
    top_n = valid_count if k is None else min(k, valid_count)
    values, indices = torch.topk(masked_scores, k=top_n)
    return [(int(idx), float(val)) for idx, val in zip(indices.tolist(), values.tolist())]


def get_rank_of_item(
    user_tensor: torch.Tensor,
    original_user_tensor: torch.Tensor,
    item_id: int,
    recommender: MLP,
    all_items_tensor: torch.Tensor,
) -> int:
    masked_scores = get_masked_scores(user_tensor, original_user_tensor, recommender, all_items_tensor)
    if not torch.isfinite(masked_scores[item_id]):
        return all_items_tensor.shape[0]
    target = masked_scores[item_id]
    return int((masked_scores > target).sum().item()) + 1


def get_user_recommended_item(
    user_tensor: torch.Tensor,
    recommender: MLP,
    all_items_tensor: torch.Tensor,
) -> int:
    masked_scores = get_masked_scores(user_tensor, user_tensor, recommender, all_items_tensor)
    return int(torch.argmax(masked_scores).item())


def sample_train_pairs(train_features: np.ndarray, pop_array: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled = train_features.copy()
    pos_idx = np.empty(sampled.shape[0], dtype=np.int64)
    neg_idx = np.empty(sampled.shape[0], dtype=np.int64)

    for row_idx in range(sampled.shape[0]):
        row = sampled[row_idx]
        ones = np.flatnonzero(row > 0)
        zeros = np.flatnonzero(row == 0)
        pos_item = int(rng.choice(ones))
        zero_weights = pop_array[zeros]
        if float(zero_weights.sum()) > 0:
            zero_weights = zero_weights / zero_weights.sum()
            neg_item = int(rng.choice(zeros, p=zero_weights))
        else:
            neg_item = int(rng.choice(zeros))
        sampled[row_idx, pos_item] = 0.0
        pos_idx[row_idx] = pos_item
        neg_idx[row_idx] = neg_item
    return sampled, pos_idx, neg_idx


def evaluate_recommender(
    recommender: MLP,
    data: ML1MData,
    all_items_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    hits10 = 0
    hits50 = 0
    hits100 = 0
    reciprocal_ranks = 0.0
    percentile_ranks = 0.0

    n = data.static_test_features.shape[0]
    for row_idx in range(n):
        pos_item = int(data.static_test_pos[row_idx])
        user_tensor = torch.tensor(data.static_test_features[row_idx], device=device)
        user_tensor[pos_item] = 0
        rank = get_rank_of_item(user_tensor, user_tensor, pos_item, recommender, all_items_tensor)
        hits10 += int(rank <= 10)
        hits50 += int(rank <= 50)
        hits100 += int(rank <= 100)
        reciprocal_ranks += 1.0 / rank
        percentile_ranks += rank / data.num_items

    return {
        "hr10": hits10 / n,
        "hr50": hits50 / n,
        "hr100": hits100 / n,
        "mrr": reciprocal_ranks / n,
        "mpr": 100.0 * percentile_ranks / n,
    }


def train_recommender(
    data: ML1MData,
    all_items_tensor: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[MLP, list[dict[str, float]]]:
    model = MLP(args.recommender_hidden_dim, data.num_items, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.recommender_lr)
    history: list[dict[str, float]] = []
    best_hr10 = -1.0
    best_state = None

    num_training = data.train_features.shape[0]
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.recommender_epochs):
        sampled, pos_idx, neg_idx = sample_train_pairs(data.train_features, data.pop_array, rng)
        perm = rng.permutation(num_training)
        batch_losses = []
        pos_losses = []
        neg_losses = []
        model.train()

        for start in range(0, num_training, args.batch_size):
            batch_indices = perm[start : start + args.batch_size]
            batch_matrix = torch.tensor(sampled[batch_indices], device=device)
            batch_pos_items = all_items_tensor[pos_idx[batch_indices]]
            batch_neg_items = all_items_tensor[neg_idx[batch_indices]]

            optimizer.zero_grad()
            pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))
            neg_output = torch.diagonal(model(batch_matrix, batch_neg_items))
            pos_loss = torch.mean((torch.ones_like(pos_output) - pos_output) ** 2)
            neg_loss = torch.mean(neg_output ** 2)
            batch_loss = pos_loss + args.beta * neg_loss
            batch_loss.backward()
            optimizer.step()

            batch_losses.append(float(batch_loss.item()))
            pos_losses.append(float(pos_loss.item()))
            neg_losses.append(float(neg_loss.item()))

        model.eval()
        metrics = evaluate_recommender(model, data, all_items_tensor, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "train_pos_loss": float(np.mean(pos_losses)),
            "train_neg_loss": float(np.mean(neg_losses)),
            **metrics,
        }
        history.append(epoch_record)
        print(
            f"[recommender] epoch={epoch} "
            f"loss={epoch_record['train_loss']:.4f} "
            f"hr10={metrics['hr10']:.4f} hr50={metrics['hr50']:.4f} "
            f"mrr={metrics['mrr']:.4f}",
            flush=True,
        )

        if metrics["hr10"] > best_hr10:
            best_hr10 = metrics["hr10"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def build_top1_maps(
    recommender: MLP,
    train_features: np.ndarray,
    train_user_ids: np.ndarray,
    test_features: np.ndarray,
    test_user_ids: np.ndarray,
    all_items_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[dict[int, int], dict[int, int]]:
    top1_train: dict[int, int] = {}
    top1_test: dict[int, int] = {}
    with torch.no_grad():
        for user_id, row in zip(train_user_ids, train_features):
            user_tensor = torch.tensor(row, device=device)
            top1_train[int(user_id)] = get_user_recommended_item(user_tensor, recommender, all_items_tensor)
        for user_id, row in zip(test_user_ids, test_features):
            user_tensor = torch.tensor(row, device=device)
            top1_test[int(user_id)] = get_user_recommended_item(user_tensor, recommender, all_items_tensor)
    return top1_train, top1_test


def compute_history_item_scores(
    explainer: Explainer,
    user_tensor: torch.Tensor,
    item_id: int,
    all_items_tensor: torch.Tensor,
) -> list[tuple[int, float]]:
    item_tensor = all_items_tensor[item_id]
    with torch.no_grad():
        scores = explainer(user_tensor, item_tensor).squeeze()
    masked_scores = scores * user_tensor
    history_items = torch.nonzero(user_tensor > 0, as_tuple=False).squeeze(-1).tolist()
    pairs = [(int(item), float(masked_scores[item].item())) for item in history_items]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def calculate_pos_neg_at_20(
    explainer: Explainer,
    recommender: MLP,
    user_tensor: torch.Tensor,
    item_id: int,
    all_items_tensor: torch.Tensor,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    ranked_items = compute_history_item_scores(explainer, user_tensor, item_id, all_items_tensor)
    user_hist_size = int(torch.sum(user_tensor).item())
    bins = [0] + [len(x) for x in np.array_split(np.arange(user_hist_size), num_bins, axis=0)]

    pos_at_20 = np.zeros(num_bins + 1, dtype=np.float32)
    neg_at_20 = np.zeros(num_bins + 1, dtype=np.float32)
    total_items = 0

    pos_ranked = ranked_items[:user_hist_size]
    neg_ranked = sorted(pos_ranked, key=lambda x: x[1])

    for idx, bucket_size in enumerate(bins):
        total_items += bucket_size
        pos_masked = user_tensor.clone()
        neg_masked = user_tensor.clone()
        for item, _ in pos_ranked[:total_items]:
            pos_masked[item] = 0
        for item, _ in neg_ranked[:total_items]:
            neg_masked[item] = 0

        pos_rank = get_rank_of_item(pos_masked, user_tensor, item_id, recommender, all_items_tensor)
        neg_rank = get_rank_of_item(neg_masked, user_tensor, item_id, recommender, all_items_tensor)
        pos_at_20[idx] = float(pos_rank <= 20)
        neg_at_20[idx] = float(neg_rank <= 20)

    return pos_at_20, neg_at_20


def train_explainer(
    recommender: MLP,
    data: ML1MData,
    top1_train: dict[int, int],
    top1_test: dict[int, int],
    all_items_tensor: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Explainer, list[dict[str, float]]]:
    explainer = Explainer(data.num_items, args.explainer_hidden_dim, device)
    optimizer = torch.optim.Adam(explainer.parameters(), lr=args.explainer_lr)
    history: list[dict[str, float]] = []

    monitor_count = min(args.monitor_users, data.test_features.shape[0])
    rng = np.random.default_rng(args.seed)
    monitor_rows = rng.choice(data.test_features.shape[0], size=monitor_count, replace=False)
    monitor_user_ids = data.test_user_ids[monitor_rows]
    test_user_index = {int(user_id): idx for idx, user_id in enumerate(data.test_user_ids)}
    train_with_user = np.concatenate(
        [data.train_features, data.train_user_ids[:, None].astype(np.float32)],
        axis=1,
    )

    best_key = None
    best_state = None

    recommender.eval()
    for epoch in range(args.explainer_epochs):
        epoch_rng = np.random.default_rng(args.seed + 1000 + epoch)
        perm = epoch_rng.permutation(train_with_user.shape[0])
        train_loss_total = 0.0
        pos_loss_total = 0.0
        neg_loss_total = 0.0
        l1_total = 0.0
        n_examples = 0
        explainer.train()

        for start in range(0, train_with_user.shape[0], args.batch_size):
            batch = train_with_user[perm[start : start + args.batch_size]]
            user_tensors = torch.tensor(batch[:, : data.num_items], device=device)
            user_ids = batch[:, data.num_items].astype(np.int64)
            item_ids = np.array([top1_train[int(user_id)] for user_id in user_ids], dtype=np.int64)
            items_tensors = all_items_tensor[item_ids]

            optimizer.zero_grad()
            pos_masks = explainer(user_tensors, items_tensors)
            neg_masks = torch.ones_like(pos_masks) - pos_masks
            x_masked_pos = user_tensors * pos_masks
            x_masked_neg = user_tensors * neg_masks

            pos_scores = torch.diagonal(recommender(x_masked_pos, items_tensors))
            neg_scores = torch.diagonal(recommender(x_masked_neg, items_tensors))
            pos_scores = torch.clamp(pos_scores, min=1e-6, max=1 - 1e-6)
            neg_scores = torch.clamp(neg_scores, min=1e-6, max=1 - 1e-6)

            pos_loss = -torch.mean(torch.log(pos_scores))
            neg_loss = torch.mean(torch.log(neg_scores))
            l1_loss = x_masked_pos[user_tensors > 0].mean()
            combined = args.lambda_pos * pos_loss + args.lambda_neg * neg_loss + args.alpha * l1_loss
            combined.backward()
            optimizer.step()

            batch_size = user_tensors.shape[0]
            train_loss_total += float(combined.item()) * batch_size
            pos_loss_total += float(pos_loss.item()) * batch_size
            neg_loss_total += float(neg_loss.item()) * batch_size
            l1_total += float(l1_loss.item()) * batch_size
            n_examples += batch_size

        explainer.eval()
        pos_monitor = np.zeros(11, dtype=np.float32)
        neg_monitor = np.zeros(11, dtype=np.float32)
        for user_id in monitor_user_ids:
            user_offset = test_user_index[int(user_id)]
            user_tensor = torch.tensor(data.test_features[user_offset], device=device)
            item_id = top1_test[int(user_id)]
            pos_at_20, neg_at_20 = calculate_pos_neg_at_20(
                explainer,
                recommender,
                user_tensor,
                item_id,
                all_items_tensor,
                num_bins=10,
            )
            pos_monitor += pos_at_20
            neg_monitor += neg_at_20

        pos_monitor_mean = float(np.mean(pos_monitor) / len(monitor_user_ids))
        neg_monitor_mean = float(np.mean(neg_monitor) / len(monitor_user_ids))
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_total / n_examples,
            "train_pos_loss": pos_loss_total / n_examples,
            "train_neg_loss": neg_loss_total / n_examples,
            "train_l1_loss": l1_total / n_examples,
            "pos_at_20_mean": pos_monitor_mean,
            "neg_at_20_mean": neg_monitor_mean,
        }
        history.append(epoch_record)
        print(
            f"[explainer] epoch={epoch} "
            f"loss={epoch_record['train_loss']:.4f} "
            f"pos@20={pos_monitor_mean:.4f} neg@20={neg_monitor_mean:.4f}",
            flush=True,
        )

        key = (pos_monitor_mean, -neg_monitor_mean)
        if best_key is None or key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in explainer.state_dict().items()}

    if best_state is not None:
        explainer.load_state_dict(best_state)
    explainer.eval()
    return explainer, history


def empirical_cdf_quantiles(history_percentiles: np.ndarray, explanation_percentiles: np.ndarray) -> np.ndarray:
    sorted_history = np.sort(history_percentiles)
    return np.searchsorted(sorted_history, explanation_percentiles, side="right") / max(len(sorted_history), 1)


def compute_alignment_metrics(
    history_items: np.ndarray,
    explanation_items: list[int],
    item_pop_percentiles: np.ndarray,
) -> tuple[float, float, float]:
    history_percentiles = item_pop_percentiles[history_items]
    explanation_percentiles = item_pop_percentiles[np.array(explanation_items, dtype=np.int64)]
    quantiles = empirical_cdf_quantiles(history_percentiles, explanation_percentiles)
    quantiles_sorted = np.sort(quantiles)
    uniform_grid = (np.arange(len(quantiles_sorted), dtype=np.float32) + 0.5) / len(quantiles_sorted)
    cf_pce = float(np.mean(np.abs(quantiles_sorted - uniform_grid)))
    sps = float(np.mean(quantiles) - 0.5)
    history_median = float(np.median(history_percentiles))
    return cf_pce, sps, history_median


def assign_groups(df: pd.DataFrame, activity_buckets: int) -> pd.DataFrame:
    if df.empty:
        return df

    bucket_count = min(activity_buckets, df["history_size"].nunique())
    if bucket_count <= 1:
        df["activity_bucket"] = "all"
    else:
        ranked_sizes = df["history_size"].rank(method="first")
        df["activity_bucket"] = pd.qcut(
            ranked_sizes,
            q=bucket_count,
            labels=[f"activity_{idx}" for idx in range(bucket_count)],
            duplicates="drop",
        ).astype(str)

    df["segment"] = "mainstream"
    for bucket, frame in df.groupby("activity_bucket"):
        threshold = frame["history_pop_median"].median()
        mask = (df["activity_bucket"] == bucket) & (df["history_pop_median"] <= threshold)
        df.loc[mask, "segment"] = "niche"
    return df


def run_prefix_cf_evaluation(
    recommender: MLP,
    explainer: Explainer,
    data: ML1MData,
    top1_test: dict[int, int],
    all_items_tensor: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(args.seed + 2000)
    eval_count = min(args.eval_users, data.test_features.shape[0])
    eval_rows = rng.choice(data.test_features.shape[0], size=eval_count, replace=False)

    records = []
    for row_idx in eval_rows:
        user_id = int(data.test_user_ids[row_idx])
        user_vector = data.test_features[row_idx]
        user_tensor = torch.tensor(user_vector, device=device)
        history_items = np.flatnonzero(user_vector > 0)
        original_item = int(top1_test[user_id])
        original_scores = get_scores_for_user(user_tensor, recommender, all_items_tensor)
        original_score = float(original_scores[original_item].item())

        ranked_items = compute_history_item_scores(explainer, user_tensor, original_item, all_items_tensor)
        candidate_items = [item for item, _ in ranked_items[: args.top_m]]

        best_subset: list[int] | None = None
        replacement_item = original_item
        masked_score = original_score

        for subset_size in range(1, min(args.max_expl_size, len(candidate_items)) + 1):
            subset = candidate_items[:subset_size]
            masked_user = user_tensor.clone()
            masked_user[subset] = 0
            new_item = get_user_recommended_item(masked_user, recommender, all_items_tensor)
            new_scores = get_scores_for_user(masked_user, recommender, all_items_tensor)
            if new_item != original_item:
                best_subset = subset
                replacement_item = int(new_item)
                masked_score = float(new_scores[original_item].item())
                break

        record = {
            "user_id": user_id,
            "history_size": int(len(history_items)),
            "original_item": original_item,
            "replacement_item": replacement_item,
            "top_m_candidates": json.dumps(candidate_items),
            "valid_cf_found": bool(best_subset is not None),
            "no_cf_found": bool(best_subset is None),
            "explanation_size": int(len(best_subset)) if best_subset is not None else math.nan,
            "explanation_items": json.dumps(best_subset) if best_subset is not None else None,
            "original_score": original_score,
            "masked_score": masked_score if best_subset is not None else math.nan,
            "score_drop": (original_score - masked_score) if best_subset is not None else math.nan,
        }

        if best_subset is not None:
            cf_pce, sps, history_pop_median = compute_alignment_metrics(
                history_items,
                best_subset,
                data.item_pop_percentiles,
            )
            record["cf_pce"] = cf_pce
            record["sps"] = sps
            record["history_pop_median"] = history_pop_median
        else:
            history_percentiles = data.item_pop_percentiles[history_items]
            record["cf_pce"] = math.nan
            record["sps"] = math.nan
            record["history_pop_median"] = float(np.median(history_percentiles))

        records.append(record)

    per_user = pd.DataFrame.from_records(records)
    per_user = assign_groups(per_user, args.activity_buckets)

    valid_df = per_user[per_user["valid_cf_found"]].copy()
    summary: dict[str, object] = {
        "eval_users": int(len(per_user)),
        "validity": float(per_user["valid_cf_found"].mean()),
        "no_cf_found_rate": float(per_user["no_cf_found"].mean()),
        "mean_explanation_size": float(valid_df["explanation_size"].mean()) if not valid_df.empty else None,
        "median_explanation_size": float(valid_df["explanation_size"].median()) if not valid_df.empty else None,
        "mean_score_drop": float(valid_df["score_drop"].mean()) if not valid_df.empty else None,
        "mean_cf_pce": float(valid_df["cf_pce"].mean()) if not valid_df.empty else None,
        "mean_sps": float(valid_df["sps"].mean()) if not valid_df.empty else None,
        "group_summary": {},
    }

    if not valid_df.empty:
        group_summary = {}
        for segment, frame in per_user.groupby("segment"):
            valid_frame = frame[frame["valid_cf_found"]]
            group_summary[segment] = {
                "users": int(len(frame)),
                "validity": float(frame["valid_cf_found"].mean()),
                "no_cf_found_rate": float(frame["no_cf_found"].mean()),
                "mean_cf_pce": float(valid_frame["cf_pce"].mean()) if not valid_frame.empty else None,
                "mean_sps": float(valid_frame["sps"].mean()) if not valid_frame.empty else None,
            }
        summary["group_summary"] = group_summary

    return per_user, summary


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    data = load_ml1m(args.data_dir, args.train_user_limit, args.test_user_limit)
    all_items_tensor = torch.eye(data.num_items, device=device)

    recommender, recommender_history = train_recommender(data, all_items_tensor, device, args)
    top1_train, top1_test = build_top1_maps(
        recommender,
        data.train_features,
        data.train_user_ids,
        data.test_features,
        data.test_user_ids,
        all_items_tensor,
        device,
    )
    explainer, explainer_history = train_explainer(
        recommender,
        data,
        top1_train,
        top1_test,
        all_items_tensor,
        device,
        args,
    )
    per_user, cf_summary = run_prefix_cf_evaluation(
        recommender,
        explainer,
        data,
        top1_test,
        all_items_tensor,
        device,
        args,
    )

    recommender_metrics = evaluate_recommender(recommender, data, all_items_tensor, device)

    torch.save(recommender.state_dict(), output_dir / "mlp_recommender.pt")
    torch.save(explainer.state_dict(), output_dir / "lxr_explainer.pt")
    per_user.to_csv(output_dir / "per_user_prefix_cf.csv", index=False)
    pd.DataFrame(recommender_history).to_csv(output_dir / "recommender_history.csv", index=False)
    pd.DataFrame(explainer_history).to_csv(output_dir / "explainer_history.csv", index=False)

    summary = {
        "timestamp_utc": timestamp,
        "elapsed_seconds": time.time() - start,
        "device": str(device),
        "config": vars(args),
        "data": {
            "num_train_users": int(data.train_features.shape[0]),
            "num_test_users": int(data.test_features.shape[0]),
            "num_items": int(data.num_items),
        },
        "recommender_metrics": recommender_metrics,
        "prefix_cf_summary": cf_summary,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "top1_train.json", top1_train)
    write_json(output_dir / "top1_test.json", top1_test)

    print(f"output_dir={output_dir}", flush=True)
    print(json.dumps(summary["recommender_metrics"], indent=2), flush=True)
    print(json.dumps(summary["prefix_cf_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
