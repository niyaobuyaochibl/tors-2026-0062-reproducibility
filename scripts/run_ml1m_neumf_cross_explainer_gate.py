#!/usr/bin/env python3
"""Run an ML1M second-backbone cross-explainer gate.

This script reuses the stable MLP cross-explainer protocol, but replaces the
recommender with a history-conditioned NeuMF-style neural backbone. The purpose
is not to introduce a new explainer path; it is to test whether the benchmark
pipeline can support more than one shared recommender backbone before expanding
the evidence package beyond the original MLP-only snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from run_ml1m_mlp_cross_explainer_smoke import (
    compute_gradient_ranked_items,
    compute_loo_ranked_items,
    compute_mc_shapley_ranked_items,
    run_prefix_cf_for_explainer,
)
from run_ml1m_mlp_lxr_smoke import (
    build_top1_maps,
    compute_history_item_scores,
    evaluate_recommender,
    load_dense_dataset,
    resolve_device,
    sample_train_pairs,
    seed_everything,
    train_explainer,
    write_json,
)


DEFAULT_DATA_DIR = Path(os.environ.get("CFE_POPCAL_ML1M_DATA_DIR", "data/ML1M"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("CFE_POPCAL_OUTPUT_ROOT", "outputs/ml1m_neumf_cross_explainer_gate"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-label", type=str, default="ML1M")
    parser.add_argument("--train-filename", type=str, default="train_data_ML1M.csv")
    parser.add_argument("--static-test-filename", type=str, default="static_test_data_ML1M.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-user-limit", type=int, default=0)
    parser.add_argument("--test-user-limit", type=int, default=0)
    parser.add_argument("--recommender-epochs", type=int, default=2)
    parser.add_argument("--explainer-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recommender-hidden-dim", type=int, default=64)
    parser.add_argument("--explainer-hidden-dim", type=int, default=64)
    parser.add_argument("--recommender-lr", type=float, default=0.003)
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
    parser.add_argument("--enable-vapc", action="store_true")
    parser.add_argument("--include-gradient", action="store_true")
    parser.add_argument("--include-shapley", action="store_true")
    parser.add_argument("--shapley-samples", type=int, default=64)
    parser.add_argument("--shapley-max-history", type=int, default=200)
    parser.add_argument("--vapc-lambda0", type=float, default=1.0)
    parser.add_argument("--vapc-alpha", type=float, default=1.0)
    parser.add_argument("--vapc-max-extra-items", type=int, default=1)
    parser.add_argument("--vapc-base-size-penalty", type=float, default=0.75)
    parser.add_argument(
        "--vapc-align-metric",
        type=str,
        default="hybrid",
        choices=["hybrid", "cf_pce", "abs_sps"],
    )
    parser.add_argument("--vapc-niche-quantile", type=float, default=0.25)
    return parser.parse_args()


class HistoryNeuMF(nn.Module):
    """NeuMF-style scorer over one-hot history vectors and one-hot item vectors."""

    def __init__(self, hidden_size: int, num_items: int, device: torch.device):
        super().__init__()
        self.device = device
        self.user_gmf = nn.Linear(num_items, hidden_size, bias=True).to(device)
        self.item_gmf = nn.Linear(num_items, hidden_size, bias=True).to(device)
        self.user_mlp = nn.Linear(num_items, hidden_size, bias=True).to(device)
        self.item_mlp = nn.Linear(num_items, hidden_size, bias=True).to(device)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size).to(device),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size).to(device),
            nn.ReLU(),
        ).to(device)
        self.output = nn.Linear(hidden_size * 2, 1).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor) -> torch.Tensor:
        if user_tensor.dim() == 1:
            user_tensor = user_tensor.unsqueeze(0)
        if item_tensor.dim() == 1:
            item_tensor = item_tensor.unsqueeze(0)

        user_tensor = user_tensor.float().to(self.device)
        item_tensor = item_tensor.float().to(self.device)

        user_gmf = self.user_gmf(user_tensor)
        item_gmf = self.item_gmf(item_tensor)
        gmf = user_gmf.unsqueeze(1) * item_gmf.unsqueeze(0)

        user_mlp = self.user_mlp(user_tensor).unsqueeze(1)
        item_mlp = self.item_mlp(item_tensor).unsqueeze(0)
        user_mlp = user_mlp.expand(-1, item_tensor.shape[0], -1)
        item_mlp = item_mlp.expand(user_tensor.shape[0], -1, -1)
        mlp_out = self.mlp(torch.cat([user_mlp, item_mlp], dim=-1))

        logits = self.output(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)
        return self.sigmoid(logits)


def train_neumf_recommender(
    data,
    all_items_tensor: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[HistoryNeuMF, list[dict[str, float]]]:
    model = HistoryNeuMF(args.recommender_hidden_dim, data.num_items, device)
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
            neg_loss = torch.mean(neg_output**2)
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
            f"[neumf-recommender] epoch={epoch} "
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
    data = load_dense_dataset(
        data_dir=args.data_dir,
        train_filename=args.train_filename,
        static_test_filename=args.static_test_filename,
        train_user_limit=args.train_user_limit,
        test_user_limit=args.test_user_limit,
    )
    all_items_tensor = torch.eye(data.num_items, device=device)

    recommender, recommender_history = train_neumf_recommender(
        data,
        all_items_tensor,
        device,
        args,
    )
    top1_train, top1_test = build_top1_maps(
        recommender,
        data.train_features,
        data.train_user_ids,
        data.test_features,
        data.test_user_ids,
        all_items_tensor,
        device,
    )

    lxr_explainer, explainer_history = train_explainer(
        recommender,
        data,
        top1_train,
        top1_test,
        all_items_tensor,
        device,
        args,
    )

    result_payloads: list[tuple[str, pd.DataFrame, dict[str, object]]] = []
    result_payloads.extend(
        run_prefix_cf_for_explainer(
            "lxr",
            lambda user_tensor, item_id: compute_history_item_scores(
                lxr_explainer,
                user_tensor,
                item_id,
                all_items_tensor,
            ),
            recommender,
            data,
            top1_test,
            all_items_tensor,
            device,
            args,
        )
    )
    result_payloads.extend(
        run_prefix_cf_for_explainer(
            "loo",
            lambda user_tensor, item_id: compute_loo_ranked_items(
                recommender,
                user_tensor,
                item_id,
                all_items_tensor,
            ),
            recommender,
            data,
            top1_test,
            all_items_tensor,
            device,
            args,
        )
    )
    if args.include_gradient:
        result_payloads.extend(
            run_prefix_cf_for_explainer(
                "grad_input",
                lambda user_tensor, item_id: compute_gradient_ranked_items(
                    recommender,
                    user_tensor,
                    item_id,
                    all_items_tensor,
                ),
                recommender,
                data,
                top1_test,
                all_items_tensor,
                device,
                args,
            )
        )

    if args.include_shapley:
        shapley_rng = np.random.default_rng(args.seed + 9100)
        result_payloads.extend(
            run_prefix_cf_for_explainer(
                "mc_shapley",
                lambda user_tensor, item_id: compute_mc_shapley_ranked_items(
                    recommender,
                    user_tensor,
                    item_id,
                    all_items_tensor,
                    args.shapley_samples,
                    args.shapley_max_history,
                    shapley_rng,
                ),
                recommender,
                data,
                top1_test,
                all_items_tensor,
                device,
                args,
            )
        )

    summary_payload: dict[str, object] = {}
    combined_frames: list[pd.DataFrame] = []
    for label, per_user, summary in result_payloads:
        summary_payload[label] = summary
        combined_frames.append(per_user)
        mean_cf_pce = summary["mean_cf_pce"]
        mean_sps = summary["mean_sps"]
        cf_pce_text = "nan" if mean_cf_pce is None else f"{mean_cf_pce:.4f}"
        sps_text = "nan" if mean_sps is None else f"{mean_sps:.4f}"
        print(
            f"[benchmark] {label} validity={summary['validity']:.4f} "
            f"cf_pce={cf_pce_text} sps={sps_text}",
            flush=True,
        )

    combined = pd.concat(combined_frames, ignore_index=True)
    recommender_metrics = evaluate_recommender(recommender, data, all_items_tensor, device)

    pd.DataFrame(recommender_history).to_csv(output_dir / "recommender_history.csv", index=False)
    pd.DataFrame(explainer_history).to_csv(output_dir / "lxr_history.csv", index=False)
    for label, per_user, _summary in result_payloads:
        per_user.to_csv(output_dir / f"per_user_{label}.csv", index=False)
    combined.to_csv(output_dir / "per_user_all.csv", index=False)
    torch.save(recommender.state_dict(), output_dir / "neumf_recommender.pt")
    torch.save(lxr_explainer.state_dict(), output_dir / "lxr_explainer.pt")
    write_json(output_dir / "top1_train.json", top1_train)
    write_json(output_dir / "top1_test.json", top1_test)

    summary = {
        "timestamp_utc": timestamp,
        "elapsed_seconds": time.time() - start,
        "device": str(device),
        "backbone_label": "history_neumf",
        "config": vars(args),
        "dataset_label": args.dataset_label,
        "data": {
            "num_train_users": int(data.train_features.shape[0]),
            "num_test_users": int(data.test_features.shape[0]),
            "num_items": int(data.num_items),
        },
        "recommender_metrics": recommender_metrics,
        "explainer_summaries": summary_payload,
    }
    write_json(output_dir / "summary.json", summary)

    print(f"output_dir={output_dir}", flush=True)
    print(json.dumps(summary["recommender_metrics"], indent=2), flush=True)
    print(json.dumps(summary["explainer_summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
