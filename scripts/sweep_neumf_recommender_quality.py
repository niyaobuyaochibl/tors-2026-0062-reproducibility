#!/usr/bin/env python3
"""Sweep NeuMF-style recommender quality without running explainers.

The benchmark breadth checks need a second backbone that is not only executable,
but also defensible as the recommender being explained. This script
reuses the existing history-conditioned NeuMF implementation and evaluates
configuration grids before spending time on LXR/LOO/Grad-input CFE runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from run_ml1m_mlp_lxr_smoke import (
    evaluate_recommender,
    load_dense_dataset,
    resolve_device,
    seed_everything,
    write_json,
)
from run_ml1m_neumf_cross_explainer_gate import train_neumf_recommender


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-label", type=str, required=True)
    parser.add_argument("--train-filename", type=str, default="train_data.csv")
    parser.add_argument("--static-test-filename", type=str, default="static_test_data.csv")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_list, default=[0])
    parser.add_argument("--epochs", type=parse_int_list, default=[20, 40])
    parser.add_argument("--hidden-dims", type=parse_int_list, default=[64, 128])
    parser.add_argument("--learning-rates", type=parse_float_list, default=[0.003, 0.001])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-user-limit", type=int, default=0)
    parser.add_argument("--test-user-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    data = load_dense_dataset(
        data_dir=args.data_dir,
        train_filename=args.train_filename,
        static_test_filename=args.static_test_filename,
        train_user_limit=args.train_user_limit,
        test_user_limit=args.test_user_limit,
    )
    all_items_tensor = torch.eye(data.num_items, device=device)

    rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    sweep_start = time.time()

    grid = list(
        itertools.product(
            args.seeds,
            args.epochs,
            args.hidden_dims,
            args.learning_rates,
        )
    )
    for index, (seed, epochs, hidden_dim, lr) in enumerate(grid, start=1):
        seed_everything(seed)
        run_args = argparse.Namespace(
            seed=seed,
            recommender_epochs=epochs,
            recommender_hidden_dim=hidden_dim,
            recommender_lr=lr,
            batch_size=args.batch_size,
            beta=args.beta,
        )
        run_id = f"seed{seed}_ep{epochs}_h{hidden_dim}_lr{lr:g}"
        print(f"[sweep] {index}/{len(grid)} {run_id}", flush=True)

        start = time.time()
        recommender, history = train_neumf_recommender(data, all_items_tensor, device, run_args)
        metrics = evaluate_recommender(recommender, data, all_items_tensor, device)
        elapsed = time.time() - start

        histories[run_id] = history
        pd.DataFrame(history).to_csv(output_dir / f"history_{run_id}.csv", index=False)
        row = {
            "dataset_label": args.dataset_label,
            "seed": seed,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "learning_rate": lr,
            "batch_size": args.batch_size,
            "beta": args.beta,
            "elapsed_seconds": elapsed,
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values(["hr10", "hr50", "mrr"], ascending=False).to_csv(
            output_dir / "sweep_results.csv",
            index=False,
        )
        print(
            f"[sweep-result] {run_id} hr10={metrics['hr10']:.4f} "
            f"hr50={metrics['hr50']:.4f} hr100={metrics['hr100']:.4f} "
            f"mrr={metrics['mrr']:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values(["hr10", "hr50", "mrr"], ascending=False)
    results.to_csv(output_dir / "sweep_results.csv", index=False)
    summary = {
        "timestamp_utc": timestamp,
        "elapsed_seconds": time.time() - sweep_start,
        "device": str(device),
        "dataset_label": args.dataset_label,
        "data": {
            "num_train_users": int(data.train_features.shape[0]),
            "num_test_users": int(data.test_features.shape[0]),
            "num_items": int(data.num_items),
        },
        "config": {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "hidden_dims": args.hidden_dims,
            "learning_rates": args.learning_rates,
            "batch_size": args.batch_size,
            "beta": args.beta,
        },
        "best_by_hr10": results.iloc[0].to_dict() if not results.empty else None,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "histories.json", histories)
    print(f"output_dir={output_dir}", flush=True)
    print(json.dumps(summary["best_by_hr10"], indent=2), flush=True)


if __name__ == "__main__":
    main()
