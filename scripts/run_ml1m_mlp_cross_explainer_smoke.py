#!/usr/bin/env python3
"""Run a small cross-explainer ML1M + MLP benchmark smoke.

This script builds on the stable `run_ml1m_mlp_lxr_smoke.py` path and adds
additional explainer baselines:

- `lxr`: the learned LXR explainer
- `loo`: leave-one-out score-drop ranking over the user history
- `grad_input`: differentiable gradient-times-input saliency
- `mc_shapley`: Monte Carlo Shapley-style coalition perturbation ranking

Both explainers are evaluated with the same prefix counterfactual protocol:

- candidate cap: `top-M`
- valid subset family: prefixes only
- success criterion: top-1 recommendation flips
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ml1m_mlp_lxr_smoke import (
    Explainer,
    assign_groups,
    build_top1_maps,
    compute_alignment_metrics,
    compute_history_item_scores,
    evaluate_recommender,
    get_scores_for_user,
    get_user_recommended_item,
    load_dense_dataset,
    resolve_device,
    seed_everything,
    train_explainer,
    train_recommender,
    write_json,
)


DEFAULT_DATA_DIR = Path("/root/autodl-tmp/lxr_processed_data/ML1M")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/cfe-popcal-runs/ml1m_mlp_cross_explainer_smoke")


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


def compute_loo_ranked_items(
    recommender,
    user_tensor: torch.Tensor,
    item_id: int,
    all_items_tensor: torch.Tensor,
) -> list[tuple[int, float]]:
    history_items = torch.nonzero(user_tensor > 0, as_tuple=False).squeeze(-1)
    if history_items.numel() == 0:
        return []

    target_item_tensor = all_items_tensor[item_id].unsqueeze(0).repeat(history_items.shape[0], 1)
    masked_batch = user_tensor.unsqueeze(0).repeat(history_items.shape[0], 1)
    row_indices = torch.arange(history_items.shape[0], device=user_tensor.device)
    masked_batch[row_indices, history_items] = 0

    with torch.no_grad():
        outputs = recommender(masked_batch, target_item_tensor)
        masked_scores = torch.diagonal(outputs)
    original_score = float(recommender(user_tensor, all_items_tensor[item_id]).squeeze().item())
    score_drop = original_score - masked_scores

    pairs = [(int(item), float(drop.item())) for item, drop in zip(history_items, score_drop)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def compute_gradient_ranked_items(
    recommender,
    user_tensor: torch.Tensor,
    item_id: int,
    all_items_tensor: torch.Tensor,
) -> list[tuple[int, float]]:
    history_items = torch.nonzero(user_tensor > 0, as_tuple=False).squeeze(-1)
    if history_items.numel() == 0:
        return []

    recommender.zero_grad(set_to_none=True)
    user_for_grad = user_tensor.detach().clone().requires_grad_(True)
    target_item_tensor = all_items_tensor[item_id]
    score = recommender(user_for_grad, target_item_tensor).squeeze()
    score.backward()

    if user_for_grad.grad is None:
        return []

    grad_input = (user_for_grad.grad.detach() * user_tensor.detach())[history_items]
    pairs = [(int(item), float(weight.item())) for item, weight in zip(history_items, grad_input)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    recommender.zero_grad(set_to_none=True)
    return pairs


def compute_mc_shapley_ranked_items(
    recommender,
    user_tensor: torch.Tensor,
    item_id: int,
    all_items_tensor: torch.Tensor,
    samples: int,
    max_history: int,
    rng: np.random.Generator,
) -> list[tuple[int, float]]:
    """Approximate removal-game Shapley values for historical items.

    The value function is the target-item score drop caused by removing a
    coalition from the user's history. For each random permutation, an item's
    marginal contribution is the score before removing it minus the score after
    removing it, conditioned on the items removed earlier in that permutation.
    """

    history_items = torch.nonzero(user_tensor > 0, as_tuple=False).squeeze(-1)
    if history_items.numel() == 0 or samples <= 0:
        return []

    history_np = history_items.detach().cpu().numpy().astype(np.int64)
    if max_history > 0 and history_np.size > max_history:
        history_np = rng.choice(history_np, size=max_history, replace=False)

    contributions = {int(item): 0.0 for item in history_np}
    target_item = all_items_tensor[item_id]

    for _ in range(samples):
        order = rng.permutation(history_np)
        states = user_tensor.unsqueeze(0).repeat(order.size + 1, 1)
        for position, item in enumerate(order):
            states[position + 1 :, int(item)] = 0

        target_batch = target_item.unsqueeze(0).repeat(order.size + 1, 1)
        with torch.no_grad():
            outputs = recommender(states, target_batch)
            scores = torch.diagonal(outputs).detach().cpu().numpy()

        marginals = scores[:-1] - scores[1:]
        for item, marginal in zip(order, marginals):
            contributions[int(item)] += float(marginal)

    inv_samples = 1.0 / float(samples)
    pairs = [(item, value * inv_samples) for item, value in contributions.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def compute_niche_strength(
    history_items: np.ndarray,
    item_pop_percentiles: np.ndarray,
    lower_quantile: float,
) -> tuple[float, float]:
    history_percentiles = item_pop_percentiles[history_items]
    if history_percentiles.size == 0:
        return 0.0, 0.5

    lower_value = float(np.quantile(history_percentiles, lower_quantile))
    niche_strength = float(np.clip((0.5 - lower_value) / 0.5, 0.0, 1.0))
    return niche_strength, lower_value


def compute_align_penalty(cf_pce: float, sps: float, metric: str) -> float:
    if metric == "cf_pce":
        return float(cf_pce)
    if metric == "abs_sps":
        return float(abs(sps))
    return float(0.5 * (cf_pce + abs(sps)))


def build_valid_prefix_candidates(
    candidate_items: list[int],
    user_tensor: torch.Tensor,
    recommender,
    all_items_tensor: torch.Tensor,
    original_item: int,
    original_score: float,
    history_items: np.ndarray,
    item_pop_percentiles: np.ndarray,
    max_expl_size: int,
) -> tuple[list[dict[str, object]], float]:
    history_percentiles = item_pop_percentiles[history_items]
    history_pop_median = float(np.median(history_percentiles)) if history_percentiles.size > 0 else 0.5

    valid_candidates: list[dict[str, object]] = []
    for subset_size in range(1, min(max_expl_size, len(candidate_items)) + 1):
        subset = candidate_items[:subset_size]
        masked_user = user_tensor.clone()
        masked_user[subset] = 0
        new_item = get_user_recommended_item(masked_user, recommender, all_items_tensor)
        new_scores = get_scores_for_user(masked_user, recommender, all_items_tensor)
        if new_item == original_item:
            continue

        masked_score = float(new_scores[original_item].item())
        cf_pce, sps, _ = compute_alignment_metrics(
            history_items,
            subset,
            item_pop_percentiles,
        )
        valid_candidates.append(
            {
                "subset": list(subset),
                "replacement_item": int(new_item),
                "masked_score": masked_score,
                "score_drop": float(original_score - masked_score),
                "cf_pce": cf_pce,
                "sps": sps,
                "history_pop_median": history_pop_median,
                "explanation_size": int(len(subset)),
            }
        )

    return valid_candidates, history_pop_median


def score_vapc_candidates(
    valid_candidates: list[dict[str, object]],
    history_items: np.ndarray,
    item_pop_percentiles: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if not valid_candidates:
        return [], None

    min_valid_size = min(int(candidate["explanation_size"]) for candidate in valid_candidates)
    admissible = [
        dict(candidate)
        for candidate in valid_candidates
        if int(candidate["explanation_size"]) <= min_valid_size + args.vapc_max_extra_items
    ]
    if not admissible:
        return [], None

    niche_strength, history_low_quantile = compute_niche_strength(
        history_items,
        item_pop_percentiles,
        args.vapc_niche_quantile,
    )
    lambda_u = float(args.vapc_lambda0 * (1.0 + args.vapc_alpha * niche_strength))

    score_values = np.array([float(candidate["score_drop"]) for candidate in admissible], dtype=np.float32)
    score_min = float(score_values.min())
    score_span = float(score_values.max() - score_values.min())

    scored_candidates: list[dict[str, object]] = []
    for candidate in admissible:
        scored_candidate = dict(candidate)
        minimality_gap = int(scored_candidate["explanation_size"]) - min_valid_size
        score_drop_norm = 0.0 if score_span <= 1e-12 else (float(scored_candidate["score_drop"]) - score_min) / score_span
        align_penalty = compute_align_penalty(
            float(scored_candidate["cf_pce"]),
            float(scored_candidate["sps"]),
            args.vapc_align_metric,
        )
        base_score = float(score_drop_norm - args.vapc_base_size_penalty * minimality_gap)
        final_score = float(base_score - lambda_u * align_penalty)

        scored_candidate.update(
            {
                "score_drop_norm": score_drop_norm,
                "align_penalty": align_penalty,
                "base_score": base_score,
                "final_score": final_score,
                "minimality_gap": minimality_gap,
                "min_valid_size": min_valid_size,
                "valid_candidate_count": len(valid_candidates),
                "user_niche_strength": niche_strength,
                "history_low_quantile": history_low_quantile,
                "vapc_lambda": lambda_u,
            }
        )
        scored_candidates.append(scored_candidate)

    admissible_count = len(scored_candidates)
    for scored_candidate in scored_candidates:
        scored_candidate["admissible_candidate_count"] = admissible_count

    selected_candidate = max(
        scored_candidates,
        key=lambda candidate: (
            round(float(candidate["final_score"]), 12),
            -int(candidate["minimality_gap"]),
            -float(candidate["align_penalty"]),
            float(candidate["score_drop"]),
        ),
    )
    return scored_candidates, selected_candidate


def build_result_record(
    explainer_label: str,
    base_explainer: str,
    repair_stage: str,
    user_id: int,
    history_items: np.ndarray,
    original_item: int,
    original_score: float,
    candidate_items: list[int],
    history_pop_median: float,
    candidate: dict[str, object] | None,
    repair_changed: bool,
    timing: dict[str, float] | None = None,
) -> dict[str, object]:
    timing = timing or {}
    record = {
        "explainer": explainer_label,
        "base_explainer": base_explainer,
        "repair_stage": repair_stage,
        "repair_changed": bool(repair_changed),
        "user_id": user_id,
        "history_size": int(len(history_items)),
        "original_item": original_item,
        "replacement_item": original_item,
        "top_m_candidates": json.dumps(candidate_items),
        "valid_cf_found": bool(candidate is not None),
        "no_cf_found": bool(candidate is None),
        "explanation_size": np.nan,
        "explanation_items": None,
        "original_score": original_score,
        "masked_score": np.nan,
        "score_drop": np.nan,
        "cf_pce": np.nan,
        "sps": np.nan,
        "history_pop_median": history_pop_median,
        "align_penalty": np.nan,
        "score_drop_norm": np.nan,
        "base_score": np.nan,
        "final_score": np.nan,
        "minimality_gap": np.nan,
        "min_valid_size": np.nan,
        "valid_candidate_count": 0,
        "admissible_candidate_count": 0,
        "user_niche_strength": np.nan,
        "history_low_quantile": np.nan,
        "vapc_lambda": np.nan,
        "target_scoring_seconds": float(timing.get("target_scoring_seconds", np.nan)),
        "ranking_seconds": float(timing.get("ranking_seconds", np.nan)),
        "candidate_eval_seconds": float(timing.get("candidate_eval_seconds", np.nan)),
        "vapc_scoring_seconds": float(timing.get("vapc_scoring_seconds", np.nan)),
        "per_user_elapsed_seconds": float(timing.get("per_user_elapsed_seconds", np.nan)),
    }

    if candidate is None:
        return record

    record.update(
        {
            "replacement_item": int(candidate["replacement_item"]),
            "explanation_size": int(candidate["explanation_size"]),
            "explanation_items": json.dumps(candidate["subset"]),
            "masked_score": float(candidate["masked_score"]),
            "score_drop": float(candidate["score_drop"]),
            "cf_pce": float(candidate["cf_pce"]),
            "sps": float(candidate["sps"]),
            "history_pop_median": float(candidate["history_pop_median"]),
            "align_penalty": float(candidate.get("align_penalty", np.nan)),
            "score_drop_norm": float(candidate.get("score_drop_norm", np.nan)),
            "base_score": float(candidate.get("base_score", np.nan)),
            "final_score": float(candidate.get("final_score", np.nan)),
            "minimality_gap": float(candidate.get("minimality_gap", 0.0)),
            "min_valid_size": float(candidate.get("min_valid_size", candidate["explanation_size"])),
            "valid_candidate_count": int(candidate.get("valid_candidate_count", 1)),
            "admissible_candidate_count": int(candidate.get("admissible_candidate_count", 1)),
            "user_niche_strength": float(candidate.get("user_niche_strength", np.nan)),
            "history_low_quantile": float(candidate.get("history_low_quantile", np.nan)),
            "vapc_lambda": float(candidate.get("vapc_lambda", np.nan)),
        }
    )
    return record


def summarize_per_user(per_user: pd.DataFrame) -> dict[str, object]:
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
        "mean_minimality_gap": float(valid_df["minimality_gap"].mean()) if "minimality_gap" in valid_df else None,
        "median_minimality_gap": float(valid_df["minimality_gap"].median()) if "minimality_gap" in valid_df else None,
        "repair_changed_rate": float(per_user["repair_changed"].mean()) if "repair_changed" in per_user else None,
        "mean_per_user_elapsed_seconds": (
            float(per_user["per_user_elapsed_seconds"].mean())
            if "per_user_elapsed_seconds" in per_user
            else None
        ),
        "median_per_user_elapsed_seconds": (
            float(per_user["per_user_elapsed_seconds"].median())
            if "per_user_elapsed_seconds" in per_user
            else None
        ),
        "mean_ranking_seconds": (
            float(per_user["ranking_seconds"].mean()) if "ranking_seconds" in per_user else None
        ),
        "mean_candidate_eval_seconds": (
            float(per_user["candidate_eval_seconds"].mean())
            if "candidate_eval_seconds" in per_user
            else None
        ),
        "mean_vapc_scoring_seconds": (
            float(per_user["vapc_scoring_seconds"].mean())
            if "vapc_scoring_seconds" in per_user
            else None
        ),
        "group_summary": {},
    }

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
    return summary


def run_prefix_cf_for_explainer(
    explainer_name: str,
    scoring_fn,
    recommender,
    data,
    top1_test: dict[int, int],
    all_items_tensor: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> list[tuple[str, pd.DataFrame, dict[str, object]]]:
    rng = np.random.default_rng(args.seed + 5000)
    eval_count = min(args.eval_users, data.test_features.shape[0])
    eval_rows = rng.choice(data.test_features.shape[0], size=eval_count, replace=False)

    base_label = explainer_name if not args.enable_vapc else f"{explainer_name}_base"
    base_records: list[dict[str, object]] = []
    vapc_records: list[dict[str, object]] = []
    for row_idx in eval_rows:
        user_id = int(data.test_user_ids[row_idx])
        user_vector = data.test_features[row_idx]
        user_tensor = torch.tensor(user_vector, device=device)
        history_items = np.flatnonzero(user_vector > 0)
        original_item = int(top1_test[user_id])
        sync_if_cuda(device)
        target_start = time.perf_counter()
        original_scores = get_scores_for_user(user_tensor, recommender, all_items_tensor)
        original_score = float(original_scores[original_item].item())
        sync_if_cuda(device)
        target_seconds = time.perf_counter() - target_start

        sync_if_cuda(device)
        ranking_start = time.perf_counter()
        ranked_items = scoring_fn(user_tensor, original_item)
        sync_if_cuda(device)
        ranking_seconds = time.perf_counter() - ranking_start
        candidate_items = [item for item, _ in ranked_items[: args.top_m]]

        sync_if_cuda(device)
        candidate_start = time.perf_counter()
        valid_candidates, history_pop_median = build_valid_prefix_candidates(
            candidate_items=candidate_items,
            user_tensor=user_tensor,
            recommender=recommender,
            all_items_tensor=all_items_tensor,
            original_item=original_item,
            original_score=original_score,
            history_items=history_items,
            item_pop_percentiles=data.item_pop_percentiles,
            max_expl_size=args.max_expl_size,
        )
        sync_if_cuda(device)
        candidate_eval_seconds = time.perf_counter() - candidate_start

        base_candidate = valid_candidates[0] if valid_candidates else None

        sync_if_cuda(device)
        vapc_start = time.perf_counter()
        scored_candidates, vapc_candidate = score_vapc_candidates(
            valid_candidates,
            history_items,
            data.item_pop_percentiles,
            args,
        )
        sync_if_cuda(device)
        vapc_scoring_seconds = time.perf_counter() - vapc_start
        scored_map = {
            tuple(int(item) for item in candidate["subset"]): candidate
            for candidate in scored_candidates
        }
        if base_candidate is not None:
            base_candidate = scored_map.get(
                tuple(int(item) for item in base_candidate["subset"]),
                base_candidate,
            )

        base_records.append(
            build_result_record(
                explainer_label=base_label,
                base_explainer=explainer_name,
                repair_stage="base",
                user_id=user_id,
                history_items=history_items,
                original_item=original_item,
                original_score=original_score,
                candidate_items=candidate_items,
                history_pop_median=history_pop_median,
                candidate=base_candidate,
                repair_changed=False,
                timing={
                    "target_scoring_seconds": target_seconds,
                    "ranking_seconds": ranking_seconds,
                    "candidate_eval_seconds": candidate_eval_seconds,
                    "vapc_scoring_seconds": 0.0,
                    "per_user_elapsed_seconds": (
                        target_seconds + ranking_seconds + candidate_eval_seconds
                    ),
                },
            )
        )

        if args.enable_vapc:
            repair_changed = False
            if base_candidate is not None and vapc_candidate is not None:
                repair_changed = tuple(base_candidate["subset"]) != tuple(vapc_candidate["subset"])
            vapc_records.append(
                build_result_record(
                    explainer_label=f"{explainer_name}_vapc",
                    base_explainer=explainer_name,
                    repair_stage="vapc",
                    user_id=user_id,
                    history_items=history_items,
                    original_item=original_item,
                    original_score=original_score,
                    candidate_items=candidate_items,
                    history_pop_median=history_pop_median,
                    candidate=vapc_candidate,
                    repair_changed=repair_changed,
                    timing={
                        "target_scoring_seconds": target_seconds,
                        "ranking_seconds": ranking_seconds,
                        "candidate_eval_seconds": candidate_eval_seconds,
                        "vapc_scoring_seconds": vapc_scoring_seconds,
                        "per_user_elapsed_seconds": (
                            target_seconds
                            + ranking_seconds
                            + candidate_eval_seconds
                            + vapc_scoring_seconds
                        ),
                    },
                )
            )

    outputs: list[tuple[str, pd.DataFrame, dict[str, object]]] = []
    base_per_user = pd.DataFrame.from_records(base_records)
    base_per_user = assign_groups(base_per_user, args.activity_buckets)
    outputs.append((base_label, base_per_user, summarize_per_user(base_per_user)))

    if args.enable_vapc:
        vapc_per_user = pd.DataFrame.from_records(vapc_records)
        vapc_per_user = assign_groups(vapc_per_user, args.activity_buckets)
        outputs.append((f"{explainer_name}_vapc", vapc_per_user, summarize_per_user(vapc_per_user)))

    return outputs


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

    lxr_outputs = run_prefix_cf_for_explainer(
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
    result_payloads.extend(lxr_outputs)

    loo_outputs = run_prefix_cf_for_explainer(
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
    result_payloads.extend(loo_outputs)

    if args.include_gradient:
        gradient_outputs = run_prefix_cf_for_explainer(
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
        result_payloads.extend(gradient_outputs)

    if args.include_shapley:
        shapley_rng = np.random.default_rng(args.seed + 9100)
        shapley_outputs = run_prefix_cf_for_explainer(
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
        result_payloads.extend(shapley_outputs)

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
            f"cf_pce={cf_pce_text} "
            f"sps={sps_text}",
            flush=True,
        )

    combined = pd.concat(combined_frames, ignore_index=True)
    recommender_metrics = evaluate_recommender(recommender, data, all_items_tensor, device)

    pd.DataFrame(recommender_history).to_csv(output_dir / "recommender_history.csv", index=False)
    pd.DataFrame(explainer_history).to_csv(output_dir / "lxr_history.csv", index=False)
    for label, per_user, _summary in result_payloads:
        per_user.to_csv(output_dir / f"per_user_{label}.csv", index=False)
    combined.to_csv(output_dir / "per_user_all.csv", index=False)
    torch.save(recommender.state_dict(), output_dir / "mlp_recommender.pt")
    torch.save(lxr_explainer.state_dict(), output_dir / "lxr_explainer.pt")
    write_json(output_dir / "top1_train.json", top1_train)
    write_json(output_dir / "top1_test.json", top1_test)

    summary = {
        "timestamp_utc": timestamp,
        "elapsed_seconds": time.time() - start,
        "device": str(device),
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
