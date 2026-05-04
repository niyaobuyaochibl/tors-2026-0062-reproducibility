#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mgprompt-py310}"
ROOT="/root/cfe-popcal-benchmark"

for seed in 0 1 2; do
  conda run -n "${ENV_NAME}" python "${ROOT}/scripts/run_ml1m_mlp_cross_explainer_smoke.py" \
    --data-dir /root/autodl-tmp/lxr_processed_data/ML1M \
    --output-root /root/autodl-tmp/cfe-popcal-runs/ml1m_mlp_m20_s10_4explainer_eval100 \
    --dataset-label ML1M_mlp_m20_s10_eval100 \
    --train-filename train_data_ML1M.csv \
    --static-test-filename static_test_data_ML1M.csv \
    --seed "${seed}" \
    --device cuda \
    --recommender-epochs 2 \
    --explainer-epochs 2 \
    --batch-size 256 \
    --recommender-hidden-dim 64 \
    --explainer-hidden-dim 64 \
    --recommender-lr 0.005 \
    --explainer-lr 0.003 \
    --monitor-users 50 \
    --eval-users 100 \
    --top-m 20 \
    --max-expl-size 10 \
    --activity-buckets 3 \
    --enable-vapc \
    --include-gradient \
    --include-shapley \
    --shapley-samples 64 \
    --shapley-max-history 200
done

for seed in 0 1 2; do
  conda run -n "${ENV_NAME}" python "${ROOT}/scripts/run_ml1m_neumf_cross_explainer_gate.py" \
    --data-dir /root/autodl-tmp/amazon-books-dense-full-u8000-i500-h10 \
    --output-root /root/autodl-tmp/cfe-popcal-runs/amazon_books_dense_full_u8000_i500_h10_neumf_m20_s10_4explainer_eval177 \
    --dataset-label AmazonBooksDense_full_u8000_i500_h10_m20_s10 \
    --train-filename train_data.csv \
    --static-test-filename static_test_data.csv \
    --seed "${seed}" \
    --device cuda \
    --recommender-epochs 40 \
    --explainer-epochs 3 \
    --batch-size 256 \
    --recommender-hidden-dim 128 \
    --explainer-hidden-dim 64 \
    --recommender-lr 0.003 \
    --explainer-lr 0.003 \
    --monitor-users 50 \
    --eval-users 177 \
    --top-m 20 \
    --max-expl-size 10 \
    --activity-buckets 3 \
    --enable-vapc \
    --include-gradient \
    --include-shapley \
    --shapley-samples 64 \
    --shapley-max-history 120
done
