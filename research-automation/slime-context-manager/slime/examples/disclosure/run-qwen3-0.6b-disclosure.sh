#!/bin/bash

# usage: bash examples/disclosure/run-qwen3-0.6b-disclosure.sh
# Progressive Evidence Disclosure training for Qwen3-0.6B (Single GPU)
# Based on roadmap: progressive-disclosure-slime-roadmap.md

# Stop previous tasks
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

set -ex

export PYTHONBUFFERED=16

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="/root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/models"

source "${PROJECT_ROOT}/scripts/models/qwen3-0.6B.sh"

# ============================================================================
# Model Checkpoint Configuration
# ============================================================================

CKPT_ARGS=(
   --hf-checkpoint "${MODELS_DIR}/qwen3-0.6B"
   --ref-load "${MODELS_DIR}/qwen3-0.6B_torch_dist"
   --load "${MODELS_DIR}/qwen3-0.6B_slime/"
   --save "${MODELS_DIR}/qwen3-0.6B_slime/"
   --save-interval 10
)

# ============================================================================
# Progressive Disclosure Rollout Configuration (Single GPU)
# ============================================================================

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.disclosure.slime_rollout.generate_rollout_disclosure
   --prompt-data "${PROJECT_ROOT}/data/slime/disclosure_train.jsonl"
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle

   --loss-type sft_loss

   --num-rollout 10
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 512
   --rollout-temperature 1

   --global-batch-size 4
   --balance-data
)

# ============================================================================
# Performance Configuration (Single GPU)
# ============================================================================

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

# ============================================================================
# Optimizer Configuration
# ============================================================================

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-5
   --lr-decay-style cosine
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.95
)

# ============================================================================
# GRPO Configuration
# ============================================================================

GRPO_ARGS=(
   --advantage-estimator grpo
)

# ============================================================================
# Miscellaneous Configuration
# ============================================================================

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============================================================================
# Disclosure-specific Arguments
# ============================================================================

DISCLOSURE_ARGS=(
   --disclosure-budget 4096
)

# ============================================================================
# Start Ray Cluster (Single GPU)
# ============================================================================

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# ============================================================================
# Submit Training Job
# ============================================================================

echo "Starting Progressive Disclosure training (single GPU)..."

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --rollout-num-gpus 1 \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${DISCLOSURE_ARGS[@]}

# ============================================================================
# Cleanup
# ============================================================================

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

echo "Training completed!"
