#!/usr/bin/env bash
# Launch SDFT training through THUDM/slime train_async.py on a single GPU.
#
# This script assumes the remote training machine has:
#   /root/slime                  THUDM/slime source checkout
#   /root/Megatron-LM            Megatron-LM source checkout
#   /public/huggingface-models/Qwen/Qwen3-0.6B
#
# It uses grpo_sdft.async_sdft_rollout.generate_rollout_sdft as the slime
# rollout function and trains with slime's native sft_loss.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

SLIME_DIR="${SLIME_DIR:-/root/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/public/huggingface-models/Qwen/Qwen3-0.6B}"
PROMPT_DATA="${PROMPT_DATA:-${PROJECT_DIR}/data/sdft_async_hf_remote/sdft_slime_sft_validation.jsonl}"
TORCH_DIST_LOAD="${TORCH_DIST_LOAD:-${PROJECT_DIR}/checkpoints/qwen3_0_6b_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/checkpoints/context_policy_sdft_slime_async}"
CONVERT_IF_MISSING="${CONVERT_IF_MISSING:-1}"
RAY_PORT="${RAY_PORT:-8265}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

MODEL_ARGS=(
  --swiglu
  --num-layers 28
  --hidden-size 1024
  --ffn-hidden-size 3072
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 8
  --kv-channels 128
  --use-rotary-position-embeddings
  --disable-bias-linear
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --rotary-base 1000000
  --max-position-embeddings 40960
  --vocab-size 151936
  --qk-layernorm
  --transformer-impl transformer_engine
)

if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "Missing SDFT prompt data: ${PROMPT_DATA}" >&2
  echo "Generate it first with scripts/run_sdft_async_hf_dataset.py" >&2
  exit 2
fi

if [[ ! -d "${SLIME_DIR}" || ! -f "${SLIME_DIR}/train_async.py" ]]; then
  echo "Missing THUDM/slime source directory: ${SLIME_DIR}" >&2
  exit 2
fi

if [[ ! -d "${MEGATRON_DIR}" ]]; then
  echo "Missing Megatron-LM source directory: ${MEGATRON_DIR}" >&2
  exit 2
fi

if [[ ! -f "${TORCH_DIST_LOAD}/latest_checkpointed_iteration.txt" ]]; then
  if [[ "${CONVERT_IF_MISSING}" != "1" ]]; then
    echo "Missing torch-dist checkpoint: ${TORCH_DIST_LOAD}" >&2
    exit 2
  fi
  echo "Converting HF checkpoint to Megatron torch-dist: ${TORCH_DIST_LOAD}"
  mkdir -p "${TORCH_DIST_LOAD}"
  cd "${SLIME_DIR}"
  python "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${TORCH_DIST_LOAD}" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --attention-backend flash
fi

mkdir -p "${SAVE_DIR}"

ray stop --force >/dev/null 2>&1 || true
ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus 1 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_PORT}"

RUNTIME_ENV_JSON="$(
  cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "${PROJECT_DIR}:${MEGATRON_DIR}:${SLIME_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}
JSON
)"

cd "${SLIME_DIR}"
ray job submit --address="http://${MASTER_ADDR}:${RAY_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_DIR}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --load "${TORCH_DIST_LOAD}" \
  --save "${SAVE_DIR}" \
  --save-interval 9999 \
  --rollout-function-path grpo_sdft.async_sdft_rollout.generate_rollout_sdft \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --rollout-batch-size 4 \
  --global-batch-size 4 \
  --n-samples-per-prompt 1 \
  --num-rollout 1 \
  --num-steps-per-rollout 1 \
  --rollout-max-prompt-len 2048 \
  --rollout-max-context-len 2048 \
  --rollout-max-response-len 128 \
  --loss-type sft_loss \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns \
  --debug-train-only \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 2048 \
  --optimizer adam \
  --lr 5e-6 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash
