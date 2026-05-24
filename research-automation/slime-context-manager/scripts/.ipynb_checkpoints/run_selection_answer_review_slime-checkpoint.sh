#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLIME_PYTHON="${SLIME_PYTHON:-/root/shared-nvme/workspace/envs/slime/bin/python}"
SLIME_BIN_DIR="$(cd "$(dirname "${SLIME_PYTHON}")" && pwd)"
RAY_CMD=("${RAY_BIN:-${SLIME_BIN_DIR}/ray}")
SLIME_DIR="${SLIME_DIR:-${ROOT_DIR}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}"
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SLIME_DIR}/scripts/models/qwen3-0.6B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${ROOT_DIR}/models/qwen3-0.6B}"
TORCH_DIST_LOAD="${TORCH_DIST_LOAD:-${ROOT_DIR}/checkpoints/qwen3_0_6b_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints/selection_review_sft}"
CONVERT_IF_MISSING="${CONVERT_IF_MISSING:-1}"
RAY_PORT="${RAY_PORT:-8265}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://${MASTER_ADDR}:${RAY_PORT}}"
NO_PROXY_ENTRIES="127.0.0.1,localhost,${MASTER_ADDR}"
export NO_PROXY="${NO_PROXY_ENTRIES}${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY_ENTRIES}${no_proxy:+,${no_proxy}}"
SPLIT="${SPLIT:-train}"
DATASET="${DATASET:-hotpotqa}"
MAX_ROWS="${MAX_ROWS:-100}"
SPANS_JSONL="${SPANS_JSONL:-${ROOT_DIR}/data/disclosure/spans_${SPLIT}.jsonl}"
RAW_JSONL="${RAW_JSONL:-${ROOT_DIR}/data/raw/hotpotqa_${SPLIT}.jsonl}"

if [[ ! -f "${MODEL_ARGS_FILE}" ]]; then
  echo "Missing model args file: ${MODEL_ARGS_FILE}" >&2
  exit 2
fi

if [[ ! -d "${SLIME_DIR}" || ! -f "${SLIME_DIR}/train_async.py" ]]; then
  echo "Missing Slime source directory: ${SLIME_DIR}" >&2
  exit 2
fi

if [[ ! -d "${MEGATRON_DIR}" ]]; then
  echo "Missing Megatron-LM source directory: ${MEGATRON_DIR}" >&2
  exit 2
fi

if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "Missing HF checkpoint directory: ${HF_CHECKPOINT}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${MODEL_ARGS_FILE}"

if [[ ! -f "${SPANS_JSONL}" ]]; then
  if [[ "${DATASET}" == "hotpotqa" && ! -f "${RAW_JSONL}" ]]; then
    "${SLIME_PYTHON}" "${ROOT_DIR}/scripts/download_datasets.py" \
      --dataset hotpotqa \
      --max-samples "${MAX_ROWS}"
  fi
  "${SLIME_PYTHON}" "${ROOT_DIR}/scripts/build_disclosure_dataset.py" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --raw-path "${RAW_JSONL}" \
    --max-rows "${MAX_ROWS}"
fi

if [[ ! -f "${TORCH_DIST_LOAD}/latest_checkpointed_iteration.txt" ]]; then
  if [[ "${CONVERT_IF_MISSING}" != "1" ]]; then
    echo "Missing torch-dist checkpoint: ${TORCH_DIST_LOAD}" >&2
    exit 2
  fi
  echo "Converting HF checkpoint to Megatron torch-dist: ${TORCH_DIST_LOAD}"
  mkdir -p "${TORCH_DIST_LOAD}"
  cd "${SLIME_DIR}"
  PYTHONPATH="${ROOT_DIR}:${MEGATRON_DIR}:${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  "${SLIME_PYTHON}" "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${TORCH_DIST_LOAD}" \
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-1}" \
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}" \
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}" \
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}" \
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}" \
    --no-rope-fusion \
    --no-masked-softmax-fusion \
    --no-persist-layer-norm \
    --no-gradient-accumulation-fusion \
    --transformer-impl "${TRANSFORMER_IMPL:-local}" \
    --attention-backend "${ATTENTION_BACKEND:-flash}"
fi

mkdir -p "${SAVE_DIR}"

"${RAY_CMD[@]}" stop --force >/dev/null 2>&1 || true
"${RAY_CMD[@]}" start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS:-1}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_PORT}"

for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 "${RAY_JOB_ADDRESS}/api/version" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS --max-time 5 "${RAY_JOB_ADDRESS}/api/version" >/dev/null; then
  echo "Ray job API is not ready at ${RAY_JOB_ADDRESS}" >&2
  exit 2
fi

RUNTIME_ENV_JSON="$(
  cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "${ROOT_DIR}:${MEGATRON_DIR}:${SLIME_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "DISCLOSURE_TOKEN_BUDGET": "${DISCLOSURE_TOKEN_BUDGET:-4096}",
    "DISCLOSURE_RESERVED_PROMPT_TOKENS": "${DISCLOSURE_RESERVED_PROMPT_TOKENS:-128}",
    "DISCLOSURE_ALLOCATOR": "${DISCLOSURE_ALLOCATOR:-greedy_margin_per_cost}",
    "SELECTION_POLICY": "${SELECTION_POLICY:-rule_expand_omission}"
  }
}
JSON
)"

cd "${SLIME_DIR}"
"${RAY_CMD[@]}" job submit --address="${RAY_JOB_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${SLIME_PYTHON}" "${SLIME_DIR}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS:-1}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --load "${TORCH_DIST_LOAD}" \
  --save "${SAVE_DIR}" \
  --save-interval "${SAVE_INTERVAL:-9999}" \
  --rollout-function-path disclosure.slime_rollout.generate_selection_answer_review_rollout \
  --prompt-data "${SPANS_JSONL}" \
  --input-key question \
  --metadata-key metadata \
  --rollout-shuffle \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-8}" \
  --n-samples-per-prompt 1 \
  --num-epoch "${NUM_EPOCH:-1}" \
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-3968}" \
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-4096}" \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-128}" \
  --loss-type sft_loss \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns \
  --debug-train-only \
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-1}" \
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}" \
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}" \
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}" \
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}" \
  --no-rope-fusion \
  --no-masked-softmax-fusion \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --transformer-impl "${TRANSFORMER_IMPL:-local}" \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}" \
  --qkv-format "${QKV_FORMAT:-bshd}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}" \
  --optimizer adam \
  --lr "${LR:-5e-6}" \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend "${ATTENTION_BACKEND:-flash}"
