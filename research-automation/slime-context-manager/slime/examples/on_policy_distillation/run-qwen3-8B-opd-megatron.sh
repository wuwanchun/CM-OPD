#!/bin/bash

# On-Policy Distillation with Megatron-based teacher model
# This example uses the original model as the teacher (self-distillation for demonstration)
# 
# IMPORTANT: This is just an example configuration!
# In practice, you should:
# 1. Use a different (stronger) model as the teacher
# 2. Adjust --opd-kl-coef based on your task
# 3. Configure proper evaluation metrics
pkill -9 sglang
ray stop --force
pkill -9 ray
pkill -9 python
pkill -9 ray
pkill -9 python


set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

Model_name="qwen3-0.6B"
MODELDIR="/root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/models"
source "/root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/slime/scripts/models/${Model_name}.sh"


CKPT_ARGS=(
   --hf-checkpoint "${MODELDIR}/${Model_name}"
   --ref-load "${MODELDIR}/${Model_name}_torch_dist"
   --load "${MODELDIR}/${Model_name}_slime/"
   --save "${MODELDIR}/${Model_name}_slime/"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --rm-type math
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 1
)

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

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo                        # Base advantage estimator (can be ppo, grpo, etc.)
   
   # OPD Configuration
   --use-opd                                          # Enable on-policy distillation
   --opd-type megatron                                # Use Megatron forward for teacher
   --opd-kl-coef 1.0                                  # CHANGE THIS: KL penalty coefficient
   # Teacher model configuration (CHANGE THIS to a stronger model!)
   --opd-teacher-load /root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/models/${Model_name}_torch_dist      # Teacher model path
   
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-8B-opd-megatron
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)


MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)




# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --colocate \
   --rollout-num-gpus 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}



####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python