# Slime Progressive Evidence Disclosure

[中文 README](README.zh-CN.md)

This directory contains the lightweight progressive evidence disclosure training and evaluation scaffold. The main model and retriever remain frozen; the trainable component learns when a cue-preserving summary is enough and when it should be expanded into raw evidence.

The current research framing is not long-term memory management. It is summary-to-raw expansion for long-context reasoning:

```text
candidate span
-> cue-preserving summary
-> action: HIDE / KEEP_SUMMARY / EXPAND_TO_RAW
-> rendered prompt: selected summaries + selected raw spans
```

Teacher-guided improvement is an implementation mechanism, not the paper identity. Teacher targets are policy-improvement signals from batched summary-level review with outcome feedback and optional success/failure contrastive runs. The teacher does not need to read the full raw long context.

## Quick Local Checks

```powershell
python -m unittest discover -s research-automation/slime-context-manager/tests -v
```

Run the complete no-network toy pipeline:

```powershell
python research-automation/slime-context-manager/scripts/run_smoke_pipeline.py
```

It generates toy data, exports slime JSONL, runs the rule policy, evaluates predictions, and writes analysis artifacts under `runs/smoke/`.

## End-To-End Commands

```powershell
cd research-automation/slime-context-manager
pip install -r requirements.txt

python scripts/download_datasets.py --dataset toy
python scripts/build_context_action_dataset.py --adapter toy
python scripts/export_slime_jsonl.py --split train
python scripts/import_hf_model.py --model Qwen/Qwen2.5-0.5B-Instruct --dry-run
python scripts/run_context_policy.py --policy rule --split test --output runs/rule_predictions.jsonl
python scripts/evaluate_predictions.py --split test --predictions runs/rule_predictions.jsonl --output runs/metrics.json
python scripts/visualize_results.py --metrics runs/metrics.json --output-dir runs/analysis
python scripts/train_sft_slime.py --train-jsonl data/slime/context_actions_train.jsonl
```

Practical launchers using local paths:

```powershell
# evaluation
python scripts/run_eval_local.py --policy rule --split test

# register an existing local HF checkpoint
python scripts/import_hf_model.py --source-local-path C:\path\to\local\checkpoint --local-dir models/qwen2_5_0_5b_instruct

# optionally copy the local checkpoint into the project model directory
python scripts/import_hf_model.py --source-local-path C:\path\to\local\checkpoint --local-dir models/qwen2_5_0_5b_instruct --copy

# slime SFT command generation using local model path
python scripts/run_train_local.py --local-model-path models/qwen2_5_0_5b_instruct

# real single-GPU HF SFT smoke train using the same slime JSONL; default dtype is float32 for stability
python scripts/run_train_local.py --backend hf-smoke --local-model-path C:\path\to\local\checkpoint --output-dir checkpoints\context_policy_hf_sft_smoke
```

`hf-smoke` is a minimal Transformers trainer for validating a local checkpoint,
CUDA runtime, exported slime JSONL, and response-token loss on one GPU. The full
slime/Ray/Megatron launcher remains the target production path; if your installed
slime package does not provide a `python -m slime` entrypoint, use `hf-smoke` for
the single-card training smoke and wire `opd.generate_rollout_opd` into the
cluster-specific slime launch script. Non-finite losses fail the smoke run.

For real HotpotQA data:

```powershell
python scripts/download_datasets.py --dataset hotpotqa --max-samples 200
python scripts/build_context_action_dataset.py --adapter hotpotqa --limit 500 --save-hf-dataset
```

For FASD-GRPO / CodeHER-GRPO, the primary test path reads directly from a Hugging Face dataset:

```powershell
python scripts/run_fasd_grpo_hf_dataset.py --dataset hotpotqa/hotpot_qa --config distractor --split validation --max-rows 8 --k-rollouts 4
```

This writes grouped rollout records, segment records, and FASD-GRPO samples under `data/grpo_sdft_hf/`.

For async teacher-guided correction from failed trajectories:

```powershell
python scripts/run_sdft_async_hf_dataset.py --dataset hotpotqa/hotpot_qa --config distractor --split validation --max-rows 8 --k-rollouts 4
```

This path filters failed selector decisions, extracts hindsight hints with an
`asyncio` hint extractor, writes source-grounded correction samples, and
validates a slime-compatible rollout batch.

Run local single-GPU training on the generated correction data:

```powershell
python scripts/train_sdft_hf_local.py --model-path C:\path\to\local\checkpoint --train-jsonl data\sdft_async_hf\sdft_slime_sft_validation.jsonl --output-dir checkpoints\context_policy_sdft --max-steps 16 --dtype float32
```

This trainer optimizes only response tokens for the corrected selector target.
The default dtype is `float32` for stability on small single-GPU runs.
It is useful when the installed slime package exposes rollout modules
but no generic `python -m slime` training entrypoint.

Run the same SDFT samples through THUDM/slime async training on the remote
single-GPU machine:

```bash
cd /root/CM-OPD/research-automation/slime-context-manager
CONDA_PREFIX=/root/miniconda3/envs/slime bash scripts/run_sdft_slime_async_qwen3_0_6b.sh
```

The launcher uses `grpo_sdft.async_sdft_rollout.generate_rollout_sdft` as
`--rollout-function-path`, reads `prompt/label/metadata` SDFT JSONL, fills
`tokens/response_length/loss_mask`, and trains with slime's native `sft_loss`.
If the Megatron torch-dist checkpoint is missing, it converts the local HF
checkpoint first. Override `TORCH_DIST_LOAD` or `SAVE_DIR` if the root
filesystem is tight.

## Training Modes

| Mode | Status | Description |
|---|---|---|
| Progressive Disclosure CE | Primary | Train `HIDE/KEEP_SUMMARY/EXPAND_TO_RAW` targets from source-grounded teacher review. |
| Visibility OPD | Compatibility | Backward-compatible JSON visibility labels for older scripts. |
| Token OPD | Scaffolded | Teacher log-probs can be attached to original selector tokens. |
| Top-K OPD | Scaffolded | Optional top-K distillation loss for future slime runs. |
| GRPO | Future | Turn-level rewards and episode records are preserved for later rollout grouping. |

## Project Layout

| Path | Purpose |
|---|---|
| `context_manager/` | Schemas, rule baseline, HF model policy wrapper. |
| `data/` | Dataset adapters and JSONL utilities. |
| `evaluation/` | Selector and task-level metrics. |
| `visualization/` | SVG and markdown analysis reports. |
| `opd/` | OpenClaw-style OPD recorder, proxy, rollout bridge, top-K loss hook. |
| `scripts/` | Download, preprocess, export, train, deploy, evaluate, visualize. |
| `configs/` | Model, dataset, slime, and eval configs. |
