# Slime Context Manager

[中文 README](README.zh-CN.md)

This directory contains the external context-policy training and evaluation scaffold for the evidence-preserving memory manager.

The current implementation focuses on OpenClaw-RL style OPD self-training:

```text
student memory action
-> next-state feedback
-> hindsight hint
-> teacher correction or teacher log-probs
-> slime-compatible sample
```

The main agent is not trained here. Only the external context manager policy is the training target.

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

For async SDFT from failed trajectories:

```powershell
python scripts/run_sdft_async_hf_dataset.py --dataset hotpotqa/hotpot_qa --config distractor --split validation --max-rows 8 --k-rollouts 4
```

This path filters failed trajectory actions, extracts hindsight hints with an
`asyncio` hint extractor, writes corrected SDFT samples, and validates a
slime-compatible SDFT rollout batch.

## OPD Modes

| Mode | Status | Description |
|---|---|---|
| Action OPD | Implemented | Teacher provides corrected memory action labels. |
| Token OPD | Scaffolded | Teacher log-probs can be attached to original action tokens. |
| Top-K OPD | Scaffolded | Optional top-K distillation loss for future slime runs. |
| GRPO | Future | Turn-level rewards and episode records are preserved for later rollout grouping. |

## Project Layout

| Path | Purpose |
|---|---|
| `context_manager/` | Schemas, rule baseline, HF model policy wrapper. |
| `data/` | Dataset adapters and JSONL utilities. |
| `evaluation/` | Action-level metrics. |
| `visualization/` | SVG and markdown analysis reports. |
| `opd/` | OpenClaw-style OPD recorder, proxy, rollout bridge, top-K loss hook. |
| `scripts/` | Download, preprocess, export, train, deploy, evaluate, visualize. |
| `configs/` | Model, dataset, slime, and eval configs. |
