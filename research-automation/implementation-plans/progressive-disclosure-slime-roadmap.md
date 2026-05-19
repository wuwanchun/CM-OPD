# Progressive Evidence Disclosure Slime Engineering Roadmap

_vNext roadmap：把 `Learning When to Expand` 落成 slime 工程。第一阶段只训练外部 disclosure policy，动作空间为 `EXPAND / KEEP / DROP`。_

---

## 1. System Boundary

固定：

```text
main LLM
retriever
summary generator
dataset split
prompt template
token budget
answer evaluator
```

训练：

```text
disclosure policy
```

动作：

```text
EXPAND
KEEP
DROP
```

首版训练：

```text
CE(action, teacher_target_action)
```

不是：

```text
GRPO
memory architecture
new self-distillation algorithm
main LLM fine-tuning
```

## 2. Target Layout

```text
research-automation/slime-context-manager/
├── disclosure/
│   ├── __init__.py
│   ├── schemas.py
│   ├── summary_builder.py
│   ├── renderer.py
│   ├── teacher_review.py
│   ├── contrastive_builder.py
│   ├── slime_sft_builder.py
│   ├── slime_rollout.py
│   └── metrics.py
├── scripts/
│   ├── build_disclosure_dataset.py
│   ├── run_disclosure_rollouts.py
│   ├── build_disclosure_teacher_targets.py
│   ├── export_disclosure_slime_jsonl.py
│   ├── train_disclosure_hf_local.py
│   ├── train_disclosure_slime.py
│   ├── run_disclosure_slime_qwen3_0_6b.sh
│   └── evaluate_disclosure_policy.py
└── tests/
    ├── test_disclosure_schemas.py
    ├── test_summary_builder.py
    ├── test_renderer.py
    ├── test_teacher_review.py
    ├── test_contrastive_builder.py
    ├── test_disclosure_slime_export.py
    └── test_disclosure_metrics.py
```

旧 `context_manager/`、`opd/`、`grpo_sdft/` 保留为兼容层，不作为新主线。

## 3. Phase 0: Compatibility Audit

任务：

```text
1. 保留旧 context_actions pipeline，但标记 legacy。
2. 新增 disclosure/ 模块，不破坏旧 tests。
3. README 明确主线是 progressive evidence disclosure。
4. 新脚本默认读 data/disclosure/，不读 data/processed/context_actions/。
```

验收：

```text
python -m unittest discover -s tests -v
```

## 4. Phase 1: Summary-First Dataset

脚本：

```text
scripts/build_disclosure_dataset.py
```

输入：

```text
--dataset hotpotqa/hotpot_qa
--config distractor
--split train
--max-rows 200
```

输出：

```text
data/disclosure/spans_train.jsonl
data/disclosure/summary_quality_train.json
```

schema：

```json
{
  "task_id": "hotpotqa_0001",
  "span_id": "doc4_sent2",
  "source_id": "hotpotqa_doc4_sent2",
  "question": "What dosage was used?",
  "raw_text": "Dosage changed from 5mg to 50mg.",
  "summary_text": "Dosage changed; exact values omitted.",
  "summary_cues": ["dosage", "changed", "exact values omitted"],
  "split": "train"
}
```

cue-preserving summary 最小要求：

```text
entity cue
relation cue
omission signal
source_id
```

## 5. Phase 2: Rendering And Rollouts

脚本：

```text
scripts/run_disclosure_rollouts.py
```

渲染：

```text
EXPAND -> raw_text
KEEP   -> summary_text
DROP   -> nothing
```

K rollout 生成：

```text
rule policy
random perturbation
budget perturbation
later HF policy
```

输出：

```text
data/disclosure/rollouts_train.jsonl
data/disclosure/rendered_prompts_train.jsonl
data/disclosure/answers_train.jsonl
```

验收：

```text
same task has K rollout ids
each rollout has action map
rendered prompt obeys budget
```

## 6. Phase 3: Evaluation

脚本：

```text
scripts/evaluate_disclosure_policy.py
```

指标：

```text
answer_em_f1
supporting_evidence_recall
expansion_precision
expansion_recall
missed_expansion_rate
unnecessary_expansion_rate
cue_preservation_rate
token_budget_usage
```

验收：

```text
runs/disclosure_eval/metrics.json
runs/disclosure_eval/analysis_report.md
```

## 7. Phase 4: Batched Teacher Review

脚本：

```text
scripts/build_disclosure_teacher_targets.py
```

teacher 输入：

```text
question
summary_text
student_action
model answer
gold/verifier outcome
optional contrastive rollout difference
optional local raw_text
```

teacher 输出：

```json
{
  "student_action": "KEEP",
  "target_action": "EXPAND",
  "label_source": "teacher_only",
  "quoted_summary": "Dosage changed; exact values omitted.",
  "quoted_raw": "Dosage changed from 5mg to 50mg.",
  "verifiable_reason": "The question asks exact dosage values."
}
```

验收：

```text
target_action in EXPAND/KEEP/DROP
quoted_summary non-empty
verifiable_reason non-empty
teacher never receives full raw long context
```

## 8. Phase 5: Slime SFT Export

脚本：

```text
scripts/export_disclosure_slime_jsonl.py
```

输出：

```text
data/slime/disclosure_train.jsonl
data/slime/disclosure_validation.jsonl
```

record：

```json
{
  "prompt": "Question: ...\nCandidate summary: ...\nChoose one action: EXPAND, KEEP, DROP.",
  "response": "{\"action\":\"EXPAND\"}",
  "metadata": {
    "task_id": "hotpotqa_0001",
    "span_id": "doc4_sent2",
    "target_action": "EXPAND"
  }
}
```

验收：

```text
response-token loss mask only covers action JSON
metadata includes task_id/span_id/source_id/label_source
local compatibility Sample can be built
```

## 9. Phase 6: Local HF Training

脚本：

```text
scripts/train_disclosure_hf_local.py
```

命令：

```bash
python scripts/train_disclosure_hf_local.py \
  --model-path /public/huggingface-models/Qwen/Qwen3-0.6B \
  --train-jsonl data/slime/disclosure_train.jsonl \
  --eval-jsonl data/slime/disclosure_validation.jsonl
```

验收：

```text
finite loss
action accuracy above random baseline
metrics_before.json and metrics_after.json
```

## 10. Phase 7: Slime Integration

新增：

```text
disclosure/slime_rollout.py
scripts/train_disclosure_slime.py
scripts/run_disclosure_slime_qwen3_0_6b.sh
```

slime hook：

```text
--rollout-function-path disclosure.slime_rollout.generate_rollout_disclosure
--loss-type sft_loss
```

远端命令：

```bash
cd /root/CM-OPD/research-automation/slime-context-manager
CONDA_PREFIX=/root/miniconda3/envs/slime bash scripts/run_disclosure_slime_qwen3_0_6b.sh
```

验收：

```text
slime rollout batch has samples
tokens/response_length/loss_mask aligned
training starts on 3090
no NaN loss
eval before/after checkpoint runs
```

## 11. Phase 8: Main Experiments

任务：

```text
HotpotQA
2WikiMultiHopQA
Qasper
```

Killer 1：

```text
same summaries + same model + same budget
summary-only vs rule expand vs learned expand
```

Killer 2：

```text
same selected spans + same summaries + same budget
KEEP vs EXPAND
```

输出：

```text
runs/disclosure_hf/metrics_before.json
runs/disclosure_hf/metrics_after.json
runs/disclosure_hf/expansion_analysis.md
runs/disclosure_hf/plots/*.svg
```

## 12. Build Order

第一批：

```text
disclosure/schemas.py
disclosure/summary_builder.py
disclosure/renderer.py
disclosure/slime_sft_builder.py
scripts/build_disclosure_dataset.py
scripts/export_disclosure_slime_jsonl.py
tests/test_disclosure_schemas.py
tests/test_disclosure_slime_export.py
```

第二批：

```text
disclosure/teacher_review.py
disclosure/contrastive_builder.py
scripts/run_disclosure_rollouts.py
scripts/build_disclosure_teacher_targets.py
tests/test_teacher_review.py
tests/test_contrastive_builder.py
```

第三批：

```text
disclosure/slime_rollout.py
scripts/train_disclosure_hf_local.py
scripts/train_disclosure_slime.py
scripts/evaluate_disclosure_policy.py
scripts/run_disclosure_slime_qwen3_0_6b.sh
```

## 13. Migration Table

| Legacy | vNext |
|---|---|
| memory item | context span |
| visibility | disclosure |
| raw/summary/hidden | expand/keep/drop |
| useful information | expansion-worthy evidence |
| self-distillation | outcome-aware policy refinement |
| context_actions_*.jsonl | disclosure_*.jsonl |
