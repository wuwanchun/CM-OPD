# Progressive Evidence Disclosure Slime Engineering Roadmap

_目标：把论文主线 `Learning When to Expand` 落成基于 slime 的训练-评测工程。第一阶段不做 GRPO，不训练主 LLM；只训练外部 progressive disclosure policy，动作空间为 `HIDE / KEEP_SUMMARY / EXPAND_TO_RAW`。_

---

## 1. System Boundary

固定组件：

```text
main LLM
retriever
dataset split
prompt template
token budget
answer evaluator
```

训练组件：

```text
progressive disclosure policy
```

policy 输入：

```text
question
candidate span summary
summary cues
source id
current token budget
retriever rank / score
optional previous disclosure actions
```

policy 输出：

```text
HIDE
KEEP_SUMMARY
EXPAND_TO_RAW
```

首版训练目标：

```text
CE(action, teacher_target_action)
```

slime 首版定位：

```text
slime-compatible SFT / CE training substrate
not full GRPO
not reward-only RL
```

## 2. Target Directory Layout

新增/调整模块：

```text
research-automation/slime-context-manager/
├── disclosure/
│   ├── __init__.py
│   ├── schemas.py
│   ├── summary_builder.py
│   ├── policy.py
│   ├── renderer.py
│   ├── teacher_review.py
│   ├── contrastive_builder.py
│   ├── slime_sft_builder.py
│   └── metrics.py
├── scripts/
│   ├── build_disclosure_dataset.py
│   ├── run_disclosure_rollouts.py
│   ├── build_disclosure_teacher_targets.py
│   ├── export_disclosure_slime_jsonl.py
│   ├── train_disclosure_hf_local.py
│   ├── train_disclosure_slime.py
│   └── evaluate_disclosure_policy.py
└── tests/
    ├── test_disclosure_schemas.py
    ├── test_summary_builder.py
    ├── test_renderer.py
    ├── test_teacher_review.py
    ├── test_contrastive_builder.py
    └── test_disclosure_slime_export.py
```

保留旧目录：

```text
opd/
grpo_sdft/
context_manager/
```

但首版 progressive disclosure 不再依赖旧 `memory action` 语义。旧模块只作为兼容层和迁移来源。

## 3. Data Schema

### 3.1 Candidate Span

```json
{
  "task_id": "hotpotqa_0001",
  "span_id": "doc4_sent2",
  "source_id": "hotpotqa_doc4_sent2",
  "question": "What dosage was used after the second trial?",
  "raw_text": "The dosage changed from 5mg to 50mg after the second trial.",
  "summary_text": "Dosage changed after second trial; exact values in raw span.",
  "summary_cues": ["dosage", "changed", "second trial", "exact values"],
  "retriever_rank": 4,
  "retriever_score": 8.3,
  "split": "train"
}
```

### 3.2 Disclosure Decision

```json
{
  "decision_id": "hotpotqa_0001_doc4_sent2_r02",
  "task_id": "hotpotqa_0001",
  "rollout_id": "hotpotqa_0001_r02",
  "span_id": "doc4_sent2",
  "student_action": "KEEP_SUMMARY",
  "target_action": "EXPAND_TO_RAW",
  "label_source": "teacher_only",
  "quoted_summary": "Dosage changed after second trial; exact values in raw span.",
  "quoted_raw": "The dosage changed from 5mg to 50mg after the second trial.",
  "verifiable_reason": "The question asks exact dosage values absent from the summary.",
  "accepted_for_training": true
}
```

### 3.3 Slime SFT Record

```json
{
  "prompt": "Question: ...\nCandidate summary: ...\nBudget left: ...\nChoose one action: HIDE, KEEP_SUMMARY, EXPAND_TO_RAW.",
  "response": "{\"action\":\"EXPAND_TO_RAW\"}",
  "metadata": {
    "task_id": "hotpotqa_0001",
    "span_id": "doc4_sent2",
    "label_source": "teacher_only",
    "target_action": "EXPAND_TO_RAW"
  }
}
```

slime rollout bridge 需要补齐：

```text
tokens
response_length
loss_mask
metadata
```

## 4. Phase Plan

### Phase 0: Compatibility Audit

目标：确认旧代码哪些能复用，哪些必须隔离。

任务：

```text
1. 标记旧 ACTIONS / MemoryItem / ContextActionSample 为 legacy。
2. 新增 disclosure.schemas，不破坏旧测试。
3. README 明确旧 OPD 是 compatibility path。
4. 所有新脚本默认读取 disclosure/ 数据，不读 context_actions/。
```

验收：

```text
python -m unittest discover -s tests -v
```

### Phase 1: Summary-First Dataset Builder

目标：从 HF 数据集生成 candidate spans 和 cue-preserving summaries。

脚本：

```text
scripts/build_disclosure_dataset.py
```

输入：

```text
--dataset hotpotqa/hotpot_qa
--config distractor
--split train
--max-rows N
```

输出：

```text
data/disclosure/raw_spans_train.jsonl
data/disclosure/candidate_spans_train.jsonl
data/disclosure/summary_quality_train.jsonl
```

summary 首版生成策略：

```text
rule summary for tests
extractive summary for HF data
optional API/teacher summary later
```

cue-preserving summary 最小规则：

```text
保留实体名
保留数字/日期/时间
保留关系触发词
对精确值使用 “exact values in raw span”
保留 source_id
```

验收：

```text
summary_cues 非空
source_id 非空
raw_text 非空
summary_text 非空
cue_preservation_rate 可计算
```

### Phase 2: Renderer And Rollout Runner

目标：给定 disclosure actions，渲染 fixed-budget prompt 并调用固定主模型/模拟模型。

模块：

```text
disclosure.renderer
scripts/run_disclosure_rollouts.py
```

渲染规则：

```text
HIDE -> no text
KEEP_SUMMARY -> summary_text
EXPAND_TO_RAW -> raw_text, optionally with summary header
```

首版 rollout policy：

```text
rule policy
random budget-aware policy
epsilon perturbation policy
HF policy later
```

K rollout 生成方式：

```text
temperature over actions
epsilon hide/expand perturbation
different token budgets
rule perturbation
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
rendered prompt obeys token budget
each action map contains all candidate span ids
```

### Phase 3: Evaluator

目标：固定 evaluator 判断 answer quality 和 evidence metrics。

模块：

```text
disclosure.metrics
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

HotpotQA 首版：

```text
answer EM/F1
supporting fact overlap
supporting source expanded or summarized
```

验收：

```text
metrics.json 可生成
每个 rollout 有 success/failure label
每个 span 可回溯 action 与 source_id
```

### Phase 4: Batched Teacher Review

目标：从成功/失败 rollout 生成 source-grounded target_action。

模块：

```text
disclosure.teacher_review
disclosure.contrastive_builder
scripts/build_disclosure_teacher_targets.py
```

teacher 输入不是 full raw context，而是：

```text
question
candidate summary
student action
model answer
gold/verifier outcome
optional contrastive action difference
optional local raw_text for selected candidate
```

target action：

```text
HIDE
KEEP_SUMMARY
EXPAND_TO_RAW
```

label source：

```text
teacher_only
contrastive_rollout
gold_support
verifier_evidence
counterfactual
```

输出：

```text
data/disclosure/teacher_targets_train.jsonl
data/disclosure/teacher_review_report_train.json
```

验收：

```text
target_action 合法
quoted_summary 非空
verifiable_reason 非空
source_id 非空
teacher 不接收完整 long raw context
```

### Phase 5: Slime SFT Export

目标：把 teacher targets 转成 slime-compatible SFT 数据。

模块：

```text
disclosure.slime_sft_builder
scripts/export_disclosure_slime_jsonl.py
```

输出：

```text
data/slime/disclosure_train.jsonl
data/slime/disclosure_validation.jsonl
```

prompt template：

```text
Question:
{question}

Candidate summary:
{summary_text}

Summary cues:
{summary_cues}

Budget left:
{budget_left}

Current model answer/outcome:
{optional_train_only_outcome}

Choose exactly one action:
HIDE, KEEP_SUMMARY, EXPAND_TO_RAW
```

response：

```json
{"action":"EXPAND_TO_RAW"}
```

验收：

```text
response token mask only covers action JSON
metadata contains task_id/span_id/source_id/label_source
JSONL can be consumed by local HF trainer and slime rollout bridge
```

### Phase 6: Local HF Training

目标：在没有完整 slime launcher 时，先用 Transformers 跑通单卡 CE/SFT。

脚本：

```text
scripts/train_disclosure_hf_local.py
```

输入：

```text
--model-path /public/huggingface-models/Qwen/Qwen3-0.6B
--train-jsonl data/slime/disclosure_train.jsonl
--eval-jsonl data/slime/disclosure_validation.jsonl
```

输出：

```text
checkpoints/disclosure_policy_hf/
runs/disclosure_hf_train/metrics.json
```

验收：

```text
loss finite
action accuracy > random baseline
checkpoint optionally saved
```

### Phase 7: Slime Integration

目标：把同一份 disclosure JSONL 接入 slime。

脚本：

```text
scripts/train_disclosure_slime.py
```

首版使用 slime SFT loss：

```text
--rollout-function-path disclosure.slime_rollout.generate_rollout_disclosure
--loss-type sft_loss
```

需要新增：

```text
disclosure/slime_rollout.py
```

rollout 函数职责：

```text
read disclosure SFT JSONL
tokenize prompt/response
build loss_mask over response only
return slime Sample / local compatibility sample
```

远端运行目标：

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

### Phase 8: End-To-End HF Dataset Run

目标：不是 toy smoke，而是基于 HF 数据集跑完整链路。

命令目标：

```bash
python scripts/build_disclosure_dataset.py --dataset hotpotqa/hotpot_qa --config distractor --split train --max-rows 200
python scripts/run_disclosure_rollouts.py --split train --k-rollouts 4
python scripts/build_disclosure_teacher_targets.py --split train
python scripts/export_disclosure_slime_jsonl.py --split train
python scripts/train_disclosure_hf_local.py --model-path /public/huggingface-models/Qwen/Qwen3-0.6B
python scripts/evaluate_disclosure_policy.py --checkpoint checkpoints/disclosure_policy_hf --split validation
```

验收报告：

```text
runs/disclosure_hf/metrics_before.json
runs/disclosure_hf/metrics_after.json
runs/disclosure_hf/expansion_analysis.md
runs/disclosure_hf/plots/*.svg
```

## 5. Slime-Specific Build Order

优先级：

```text
1. disclosure.schemas
2. disclosure.slime_sft_builder
3. disclosure.slime_rollout local compatibility stubs
4. unit tests for Sample contract
5. train_disclosure_hf_local.py
6. train_disclosure_slime.py command generator
7. remote slime launcher
```

不要先做：

```text
GRPO reward
custom loss
teacher top-k logprobs
main LLM fine-tuning
ALFWorld / ScienceWorld rollout
```

这些会等 CE/SFT 路径稳定后再加。

## 6. Near-Term Code Tasks

第一批 PR / commit 应完成：

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
remote run_disclosure_slime_qwen3_0_6b.sh
```

## 7. Migration Notes

旧名到新名：

| Legacy | New |
|---|---|
| `ContextActionSample` | `DisclosureSample` |
| `MemoryItem` | `CandidateSpan` |
| `gold_action` | `target_action` |
| `PIN/KEEP/COMPRESS/DROP` | `HIDE/KEEP_SUMMARY/EXPAND_TO_RAW` |
| `context_actions_*.jsonl` | `disclosure_*.jsonl` |
| `Visibility OPD` | compatibility path |
| `FASD-Mem` | deprecated framing |

旧测试可以保留，但新测试必须覆盖 progressive disclosure 主路径。
