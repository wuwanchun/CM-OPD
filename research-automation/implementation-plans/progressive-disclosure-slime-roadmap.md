# Progressive Evidence Disclosure Slime Design

_本文档把 `Learning When to Expand: Progressive Evidence Disclosure for Long-Context Reasoning` 落成基于 slime 的工程设计。主线是 summary-first 长文本推理中的 `EXPAND / KEEP / DROP` 外部 disclosure policy，不是 agent memory、不训练主 LLM、首版不做 GRPO。训练机制是 policy self-training / on-policy distillation：当前 policy 先在自己的状态分布上生成 disclosure action，再由带 outcome 的 reviewer 产生修正目标，最后蒸馏回 policy。_

---

## 1. 论文 Claim 到工程边界

核心研究问题：

```text
Given a summary-first view of long context, can a lightweight policy learn
when compressed summaries are insufficient and raw evidence should be disclosed?
```

工程上必须固定：

```text
main LLM
retriever / candidate span source
summary generator (fixed upstream component; not a trainable contribution)
prompt template
token budget
answer evaluator
dataset split
```

工程上只训练：

```text
disclosure policy
```

训练闭环不是一次性离线 SFT，而是迭代式 self-training：

```text
π_t generates actions on current tasks
-> renderer builds prompts under fixed budget
-> fixed main LLM answers
-> evaluator produces outcome on train split
-> self-reviewer creates corrected disclosure targets
-> train π_{t+1} with on-policy distillation + replay
```

角色边界：

| Component | 是否训练 | 推理时可见信息 | 训练时额外权限 |
|---|---|---|---|
| `disclosure policy π` | 是 | question、summary view、budget state、previous allocation trace | 无，训练样本来自自身 rollout 状态 |
| `fixed main LLM M` | 否 | rendered prompt | 无，只负责答题 |
| `self-reviewer T` | 否或弱更新 | 不参与部署 | train split outcome、gold answer/evidence、失败答案、局部 raw span |
| `validator` | 否 | teacher target schema | source grounding、leakage check、confidence gating |
| `replay buffer` | 否 | 历史 accepted targets | 防止迭代漂移和遗忘 |

动作空间：

| Action | Rendered context | 含义 |
|---|---|---|
| `EXPAND` | `raw_text` | summary 信息不足，需要展开原文证据。 |
| `KEEP` | `summary_text` | summary 已足够支持当前推理。 |
| `DROP` | nothing | 当前 span 无关或预算不允许展示。 |

首版训练目标：

```text
L_disclosure = CE(target_action | state_onpolicy)
             + λ_replay CE(target_action | state_replay)
             + λ_kl KL(π_{t+1} || π_ref)
```

其中 `state_onpolicy` 必须来自当前 policy `π_t` 的真实 rollout，而不是纯规则或 oracle 状态。`π_ref` 可以是初始 policy、上一轮 policy 或 EMA policy。首版不做 policy gradient；answer outcome 只用于生成 corrected action target。

推理时必须额外有一个确定性的 budget-aware renderer，负责把 per-span action scores 变成全局可行 prompt：

```text
policy scores over EXPAND/KEEP/DROP
-> budget-aware renderer
-> final rendered prompt under token_budget
```

不作为首版主线：

```text
GRPO
main LLM fine-tuning
long-term memory
memory write / forgetting / consolidation
teacher full raw long-context prompting
new self-distillation algorithm claim
```

## 2. 系统总览

```mermaid
flowchart LR
    accTitle: Progressive Disclosure Slime Pipeline
    accDescr: The pipeline builds cue-preserving summaries, lets the current policy generate disclosure rollouts, evaluates fixed-model answers, creates outcome-aware self-review targets, exports on-policy distillation samples, trains a lightweight policy, and evaluates held-out disclosure quality.

    D["HF datasets<br/>HotpotQA / 2Wiki / Qasper"]
    SEG["Span segmentation"]
    SUM["Cue-preserving summaries"]
    ROLL["On-policy disclosure rollouts<br/>π_t + exploration baselines"]
    REN["Budget-aware renderer<br/>EXPAND / KEEP / DROP"]
    LLM["Fixed main LLM"]
    EVAL["Answer evaluator"]
    TCH["Outcome-aware self-review<br/>correct policy actions"]
    SFT["On-policy distillation JSONL"]
    TRAIN["Slime SFT / local HF CE<br/>π_t -> π_t+1"]
    MET["Held-out metrics"]

    D --> SEG --> SUM --> ROLL --> REN --> LLM --> EVAL
    EVAL --> TCH
    SUM --> TCH
    REN --> TCH
    TCH --> SFT --> TRAIN --> MET
    TRAIN --> ROLL
```

关键归因原则：

```text
same summaries
same candidate spans
same main model
same token budget
different disclosure policy
```

如果 `learned disclosure policy` 优于 `summary-only`、`rule expand` 和 `static SFT`，论文主 claim 才成立。

这里的 `K rollout` 有两种用途，不能混淆：

| 用途 | 是否必须 | 作用 |
|---|---|---|
| on-policy distillation | 必须至少 1 条来自当前 `π_t` 的 rollout | 让 policy 在自己真实会遇到的状态上学习 corrected action。 |
| contrastive review | 可选，建议 K>1 | 用同题成功/失败 rollout 差异辅助 reviewer 判断哪个 span 应该 `EXPAND / KEEP / DROP`。 |

因此不做 GRPO 时也可以运行：`K=1 current-policy rollout -> outcome-aware self-review -> CE distillation`。多条 rollout 只是提升 reviewer 质量，不是训练算法的必要条件。

## 3. Target Layout

新增主线模块，不继续把新逻辑塞进 legacy `context_manager/`、`opd/` 或 `grpo_sdft/`：

```text
research-automation/slime-context-manager/
├── disclosure/
│   ├── __init__.py
│   ├── schemas.py
│   ├── summary_builder.py
│   ├── renderer.py
│   ├── budget_allocator.py
│   ├── rollout_policy.py
│   ├── teacher_review.py
│   ├── self_review.py
│   ├── on_policy_buffer.py
│   ├── contrastive_builder.py
│   ├── slime_sft_builder.py
│   ├── slime_rollout.py
│   └── metrics.py
├── scripts/
│   ├── build_disclosure_dataset.py
│   ├── run_disclosure_rollouts.py
│   ├── build_disclosure_teacher_targets.py
│   ├── build_onpolicy_distillation_targets.py
│   ├── export_disclosure_slime_jsonl.py
│   ├── train_disclosure_hf_local.py
│   ├── train_disclosure_onpolicy.py
│   ├── train_disclosure_slime.py
│   ├── run_disclosure_self_training.py
│   ├── run_disclosure_slime_qwen3_0_6b.sh
│   └── evaluate_disclosure_policy.py
└── tests/
    ├── test_disclosure_schemas.py
    ├── test_summary_builder.py
    ├── test_renderer.py
    ├── test_budget_allocator.py
    ├── test_teacher_review.py
    ├── test_self_review.py
    ├── test_on_policy_buffer.py
    ├── test_contrastive_builder.py
    ├── test_disclosure_slime_export.py
    └── test_disclosure_metrics.py
```

兼容策略：

| Legacy | 新主线 |
|---|---|
| `context_manager/` | 保留旧 rule / HF policy wrapper，逐步迁移到 `disclosure/`。 |
| `opd/` | 保留 OpenClaw-style OPD scaffold，不作为论文 identity。 |
| `grpo_sdft/` | 保留未来 RL 扩展，不进入首版实验主表。 |
| `context_actions_*.jsonl` | 新增 `disclosure_*.jsonl`，避免语义混用。 |

## 4. Data Contracts

### 4.1 Context Span

`ContextSpan` 是最小候选证据单元。它不是 memory item，也不带长期状态。

```json
{
  "task_id": "hotpotqa_0001",
  "span_id": "doc4_sent2",
  "source_id": "hotpotqa_doc4_sent2",
  "question": "What dosage was used?",
  "raw_text": "Dosage changed from 5mg to 50mg.",
  "summary_text": "Dosage changed; exact values omitted.",
  "summary_cues": ["dosage", "changed", "exact values omitted"],
  "span_type": "supporting_sentence",
  "split": "train"
}
```

`summary_text` 的要求：

```text
preserve entity cue
preserve relation cue
preserve omission signal
keep source_id traceable
```

注意：本文不研究如何训练 summary generator。summary 是固定输入表示，工程只做最小 cue audit，避免把论文重心拖回 summarization。

坏例子：

```text
Trial procedure changed.
```

好例子：

```text
Dosage changed; exact values omitted.
```

### 4.2 Disclosure Rollout

一个 task 生成 K 个 disclosure plan。K rollout 只是为了得到 outcome-aware / contrastive teacher targets，首版不做 GRPO advantage。

```json
{
  "rollout_id": "hotpotqa_0001_r02",
  "task_id": "hotpotqa_0001",
  "policy_name": "rule_expand_omission",
  "token_budget": 4096,
  "actions": [
    {"span_id": "doc1_sent1", "action": "KEEP"},
    {"span_id": "doc4_sent2", "action": "EXPAND"},
    {"span_id": "doc9_sent3", "action": "DROP"}
  ],
  "rendered_prompt": "Question: ...",
  "answer": "50mg",
  "answer_score": 1.0,
  "supporting_evidence_hit": true
}
```

### 4.3 Budget Allocation Record

per-span 分类只是 policy 输出，最终 prompt 由 budget allocator 统一裁决。这样可以处理多个 span 同时想 `EXPAND` 但预算不足的情况。

```json
{
  "rollout_id": "hotpotqa_0001_r02",
  "token_budget": 4096,
  "reserved_prompt_tokens": 420,
  "allocator": "greedy_margin_per_cost",
  "candidate_decisions": [
    {
      "span_id": "doc4_sent2",
      "policy_action": "EXPAND",
      "expand_logit": 4.2,
      "keep_logit": 1.7,
      "drop_logit": -0.4,
      "summary_tokens": 12,
      "raw_tokens": 31,
      "upgrade_cost": 19,
      "allocation_score": 0.132
    }
  ],
  "final_actions": [
    {"span_id": "doc4_sent2", "action": "EXPAND"}
  ],
  "budget_used": 3180,
  "budget_violation": false
}
```

首版支持两个 allocator：

| Allocator | 用途 |
|---|---|
| `greedy_margin_per_cost` | 默认实现，按 `EXPAND` over `KEEP` margin / upgrade cost 排序。 |
| `knapsack_analysis` | 小规模分析用，用 policy logits 做 utility、token cost 做 weight。 |

失败条件：

```text
budget_used > token_budget
final action not in EXPAND/KEEP/DROP
EXPAND selected but raw_text missing
KEEP selected but summary_text missing
```

### 4.4 Teacher Target

Teacher target 是 policy-improvement signal，不是客观 usefulness 真值。

```json
{
  "decision_id": "hotpotqa_0001_doc4_sent2_r02",
  "task_id": "hotpotqa_0001",
  "rollout_id": "hotpotqa_0001_r02",
  "span_id": "doc4_sent2",
  "source_id": "hotpotqa_doc4_sent2",
  "student_action": "KEEP",
  "target_action": "EXPAND",
  "label_source": "teacher_only",
  "quoted_summary": "Dosage changed; exact values omitted.",
  "quoted_raw": "Dosage changed from 5mg to 50mg.",
  "verifiable_reason": "The question asks exact dosage values.",
  "contrast_rollout_id": "hotpotqa_0001_r01",
  "accepted_for_training": true
}
```

进入训练的 target 必须满足：

```text
target_action in EXPAND/KEEP/DROP
quoted_summary non-empty
verifiable_reason non-empty
source_id traceable
accepted_for_training = true
```

若缺少 `quoted_raw`，允许进入诊断集；默认不进入高置信训练集，除非 `label_source = verifier_evidence` 或 `gold_support`。

### 4.5 Slime SFT Record

导出给 slime 的首版格式保持简单，训练 response action tokens。

```json
{
  "prompt": "Question: ...\nCandidate summary: Dosage changed; exact values omitted.\nSummary cues: dosage; changed; exact values omitted\nBudget left: 1220\nChoose one action: EXPAND, KEEP, DROP.",
  "response": "{\"action\":\"EXPAND\"}",
  "metadata": {
    "task_id": "hotpotqa_0001",
    "span_id": "doc4_sent2",
    "source_id": "hotpotqa_doc4_sent2",
    "label_source": "teacher_only",
    "target_action": "EXPAND",
    "summary_tokens": 12,
    "raw_tokens": 31,
    "upgrade_cost": 19,
    "token_budget": 4096
  }
}
```

`disclosure.slime_rollout` 在 slime 侧补齐：

```text
tokens
response_length
loss_mask
metadata
```

`loss_mask` 只覆盖 response JSON action，不训练 prompt。

### 4.6 On-Policy Episode Record

`OnPolicyEpisodeRecord` 保存当前 policy 的真实 disclosure 行为、主模型答案和 evaluator outcome。它是 self-training 的核心数据，不是离线 oracle label。

```json
{
  "iteration": 2,
  "policy_checkpoint": "checkpoints/disclosure_policy_iter02",
  "task_id": "hotpotqa_train_0001",
  "rollout_id": "hotpotqa_train_0001_pi02_r00",
  "policy_name": "current_policy",
  "token_budget": 4096,
  "actions": [
    {
      "span_id": "doc4_sent2",
      "source_id": "hotpotqa_doc4_sent2",
      "student_action": "KEEP",
      "student_action_logprob": -0.31,
      "summary_tokens": 12,
      "raw_tokens": 31,
      "budget_left_before": 1220
    }
  ],
  "rendered_prompt_id": "prompt_hotpotqa_train_0001_pi02_r00",
  "main_model_answer": "5mg",
  "outcome": {
    "answer_em": 0.0,
    "answer_f1": 0.0,
    "supporting_evidence_recall": 0.5
  }
}
```

关键要求：

```text
iteration and policy_checkpoint are mandatory
student_action must be produced by the current policy checkpoint
main_model_answer comes from fixed main LLM
held-out test records never include gold answer in review fields
```

### 4.7 Self-Review Target

`SelfReviewTarget` 是带 outcome 的 reviewer 对当前 policy action 的修正。它不是客观 useful evidence 真值，而是用于改进 `π_t` 的 distillation target。

```json
{
  "iteration": 2,
  "decision_id": "hotpotqa_train_0001_doc4_sent2_pi02",
  "task_id": "hotpotqa_train_0001",
  "rollout_id": "hotpotqa_train_0001_pi02_r00",
  "span_id": "doc4_sent2",
  "source_id": "hotpotqa_doc4_sent2",
  "student_action": "KEEP",
  "target_action": "EXPAND",
  "reviewer_mode": "self_snapshot",
  "review_context": {
    "summary_view": "Dosage changed; exact values omitted.",
    "student_answer": "5mg",
    "outcome": "incorrect",
    "gold_answer": "50mg",
    "local_raw_inspected": true
  },
  "quoted_summary": "Dosage changed; exact values omitted.",
  "quoted_raw": "Dosage changed from 5mg to 50mg.",
  "verifiable_reason": "The summary signals omitted exact values, and the wrong answer used the old value.",
  "confidence": 0.86,
  "accepted_for_distillation": true
}
```

`reviewer_mode` 首版支持：

| Mode | 含义 |
|---|---|
| `self_snapshot` | 用当前 policy checkpoint 的 frozen copy 加 review prompt 生成 correction。 |
| `ema_snapshot` | 用 EMA policy 做 reviewer，减少单轮噪声。 |
| `strong_teacher` | 用 7B/14B/API reviewer 做上界或低资源 bootstrap。 |
| `verifier_rule` | 有 supporting facts 时由规则直接产生高置信 correction。 |

进入 on-policy distillation 的 target 必须满足：

```text
accepted_for_distillation = true
confidence >= threshold
source_id traceable
quoted_summary appears in summary_text
quoted_raw appears in raw_text if local_raw_inspected = true
review uses train/validation gold only, never held-out test gold
```

### 4.8 Policy Iteration Manifest

每轮 self-training 都要写 manifest，保证可复现和可回滚：

```json
{
  "iteration": 2,
  "student_policy_in": "checkpoints/disclosure_policy_iter02",
  "student_policy_out": "checkpoints/disclosure_policy_iter03",
  "reviewer_mode": "self_snapshot",
  "train_split": "hotpotqa_train",
  "num_onpolicy_episodes": 2000,
  "num_accepted_targets": 6140,
  "num_rejected_targets": 930,
  "replay_ratio": 0.25,
  "kl_reference": "checkpoints/disclosure_policy_iter02",
  "heldout_eval_file": "runs/disclosure_eval/iter03_hotpotqa_dev.json"
}
```

### 4.9 HotpotQA On-Policy Distillation Example

针对 HotpotQA，一条训练样本的完整数据流是：

```text
1. Dataset row:
   question + context documents + supporting_facts + answer

2. Span builder:
   each document sentence -> ContextSpan(raw_text, summary_text, source_id)

3. Current disclosure policy π_t:
   input = question + span summary + budget state
   output = EXPAND / KEEP / DROP for each span

4. Budget-aware renderer:
   actions -> rendered prompt under fixed token budget

5. Fixed main LLM:
   rendered prompt -> predicted answer

6. Evaluator:
   predicted answer vs gold answer
   rendered source_ids vs supporting_facts
   -> answer score + evidence recall

7. Self-reviewer:
   sees summary view, student action, wrong/correct outcome, gold answer on train split,
   optional local raw span, and optional contrastive rollout
   -> corrected target_action with grounded reason

8. Validator:
   filters ungrounded / low-confidence / leakage-risk corrections

9. Slime export:
   prompt = policy state seen by π_t
   response = corrected target action
   loss_mask = response action only

10. Train π_{t+1}:
   on-policy CE + replay CE + optional KL to π_t
```

关键点：主模型只跑答题推理；policy model 先跑 disclosure action，训练后再进入下一轮 rollout。reviewer 只在训练阶段看到 outcome，不参与测试部署。

## 5. Pipeline Phases

### Phase 0: Compatibility Audit

任务：

```text
1. 保留旧 context_actions pipeline，但 README 标记为 legacy。
2. 新增 disclosure/ 模块，不破坏旧 tests。
3. 新脚本默认读 data/disclosure/。
4. 新训练数据默认写 data/slime/disclosure_*.jsonl。
```

验收：

```bash
python -m unittest discover -s tests -v
```

### Phase 1: Summary-First Dataset

脚本：

```text
scripts/build_disclosure_dataset.py
```

典型输入：

```bash
python scripts/build_disclosure_dataset.py \
  --dataset hotpotqa/hotpot_qa \
  --config distractor \
  --split train \
  --max-rows 200
```

输出：

```text
data/disclosure/spans_train.jsonl
data/disclosure/summary_quality_train.json
```

验收：

```text
each row has raw_text and summary_text
summary_cues is non-empty
source_id can be traced back to original dataset item
cue_preservation_rate is reported as an audit metric
no training objective updates the summary generator
```

### Phase 2: Rendering And Rollouts

脚本：

```text
scripts/run_disclosure_rollouts.py
```

rollout policy：

```text
summary_only
full_raw_topk
rule_expand_omission
random_budget_perturbation
hf_policy_checkpoint
```

输出：

```text
data/disclosure/rollouts_train.jsonl
data/disclosure/budget_allocations_train.jsonl
data/disclosure/rendered_prompts_train.jsonl
data/disclosure/answers_train.jsonl
```

验收：

```text
same task has K rollout ids
each rollout has complete action map
rendered prompt obeys token_budget
main LLM is fixed across rollouts
allocator decision is reproducible from policy scores and token costs
```

### Phase 2.5: Budget-Aware Renderer

模块：

```text
disclosure/budget_allocator.py
```

输入：

```text
candidate spans
policy logits or action confidences
summary token counts
raw token counts
reserved prompt tokens
token_budget
```

输出：

```text
final_actions
budget_used
budget_violation
allocation_trace
```

默认策略：

```text
1. reserve system/task prompt tokens
2. include high-confidence KEEP summaries when possible
3. rank EXPAND upgrades by margin(EXPAND, KEEP) / upgrade_cost
4. apply upgrades while budget remains
5. DROP low-confidence or over-budget spans
```

分析策略：

```text
knapsack_analysis
```

验收：

```text
never exceeds token_budget
deterministic given same scores
greedy and knapsack traces can be compared
budget_violation_rate is reported
```

### Phase 3: Evaluation

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
budget_violation_rate
latency
teacher_tokens_per_correction
accepted_correction_rate
```

输出：

```text
runs/disclosure_eval/metrics.json
runs/disclosure_eval/analysis_report.md
runs/disclosure_eval/plots/*.svg
```

### Phase 4: On-Policy Self-Review

脚本：

```text
scripts/build_disclosure_teacher_targets.py
scripts/build_onpolicy_distillation_targets.py
```

self-review 输入：

```text
question
candidate summary
student action generated by π_t
model answer
gold/verifier outcome on train split
optional contrastive rollout difference
optional local raw_text for inspected span
```

禁止：

```text
reviewer reads full raw long context
reviewer writes ungrounded reflection
reviewer label treated as objective truth
reviewer sees held-out test gold answer
reviewer corrects actions that were not produced by current π_t unless marked as replay/bootstrap
```

输出：

```text
data/disclosure/teacher_targets_train.jsonl
data/disclosure/teacher_targets_rejected_train.jsonl
data/disclosure/teacher_cost_train.json
data/disclosure/onpolicy_targets_iterXX_train.jsonl
data/disclosure/policy_iteration_iterXX.json
```

reviewer 类型：

| Reviewer | 用途 |
|---|---|
| `self_snapshot` | 默认 self-training reviewer，使用当前 policy 的 frozen copy 加 review prompt。 |
| `ema_snapshot` | 更稳定的 self-reviewer，减少单轮噪声。 |
| `llm_teacher` / `strong_teacher` | 自动标注上界或低资源 bootstrap，输入受 schema 限制。 |
| `human_audit` | 小样本人工核查 teacher 质量。 |
| `verifier_evidence` | 有 supporting facts / exact evidence 时生成高置信 target。 |

target validator 必须检查：

```text
target_action valid
source_id exists
quoted_summary appears in candidate summary
quoted_raw appears in raw span if provided
verifiable_reason non-empty
split is train or validation, never held-out test
student_action source is current π_t, unless replay/bootstrap flag is set
confidence >= threshold for self-review labels
```

成本日志：

```text
teacher_calls
teacher_prompt_tokens
teacher_completion_tokens
accepted_targets
rejected_targets
cost_per_accepted_target
```

### Phase 4.5: On-Policy Distillation Loop

脚本：

```text
scripts/run_disclosure_self_training.py
scripts/train_disclosure_onpolicy.py
```

最小闭环：

```text
initialize π_0 from local model or rule-SFT checkpoint
for iteration t in 0..T-1:
  run π_t disclosure rollouts on train tasks
  render prompts with fixed budget allocator
  query fixed main LLM
  evaluate answers and evidence recall
  build self-review targets from failed and successful rollouts
  filter targets with validator
  mix D_onpolicy^t with replay buffer D_replay
  train π_{t+1}
  evaluate π_{t+1} on held-out dev without gold in prompt
```

训练目标：

```text
L = L_onpolicy_ce
  + λ_replay L_replay_ce
  + λ_kl KL(π_{t+1} || π_ref)
```

其中：

| Term | 作用 |
|---|---|
| `L_onpolicy_ce` | 在当前 `π_t` 真实生成的状态上学习 corrected action。 |
| `L_replay_ce` | 混入 bootstrap / 高置信历史样本，防止 self-training 漂移。 |
| `L_kl` | 约束 `π_{t+1}` 不要突然偏离上一轮或初始 reference。 |

这一步叫 on-policy distillation，而不是 GRPO。它不需要 group advantage，也不直接最大化 reward；reward/outcome 的作用是帮助 reviewer 产生 action-level 修正。

验收：

```text
each training record has policy_iteration and policy_checkpoint
accepted target must point to an on-policy rollout_id
held-out eval never includes gold answer in reviewer input
iter t+1 action accuracy and downstream answer metrics are reported
replay ratio and KL reference are written to manifest
```

### Phase 5: Slime SFT Export

脚本：

```text
scripts/export_disclosure_slime_jsonl.py
```

输出：

```text
data/slime/disclosure_train.jsonl
data/slime/disclosure_validation.jsonl
data/slime/disclosure_onpolicy_iterXX_train.jsonl
```

验收：

```text
response parses as JSON
action in EXPAND/KEEP/DROP
metadata includes task_id/span_id/source_id/label_source
metadata includes token_budget/summary_tokens/raw_tokens/upgrade_cost
metadata includes policy_iteration/policy_checkpoint/rollout_id for on-policy records
local compatibility Sample can be built
```

### Phase 6: Local HF Training

脚本：

```text
scripts/train_disclosure_hf_local.py
```

命令：

```bash
python scripts/train_disclosure_hf_local.py \
  --model-path /public/huggingface-models/Qwen/Qwen3-0.6B \
  --train-jsonl data/slime/disclosure_train.jsonl \
  --eval-jsonl data/slime/disclosure_validation.jsonl \
  --output-dir checkpoints/disclosure_policy_qwen3_0_6b
```

on-policy distillation 命令：

```bash
python scripts/train_disclosure_onpolicy.py \
  --model-path checkpoints/disclosure_policy_iter02 \
  --train-jsonl data/slime/disclosure_onpolicy_iter02_train.jsonl \
  --replay-jsonl data/slime/disclosure_replay.jsonl \
  --eval-jsonl data/slime/disclosure_validation.jsonl \
  --kl-reference checkpoints/disclosure_policy_iter02 \
  --output-dir checkpoints/disclosure_policy_iter03
```

验收：

```text
finite loss
action accuracy above random baseline
eval before/after checkpoint runs
on-policy validation improves or failure analysis is generated
```

### Phase 7: Slime Integration

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

on-policy distillation 在 slime 侧仍然走 SFT-style loss；区别在于 JSONL 来源是当前 policy rollout 后的 corrected target：

```text
--train-data data/slime/disclosure_onpolicy_iterXX_train.jsonl
--eval-data data/slime/disclosure_validation.jsonl
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
tokens / response_length / loss_mask aligned
metadata can reconstruct budget allocation
training starts on 3090
no NaN loss
eval before/after checkpoint runs
```

## 6. Main Experiments

首版数据集：

```text
HotpotQA
2WikiMultiHopQA
Qasper
```

Killer experiment 1：

```text
same summaries
same candidate spans
same main model
same token budget
compare: summary-only vs rule expand vs static SFT vs on-policy distilled expand
```

Killer experiment 2：

```text
same selected spans
same summaries
same token budget
compare: KEEP summaries vs EXPAND raw evidence vs learned mixed rendering
```

主结果表：

| Method | HotpotQA EM/F1 | 2Wiki EM/F1 | Qasper F1 | Evidence recall | Token budget |
|---|---:|---:|---:|---:|---:|
| Summary-only | TBD | TBD | TBD | TBD | TBD |
| Full raw / top-k raw | TBD | TBD | TBD | TBD | TBD |
| BM25 / frozen dense top-k | TBD | TBD | TBD | TBD | TBD |
| LLMLingua / LongLLMLingua | TBD | TBD | TBD | TBD | TBD |
| ReSP-style control | TBD | TBD | TBD | TBD | TBD |
| SentGraph-style structure | TBD | TBD | TBD | TBD | TBD |
| Graph/decomposition RAG | TBD | TBD | TBD | TBD | TBD |
| Question-centric RAG | TBD | TBD | TBD | TBD | TBD |
| Rule expand | TBD | TBD | TBD | TBD | TBD |
| Static expansion SFT | TBD | TBD | TBD | TBD | TBD |
| On-policy distilled disclosure policy | TBD | TBD | TBD | TBD | TBD |
| Oracle supporting-fact expansion | TBD | TBD | TBD | TBD | TBD |

所有数字必须来自真实评测，文档和论文中保持 `TBD` 直到测量完成。

预算扫描：

| Budget | Required outputs |
|---:|---|
| 1k | answer metrics, evidence metrics, token usage, latency |
| 2k | answer metrics, evidence metrics, token usage, latency |
| 4k | answer metrics, evidence metrics, token usage, latency |
| 8k | answer metrics, evidence metrics, token usage, latency |

迁移测试：

```text
train HotpotQA -> test 2Wiki / Qasper
train 2Wiki -> test HotpotQA
train Qasper -> test HotpotQA
```

自训练迭代曲线：

| Iteration | Reviewer mode | Accepted targets | Answer EM/F1 | Evidence recall | Missed expansion | Token budget |
|---:|---|---:|---:|---:|---:|---:|
| 0 | bootstrap | TBD | TBD | TBD | TBD | TBD |
| 1 | self_snapshot | TBD | TBD | TBD | TBD | TBD |
| 2 | self_snapshot | TBD | TBD | TBD | TBD | TBD |
| 3 | ema_snapshot | TBD | TBD | TBD | TBD | TBD |

必须报告：

```text
accepted_target_rate
reviewer_token_cost
policy_drift_from_reference
train-to-heldout gap
failure cases where self-review worsens expansion
```

baseline 归一化原则：

```text
same retriever when possible
same candidate spans when possible
same token budget
report latency and token cost
if a baseline changes retrieval/decomposition, mark it as pipeline-level comparison
```

## 7. Slime Implementation Notes

首版不要依赖通用 `python -m slime` 入口存在。设计上分两层：

| Layer | Purpose |
|---|---|
| Local HF CE | 单卡快速验证数据、loss mask、checkpoint、action accuracy。 |
| Slime SFT | 生产训练路径，复用 slime rollout/sample/loss 机制。 |

`disclosure.slime_rollout.generate_rollout_disclosure` 负责把 JSONL record 转成 slime-compatible sample：

```text
prompt + response
-> tokenizer
-> tokens
-> response_length
-> loss_mask over response tokens
-> metadata passthrough
```

失败条件：

```text
response_length <= 0
loss_mask sum != response_length
target_action not in action set
missing task_id/span_id/source_id
missing budget metadata
non-finite loss
```

## 8. Test Plan

Unit tests：

```text
test_disclosure_schemas.py
test_summary_builder.py
test_renderer.py
test_budget_allocator.py
test_teacher_review.py
test_self_review.py
test_on_policy_buffer.py
test_contrastive_builder.py
test_disclosure_slime_export.py
test_disclosure_metrics.py
```

关键断言：

```text
EXPAND / KEEP / DROP parser works
renderer maps actions deterministically
budget allocator never exceeds budget
greedy allocator deterministic
knapsack allocator matches brute-force result on small cases
summary_cues are present
teacher target has quoted_summary and verifiable_reason
teacher target cannot use held-out test gold answer
teacher cost logger records accepted/rejected counts
unaccepted teacher target is excluded from train JSONL
on-policy target must reference current policy_checkpoint
policy iteration manifest records input/output checkpoint
replay buffer mixes historical samples without overwriting on-policy metadata
low-confidence self-review is rejected
slime JSONL masks only response action tokens
K rollout diff finds changed actions
metrics compute expansion precision/recall
metrics compute budget sweep and cost fields
```

Integration tests：

```text
build 5 HotpotQA rows
generate cue-preserving summaries
run summary-only and rule-expand rollouts
evaluate fixed answers or mocked answers
build teacher targets from mocked outcomes
build self-review targets from mocked failed and successful rollouts
export slime JSONL
construct slime-compatible samples
train local HF CE for a few steps
train one on-policy distillation iteration with replay
evaluate before/after action accuracy
run budget sweep with mocked scores
```

## 9. Build Order

第一批：数据与渲染闭环。

```text
disclosure/schemas.py
disclosure/summary_builder.py
disclosure/renderer.py
disclosure/budget_allocator.py
disclosure/slime_sft_builder.py
scripts/build_disclosure_dataset.py
scripts/export_disclosure_slime_jsonl.py
tests/test_disclosure_schemas.py
tests/test_budget_allocator.py
tests/test_disclosure_slime_export.py
```

第二批：rollout、teacher target、对比样本。

```text
disclosure/rollout_policy.py
disclosure/teacher_review.py
disclosure/self_review.py
disclosure/on_policy_buffer.py
disclosure/contrastive_builder.py
scripts/run_disclosure_rollouts.py
scripts/build_disclosure_teacher_targets.py
scripts/build_onpolicy_distillation_targets.py
scripts/run_disclosure_self_training.py
tests/test_teacher_review.py
tests/test_self_review.py
tests/test_on_policy_buffer.py
tests/test_contrastive_builder.py
```

第三批：on-policy distillation 训练、slime、评测。

```text
disclosure/slime_rollout.py
disclosure/metrics.py
scripts/train_disclosure_hf_local.py
scripts/train_disclosure_onpolicy.py
scripts/train_disclosure_slime.py
scripts/evaluate_disclosure_policy.py
scripts/run_disclosure_slime_qwen3_0_6b.sh
tests/test_disclosure_metrics.py
```

## 10. Future Extensions

这些是后续扩展，不进入首版主线：

| Extension | Condition |
|---|---|
| GRPO | 当 on-policy distillation 已经强于 rule expand，再引入 K rollout reward 和 group advantage。 |
| Token-level OPD | 当 action CE 不足以学习复杂 JSON/tool-call action，再接 teacher log-probs。 |
| Preference learning | 当 pairwise budget/rendering trade-off 比三分类 CE 更重要时再引入。 |
| Learned summarizer | 非本文重点；只有当 fixed summary cue audit 成为瓶颈时再考虑。 |
| Main LLM memory tool policy | 单独写另一条论文线，不能混入本实验归因。 |
| ALFWorld / ScienceWorld | 等 QA/document reasoning 主线跑通后再迁移到多轮环境。 |
| Coding agent traces | 作为 CodeHER-style specialization，而不是 Progressive Disclosure 首版必需项。 |

## 11. Terminology Lock

| 不再使用为主线 | 使用 |
|---|---|
| memory item | context span |
| visibility | disclosure |
| raw / summary / hidden | EXPAND / KEEP / DROP |
| useful information | expansion-worthy evidence |
| self-distillation as paper identity | on-policy distillation as training mechanism |
| memory manager | disclosure policy |
