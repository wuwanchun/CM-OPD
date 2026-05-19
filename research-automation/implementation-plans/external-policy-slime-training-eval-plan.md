# Progressive Evidence Disclosure Training And Evaluation Plan

_vNext：基于 slime 搭建 summary-first 长文本推理训练链路。首版不做 GRPO，不训练主 LLM；只训练外部 disclosure policy，动作空间为 `EXPAND / KEEP / DROP`。_

Roadmap:

```text
research-automation/implementation-plans/progressive-disclosure-slime-roadmap.md
```

---

## 1. Goal

核心工程问题：

```text
Given cue-preserving summaries, train a policy to decide when raw evidence should be disclosed.
```

固定组件：

```text
main LLM
retriever
summary generator
prompt template
token budget
answer evaluator
```

训练组件：

```text
disclosure policy
```

动作：

```text
EXPAND  -> render raw_text
KEEP    -> render summary_text
DROP    -> render nothing
```

## 2. Data Model

### 2.1 Context Span

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

### 2.2 Cue-Preserving Summary

summary 不是最终答案材料，而是 raw preview。最少保留：

```text
entity cue
relation cue
omission signal
source_id
```

例子：

```text
bad:  Trial procedure changed.
good: Dosage changed; exact values omitted.
```

### 2.3 Disclosure Target

```json
{
  "decision_id": "hotpotqa_0001_doc4_sent2_r02",
  "task_id": "hotpotqa_0001",
  "rollout_id": "hotpotqa_0001_r02",
  "span_id": "doc4_sent2",
  "student_action": "KEEP",
  "target_action": "EXPAND",
  "label_source": "teacher_only",
  "quoted_summary": "Dosage changed; exact values omitted.",
  "quoted_raw": "Dosage changed from 5mg to 50mg.",
  "verifiable_reason": "The question asks exact dosage values.",
  "accepted_for_training": true
}
```

## 3. Pipeline

```text
HF dataset
-> span segmentation
-> cue-preserving summaries
-> K disclosure rollouts per task
-> fixed main LLM answer
-> evaluator success/failure
-> batched teacher review
-> disclosure targets
-> slime SFT JSONL
-> local HF CE/SFT
-> slime SFT integration
-> held-out evaluation
```

## 4. Teacher Review

Teacher 是 hindsight reviewer，不是 truth oracle。

teacher 输入：

```text
question
candidate summary
student action
model answer
gold/verifier outcome on train split
optional contrastive rollout difference
optional local raw_text for inspected span
```

teacher 输出：

```text
EXPAND / KEEP / DROP
quoted_summary
quoted_raw when inspected
verifiable_reason
label_source
```

禁止首版使用：

```text
teacher full raw long-context prompt
```

## 5. Contrastive Rollouts Without GRPO

多 rollout 用来增强标注，不用来做 GRPO：

```text
same task -> K disclosure plans
-> evaluator marks success/failure
-> compare action maps
-> teacher reviews differing summaries
-> generate better targets
-> CE/SFT train policy
```

不做：

```text
group advantage
policy-gradient update
reward normalization
```

## 6. Slime SFT Format

```json
{
  "prompt": "Question: ...\nCandidate summary: ...\nSummary cues: ...\nBudget left: ...\nChoose one action: EXPAND, KEEP, DROP.",
  "response": "{\"action\":\"EXPAND\"}",
  "metadata": {
    "task_id": "hotpotqa_0001",
    "span_id": "doc4_sent2",
    "source_id": "hotpotqa_doc4_sent2",
    "label_source": "teacher_only",
    "target_action": "EXPAND"
  }
}
```

slime rollout bridge 补齐：

```text
tokens
response_length
loss_mask
metadata
```

## 7. Scripts To Build

第一批：

```text
scripts/build_disclosure_dataset.py
scripts/export_disclosure_slime_jsonl.py
scripts/evaluate_disclosure_policy.py
```

第二批：

```text
scripts/run_disclosure_rollouts.py
scripts/build_disclosure_teacher_targets.py
```

第三批：

```text
scripts/train_disclosure_hf_local.py
scripts/train_disclosure_slime.py
scripts/run_disclosure_slime_qwen3_0_6b.sh
```

## 8. Modules To Build

```text
disclosure/schemas.py
disclosure/summary_builder.py
disclosure/renderer.py
disclosure/teacher_review.py
disclosure/contrastive_builder.py
disclosure/slime_sft_builder.py
disclosure/slime_rollout.py
disclosure/metrics.py
```

## 9. Main Experiments

首版只做：

```text
HotpotQA
2WikiMultiHopQA
Qasper
```

Killer experiment 1:

```text
same summaries
same retriever
same budget
same main model
compare: summary-only vs rule expand vs learned expand
```

Killer experiment 2:

```text
same selected spans
same summaries
same budget
compare: KEEP vs EXPAND
```

## 10. Metrics

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

## 11. Tests

```text
test_disclosure_schemas.py
test_summary_builder.py
test_renderer.py
test_teacher_review.py
test_contrastive_builder.py
test_disclosure_slime_export.py
test_disclosure_metrics.py
```

Acceptance:

```text
EXPAND / KEEP / DROP parser works
summary_cues are present
teacher target has quoted_summary and verifiable_reason
slime JSONL masks only response action tokens
K rollout diff finds changed actions
metrics compute expansion precision/recall
```

## 12. Slime Integration

首版接 slime SFT：

```text
--rollout-function-path disclosure.slime_rollout.generate_rollout_disclosure
--loss-type sft_loss
```

远端目标：

```bash
cd /root/CM-OPD/research-automation/slime-context-manager
CONDA_PREFIX=/root/miniconda3/envs/slime bash scripts/run_disclosure_slime_qwen3_0_6b.sh
```

不优先做：

```text
GRPO reward
custom loss
teacher top-k logprobs
main LLM fine-tuning
ALFWorld / ScienceWorld rollout
```

## 13. Related Work Boundary

| Work | Boundary |
|---|---|
| LLMLingua / LongLLMLingua | compression |
| RECOMP | retrieve + compress |
| PRISM | incremental structured memory/revision |
| RAG / reranking | candidate source/order |
| This project | learning when compressed summaries are insufficient and raw evidence should be disclosed |
