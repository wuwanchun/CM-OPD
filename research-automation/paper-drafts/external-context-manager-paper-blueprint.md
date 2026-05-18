# Learning When to Expand: Progressive Evidence Disclosure for Long-Context Reasoning

_主线版本：不训练主 LLM，不 claim agent memory system，不把 self-distillation 当论文身份。核心问题是：给定长文本的摘要视图，模型能否学会判断哪些片段值得进一步展开为原文，以支持更可靠的推理。日期：2026-05-18。_

---

## 1. Scope

本文研究的是 **progressive evidence disclosure for long-context reasoning**。它不是长期记忆系统，也不是新的蒸馏算法。系统固定主模型、retriever、prompt template 和 token budget，只训练一个轻量外部 policy 来决定候选片段如何进入当前 prompt。

固定项：

```text
main LLM
retriever
prompt template
token budget
benchmark split
```

唯一替换项：

```text
summary-to-raw disclosure policy
```

推荐标题：

```text
Learning When to Expand: Progressive Evidence Disclosure for Long-Context Reasoning
```

备选标题：

```text
From Summary to Raw Evidence: Learning Progressive Context Disclosure for Long-Context Reasoning
```

## 2. Core Claim

长文本推理不一定需要一开始把所有原文塞进 prompt。更自然的工作流是：

```text
先看 cue-preserving summary
-> 判断哪些摘要不够充分
-> 展开少量 RAW evidence
-> 在 summary + selected raw 下推理
```

本文的核心不再是泛泛的 useful information selection，而是：

```text
learning summary-to-raw expansion for long-context reasoning
```

三类动作重新解释为：

| Action | Meaning |
|---|---|
| `HIDE` | summary 也不放入当前 prompt，表示当前无关或预算不允许。 |
| `KEEP_SUMMARY` | 保留摘要视图，说明语义线索足够。 |
| `EXPAND_TO_RAW` | 从摘要升级为原文，说明摘要不足以支撑可靠推理。 |

三点贡献：

1. 把长文本推理中的上下文选择问题收紧为 summary-to-raw escalation，而不是宽泛的 memory management 或 useful-span classification。
2. 提出 cue-preserving summary：摘要不是最终答案材料，而是 raw evidence 的可展开索引。
3. 用 outcome-aware teacher review 和成功/失败对比轨迹学习何时摘要不足、何时需要展开原文。

## 3. Abstract Draft

Long-context reasoning systems often face a trade-off between showing raw evidence and staying within a limited prompt budget. Instead of asking a model to read all raw evidence at once, we study progressive evidence disclosure: all candidate spans are first represented as cue-preserving summaries, and a lightweight policy learns which summaries should be hidden, kept as summaries, or expanded into raw evidence. This reframes context selection as a summary-to-raw escalation problem. The teacher does not require access to the full raw context; it performs batched summary-level review with outcome feedback and, when available, success/failure contrastive runs. Training uses source-grounded teacher targets as policy-improvement signals, while final validity is measured by held-out answer accuracy, supporting-evidence recall, exact-span preservation, expansion precision, and token efficiency. All empirical numbers are `TBD`.

## 4. Why This Is Cleaner

| Old framing | New framing |
|---|---|
| useful information selection | progressive evidence disclosure |
| RAW/SUMMARY/HIDDEN as parallel labels | `EXPAND_TO_RAW` as escalation from summary |
| teacher judges whether span is useful | teacher judges whether summary is sufficient |
| teacher may need full long context | teacher performs batched summary-level review |
| summary is compressed evidence | summary is a cue-preserving raw pointer |

This makes the teacher more realistic: it behaves like a reviewer who scans summaries, notices underspecified evidence, and asks to open the raw source only where needed.

## 5. Method

### 5.1 Stage 1: Cue-Preserving Summary Index

Every candidate span is converted into a summary that serves as an index into the raw text, not as a final substitute for the raw text.

Bad summary:

```text
The trial procedure changed.
```

Good cue-preserving summary:

```text
Dosage changed after the second trial; exact values are in the raw span.
```

The summary must preserve expansion cues:

```text
entity cue
relation cue
number/time cue or placeholder
uncertainty cue
source cue
raw access handle
```

Minimum span record:

```json
{
  "span_id": "doc4_sent2",
  "source_id": "hotpotqa_doc4_sent2",
  "raw_text": "The dosage changed from 5mg to 50mg after the second trial.",
  "summary_text": "Dosage changed after the second trial; exact values in raw span.",
  "summary_cues": ["dosage", "changed", "second trial", "exact values"]
}
```

If summaries are too generic, the policy cannot know when to expand. Therefore cue preservation is a core method assumption, not a preprocessing detail.

### 5.2 Stage 2: Progressive Raw Expansion

Given the summary-level view, policy chooses:

```text
a_i in {HIDE, KEEP_SUMMARY, EXPAND_TO_RAW}
```

Rendering rule:

```text
HIDE          -> render nothing for this span
KEEP_SUMMARY  -> render summary_text
EXPAND_TO_RAW -> render raw_text, optionally with summary as header
```

Training loss can remain simple:

```text
L = CE(a_i, a_i*)
```

The important shift is semantic: `RAW` is no longer just another visibility class. It is an **escalation action** triggered when the summary is insufficient.

### 5.3 Stage 3: Outcome-Aware Refinement

After the fixed main model answers, teacher reviews the run:

```text
summary-level context
selected raw expansions
model answer
gold/verifier outcome on train split
optional success/failure contrast
```

Teacher target examples:

```json
{
  "span_id": "doc4_sent2",
  "student_action": "KEEP_SUMMARY",
  "teacher_action": "EXPAND_TO_RAW",
  "quoted_summary": "Dosage changed after the second trial; exact values in raw span.",
  "quoted_raw": "The dosage changed from 5mg to 50mg after the second trial.",
  "verifiable_reason": "The question asks for the exact dosage values, which are absent from the summary."
}
```

Teacher target is not objective truth. It is a policy-improvement signal. Its value is judged by whether the trained policy improves held-out task performance.

## 6. Teacher Privilege

Teacher should not be described as reading the full raw long context. It has three more realistic privileges:

| Privilege | Mechanism | Why It Helps |
|---|---|---|
| Progressive disclosure | Teacher first sees summaries, then may inspect raw for selected spans. | Same context scale as student, but can request local expansion. |
| Outcome privilege | Teacher sees train-split final answer, failed answer, verifier feedback, or gold answer. | It can diagnose which summary may have been insufficient. |
| Contrastive privilege | Same task has multiple rendering plans, some correct and some wrong. | Differences reveal which expansions likely mattered. |

Contrastive example:

```text
Rollout A -> correct
  span_7 = EXPAND_TO_RAW

Rollout B -> wrong
  span_7 = KEEP_SUMMARY

teacher reviews span_7 summary and raw
-> target: EXPAND_TO_RAW
```

Without GRPO, multiple rollouts are still useful. They are not used for policy-gradient advantage; they are used as a **label amplifier** for better teacher correction.

## 7. Data Flow Without GRPO

```text
1. Build cue-preserving summaries for candidate spans.
2. Current policy samples K disclosure plans for the same task.
3. Fixed main model answers each rendered prompt.
4. Evaluator splits success and failure.
5. Teacher compares summary/raw decisions across success and failure.
6. Teacher generates source-grounded escalation targets.
7. Train policy with CE on HIDE / KEEP_SUMMARY / EXPAND_TO_RAW.
8. Evaluate on fixed held-out split.
9. Iterate.
```

This is not GRPO:

```text
No group advantage
No policy-gradient update
No reward normalization
```

It is:

```text
contrastive teacher-guided supervised improvement
```

## 8. Data Schema

```json
{
  "sample_id": "ped_task001_span017",
  "task_id": "task001",
  "question": "What dosage was used after the second trial?",
  "candidate_span": {
    "span_id": "span017",
    "source_id": "doc_4_sent_2",
    "raw_text": "The dosage changed from 5mg to 50mg after the second trial.",
    "summary_text": "Dosage changed after second trial; exact values in raw span.",
    "summary_cues": ["dosage", "changed", "second trial", "exact values"]
  },
  "student_action": "KEEP_SUMMARY",
  "teacher_action": "EXPAND_TO_RAW",
  "label_source": "teacher_only",
  "quoted_summary": "Dosage changed after second trial; exact values in raw span.",
  "quoted_raw": "The dosage changed from 5mg to 50mg after the second trial.",
  "verifiable_reason": "Exact dosage values are required for the answer.",
  "contrast": {
    "success_rollout_action": "EXPAND_TO_RAW",
    "failed_rollout_action": "KEEP_SUMMARY"
  },
  "accepted_for_training": true
}
```

## 9. Experiments

### 9.1 Main Tasks

首版主实验聚焦长文本 QA / evidence reasoning：

| Task | Why |
|---|---|
| HotpotQA | supporting facts 明确，适合测 raw expansion 是否找对证据。 |
| 2WikiMultiHopQA | 跨文档实体关系，适合测 summary cue 是否能引导展开。 |
| MuSiQue | 多跳组合推理，适合测 expansion credit assignment。 |
| Qasper / NarrativeQA | 长文档问答，适合测 summary-to-raw escalation。 |
| LongBench 子集 | 测跨任务稳定性。 |

### 9.2 Baselines

| Type | Baseline |
|---|---|
| No expansion | summary-only |
| All expansion | full raw / top-k raw |
| Window | sliding window |
| Retrieval | BM25 / frozen dense top-k |
| Compression | LLMLingua / LongLLMLingua |
| Rule | expand if summary contains numbers/entities/uncertainty cue |
| Learning | static expansion SFT |
| Ours | progressive disclosure policy with teacher-guided refinement |
| Upper bound | oracle supporting fact expansion on train-only analysis |

### 9.3 Metrics

| Metric | Meaning |
|---|---|
| `answer_em_f1` | 最终答案质量。 |
| `supporting_evidence_recall` | supporting facts 是否被 summary 保留或 raw 展开。 |
| `expansion_precision` | 展开的 raw span 中真正有用的比例。 |
| `expansion_recall` | 需要原文的 span 被展开的比例。 |
| `missed_expansion_rate` | 应展开但只保留 summary 或 hidden 的比例。 |
| `unnecessary_expansion_rate` | 展开但对答案无帮助的 raw span 比例。 |
| `cue_preservation_rate` | summary 是否保留足以触发展开的线索。 |
| `token_budget_usage` | raw expansion 消耗的 token。 |
| `policy_improvement_delta` | 每轮迭代后 held-out 分数变化。 |

## 10. Ablations

| Ablation | Question |
|---|---|
| bad summary vs cue-preserving summary | summary 是否能作为 raw expansion index？ |
| no raw expansion | summary-only 是否足够？ |
| expand all | 全展开是否只是靠更多 token？ |
| random expansion under same budget | policy 是否真学会展开？ |
| w/o outcome privilege | teacher 看不到结果是否变弱？ |
| w/o contrastive rollout | 成功/失败对比是否提升 target 质量？ |
| batched teacher vs full-context teacher | teacher 是否需要完整 raw context？ |
| single rollout vs K rollout | 多 rollout 是否只是增强标注质量？ |
| teacher-only vs counterfactual-audited | 反事实审计是否提升稳定性？ |

## 11. Reviewer Risks

### 11.1 这是不是 reranking？

不是。Reranking 只排序候选文本；本文学习的是从 summary view 到 raw evidence 的展开决策。输入是压缩视图，动作是 `HIDE / KEEP_SUMMARY / EXPAND_TO_RAW`。

### 11.2 teacher 看不到完整长文本，怎么标注？

teacher 不需要一开始看 full raw context。它先看 summary-level context，再对可疑 span 做局部 raw expansion。优势来自 progressive disclosure、outcome feedback 和 contrastive runs。

### 11.3 summary 如果丢了关键线索怎么办？

这是核心前提，所以本文必须引入 cue-preserving summary。摘要不是最终答案材料，而是 raw pointer。它至少要保留实体、关系、数字/时间占位、uncertainty 和 source cue。

### 11.4 RAW/SUMMARY/HIDDEN 是否随意？

新语义下不随意：`RAW` 是 escalation，`SUMMARY` 是默认压缩视图，`HIDDEN` 是当前不展示。policy 学的是“什么时候 summary 不够，需要展开 raw”。

### 11.5 不做 GRPO 是否合理？

合理。K 条 rollout 不用于 group advantage，而用于成功/失败对比，帮助 teacher 生成更好的 expansion targets。训练仍然是稳定的 CE/SFT。

## 12. Minimal Publishable Version

```text
Tasks:
  HotpotQA + 2WikiMultiHopQA + MuSiQue

Main model:
  fixed 7B/14B/API model

Policy:
  classifier or 0.5B-1.5B model

Actions:
  HIDE
  KEEP_SUMMARY
  EXPAND_TO_RAW

Teacher:
  summary-level batched review
  optional raw expansion for selected spans
  outcome + contrastive rollout feedback

Key proof:
  same backbone + same retriever + same budget
  learned expansion policy improves answer quality and evidence preservation
```

If this minimal version works, the paper has a clean claim: **long-context reasoning improves when the system learns when summaries are insufficient and raw evidence should be progressively disclosed**.
