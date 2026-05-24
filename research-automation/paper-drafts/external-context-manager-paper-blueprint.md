# Learning When to Expand: Progressive Evidence Disclosure for Long-Context Reasoning

_vNext 主线：不做 memory，不主打 self-distillation，不 claim general useful-information selection。本文只研究 summary-first 长文本推理中的一个具体控制问题：什么时候摘要不足，需要展开 raw evidence。日期：2026-05-19。_

---

## 1. Positioning

本文唯一主线是：

```text
Progressive Evidence Disclosure for Long-Context Reasoning
```

核心问题：

```text
Given a summary-first view of long context, can a model learn when compressed
summaries are insufficient and raw evidence should be disclosed?
```

中文表述：

```text
在 summary-first 的长文本推理中，模型能否学习什么时候摘要不足，需要展开原文证据？
```

本文明确不主打：

```text
memory
agent memory
general useful information selection
self-distillation framework
retriever training
main LLM training
```

训练机制可以使用 outcome-aware policy refinement，但它不是论文 identity。

## 2. Narrative

全文围绕这一段：

> Long-context reasoning should not begin with fully expanded raw evidence. Instead, models should first reason over a compressed summary view and selectively expand raw evidence only when summaries become insufficient. We formulate this as a progressive evidence disclosure problem and learn an expansion policy refined through outcome-aware review.

这篇论文的身份是：

```text
summary-first + progressive raw expansion
```

不是：

```text
memory / self-distillation / useful information selection
```

## 3. Method

### 3.1 Stage 1: Summary-First Context

长文本先切成 context spans：

```text
long context -> spans
```

每个 span 生成一个 **cue-preserving summary**。这个 summary 不是最终答案材料，而是 raw evidence 的 preview / pointer。

最简定义：

```text
summary must preserve entity, relation, and omission signal.
```

例子：

```text
raw:
Dosage changed from 5mg to 50mg.

bad summary:
The trial procedure changed.

good summary:
Dosage changed; exact values omitted.
```

这个设计非常关键：如果 summary 把展开线索也丢了，policy 就无法知道哪里需要 raw。

### 3.2 Stage 2: Progressive Expansion Policy

动作空间统一为：

```text
EXPAND
KEEP
DROP
```

语义：

| Action | Meaning |
|---|---|
| `EXPAND` | 展开 raw evidence。summary 信息不足，需要原文支持推理。 |
| `KEEP` | 保留 summary。摘要视图足够。 |
| `DROP` | 当前 prompt 不展示该 summary。当前无关或预算不允许。 |

渲染规则：

```text
EXPAND -> render raw_text
KEEP   -> render summary_text
DROP   -> render nothing
```

`EXPAND` 不是静态标签，而是 escalation action。

训练目标首版保持简单：

```text
L = CE(action, target_action)
```

### 3.3 Stage 3: Reasoning

最终 prompt 是：

```text
summary context + selected raw expansions
```

主模型只负责最终推理，不更新参数。

### 3.4 Stage 4: Outcome-Aware Teacher Review

teacher 不是 truth oracle，也不是 full-context supermodel。teacher 是 hindsight reviewer。

teacher 看到：

```text
summary-level context
selected raw expansions
model answer
gold/verifier outcome on train split
optional success/failure contrastive runs
```

teacher 纠正：

```text
which summaries should EXPAND
which EXPAND actions wasted budget
which DROP actions were wrong
```

teacher target 是：

```text
policy-improvement signal
```

不是：

```text
ground-truth usefulness label
```

### 3.5 Stage 5: Policy Refinement

训练的是 expansion policy：

```text
student action -> teacher target action -> CE/SFT update
```

如果使用 distillation/SFT/OPD，它们只是 training mechanism，不是 paper identity。

## 4. Teacher Privilege

teacher 不需要一开始看 full raw context。它的优势来自：

| Privilege | Description |
|---|---|
| Progressive disclosure | 先看 summary，必要时只展开局部 raw。 |
| Outcome privilege | 训练集可看 final answer correct/wrong、gold answer、failed answer、verifier feedback。 |
| Contrastive privilege | 同题多个 disclosure plans，有成功也有失败，可比较差异。 |

推荐表述：

```text
teacher performs batched summary-level review with outcome and contrastive runs
```

避免表述：

```text
teacher reads full long context
```

## 5. Data Schema

```json
{
  "sample_id": "ped_task001_span017",
  "task_id": "task001",
  "question": "What dosage was used?",
  "candidate_span": {
    "span_id": "span017",
    "source_id": "doc_4_sent_2",
    "raw_text": "Dosage changed from 5mg to 50mg.",
    "summary_text": "Dosage changed; exact values omitted.",
    "summary_cues": ["dosage", "changed", "exact values omitted"]
  },
  "student_action": "KEEP",
  "teacher_action": "EXPAND",
  "label_source": "teacher_only",
  "quoted_summary": "Dosage changed; exact values omitted.",
  "quoted_raw": "Dosage changed from 5mg to 50mg.",
  "verifiable_reason": "The question asks for exact dosage values.",
  "contrast": {
    "success_rollout_action": "EXPAND",
    "failed_rollout_action": "KEEP"
  },
  "accepted_for_training": true
}
```

## 6. Experiments

### 6.1 Main Tasks

首版只打透三类任务：

| Task | Why |
|---|---|
| HotpotQA | 多跳 supporting facts，能测 raw expansion 是否展开了关键证据。 |
| 2WikiMultiHopQA | 关系链推理，适合测 summary cue 是否足以触发展开。 |
| Qasper | 长文档问答，适合测 summary-first 到 raw expansion。 |

LongBench、MuSiQue、coding、legal、biomedical 都放 supplementary 或后续扩展，首版不要铺太散。

### 6.2 Killer Experiment 1: Same Summary, Different Expansion Policy

固定：

```text
same summaries
same retriever
same token budget
same main model
```

只改：

```text
expansion policy
```

比较：

```text
summary-only
rule expand
learned expand
```

如果 learned expand 明显更好，主 claim 成立。

### 6.3 Killer Experiment 2: Expansion Is Not Reranking

固定：

```text
same selected spans
same summaries
same main model
same budget
```

只改：

```text
KEEP vs EXPAND
```

目的：

```text
证明提升不是“选了哪些 span”，而是“什么时候需要 raw evidence”。
```

### 6.4 Baselines

| Type | Baseline |
|---|---|
| No expansion | summary-only |
| All expansion | full raw / top-k raw |
| Window | sliding window |
| Retrieval | BM25 / frozen dense top-k |
| Compression | LLMLingua / LongLLMLingua |
| Rule | expand if summary contains entity/number/omission cue |
| Learning | static expansion SFT |
| Ours | progressive disclosure policy with outcome-aware refinement |
| Upper bound | oracle supporting fact expansion on train-only analysis |

### 6.5 Metrics

| Metric | Meaning |
|---|---|
| `answer_em_f1` | 最终答案质量。 |
| `supporting_evidence_recall` | 关键证据是否被 summary 保留或 raw 展开。 |
| `expansion_precision` | 展开的 raw span 中真正有帮助的比例。 |
| `expansion_recall` | 需要 raw 的 span 被展开的比例。 |
| `missed_expansion_rate` | 应展开但只 KEEP 或 DROP 的比例。 |
| `unnecessary_expansion_rate` | 展开但浪费 budget 的比例。 |
| `cue_preservation_rate` | summary 是否保留 entity/relation/omission cue。 |
| `token_budget_usage` | token 成本。 |

## 7. Related Work Boundary

| Work | Boundary |
|---|---|
| LLMLingua / LongLLMLingua | prompt compression；本文研究什么时候压缩摘要不足，需要 raw expansion。 |
| RECOMP | retrieve + compress；本文固定候选，学习 disclosure action。 |
| PRISM | structured incremental memory / revision；本文保留 summary preview 和 raw pointer，学习何时展开原文。 |
| RAG / reranking | 决定候选来源或排序；本文决定 summary view 是否升级为 raw evidence。 |
| Memory systems | 研究 write/forget/retrieve；本文不研究长期记忆，只研究 summary-first prompt 中的 evidence disclosure。 |

可用 related work 句子：

> Prior work compresses or schedules context. We study when compressed summaries become insufficient and raw evidence should be disclosed.

## 8. Risk Control

### 8.1 Summary Quality

最大风险是 summary 不能作为 raw index。因此必须报告：

```text
cue_preservation_rate
bad summary vs cue-preserving summary ablation
```

### 8.2 Teacher Subjectivity

teacher label 是训练信号，不是真值。必须记录：

```text
source_id
quoted_summary
quoted_raw if inspected
verifiable_reason
label_source
```

### 8.3 Scope Control

明确写：

```text
We study a specific but important control problem in summary-first long-context reasoning.
```

不要写：

```text
general long-context framework
new memory architecture
new self-distillation algorithm
```

## 9. Minimal Publishable Version

```text
Tasks:
  HotpotQA
  2WikiMultiHopQA
  Qasper

Main model:
  fixed 7B/14B/API model

Policy:
  classifier or 0.5B-1.5B model

Actions:
  EXPAND
  KEEP
  DROP

Teacher:
  summary-level batched review
  optional local raw inspection
  outcome + contrastive rollout feedback

Core proof:
  same summaries + same model + same budget
  learned expand policy > summary-only / rule expand
```

If this works, the paper has a clean claim:

```text
Summary-first long-context reasoning improves when the system learns when summaries are insufficient and raw evidence should be disclosed.
```
