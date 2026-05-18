# Progressive Evidence Disclosure Training And Evaluation Plan

_Implementation plan for the lightweight summary-to-raw expansion policy. OPD, token distillation, and GRPO hooks are optional extensions; the first runnable path is supervised training of a progressive disclosure policy._

---

For the build-order roadmap, see:

```text
research-automation/implementation-plans/progressive-disclosure-slime-roadmap.md
```

## 1. Goal

The main model and retriever remain frozen. The trainable component is a lightweight policy that decides how each candidate span should be disclosed under a fixed token budget:

```text
HIDE
KEEP_SUMMARY
EXPAND_TO_RAW
```

Research object:

```text
summary-to-raw escalation policy
```

Not the research object:

```text
retriever fine-tuning
main LLM fine-tuning
long-term memory management
new distillation algorithm
```

## 2. Summary-First Data Model

Every candidate span must keep both raw text and a cue-preserving summary:

```json
{
  "span_id": "span_001",
  "source_id": "hotpotqa_doc4_sent2",
  "raw_text": "The dosage changed from 5mg to 50mg after the second trial.",
  "summary_text": "Dosage changed after second trial; exact values in raw span.",
  "summary_cues": ["dosage", "changed", "second trial", "exact values"],
  "action": "KEEP_SUMMARY"
}
```

The summary is not a final answer substitute. It is a raw-access index. It should preserve:

```text
entity cue
relation cue
number/time cue or placeholder
uncertainty cue
source cue
raw access handle
```

Bad summaries should be detected in preprocessing or counted in evaluation:

```text
bad:  Details about trial procedure.
good: Dosage changed after second trial; exact values in raw span.
```

## 3. Rendering Rules

```text
HIDE          -> render nothing for this span
KEEP_SUMMARY  -> render summary_text
EXPAND_TO_RAW -> render raw_text, optionally preceded by summary_text
```

The first implementation can train a single action head:

```text
L = CE(action, target_action)
```

The paper-facing action space is progressive disclosure.

## 4. Teacher-Guided Label Builder

Teacher review must not assume a full raw long-context prompt. It should use summary-level batched review:

```text
question
candidate summary
current action
model answer
gold/verifier outcome on train split
optional contrastive rollout difference
optional raw text only when reviewing an expansion candidate
```

Teacher emits a source-grounded policy-improvement target:

```json
{
  "span_id": "span017",
  "student_action": "KEEP_SUMMARY",
  "target_action": "EXPAND_TO_RAW",
  "label_source": "teacher_only",
  "quoted_summary": "Dosage changed after second trial; exact values in raw span.",
  "quoted_raw": "The dosage changed from 5mg to 50mg after the second trial.",
  "verifiable_reason": "The question asks for exact values that are absent from the summary.",
  "accepted_for_training": true
}
```

Teacher-only labels are allowed in training, but every experiment must track them separately from:

```text
counterfactual
gold_support
verifier_evidence
contrastive_rollout
```

## 5. Contrastive Rollout Review Without GRPO

Multiple rollouts are used as a label amplifier, not as GRPO:

```text
same task -> sample K disclosure plans
-> fixed main model answers each prompt
-> evaluator splits success/failure
-> diff success and failure action maps
-> teacher reviews only differing spans in summary-first batches
-> write source-grounded target_action labels
-> train with CE/SFT
```

No GRPO in the first version:

```text
no group advantage
no policy-gradient update
no reward normalization
```

## 6. Counterfactual Audit

Optional stronger validation:

```text
hide_expanded_span
summary_instead_of_raw
expand_summary_to_raw
```

Useful audit signals:

```json
{
  "counterfactuals": [
    {"variant": "summary_instead_of_raw", "score_before": 1.0, "score_after": 0.4, "score_delta": 0.6},
    {"variant": "hide_span", "score_before": 1.0, "score_after": 0.0, "score_delta": 1.0}
  ]
}
```

Counterfactual audit is not required for every training sample. It is most valuable for:

```text
gold supporting facts
teacher-selected expansion candidates
success/failure differing spans
high-cost raw expansions
```

## 7. Dataset Pipeline

Initial HF datasets:

```text
HotpotQA
2WikiMultiHopQA
MuSiQue
Qasper / NarrativeQA extension
LongBench subset extension
```

Processing steps:

```text
download dataset
split documents into candidate spans
generate cue-preserving summaries
build fixed retriever top-k candidates
run disclosure policy
render prompt
evaluate answer
build teacher targets
export CE/SFT JSONL
```

## 8. Slime / Local Training Paths

Primary low-resource path:

```text
Progressive Disclosure CE
```

Compatibility and future paths:

| Path | Status | Purpose |
|---|---|---|
| `Progressive Disclosure CE` | primary | Train `HIDE/KEEP_SUMMARY/EXPAND_TO_RAW`. |
| `Visibility OPD` | compatibility | Reuse older `RAW/SUMMARY/HIDDEN` JSON labels. |
| `Token OPD` | optional | Attach teacher log-probs to selector output tokens. |
| `Top-K OPD` | optional | Reverse-KL distillation for slime custom loss. |
| `GRPO` | future | Use grouped rollouts and process rewards after CE path is stable. |

Reserved slime hooks:

```text
--rollout-function-path opd.opd_rollout.generate_rollout_opd
--custom-rm-path opd.opd_rollout.custom_rm
--loss-type custom_loss
--custom-loss-function-path opd.topk_distillation_loss.topk_distillation_loss_function
```

## 9. Evaluation Metrics

```text
answer_em_f1
supporting_evidence_recall
expansion_precision
expansion_recall
missed_expansion_rate
unnecessary_expansion_rate
cue_preservation_rate
token_budget_usage
policy_improvement_delta
teacher_only_label_rate
```

## 10. Test And Acceptance Criteria

- Parser handles `HIDE`, `KEEP_SUMMARY`, `EXPAND_TO_RAW`, unknown actions, and invalid JSON.
- Cue-preserving summary builder records `summary_cues` and `source_id`.
- Teacher target builder requires `quoted_summary`, `verifiable_reason`, and `label_source`.
- Optional raw review is span-local, not full-context.
- Contrastive rollout builder can diff success/failure action maps.
- CE sample builder masks only target action tokens.
- Held-out evaluator runs after every training round.
- Counterfactual audit computes `hide_span`, `summary_instead_of_raw`, and `expand_summary_to_raw` deltas when enabled.

## 11. References

- OpenClaw-RL OPD README: https://github.com/Gen-Verse/OpenClaw-RL/blob/main/openclaw-opd/README.md
- slime usage guide: https://github.com/THUDM/slime/blob/main/docs/en/get_started/usage.md
- LLMLingua: https://arxiv.org/abs/2310.05736
- LongLLMLingua: https://arxiv.org/abs/2310.06839
- RECOMP: https://arxiv.org/abs/2310.04408
