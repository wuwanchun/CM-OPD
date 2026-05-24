"""Build slime SFT records for disclosure targets."""

from __future__ import annotations

from .schemas import ContextSpan, DisclosureTarget, SlimeSFTRecord


def build_disclosure_prompt(span: ContextSpan, *, token_budget: int, budget_left: int | None = None) -> str:
    left = token_budget if budget_left is None else budget_left
    return (
        "You are a progressive evidence disclosure policy.\n"
        "Choose exactly one action: EXPAND, SUMMARY, HIDDEN.\n"
        "EXPAND shows raw evidence, SUMMARY shows only the summary, HIDDEN hides the span.\n\n"
        f"Question: {span.question}\n"
        f"Source ID: {span.source_id}\n"
        f"Candidate summary: {span.summary_text}\n"
        f"Summary cues: {'; '.join(span.summary_cues)}\n"
        f"Summary tokens: {span.summary_tokens}\n"
        f"Raw tokens: {span.raw_tokens}\n"
        f"Upgrade cost: {span.upgrade_cost}\n"
        f"Token budget: {token_budget}\n"
        f"Budget left: {left}\n"
        "Return only JSON like {\"action\":\"EXPAND\"}."
    )


def build_slime_sft_record(span: ContextSpan, target: DisclosureTarget, *, token_budget: int) -> SlimeSFTRecord:
    target.validate(span)
    metadata = {
        "task_id": span.task_id,
        "span_id": span.span_id,
        "source_id": span.source_id,
        "split": span.split,
        "label_source": target.label_source,
        "target_action": target.target_action,
        "student_action": target.student_action,
        "decision_id": target.decision_id,
        "summary_tokens": span.summary_tokens,
        "raw_tokens": span.raw_tokens,
        "upgrade_cost": span.upgrade_cost,
        "token_budget": token_budget,
    }
    metadata.update(target.metadata)
    return SlimeSFTRecord(
        prompt=build_disclosure_prompt(span, token_budget=token_budget),
        response=SlimeSFTRecord.action_response(target.target_action),
        metadata=metadata,
    )
