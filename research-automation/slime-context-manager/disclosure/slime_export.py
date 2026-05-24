"""Export Review-SFT and OPD targets as Slime JSONL records."""

from __future__ import annotations

from .opd_distillation_builder import build_onpolicy_distillation_targets
from .review_sft_builder import build_review_prompt
from .schemas import ContextSpan, ReviewSFTTarget, SlimeSFTRecord


def export_review_sft_records(spans: list[ContextSpan], targets: list[ReviewSFTTarget]) -> list[dict]:
    span_by_id = {span.span_id: span for span in spans}
    records = []
    for target in targets:
        span = span_by_id.get(target.span_id)
        if span is None:
            continue
        metadata = dict(target.metadata)
        metadata.update(
            {
                "task_id": target.task_id,
                "span_id": target.span_id,
                "source_id": target.source_id,
                "selection_id": target.selection_id,
                "answer_id": target.answer_id,
                "token_budget": target.token_budget,
                "target_state": target.target_state,
            }
        )
        records.append(
            SlimeSFTRecord(
                prompt=build_review_prompt(span, target),
                response=SlimeSFTRecord.action_response(target.target_state),
                metadata=metadata,
            ).to_dict()
        )
    return records


def export_opd_records(spans: list[ContextSpan], targets: list[ReviewSFTTarget]) -> list[dict]:
    return [record.to_dict() for record in build_onpolicy_distillation_targets(spans, targets, distill_mode="ce_fallback")]
