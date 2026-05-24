"""Build OPD-style CE fallback distillation targets."""

from __future__ import annotations

from .review_sft_builder import build_review_prompt
from .schemas import ContextSpan, OnPolicyDistillationTarget, ReviewSFTTarget, SlimeSFTRecord


def build_onpolicy_distillation_targets(
    spans: list[ContextSpan],
    review_targets: list[ReviewSFTTarget],
    *,
    distill_mode: str = "ce_fallback",
) -> list[OnPolicyDistillationTarget]:
    span_by_id = {span.span_id: span for span in spans}
    out: list[OnPolicyDistillationTarget] = []
    for target in review_targets:
        span = span_by_id.get(target.span_id)
        if span is None:
            continue
        metadata = dict(target.metadata)
        metadata.update({"distill_mode": distill_mode, "target_state": target.target_state})
        out.append(
            OnPolicyDistillationTarget(
                target_id=target.target_id,
                task_id=target.task_id,
                span_id=target.span_id,
                source_id=target.source_id,
                selection_id=target.selection_id,
                answer_id=target.answer_id,
                prompt=build_review_prompt(span, target),
                response=SlimeSFTRecord.action_response(target.target_state),
                distill_mode=distill_mode,
                teacher_logprobs=[],
                teacher_topk=[],
                metadata=metadata,
            )
        )
    return out
