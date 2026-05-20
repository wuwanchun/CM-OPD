"""Teacher target validation and deterministic target builders."""

from __future__ import annotations

from .schemas import ContextSpan, DisclosureDecision, DisclosureTarget


def validate_teacher_target(target: DisclosureTarget, span: ContextSpan) -> bool:
    target.validate(span)
    return True


def heuristic_teacher_target(
    span: ContextSpan,
    decision: DisclosureDecision,
    *,
    rollout_id: str,
    label_source: str = "verifier_evidence",
) -> DisclosureTarget:
    raw_lower = span.raw_text.lower()
    summary_lower = span.summary_text.lower()
    exact_cue_missing = any(ch.isdigit() for ch in span.raw_text) and not any(ch.isdigit() for ch in span.summary_text)
    if exact_cue_missing or "omitted" in summary_lower or "details omitted" in summary_lower:
        target_action = "EXPAND"
        reason = "The summary signals omitted details or missing exact values needed for verification."
    elif any(token.lower() in raw_lower for token in span.question.split()[:6]):
        target_action = "KEEP"
        reason = "The summary keeps enough question-relevant cues for this span."
    else:
        target_action = "DROP"
        reason = "The span does not expose question-relevant cues."
    quoted_summary = span.summary_text[: min(len(span.summary_text), 200)]
    quoted_raw = span.raw_text[: min(len(span.raw_text), 200)] if target_action == "EXPAND" else ""
    target = DisclosureTarget(
        decision_id=f"{span.task_id}_{span.span_id}_{rollout_id}",
        task_id=span.task_id,
        rollout_id=rollout_id,
        span_id=span.span_id,
        source_id=span.source_id,
        student_action=decision.action,
        target_action=target_action,
        label_source=label_source,
        quoted_summary=quoted_summary,
        quoted_raw=quoted_raw,
        verifiable_reason=reason,
        accepted_for_training=span.split != "test",
        split=span.split,
    )
    if target.accepted_for_training:
        target.validate(span)
    return target
