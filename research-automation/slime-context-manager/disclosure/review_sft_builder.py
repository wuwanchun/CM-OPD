"""Build grounded Review-SFT targets for selection training."""

from __future__ import annotations

from .schemas import ContextSpan, ReviewSFTTarget
from .selection_prompt import coerce_selection_states
from .teacher_review import heuristic_teacher_target


def build_review_sft_targets(
    spans: list[ContextSpan],
    selections: list[dict],
    answers: list[dict],
) -> list[ReviewSFTTarget]:
    span_by_id = {span.span_id: span for span in spans}
    answer_by_selection = {str(answer.get("selection_id", "")): answer for answer in answers}
    targets: list[ReviewSFTTarget] = []
    for selection in selections:
        selection_id = str(selection.get("selection_id", ""))
        answer = answer_by_selection.get(selection_id, {})
        answer_id = str(answer.get("answer_id", ""))
        token_budget = int(selection.get("token_budget", answer.get("token_budget", 0)) or 0)
        for state in coerce_selection_states(selection):
            span = span_by_id.get(state.span_id)
            if span is None or span.split == "test":
                continue
            teacher = heuristic_teacher_target(span, state, rollout_id=selection_id, label_source="answer_review")
            metadata = {
                "task_id": span.task_id,
                "span_id": span.span_id,
                "source_id": span.source_id,
                "selection_id": selection_id,
                "answer_id": answer_id,
                "token_budget": token_budget,
                "answer_score": answer.get("answer_score", 0.0),
                "supporting_evidence_hit": answer.get("supporting_evidence_hit", False),
            }
            targets.append(
                ReviewSFTTarget(
                    target_id=f"{selection_id}_{answer_id}_{span.span_id}",
                    task_id=span.task_id,
                    span_id=span.span_id,
                    source_id=span.source_id,
                    selection_id=selection_id,
                    answer_id=answer_id,
                    student_state=state.action,
                    target_state=teacher.target_action,
                    quoted_summary=teacher.quoted_summary,
                    verifiable_reason=teacher.verifiable_reason,
                    token_budget=token_budget,
                    split=span.split,
                    metadata=metadata,
                )
            )
    return targets


def build_review_prompt(span: ContextSpan, target: ReviewSFTTarget) -> str:
    return (
        "You are reviewing a progressive disclosure decision.\n"
        "Return only JSON with the corrected state for this span.\n"
        "Allowed states: EXPAND, SUMMARY, HIDDEN.\n\n"
        f"Question: {span.question}\n"
        f"Source ID: {span.source_id}\n"
        f"Candidate summary: {span.summary_text}\n"
        f"Summary cues: {'; '.join(span.summary_cues)}\n"
        f"Student state: {target.student_state}\n"
        f"Review reason: {target.verifiable_reason}\n"
        f"Token budget: {target.token_budget}\n"
        "Return JSON like {\"action\":\"SUMMARY\"}."
    )
