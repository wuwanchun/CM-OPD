"""Answer rollout generation from selection states."""

from __future__ import annotations

import json
import urllib.request

from .budget_allocator import allocate_budget
from .renderer import render_prompt
from .schemas import AnswerRollout, ContextSpan
from .selection_prompt import coerce_selection_states


def run_answer_rollouts(
    spans: list[ContextSpan],
    selections: list[dict],
    *,
    token_budget: int = 4096,
    reserved_prompt_tokens: int = 128,
    allocator: str = "greedy_margin_per_cost",
    sglang_endpoint: str = "",
) -> list[AnswerRollout]:
    spans_by_task: dict[str, list[ContextSpan]] = {}
    for span in spans:
        spans_by_task.setdefault(span.task_id, []).append(span)

    answers = []
    for index, selection in enumerate(selections):
        task_id = str(selection.get("task_id", ""))
        task_spans = spans_by_task.get(task_id, [])
        if not task_spans:
            continue
        selection_id = str(selection.get("selection_id", f"selection_{index:04d}"))
        states = coerce_selection_states(selection)
        allocation = allocate_budget(
            task_spans,
            states,
            rollout_id=selection_id,
            token_budget=token_budget,
            reserved_prompt_tokens=reserved_prompt_tokens,
            allocator=allocator,
        )
        prompt = render_prompt(task_spans[0].question, task_spans, allocation.final_actions)
        answer_text = _call_sglang(sglang_endpoint, prompt) if sglang_endpoint else _mock_answer(task_spans, allocation.final_actions)
        answer_id = f"{selection_id}_ans_{index:04d}"
        answers.append(
            AnswerRollout(
                answer_id=answer_id,
                task_id=task_id,
                selection_id=selection_id,
                token_budget=token_budget,
                rendered_prompt=prompt,
                answer=answer_text,
                answer_score=1.0 if answer_text else 0.0,
                supporting_evidence_hit=any(action.action == "EXPAND" for action in allocation.final_actions),
                metadata={"allocator": allocator, "budget_used": allocation.budget_used, "budget_violation": allocation.budget_violation},
            )
        )
    return answers


def _mock_answer(spans: list[ContextSpan], states) -> str:
    expanded = {state.span_id for state in states if state.action == "EXPAND"}
    summarized = {state.span_id for state in states if state.action == "SUMMARY"}
    for span in spans:
        if span.span_id in expanded:
            return f"Mock answer grounded in raw evidence from {span.source_id}: {span.raw_text}"
    for span in spans:
        if span.span_id in summarized:
            return f"Mock answer grounded in summary from {span.source_id}: {span.summary_text}"
    return "Mock answer: no disclosed evidence."


def _call_sglang(endpoint: str, prompt: str) -> str:
    payload = json.dumps({"prompt": prompt, "temperature": 0.0, "max_tokens": 512}).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("text", data.get("response", data.get("generated_text", ""))))
