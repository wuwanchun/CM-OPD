"""Selection rollout builders for bootstrap and SGLang policies."""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from typing import Any

from .budget_allocator import allocate_budget
from .rollout_policy import policy_from_name
from .schemas import ContextSpan, SelectionRollout
from .selection_prompt import build_selection_prompt, parse_selection_response, selection_response_from_decisions


def group_spans_by_task(spans: list[ContextSpan]) -> dict[str, list[ContextSpan]]:
    grouped: dict[str, list[ContextSpan]] = defaultdict(list)
    for span in spans:
        grouped[span.task_id].append(span)
    return dict(grouped)


def run_selection_rollouts(
    spans: list[ContextSpan],
    *,
    policy_name: str = "rule_expand_omission",
    token_budget: int = 4096,
    reserved_prompt_tokens: int = 128,
    allocator: str = "greedy_margin_per_cost",
    sglang_endpoint: str = "",
) -> list[SelectionRollout]:
    rollouts = []
    policy = policy_from_name(policy_name)
    for index, (task_id, task_spans) in enumerate(sorted(group_spans_by_task(spans).items())):
        prompt = build_selection_prompt(task_spans, token_budget=token_budget)
        selection_id = f"{task_id}_sel_{index:04d}"
        if sglang_endpoint:
            response = _call_sglang(sglang_endpoint, prompt)
            decisions = parse_selection_response(response, task_spans)
        else:
            candidate_decisions = [policy.decide(span) for span in task_spans]
            allocation = allocate_budget(
                task_spans,
                candidate_decisions,
                rollout_id=selection_id,
                token_budget=token_budget,
                reserved_prompt_tokens=reserved_prompt_tokens,
                allocator=allocator,
            )
            decisions = allocation.final_actions
            response = selection_response_from_decisions(decisions)
        rollouts.append(
            SelectionRollout(
                selection_id=selection_id,
                task_id=task_id,
                policy_name=policy_name if not sglang_endpoint else "sglang_policy",
                token_budget=token_budget,
                states=decisions,
                prompt=prompt,
                response=response,
                metadata={"allocator": allocator, "reserved_prompt_tokens": reserved_prompt_tokens},
            )
        )
    return rollouts


def _call_sglang(endpoint: str, prompt: str) -> str:
    payload = json.dumps({"prompt": prompt, "temperature": 0.0, "max_tokens": 256}).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("text", data.get("response", data.get("generated_text", ""))))
