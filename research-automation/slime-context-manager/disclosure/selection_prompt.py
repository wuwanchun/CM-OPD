"""Prompt contract for summary-first span selection."""

from __future__ import annotations

import json
from typing import Any

from .schemas import ContextSpan, DisclosureDecision, STATES, normalize_state


def build_selection_prompt(spans: list[ContextSpan], *, token_budget: int) -> str:
    if not spans:
        raise ValueError("selection prompt requires at least one span")
    sections = [
        "You are a progressive evidence disclosure selector.",
        "Choose one state for every span: EXPAND, SUMMARY, or HIDDEN.",
        "EXPAND reveals raw_text. SUMMARY reveals summary_text. HIDDEN omits the span from the answer prompt.",
        "Return only a JSON object whose keys are span_id values and whose values are states.",
        f"Question: {spans[0].question}",
        f"Token budget: {token_budget}",
        "Candidate summaries:",
    ]
    for span in spans:
        sections.append(
            "\n".join(
                [
                    f"- span_id: {span.span_id}",
                    f"  source_id: {span.source_id}",
                    f"  summary_text: {span.summary_text}",
                    f"  summary_cues: {'; '.join(span.summary_cues)}",
                    f"  summary_tokens: {span.summary_tokens}",
                    f"  raw_tokens: {span.raw_tokens}",
                    f"  upgrade_cost: {span.upgrade_cost}",
                ]
            )
        )
    return "\n\n".join(sections)


def parse_selection_response(response: str, spans: list[ContextSpan]) -> list[DisclosureDecision]:
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"selection response must be JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("selection response must be a JSON object")

    expected = {span.span_id for span in spans}
    unknown = set(str(key) for key in data) - expected
    if unknown:
        raise ValueError(f"selection response contains unknown span_id values: {sorted(unknown)}")

    decisions = []
    for span in spans:
        if span.span_id not in data:
            raise ValueError(f"selection response missing span_id: {span.span_id}")
        state = normalize_state(str(data[span.span_id]))
        logits = {name: 1.0 if name == state else 0.0 for name in STATES}
        decisions.append(DisclosureDecision(span.span_id, state, logits))
    return decisions


def selection_response_from_decisions(decisions: list[DisclosureDecision]) -> str:
    return json.dumps({decision.span_id: normalize_state(decision.action) for decision in decisions}, ensure_ascii=False, separators=(",", ":"))


def coerce_selection_states(record: dict[str, Any]) -> list[DisclosureDecision]:
    raw_states = record.get("states", record.get("actions", []))
    return [DisclosureDecision.from_dict(item) for item in raw_states]
