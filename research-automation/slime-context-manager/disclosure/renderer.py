"""Budget-aware prompt rendering for disclosure decisions."""

from __future__ import annotations

from .schemas import ContextSpan, DisclosureDecision, normalize_action, token_count


def render_span(span: ContextSpan, action: str) -> str:
    action = normalize_action(action)
    if action == "DROP":
        return ""
    if action == "EXPAND":
        if not span.raw_text:
            raise ValueError(f"EXPAND selected but raw_text is missing for {span.span_id}")
        body = span.raw_text
        label = "RAW"
    else:
        if not span.summary_text:
            raise ValueError(f"KEEP selected but summary_text is missing for {span.span_id}")
        body = span.summary_text
        label = "SUMMARY"
    return f"[{label} source_id={span.source_id} span_id={span.span_id}]\n{body}"


def render_prompt(
    question: str,
    spans: list[ContextSpan],
    decisions: list[DisclosureDecision],
    *,
    system_prompt: str = "Answer the question using only the disclosed evidence.",
) -> str:
    span_by_id = {span.span_id: span for span in spans}
    sections: list[str] = [system_prompt, f"Question: {question}", "Evidence:"]
    for decision in decisions:
        span = span_by_id.get(decision.span_id)
        if span is None:
            continue
        rendered = render_span(span, decision.action)
        if rendered:
            sections.append(rendered)
    sections.append("Answer:")
    return "\n\n".join(sections)


def rendered_token_count(prompt: str) -> int:
    return token_count(prompt)
