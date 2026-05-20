"""Cue-preserving summary helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import ContextSpan


_TOKEN_RE = re.compile(r"[A-Za-z0-9_.$%-]+")


@dataclass
class SummaryAudit:
    num_spans: int
    spans_with_cues: int
    cue_preservation_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "num_spans": self.num_spans,
            "spans_with_cues": self.spans_with_cues,
            "cue_preservation_rate": self.cue_preservation_rate,
        }


def build_summary(raw_text: str, *, max_words: int = 32) -> tuple[str, list[str]]:
    words = str(raw_text).split()
    summary = " ".join(words[:max_words])
    if len(words) > max_words:
        summary = f"{summary} ... details omitted"
    cues = extract_cues(raw_text, limit=8)
    if len(words) > max_words and "details omitted" not in cues:
        cues.append("details omitted")
    return summary, cues


def extract_cues(text: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    cues: list[str] = []
    for token in _TOKEN_RE.findall(str(text)):
        normalized = token.strip()
        key = normalized.lower()
        if len(normalized) < 3 or key in seen:
            continue
        if normalized[0].isupper() or any(ch.isdigit() for ch in normalized) or len(normalized) >= 7:
            cues.append(normalized)
            seen.add(key)
        if len(cues) >= limit:
            break
    if not cues:
        cues = [word for word in str(text).split()[: min(limit, 3)] if word]
    return cues


def span_from_raw(
    *,
    task_id: str,
    span_id: str,
    source_id: str,
    question: str,
    raw_text: str,
    split: str,
    span_type: str = "context",
    metadata: dict | None = None,
    max_summary_words: int = 32,
) -> ContextSpan:
    summary, cues = build_summary(raw_text, max_words=max_summary_words)
    return ContextSpan(
        task_id=task_id,
        span_id=span_id,
        source_id=source_id,
        question=question,
        raw_text=raw_text,
        summary_text=summary,
        summary_cues=cues,
        span_type=span_type,
        split=split,
        metadata=dict(metadata or {}),
    )


def audit_summaries(spans: list[ContextSpan]) -> SummaryAudit:
    total = len(spans)
    with_cues = sum(1 for span in spans if span.summary_cues)
    return SummaryAudit(total, with_cues, with_cues / total if total else 0.0)
