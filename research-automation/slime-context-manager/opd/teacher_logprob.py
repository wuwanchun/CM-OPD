"""Teacher log-prob helpers for OPD.

The real token log-prob query is served by SGLang/slime infrastructure. This
module keeps the interface small and testable without those dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeacherLogprobResult:
    """Teacher signal aligned to response tokens."""

    teacher_log_probs: list[float]
    teacher_topk_log_probs: list[list[float]] = field(default_factory=list)
    teacher_topk_indices: list[list[int]] = field(default_factory=list)


def append_hint_to_messages(messages: list[dict[str, Any]], hint: str) -> list[dict[str, Any]]:
    """Append a hindsight hint to the last user message."""

    cloned = [dict(message) for message in messages]
    if not cloned:
        return [{"role": "user", "content": f"[hindsight hint]\n{hint.strip()}"}]

    target_idx = None
    for idx in range(len(cloned) - 1, -1, -1):
        if cloned[idx].get("role") == "user":
            target_idx = idx
            break
    if target_idx is None:
        target_idx = len(cloned) - 1

    old_content = str(cloned[target_idx].get("content", ""))
    cloned[target_idx]["content"] = f"{old_content}\n\n[hindsight hint]\n{hint.strip()}".strip()
    return cloned


def align_log_probs(log_probs: list[float], response_len: int, pad_value: float = 0.0) -> list[float]:
    """Pad or truncate teacher log-probs to response length."""

    values = [float(value) for value in log_probs]
    if len(values) > response_len:
        return values[:response_len]
    if len(values) < response_len:
        return values + [pad_value] * (response_len - len(values))
    return values


class TeacherLogprobClient:
    """Minimal async SGLang teacher client.

    The client expects a `/generate` endpoint compatible with SGLang:
    input ids are passed in, log-probs are read from `meta_info`.
    """

    def __init__(self, generate_url: str, *, distill_topk: int = 0, timeout: float | None = None):
        self.generate_url = generate_url
        self.distill_topk = int(distill_topk)
        self.timeout = timeout

    async def compute(self, input_ids: list[int], response_len: int) -> TeacherLogprobResult:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on optional deployment deps
            raise RuntimeError("httpx is required for TeacherLogprobClient") from exc

        start_len = max(0, len(input_ids) - response_len)
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": start_len,
        }
        if self.distill_topk > 0:
            payload["top_logprobs_num"] = self.distill_topk

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.generate_url, json=payload)
            response.raise_for_status()
            result = response.json()

        meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
        input_token_logprobs = meta.get("input_token_logprobs", [])
        teacher_log_probs = _extract_1d_log_probs(input_token_logprobs, response_len)

        if self.distill_topk <= 0:
            return TeacherLogprobResult(teacher_log_probs=teacher_log_probs)

        topk_log_probs, topk_indices = _extract_topk(meta.get("input_top_logprobs", []), response_len, self.distill_topk)
        return TeacherLogprobResult(
            teacher_log_probs=teacher_log_probs,
            teacher_topk_log_probs=topk_log_probs,
            teacher_topk_indices=topk_indices,
        )


def _extract_1d_log_probs(raw: Any, response_len: int) -> list[float]:
    values: list[float] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and item:
                values.append(float(item[0] or 0.0))
            elif isinstance(item, dict):
                values.append(float(item.get("logprob", 0.0)))
            elif isinstance(item, (int, float)):
                values.append(float(item))
    if len(values) > 1:
        values = values[1:]
    return align_log_probs(values[-response_len:], response_len)


def _extract_topk(raw: Any, response_len: int, k: int) -> tuple[list[list[float]], list[list[int]]]:
    log_probs: list[list[float]] = []
    indices: list[list[int]] = []
    if isinstance(raw, list):
        rows = raw[1:] if len(raw) > 1 else raw
        for row in rows:
            row_lp: list[float] = []
            row_idx: list[int] = []
            if isinstance(row, list):
                for entry in row:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        row_lp.append(float(entry[0] or 0.0))
                        row_idx.append(int(entry[1] or 0))
                    elif isinstance(entry, dict):
                        row_lp.append(float(entry.get("logprob", 0.0)))
                        row_idx.append(int(entry.get("token_id", 0)))
            while len(row_lp) < k:
                row_lp.append(0.0)
                row_idx.append(0)
            log_probs.append(row_lp[:k])
            indices.append(row_idx[:k])

    if len(log_probs) > response_len:
        return log_probs[-response_len:], indices[-response_len:]
    while len(log_probs) < response_len:
        log_probs.insert(0, [0.0] * k)
        indices.insert(0, [0] * k)
    return log_probs, indices
