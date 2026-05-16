"""Parsing helpers for memory action tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


ACTIONS = frozenset({"KEEP", "COMPRESS", "ARCHIVE", "RETRIEVE", "UPDATE", "DROP", "PIN"})


@dataclass
class ParsedMemoryAction:
    """Normalized memory action parse result."""

    valid: bool
    action: str | None = None
    target_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = _JSON_OBJECT_RE.search(stripped)
    return match.group(0) if match else None


def parse_memory_action(text: str | dict[str, Any]) -> ParsedMemoryAction:
    """Parse a memory action from JSON text or a dict.

    Accepted minimal shape:

    {"action": "KEEP", "target_ids": ["mem_001"]}
    """

    errors: list[str] = []
    if isinstance(text, dict):
        payload = dict(text)
    elif isinstance(text, str):
        obj = _extract_json_object(text)
        if obj is None:
            return ParsedMemoryAction(valid=False, errors=["no_json_object"])
        try:
            payload = json.loads(obj)
        except json.JSONDecodeError as exc:
            return ParsedMemoryAction(valid=False, errors=[f"json_decode_error:{exc.msg}"])
    else:
        return ParsedMemoryAction(valid=False, errors=["unsupported_input_type"])

    raw_action = payload.get("action") or payload.get("tool") or payload.get("name")
    if not isinstance(raw_action, str) or not raw_action.strip():
        errors.append("missing_action")
        action = None
    else:
        action = raw_action.rsplit(".", 1)[-1].upper()
        if action not in ACTIONS:
            errors.append(f"unknown_action:{action}")

    raw_targets = payload.get("target_ids", payload.get("memory_id", payload.get("target_id", [])))
    if isinstance(raw_targets, str):
        target_ids = [raw_targets]
    elif isinstance(raw_targets, list):
        target_ids = [str(item) for item in raw_targets if item is not None]
    elif raw_targets is None:
        target_ids = []
    else:
        target_ids = [str(raw_targets)]

    valid = not errors
    return ParsedMemoryAction(
        valid=valid,
        action=action if valid else action,
        target_ids=target_ids,
        payload=payload,
        errors=errors,
    )


def render_memory_action(action: str, target_ids: list[str] | None = None, **extra: Any) -> str:
    """Render a normalized memory action as compact JSON."""

    action_norm = action.upper()
    if action_norm not in ACTIONS:
        raise ValueError(f"unknown memory action: {action}")
    payload: dict[str, Any] = {"action": action_norm, "target_ids": target_ids or []}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
