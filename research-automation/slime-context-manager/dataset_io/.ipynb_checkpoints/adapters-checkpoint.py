"""Dataset adapters for small local policy datasets."""

from __future__ import annotations

from typing import Any

from context_manager.schemas import ACTIONS


class ToyAdapter:
    """Build a tiny deterministic context-action dataset for tests and demos."""

    def build_samples(self, *, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
        rows = [
            _sample(split, 0, "task_goal", "Always cite exact dosage changes.", "PIN", {"has_user_constraint": True}),
            _sample(split, 1, "test_failure", "pytest failed in dosage parser.", "KEEP", {"has_error": True}),
            _sample(split, 2, "retrieved_doc", "Protocol appendix source pointer.", "ARCHIVE", {"token_count": 320}),
            _sample(split, 3, "scratch_note", "Long exploratory note.", "COMPRESS", {"token_count": 1200}, budget_used=95),
            _sample(split, 4, "misc", "Uncertain context item.", "KEEP", {}),
        ]
        return rows[:limit] if limit is not None else rows


def get_adapter(name: str) -> ToyAdapter:
    normalized = name.lower()
    if normalized in {"toy", "hotpotqa", "alfworld", "scienceworld"}:
        return ToyAdapter()
    raise ValueError(f"unknown dataset adapter: {name}")


def _sample(
    split: str,
    index: int,
    item_type: str,
    summary: str,
    gold_action: str,
    metadata: dict[str, Any],
    *,
    budget_used: int = 40,
) -> dict[str, Any]:
    return {
        "sample_id": f"toy_{split}_{index:03d}",
        "episode_id": f"toy_{split}",
        "split": split,
        "task_family": "toy",
        "task_goal": "Maintain useful context under budget.",
        "recent_context": "The agent is solving a small context-management task.",
        "budget": 100,
        "budget_used": budget_used,
        "memory_item": {
            "id": f"mem_{index}",
            "type": item_type,
            "summary": summary,
            "raw_pointer": f"raw_{index}",
            "metadata": metadata,
        },
        "candidate_actions": list(ACTIONS),
        "gold_action": gold_action,
        "gold_evidence_ids": [f"raw_{index}"] if gold_action in {"KEEP", "PIN"} else [],
        "labels": {"forbidden_actions": ["DROP"]} if gold_action in {"KEEP", "PIN"} else {},
        "metadata": {"task_family": "toy"},
    }
