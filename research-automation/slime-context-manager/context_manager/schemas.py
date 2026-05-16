"""Shared schemas for context-action training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ACTIONS = ("KEEP", "COMPRESS", "ARCHIVE", "RETRIEVE", "UPDATE", "DROP", "PIN")


@dataclass
class MemoryItem:
    id: str
    type: str
    summary: str
    raw_pointer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        return cls(
            id=str(data.get("id", "")),
            type=str(data.get("type", "")),
            summary=str(data.get("summary", "")),
            raw_pointer=str(data.get("raw_pointer", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "summary": self.summary,
            "raw_pointer": self.raw_pointer,
            "metadata": self.metadata,
        }


@dataclass
class ContextActionSample:
    sample_id: str
    episode_id: str
    split: str
    task_family: str
    task_goal: str
    recent_context: str
    budget: int
    budget_used: int
    memory_item: MemoryItem
    candidate_actions: list[str]
    gold_action: str
    gold_evidence_ids: list[str] = field(default_factory=list)
    labels: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextActionSample":
        state = data.get("state", {})
        return cls(
            sample_id=str(data.get("sample_id", "")),
            episode_id=str(data.get("episode_id", "")),
            split=str(data.get("split", "train")),
            task_family=str(data.get("task_family", data.get("metadata", {}).get("task_family", ""))),
            task_goal=str(data.get("task_goal", state.get("task_goal", ""))),
            recent_context=str(data.get("recent_context", state.get("recent_context", ""))),
            budget=int(data.get("budget", state.get("budget", 0)) or 0),
            budget_used=int(data.get("budget_used", state.get("budget_used", 0)) or 0),
            memory_item=MemoryItem.from_dict(data.get("memory_item", {})),
            candidate_actions=list(data.get("candidate_actions", ACTIONS)),
            gold_action=str(data.get("gold_action", "")).upper(),
            gold_evidence_ids=[str(x) for x in data.get("gold_evidence_ids", [])],
            labels=dict(data.get("labels", {})),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def budget_pressure(self) -> float:
        if self.budget <= 0:
            return 0.0
        return min(1.0, max(0.0, self.budget_used / self.budget))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "split": self.split,
            "task_family": self.task_family,
            "task_goal": self.task_goal,
            "recent_context": self.recent_context,
            "budget": self.budget,
            "budget_used": self.budget_used,
            "memory_item": self.memory_item.to_dict(),
            "candidate_actions": self.candidate_actions,
            "gold_action": self.gold_action,
            "gold_evidence_ids": self.gold_evidence_ids,
            "labels": self.labels,
            "metadata": self.metadata,
        }


@dataclass
class Prediction:
    sample_id: str
    pred_action: str
    pred_rationale: str = ""
    confidence: float | None = None
    latency_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "pred_action": self.pred_action,
            "pred_rationale": self.pred_rationale,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "raw_output": self.raw_output,
        }
