"""Schemas for progressive evidence disclosure."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


STATES = ("EXPAND", "SUMMARY", "HIDDEN")
ACTIONS = STATES


def normalize_action(action: str) -> str:
    normalized = str(action or "").upper()
    if normalized not in STATES:
        raise ValueError(f"invalid disclosure state: {action}")
    return normalized


normalize_state = normalize_action


@dataclass
class ContextSpan:
    task_id: str
    span_id: str
    source_id: str
    question: str
    raw_text: str
    summary_text: str
    summary_cues: list[str] = field(default_factory=list)
    span_type: str = "context"
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextSpan":
        return cls(
            task_id=str(data.get("task_id", "")),
            span_id=str(data.get("span_id", "")),
            source_id=str(data.get("source_id", data.get("span_id", ""))),
            question=str(data.get("question", "")),
            raw_text=str(data.get("raw_text", "")),
            summary_text=str(data.get("summary_text", "")),
            summary_cues=[str(x) for x in data.get("summary_cues", [])],
            span_type=str(data.get("span_type", "context")),
            split=str(data.get("split", "train")),
            metadata=dict(data.get("metadata", {})),
        )

    def validate(self) -> None:
        missing = [
            name
            for name in ("task_id", "span_id", "source_id", "question", "raw_text", "summary_text")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"ContextSpan missing required fields: {', '.join(missing)}")
        if not self.summary_cues:
            raise ValueError("ContextSpan summary_cues must be non-empty")

    @property
    def summary_tokens(self) -> int:
        return token_count(self.summary_text)

    @property
    def raw_tokens(self) -> int:
        return token_count(self.raw_text)

    @property
    def upgrade_cost(self) -> int:
        return max(0, self.raw_tokens - self.summary_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "span_id": self.span_id,
            "source_id": self.source_id,
            "question": self.question,
            "raw_text": self.raw_text,
            "summary_text": self.summary_text,
            "summary_cues": self.summary_cues,
            "span_type": self.span_type,
            "split": self.split,
            "metadata": self.metadata,
        }


@dataclass
class DisclosureDecision:
    span_id: str
    action: str
    logits: dict[str, float] = field(default_factory=dict)
    score: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DisclosureDecision":
        return cls(
            span_id=str(data.get("span_id", "")),
            action=normalize_action(str(data.get("action", ""))),
            logits={str(k).upper(): float(v) for k, v in dict(data.get("logits", {})).items()},
            score=float(data.get("score", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"span_id": self.span_id, "action": self.action, "logits": self.logits, "score": self.score}


@dataclass
class SelectionRollout:
    selection_id: str
    task_id: str
    policy_name: str
    token_budget: int
    states: list[DisclosureDecision]
    prompt: str = ""
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "task_id": self.task_id,
            "policy_name": self.policy_name,
            "token_budget": self.token_budget,
            "states": [state.to_dict() for state in self.states],
            "actions": [state.to_dict() for state in self.states],
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
        }


@dataclass
class AnswerRollout:
    answer_id: str
    task_id: str
    selection_id: str
    token_budget: int
    rendered_prompt: str
    answer: str
    answer_score: float = 0.0
    supporting_evidence_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "task_id": self.task_id,
            "selection_id": self.selection_id,
            "token_budget": self.token_budget,
            "rendered_prompt": self.rendered_prompt,
            "answer": self.answer,
            "answer_score": self.answer_score,
            "supporting_evidence_hit": self.supporting_evidence_hit,
            "metadata": self.metadata,
        }


@dataclass
class DisclosureRollout:
    rollout_id: str
    task_id: str
    policy_name: str
    token_budget: int
    actions: list[DisclosureDecision]
    rendered_prompt: str = ""
    answer: str = ""
    answer_score: float = 0.0
    supporting_evidence_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "task_id": self.task_id,
            "policy_name": self.policy_name,
            "token_budget": self.token_budget,
            "actions": [action.to_dict() for action in self.actions],
            "rendered_prompt": self.rendered_prompt,
            "answer": self.answer,
            "answer_score": self.answer_score,
            "supporting_evidence_hit": self.supporting_evidence_hit,
            "metadata": self.metadata,
        }


@dataclass
class BudgetAllocationRecord:
    rollout_id: str
    token_budget: int
    reserved_prompt_tokens: int
    allocator: str
    candidate_decisions: list[dict[str, Any]]
    final_actions: list[DisclosureDecision]
    budget_used: int
    budget_violation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "token_budget": self.token_budget,
            "reserved_prompt_tokens": self.reserved_prompt_tokens,
            "allocator": self.allocator,
            "candidate_decisions": self.candidate_decisions,
            "final_actions": [action.to_dict() for action in self.final_actions],
            "budget_used": self.budget_used,
            "budget_violation": self.budget_violation,
        }


@dataclass
class DisclosureTarget:
    decision_id: str
    task_id: str
    rollout_id: str
    span_id: str
    source_id: str
    student_action: str
    target_action: str
    label_source: str
    quoted_summary: str
    verifiable_reason: str
    quoted_raw: str = ""
    contrast_rollout_id: str = ""
    accepted_for_training: bool = True
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DisclosureTarget":
        return cls(
            decision_id=str(data.get("decision_id", "")),
            task_id=str(data.get("task_id", "")),
            rollout_id=str(data.get("rollout_id", "")),
            span_id=str(data.get("span_id", "")),
            source_id=str(data.get("source_id", "")),
            student_action=normalize_action(str(data.get("student_action", "SUMMARY"))),
            target_action=normalize_action(str(data.get("target_action", ""))),
            label_source=str(data.get("label_source", "teacher_only")),
            quoted_summary=str(data.get("quoted_summary", "")),
            verifiable_reason=str(data.get("verifiable_reason", "")),
            quoted_raw=str(data.get("quoted_raw", "")),
            contrast_rollout_id=str(data.get("contrast_rollout_id", "")),
            accepted_for_training=bool(data.get("accepted_for_training", True)),
            split=str(data.get("split", "train")),
            metadata=dict(data.get("metadata", {})),
        )

    def validate(self, span: ContextSpan | None = None) -> None:
        normalize_action(self.student_action)
        normalize_action(self.target_action)
        if self.split == "test":
            raise ValueError("teacher targets must not be built from held-out test split")
        if not self.source_id:
            raise ValueError("DisclosureTarget source_id is required")
        if not self.quoted_summary:
            raise ValueError("DisclosureTarget quoted_summary is required")
        if not self.verifiable_reason:
            raise ValueError("DisclosureTarget verifiable_reason is required")
        if span is not None:
            if self.source_id != span.source_id:
                raise ValueError("DisclosureTarget source_id does not match span")
            if self.quoted_summary not in span.summary_text:
                raise ValueError("quoted_summary must appear in candidate summary")
            if self.quoted_raw and self.quoted_raw not in span.raw_text:
                raise ValueError("quoted_raw must appear in raw span")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "span_id": self.span_id,
            "source_id": self.source_id,
            "student_action": self.student_action,
            "target_action": self.target_action,
            "label_source": self.label_source,
            "quoted_summary": self.quoted_summary,
            "quoted_raw": self.quoted_raw,
            "verifiable_reason": self.verifiable_reason,
            "contrast_rollout_id": self.contrast_rollout_id,
            "accepted_for_training": self.accepted_for_training,
            "split": self.split,
            "metadata": self.metadata,
        }


@dataclass
class SlimeSFTRecord:
    prompt: str
    response: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "label": self.response,
            "response": self.response,
            "metadata": self.metadata,
        }

    @classmethod
    def action_response(cls, action: str) -> str:
        return json.dumps({"action": normalize_action(action)}, ensure_ascii=False, separators=(",", ":"))


@dataclass
class ReviewSFTTarget:
    target_id: str
    task_id: str
    span_id: str
    source_id: str
    selection_id: str
    answer_id: str
    student_state: str
    target_state: str
    quoted_summary: str
    verifiable_reason: str
    token_budget: int
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewSFTTarget":
        return cls(
            target_id=str(data.get("target_id", data.get("decision_id", ""))),
            task_id=str(data.get("task_id", "")),
            span_id=str(data.get("span_id", "")),
            source_id=str(data.get("source_id", "")),
            selection_id=str(data.get("selection_id", "")),
            answer_id=str(data.get("answer_id", "")),
            student_state=normalize_action(str(data.get("student_state", data.get("student_action", "SUMMARY")))),
            target_state=normalize_action(str(data.get("target_state", data.get("target_action", "")))),
            quoted_summary=str(data.get("quoted_summary", "")),
            verifiable_reason=str(data.get("verifiable_reason", "")),
            token_budget=int(data.get("token_budget", 0)),
            split=str(data.get("split", "train")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "decision_id": self.target_id,
            "task_id": self.task_id,
            "span_id": self.span_id,
            "source_id": self.source_id,
            "selection_id": self.selection_id,
            "answer_id": self.answer_id,
            "student_state": self.student_state,
            "target_state": self.target_state,
            "target_action": self.target_state,
            "quoted_summary": self.quoted_summary,
            "verifiable_reason": self.verifiable_reason,
            "token_budget": self.token_budget,
            "split": self.split,
            "metadata": self.metadata,
        }


@dataclass
class OnPolicyDistillationTarget:
    target_id: str
    task_id: str
    span_id: str
    source_id: str
    selection_id: str
    answer_id: str
    prompt: str
    response: str
    distill_mode: str = "ce_fallback"
    teacher_logprobs: list[float] = field(default_factory=list)
    teacher_topk: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "task_id": self.task_id,
            "span_id": self.span_id,
            "source_id": self.source_id,
            "selection_id": self.selection_id,
            "answer_id": self.answer_id,
            "prompt": self.prompt,
            "response": self.response,
            "label": self.response,
            "distill_mode": self.distill_mode,
            "teacher_logprobs": self.teacher_logprobs,
            "teacher_topk": self.teacher_topk,
            "metadata": self.metadata,
        }


def token_count(text: str) -> int:
    return max(0, len(str(text).split()))
