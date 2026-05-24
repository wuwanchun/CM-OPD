"""Schemas for FASD-GRPO trajectory and training samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ACTION_TYPES = ("memory_action", "tool_call", "edit", "test")
FAILURE_TYPES = (
    "none",
    "missing_evidence",
    "wrong_edit",
    "test_regression",
    "repeated_tool",
    "format_error",
)


@dataclass
class LossMasks:
    """Which objective should consume this action sample."""

    grpo: bool = True
    sdft: bool = False
    replay: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LossMasks":
        data = data or {}
        return cls(grpo=bool(data.get("grpo", True)), sdft=bool(data.get("sdft", False)), replay=bool(data.get("replay", False)))

    def to_dict(self) -> dict[str, bool]:
        return {"grpo": self.grpo, "sdft": self.sdft, "replay": self.replay}


@dataclass
class TrajectoryAction:
    """One credit-assignment unit inside a long-horizon rollout."""

    task_id: str
    rollout_id: str
    group_id: str
    segment_id: str
    turn_id: int
    action_type: str
    student_action: str
    teacher_action: str = ""
    episode_reward: float = 0.0
    segment_reward: float = 0.0
    action_reward: float = 0.0
    advantage: float = 0.0
    failure_type: str = "none"
    raw_evidence_ids: list[str] = field(default_factory=list)
    hindsight_hint: str = ""
    confidence: float = 0.0
    loss_masks: LossMasks = field(default_factory=LossMasks)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryAction":
        action_type = str(data.get("action_type", "memory_action"))
        if action_type not in ACTION_TYPES:
            raise ValueError(f"unknown action_type: {action_type}")
        failure_type = str(data.get("failure_type", "none"))
        if failure_type not in FAILURE_TYPES:
            raise ValueError(f"unknown failure_type: {failure_type}")
        return cls(
            task_id=str(data["task_id"]),
            rollout_id=str(data["rollout_id"]),
            group_id=str(data.get("group_id", data["task_id"])),
            segment_id=str(data["segment_id"]),
            turn_id=int(data["turn_id"]),
            action_type=action_type,
            student_action=str(data.get("student_action", "")),
            teacher_action=str(data.get("teacher_action", "")),
            episode_reward=float(data.get("episode_reward", 0.0) or 0.0),
            segment_reward=float(data.get("segment_reward", 0.0) or 0.0),
            action_reward=float(data.get("action_reward", 0.0) or 0.0),
            advantage=float(data.get("advantage", 0.0) or 0.0),
            failure_type=failure_type,
            raw_evidence_ids=[str(x) for x in data.get("raw_evidence_ids", [])],
            hindsight_hint=str(data.get("hindsight_hint", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            loss_masks=LossMasks.from_dict(data.get("loss_masks")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "group_id": self.group_id,
            "segment_id": self.segment_id,
            "turn_id": self.turn_id,
            "action_type": self.action_type,
            "student_action": self.student_action,
            "teacher_action": self.teacher_action,
            "episode_reward": self.episode_reward,
            "segment_reward": self.segment_reward,
            "action_reward": self.action_reward,
            "advantage": self.advantage,
            "failure_type": self.failure_type,
            "raw_evidence_ids": self.raw_evidence_ids,
            "hindsight_hint": self.hindsight_hint,
            "confidence": self.confidence,
            "loss_masks": self.loss_masks.to_dict(),
            "metadata": self.metadata,
        }

    @property
    def has_raw_evidence(self) -> bool:
        return bool(self.raw_evidence_ids)


@dataclass
class SegmentRecord:
    task_id: str
    rollout_id: str
    group_id: str
    segment_id: str
    actions: list[TrajectoryAction]
    segment_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "group_id": self.group_id,
            "segment_id": self.segment_id,
            "segment_reward": self.segment_reward,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass
class RolloutRecord:
    task_id: str
    rollout_id: str
    group_id: str
    episode_reward: float
    actions: list[TrajectoryAction]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_actions(cls, actions: list[TrajectoryAction]) -> "RolloutRecord":
        if not actions:
            raise ValueError("rollout must contain at least one action")
        first = actions[0]
        return cls(
            task_id=first.task_id,
            rollout_id=first.rollout_id,
            group_id=first.group_id,
            episode_reward=first.episode_reward,
            actions=actions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "group_id": self.group_id,
            "episode_reward": self.episode_reward,
            "actions": [action.to_dict() for action in self.actions],
            "metadata": self.metadata,
        }


@dataclass
class GRPOSDFTSample:
    """Flattened training sample consumed by HF dataset tests or future slime bridge."""

    sample_id: str
    prompt: str
    response: str
    task_id: str
    rollout_id: str
    group_id: str
    segment_id: str
    turn_id: int
    advantage: float
    reward: dict[str, float]
    loss_masks: LossMasks
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "response": self.response,
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "group_id": self.group_id,
            "segment_id": self.segment_id,
            "turn_id": self.turn_id,
            "advantage": self.advantage,
            "reward": self.reward,
            "loss_masks": self.loss_masks.to_dict(),
            "metadata": self.metadata,
        }
