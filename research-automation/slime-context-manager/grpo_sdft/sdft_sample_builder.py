"""Build SDFT action-correction samples from extracted hints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .sdft_hint_extractor import SDFTHint
from .trajectory_schema import LossMasks, TrajectoryAction


@dataclass
class SDFTTrainingSample:
    sample_id: str
    prompt: str
    response: str
    task_id: str
    rollout_id: str
    segment_id: str
    turn_id: int
    raw_evidence_id: str
    failure_type: str
    loss_masks: LossMasks = field(default_factory=lambda: LossMasks(grpo=False, sdft=True, replay=False))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "response": self.response,
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "segment_id": self.segment_id,
            "turn_id": self.turn_id,
            "raw_evidence_id": self.raw_evidence_id,
            "failure_type": self.failure_type,
            "loss_masks": self.loss_masks.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SDFTTrainingSample":
        return cls(
            sample_id=str(data["sample_id"]),
            prompt=str(data["prompt"]),
            response=str(data["response"]),
            task_id=str(data["task_id"]),
            rollout_id=str(data["rollout_id"]),
            segment_id=str(data["segment_id"]),
            turn_id=int(data["turn_id"]),
            raw_evidence_id=str(data["raw_evidence_id"]),
            failure_type=str(data.get("failure_type", "none")),
            loss_masks=LossMasks.from_dict(data.get("loss_masks")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_slime_sample(self) -> Any:
        reward = {"score": 1.0, "sdft": 1.0}
        try:
            from slime.utils.types import Sample  # type: ignore

            sample = Sample()
            sample.prompt = self.prompt
            sample.response = self.response
            sample.reward = reward
            sample.metadata = self.metadata
            sample.status = Sample.Status.COMPLETED
            return sample
        except Exception:
            return SDFTSlimeSampleStub(prompt=self.prompt, response=self.response, reward=reward, metadata=self.metadata)


@dataclass
class SDFTSlimeSampleStub:
    prompt: str
    response: str
    reward: dict[str, float]
    metadata: dict[str, Any]
    status: str = "COMPLETED"


def build_sdft_samples(
    actions: list[TrajectoryAction],
    hints: list[SDFTHint],
    *,
    min_confidence: float = 0.7,
) -> list[SDFTTrainingSample]:
    """Build SDFT samples by joining failed actions with accepted hints."""

    action_index = {(action.rollout_id, action.segment_id, action.turn_id): action for action in actions}
    samples: list[SDFTTrainingSample] = []
    for hint in hints:
        if hint.confidence < min_confidence:
            continue
        action = action_index.get((hint.rollout_id, hint.segment_id, hint.turn_id))
        if action is None:
            continue
        samples.append(_build_one(action, hint))
    return samples


def _build_one(action: TrajectoryAction, hint: SDFTHint) -> SDFTTrainingSample:
    sample_id = f"sdft_{action.rollout_id}_{action.segment_id}_t{action.turn_id:04d}"
    prompt_payload = {
        "task_id": action.task_id,
        "rollout_id": action.rollout_id,
        "segment_id": action.segment_id,
        "turn_id": action.turn_id,
        "action_type": action.action_type,
        "failed_action": action.student_action,
        "failure_type": action.failure_type,
        "raw_evidence_id": hint.raw_evidence_id,
        "hindsight_hint": hint.hint,
    }
    teacher_prompt_payload = dict(prompt_payload)
    teacher_prompt_payload["teacher_instruction"] = "Use the hindsight hint to correct the failed action."
    prompt = (
        "You are doing SDFT for a long-horizon agent.\n"
        "Given the failed trajectory action and hindsight hint, output the corrected action only.\n\n"
        f"Input:\n{json.dumps(prompt_payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    metadata = dict(action.metadata)
    metadata.update(
        {
            "task_id": action.task_id,
            "rollout_id": action.rollout_id,
            "segment_id": action.segment_id,
            "turn_id": action.turn_id,
            "failure_type": action.failure_type,
            "raw_evidence_id": hint.raw_evidence_id,
            "hindsight_hint": hint.hint,
            "confidence": hint.confidence,
            "teacher_prompt": json.dumps(teacher_prompt_payload, ensure_ascii=False, separators=(",", ":")),
            "hint_votes": hint.votes,
            "sdft_source": "failed_trajectory_hindsight",
        }
    )
    return SDFTTrainingSample(
        sample_id=sample_id,
        prompt=prompt,
        response=hint.corrected_action,
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        segment_id=action.segment_id,
        turn_id=action.turn_id,
        raw_evidence_id=hint.raw_evidence_id,
        failure_type=action.failure_type,
        metadata=metadata,
    )
