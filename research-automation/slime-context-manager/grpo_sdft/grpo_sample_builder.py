"""Build flattened FASD-GRPO training samples."""

from __future__ import annotations

import json

from .hindsight_relabeler import correction_target
from .reward_decomposer import blended_reward
from .trajectory_schema import GRPOSDFTSample, TrajectoryAction


def build_training_sample(action: TrajectoryAction) -> GRPOSDFTSample:
    """Convert one trajectory action into a joint-objective sample."""

    target = correction_target(action)
    sample_id = f"{action.rollout_id}_{action.segment_id}_t{action.turn_id:04d}"
    prompt = _render_prompt(action)
    reward = {
        "score": blended_reward(action),
        "episode_reward": action.episode_reward,
        "segment_reward": action.segment_reward,
        "action_reward": action.action_reward,
        "advantage": action.advantage,
    }
    metadata = dict(action.metadata)
    metadata.update(
        {
            "task_id": action.task_id,
            "rollout_id": action.rollout_id,
            "group_id": action.group_id,
            "segment_id": action.segment_id,
            "turn_id": action.turn_id,
            "action_type": action.action_type,
            "student_action": action.student_action,
            "teacher_action": action.teacher_action,
            "failure_type": action.failure_type,
            "raw_evidence_ids": action.raw_evidence_ids,
            "hindsight_hint": action.hindsight_hint,
            "confidence": action.confidence,
        }
    )
    return GRPOSDFTSample(
        sample_id=sample_id,
        prompt=prompt,
        response=target,
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        group_id=action.group_id,
        segment_id=action.segment_id,
        turn_id=action.turn_id,
        advantage=action.advantage,
        reward=reward,
        loss_masks=action.loss_masks,
        metadata=metadata,
    )


def build_training_samples(actions: list[TrajectoryAction]) -> list[GRPOSDFTSample]:
    return [build_training_sample(action) for action in actions]


def _render_prompt(action: TrajectoryAction) -> str:
    payload = {
        "task_id": action.task_id,
        "rollout_id": action.rollout_id,
        "segment_id": action.segment_id,
        "turn_id": action.turn_id,
        "action_type": action.action_type,
        "student_action": action.student_action,
        "failure_type": action.failure_type,
        "raw_evidence_ids": action.raw_evidence_ids,
        "hindsight_hint": action.hindsight_hint,
    }
    return (
        "You are training a long-horizon agent with FASD-GRPO.\n"
        "Return the corrected action for this trajectory step.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
