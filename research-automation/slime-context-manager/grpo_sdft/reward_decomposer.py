"""Reward decomposition and GRPO advantage helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .trajectory_schema import RolloutRecord, TrajectoryAction


def assign_segment_and_action_rewards(actions: list[TrajectoryAction]) -> list[TrajectoryAction]:
    """Fill action and segment rewards when they are not already provided."""

    by_segment: dict[tuple[str, str], list[TrajectoryAction]] = defaultdict(list)
    for action in actions:
        if action.action_reward == 0.0:
            action.action_reward = _default_action_reward(action)
        by_segment[(action.rollout_id, action.segment_id)].append(action)

    for segment_actions in by_segment.values():
        segment_reward = sum(action.action_reward for action in segment_actions) / max(1, len(segment_actions))
        for action in segment_actions:
            if action.segment_reward == 0.0:
                action.segment_reward = segment_reward
    return actions


def compute_group_advantages(rollouts: list[RolloutRecord], eps: float = 1e-6) -> list[RolloutRecord]:
    """Compute group-relative advantages from episode rewards."""

    by_group: dict[str, list[RolloutRecord]] = defaultdict(list)
    for rollout in rollouts:
        by_group[rollout.group_id].append(rollout)

    for group_rollouts in by_group.values():
        rewards = [rollout.episode_reward for rollout in group_rollouts]
        mean = sum(rewards) / max(1, len(rewards))
        variance = sum((reward - mean) ** 2 for reward in rewards) / max(1, len(rewards))
        std = math.sqrt(variance)
        for rollout in group_rollouts:
            advantage = 0.0 if std < eps else (rollout.episode_reward - mean) / (std + eps)
            for action in rollout.actions:
                action.advantage = advantage
    return rollouts


def blended_reward(action: TrajectoryAction, *, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.25) -> float:
    """Return a scalar reward combining episode, segment, and action signals."""

    return alpha * action.episode_reward + beta * action.segment_reward + gamma * action.action_reward


async def custom_rm(_: Any, sample_or_samples: Any, **__: Any) -> Any:
    """Slime-compatible reward hook for future FASD-GRPO rollout integration."""

    def score_one(sample: Any) -> dict[str, float]:
        reward = getattr(sample, "reward", None)
        if isinstance(reward, dict):
            return {"score": float(reward.get("score", 0.0) or 0.0)}
        return {"score": float(reward or 0.0)}

    if isinstance(sample_or_samples, list):
        return [score_one(sample) for sample in sample_or_samples]
    return score_one(sample_or_samples)


def _default_action_reward(action: TrajectoryAction) -> float:
    if action.failure_type == "none" and action.teacher_action and action.teacher_action == action.student_action:
        return 1.0
    if action.failure_type != "none" and action.teacher_action and action.teacher_action != action.student_action:
        return -1.0
    if action.failure_type == "format_error":
        return -1.0
    return 0.0
