"""Trajectory grouping and segment construction for FASD-GRPO."""

from __future__ import annotations

from collections import defaultdict

from .trajectory_schema import RolloutRecord, SegmentRecord, TrajectoryAction


def group_actions_by_rollout(actions: list[TrajectoryAction]) -> list[RolloutRecord]:
    """Group flat action records into rollout records."""

    grouped: dict[tuple[str, str], list[TrajectoryAction]] = defaultdict(list)
    for action in actions:
        grouped[(action.group_id, action.rollout_id)].append(action)

    rollouts: list[RolloutRecord] = []
    for _, items in sorted(grouped.items()):
        items.sort(key=lambda action: (action.segment_id, action.turn_id))
        rollouts.append(RolloutRecord.from_actions(items))
    return rollouts


def segment_rollout(rollout: RolloutRecord) -> list[SegmentRecord]:
    """Split a rollout into segment records using explicit segment ids."""

    grouped: dict[str, list[TrajectoryAction]] = defaultdict(list)
    for action in rollout.actions:
        grouped[action.segment_id].append(action)

    segments: list[SegmentRecord] = []
    for segment_id, actions in sorted(grouped.items()):
        actions.sort(key=lambda action: action.turn_id)
        reward = sum(action.action_reward for action in actions) / max(1, len(actions))
        for action in actions:
            action.segment_reward = reward
        segments.append(
            SegmentRecord(
                task_id=rollout.task_id,
                rollout_id=rollout.rollout_id,
                group_id=rollout.group_id,
                segment_id=segment_id,
                actions=actions,
                segment_reward=reward,
            )
        )
    return segments


def assert_group_has_k_rollouts(rollouts: list[RolloutRecord], expected_k: int) -> None:
    """Validate each task group has the requested number of rollouts."""

    counts: dict[str, int] = defaultdict(int)
    for rollout in rollouts:
        counts[rollout.group_id] += 1
    bad = {group_id: count for group_id, count in counts.items() if count != expected_k}
    if bad:
        raise ValueError(f"rollout group size mismatch: expected {expected_k}, got {bad}")
