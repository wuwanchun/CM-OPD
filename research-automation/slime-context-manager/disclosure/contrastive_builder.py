"""Build target candidates from differing disclosure rollouts."""

from __future__ import annotations

from .schemas import DisclosureRollout


def differing_actions(rollouts: list[DisclosureRollout]) -> list[dict[str, str]]:
    by_task: dict[str, list[DisclosureRollout]] = {}
    for rollout in rollouts:
        by_task.setdefault(rollout.task_id, []).append(rollout)

    diffs: list[dict[str, str]] = []
    for task_rollouts in by_task.values():
        for left_index, left in enumerate(task_rollouts):
            left_actions = {action.span_id: action.action for action in left.actions}
            for right in task_rollouts[left_index + 1 :]:
                right_actions = {action.span_id: action.action for action in right.actions}
                for span_id in sorted(set(left_actions) | set(right_actions)):
                    if left_actions.get(span_id) != right_actions.get(span_id):
                        diffs.append(
                            {
                                "task_id": left.task_id,
                                "span_id": span_id,
                                "rollout_id": left.rollout_id,
                                "contrast_rollout_id": right.rollout_id,
                                "left_action": left_actions.get(span_id, ""),
                                "right_action": right_actions.get(span_id, ""),
                            }
                        )
    return diffs
