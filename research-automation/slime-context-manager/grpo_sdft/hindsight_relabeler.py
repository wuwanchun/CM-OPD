"""Hindsight relabeling and confidence gating for FASD-GRPO."""

from __future__ import annotations

from .trajectory_schema import LossMasks, TrajectoryAction


def should_accept_correction(
    action: TrajectoryAction,
    *,
    min_confidence: float = 0.7,
    require_raw_evidence: bool = True,
) -> bool:
    """Return whether an action should enter the SDFT correction set."""

    if action.confidence < min_confidence:
        return False
    if require_raw_evidence and not action.raw_evidence_ids:
        return False
    if not action.teacher_action:
        return False
    if not action.hindsight_hint:
        return False
    return True


def apply_hindsight_masks(
    actions: list[TrajectoryAction],
    *,
    min_confidence: float = 0.7,
    require_raw_evidence: bool = True,
) -> list[TrajectoryAction]:
    """Set GRPO/SDFT/replay masks from success and failure hindsight signals."""

    for action in actions:
        accept = should_accept_correction(
            action,
            min_confidence=min_confidence,
            require_raw_evidence=require_raw_evidence,
        )
        replay = action.failure_type == "none" and action.teacher_action == action.student_action and action.confidence >= min_confidence
        action.loss_masks = LossMasks(grpo=True, sdft=accept, replay=replay)
    return actions


def correction_target(action: TrajectoryAction) -> str:
    """Return the action target used by SDFT; fallback to student action for replay."""

    return action.teacher_action or action.student_action
