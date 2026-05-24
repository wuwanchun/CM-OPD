"""Async hindsight hint extraction for SDFT training."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from opd.action_parser import parse_memory_action
from opd.hindsight_judge import HintVote, select_best_hint

from .trajectory_schema import TrajectoryAction


@dataclass
class SDFTHint:
    """Evidence-grounded correction extracted from a failed trajectory action."""

    task_id: str
    rollout_id: str
    segment_id: str
    turn_id: int
    raw_evidence_id: str
    bad_action: str
    corrected_action: str
    hint: str
    failure_type: str
    confidence: float
    votes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "segment_id": self.segment_id,
            "turn_id": self.turn_id,
            "raw_evidence_id": self.raw_evidence_id,
            "bad_action": self.bad_action,
            "corrected_action": self.corrected_action,
            "hint": self.hint,
            "failure_type": self.failure_type,
            "confidence": self.confidence,
            "votes": self.votes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SDFTHint":
        return cls(
            task_id=str(data["task_id"]),
            rollout_id=str(data["rollout_id"]),
            segment_id=str(data["segment_id"]),
            turn_id=int(data["turn_id"]),
            raw_evidence_id=str(data["raw_evidence_id"]),
            bad_action=str(data["bad_action"]),
            corrected_action=str(data["corrected_action"]),
            hint=str(data["hint"]),
            failure_type=str(data.get("failure_type", "none")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            votes=[dict(vote) for vote in data.get("votes", [])],
            metadata=dict(data.get("metadata", {})),
        )


def failed_actions(actions: list[TrajectoryAction]) -> list[TrajectoryAction]:
    """Return trajectory actions that need SDFT correction."""

    return [action for action in actions if action.failure_type != "none"]


async def extract_hints_async(
    actions: list[TrajectoryAction],
    *,
    concurrency: int = 8,
    mode: str = "rule",
    votes_per_action: int = 3,
) -> list[SDFTHint]:
    """Extract hindsight hints concurrently.

    `mode="rule"` is deterministic and runs without an external model. It is
    the default CI/remote test path. Future teacher endpoints can be added
    behind this interface without changing downstream SDFT sample construction.
    """

    if mode != "rule":
        raise ValueError(f"unsupported hint extractor mode: {mode}")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(action: TrajectoryAction) -> SDFTHint | None:
        async with semaphore:
            await asyncio.sleep(0)
            return _rule_extract_hint(action, votes_per_action=votes_per_action)

    results = await asyncio.gather(*(run_one(action) for action in failed_actions(actions)))
    return [hint for hint in results if hint is not None]


def _rule_extract_hint(action: TrajectoryAction, *, votes_per_action: int) -> SDFTHint | None:
    if not action.raw_evidence_ids:
        return None
    if not action.teacher_action:
        return None

    raw_evidence_id = action.raw_evidence_ids[0]
    bad = parse_memory_action(action.student_action)
    corrected = parse_memory_action(action.teacher_action)
    bad_name = bad.action or "INVALID"
    corrected_name = corrected.action or action.teacher_action
    if bad.valid and corrected.valid and bad_name == corrected_name:
        return None

    title = str(action.metadata.get("title", "UNKNOWN"))
    reason = _reason_for_failure(action.failure_type, title)
    votes = _rule_votes(action, raw_evidence_id, bad_name, corrected_name, reason, votes_per_action=votes_per_action)
    selected = select_best_hint(votes)
    if selected is None:
        return None
    confidence = 0.92 if action.failure_type == "missing_evidence" else 0.82
    return SDFTHint(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        segment_id=action.segment_id,
        turn_id=action.turn_id,
        raw_evidence_id=raw_evidence_id,
        bad_action=action.student_action,
        corrected_action=action.teacher_action,
        hint=selected.hint,
        failure_type=action.failure_type,
        confidence=confidence,
        votes=[{"vote_id": vote.vote_id, "score": vote.score, "hint": vote.hint, "raw": vote.raw} for vote in votes],
        metadata={
            "action_type": action.action_type,
            "title": title,
            "source": "rule_hindsight_extractor",
        },
    )


def _rule_votes(
    action: TrajectoryAction,
    raw_evidence_id: str,
    bad_name: str,
    corrected_name: str,
    reason: str,
    *,
    votes_per_action: int,
) -> list[HintVote]:
    votes: list[HintVote] = []
    for vote_id in range(max(1, votes_per_action)):
        detail = "" if vote_id == 0 else f" vote={vote_id} title={action.metadata.get('title', 'UNKNOWN')}"
        hint = (
            f"turn_id={action.turn_id} "
            f"memory_id={_target_memory_id(action)} "
            f"raw_evidence_id={raw_evidence_id} "
            f"bad_action={bad_name} "
            f"correct_action={corrected_name} "
            f"verifiable_reason={reason}{detail}"
        )
        votes.append(HintVote(vote_id=vote_id, score=1, hint=hint, raw=rf"\boxed{{1}}[HINT_START]{hint}[HINT_END]"))
    return votes


def _target_memory_id(action: TrajectoryAction) -> str:
    parsed = parse_memory_action(action.teacher_action) if action.teacher_action else parse_memory_action(action.student_action)
    if parsed.payload:
        ids = parsed.payload.get("target_ids")
        if isinstance(ids, list) and ids:
            return str(ids[0])
    return str(action.metadata.get("memory_id", "UNKNOWN"))


def _reason_for_failure(failure_type: str, title: str) -> str:
    if failure_type == "missing_evidence":
        return f"the failed rollout did not preserve supporting evidence from title '{title}'"
    if failure_type == "format_error":
        return "the failed rollout emitted an invalid memory action format"
    if failure_type == "repeated_tool":
        return f"the failed rollout wasted budget on a non-supporting paragraph titled '{title}'"
    if failure_type == "wrong_edit":
        return "the failed rollout used evidence incorrectly before editing"
    if failure_type == "test_regression":
        return "the failed rollout ignored regression feedback"
    return "the failed rollout action was inconsistent with verifier feedback"
