"""Hindsight judge prompts and parsers for OPD."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_BOXED_RE = re.compile(r"\\?boxed\{\s*([-+]?\d)\s*\}")
_HINT_RE = re.compile(r"\[HINT_START\](.*?)\[HINT_END\]", re.DOTALL)


@dataclass(frozen=True)
class HintVote:
    """A single hindsight judge vote."""

    vote_id: int
    score: int | None
    hint: str
    raw: str = ""


def parse_judge_result(text: str) -> tuple[int | None, str]:
    """Parse a judge response into score and hint.

    Scores follow OpenClaw-style boxed labels:
    ``\\boxed{1}``, ``\\boxed{-1}``, or ``\\boxed{0}``.
    """

    boxed = _BOXED_RE.findall(text or "")
    score = int(boxed[-1]) if boxed else None
    if score not in (1, -1, 0):
        score = None

    hints = _HINT_RE.findall(text or "")
    hint = hints[-1].strip() if hints else ""
    return score, hint


def make_vote(text: str, vote_id: int = 0) -> HintVote:
    """Create a vote from raw judge text."""

    score, hint = parse_judge_result(text)
    return HintVote(vote_id=vote_id, score=score, hint=hint, raw=text)


def select_best_hint(votes: list[HintVote | dict[str, Any]], min_chars: int = 10) -> HintVote | None:
    """Select the longest positive non-trivial hint."""

    normalized: list[HintVote] = []
    for idx, vote in enumerate(votes):
        if isinstance(vote, HintVote):
            normalized.append(vote)
        else:
            normalized.append(
                HintVote(
                    vote_id=int(vote.get("vote_id", idx)),
                    score=vote.get("score"),
                    hint=str(vote.get("hint", "")),
                    raw=str(vote.get("raw", "")),
                )
            )

    positive = [vote for vote in normalized if vote.score == 1 and len(vote.hint.strip()) >= min_chars]
    if not positive:
        return None
    return max(positive, key=lambda vote: len(vote.hint.strip()))


def majority_process_reward(votes: list[HintVote | dict[str, Any]]) -> float:
    """Return a turn-level process reward from votes."""

    scores: list[int] = []
    for vote in votes:
        score = vote.score if isinstance(vote, HintVote) else vote.get("score")
        if score in (1, -1, 0):
            scores.append(int(score))
    if not scores:
        return 0.0
    counts = {score: scores.count(score) for score in (1, -1, 0)}
    best_score, best_count = max(counts.items(), key=lambda item: item[1])
    if list(counts.values()).count(best_count) > 1:
        return 0.0
    return float(best_score)


def build_hindsight_judge_messages(
    student_action_text: str,
    next_state: dict[str, Any],
    *,
    turn_id: int,
    memory_id: str,
    raw_evidence_id: str | None = None,
) -> list[dict[str, str]]:
    """Build an evidence-grounded judge prompt for a memory action."""

    role = str(next_state.get("role", "env"))
    content = str(next_state.get("content", ""))
    evidence_line = raw_evidence_id or "UNKNOWN"
    system = (
        "You are a process reward model for an external context manager.\n"
        "Judge whether the memory action at turn t was useful given the next state.\n"
        "Return exactly one final score: \\boxed{1}, \\boxed{0}, or \\boxed{-1}.\n"
        "If score is \\boxed{1}, include one concise evidence-grounded hint between "
        "[HINT_START] and [HINT_END].\n"
        "A valid hint must mention turn_id, memory_id, raw_evidence_id, suggested_action, "
        "and verifiable_reason."
    )
    user = (
        f"turn_id: {turn_id}\n"
        f"memory_id: {memory_id}\n"
        f"raw_evidence_id: {evidence_line}\n\n"
        f"student_memory_action:\n{student_action_text}\n\n"
        f"next_state_role: {role}\n"
        f"next_state_content:\n{content}\n\n"
        "Decide whether hindsight feedback can improve this memory action."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
