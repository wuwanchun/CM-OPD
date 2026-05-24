"""Core OPD recorder and optional API server for external context policies."""

from __future__ import annotations

import json
import queue
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from .action_parser import parse_memory_action
from .hindsight_judge import HintVote, majority_process_reward, select_best_hint
from .teacher_logprob import align_log_probs


@dataclass
class NextState:
    """Next-state feedback for a memory decision."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class PendingTurn:
    """A memory action waiting for next-state hindsight feedback."""

    session_id: str
    turn_id: int
    memory_id: str
    student_prompt: str
    student_action_text: str
    prompt_ids: list[int]
    response_ids: list[int]
    rollout_log_probs: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OPDSampleRecord:
    """Serializable OPD training sample."""

    sample_id: str
    session_id: str
    turn_id: int
    memory_id: str
    student_prompt: str
    student_action_text: str
    next_state: dict[str, Any]
    hint: str
    teacher_action: str | None
    teacher_log_probs: list[float]
    process_reward: float
    metadata: dict[str, Any]
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)
    teacher_topk_log_probs: list[list[float]] = field(default_factory=list)
    teacher_topk_indices: list[list[int]] = field(default_factory=list)

    @property
    def response_length(self) -> int:
        return len(self.response_ids)

    @property
    def loss_mask(self) -> list[int]:
        return [1] * self.response_length

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "memory_id": self.memory_id,
            "student_prompt": self.student_prompt,
            "student_action_text": self.student_action_text,
            "next_state": self.next_state,
            "hint": self.hint,
            "teacher_action": self.teacher_action,
            "teacher_log_probs": self.teacher_log_probs,
            "teacher_topk_log_probs": self.teacher_topk_log_probs,
            "teacher_topk_indices": self.teacher_topk_indices,
            "process_reward": self.process_reward,
            "metadata": self.metadata,
        }

    def to_slime_sample(self) -> Any:
        """Convert to slime Sample if available, otherwise return a stub."""

        reward = {
            "score": self.process_reward,
            "process_reward": self.process_reward,
            "episode_reward": self.metadata.get("episode_reward"),
        }
        try:
            from slime.utils.types import Sample  # type: ignore

            sample = Sample()
            sample.prompt = self.student_prompt
            sample.response = self.student_action_text
            sample.tokens = self.prompt_ids + self.response_ids
            sample.response_length = self.response_length
            sample.loss_mask = self.loss_mask
            sample.rollout_log_probs = self.rollout_log_probs
            sample.teacher_log_probs = self.teacher_log_probs
            sample.reward = reward
            sample.metadata = self.metadata
            sample.status = Sample.Status.COMPLETED
            return sample
        except Exception:
            return SlimeSampleStub(
                prompt=self.student_prompt,
                response=self.student_action_text,
                tokens=self.prompt_ids + self.response_ids,
                response_length=self.response_length,
                loss_mask=self.loss_mask,
                rollout_log_probs=self.rollout_log_probs,
                teacher_log_probs=self.teacher_log_probs,
                reward=reward,
                metadata=self.metadata,
            )


@dataclass
class SlimeSampleStub:
    """Local compatibility object used when slime is not installed."""

    prompt: str
    response: str
    tokens: list[int]
    response_length: int
    loss_mask: list[int]
    rollout_log_probs: list[float]
    teacher_log_probs: list[float]
    reward: dict[str, Any]
    metadata: dict[str, Any]
    status: str = "COMPLETED"


@dataclass
class EpisodeRecord:
    """Episode-level record preserved for future GRPO."""

    episode_id: str
    turn_rewards: list[dict[str, Any]] = field(default_factory=list)
    final_reward: float | None = None
    evidence_recall: float | None = None
    token_cost: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "turn_rewards": self.turn_rewards,
            "final_reward": self.final_reward,
            "evidence_recall": self.evidence_recall,
            "token_cost": self.token_cost,
        }


class OPDRecorder:
    """Manage pending turns and accepted OPD samples."""

    def __init__(self, *, require_raw_evidence: bool = True):
        self.require_raw_evidence = require_raw_evidence
        self.pending: dict[str, dict[int, PendingTurn]] = {}
        self.samples: queue.Queue[OPDSampleRecord] = queue.Queue()
        self.episodes: dict[str, EpisodeRecord] = {}
        self._sample_counter = count(0)

    def record_turn(self, turn: PendingTurn) -> None:
        self.pending.setdefault(turn.session_id, {})[turn.turn_id] = turn
        self.episodes.setdefault(turn.session_id, EpisodeRecord(episode_id=turn.session_id))

    def receive_next_state(
        self,
        *,
        session_id: str,
        turn_id: int,
        next_state: NextState | dict[str, Any],
        votes: list[HintVote | dict[str, Any]],
        teacher_log_probs: list[float] | None = None,
        teacher_topk_log_probs: list[list[float]] | None = None,
        teacher_topk_indices: list[list[int]] | None = None,
        teacher_action: str | None = None,
    ) -> OPDSampleRecord | None:
        turn = self.pending.get(session_id, {}).pop(turn_id, None)
        if turn is None:
            return None

        process_reward = majority_process_reward(votes)
        episode = self.episodes.setdefault(session_id, EpisodeRecord(episode_id=session_id))
        episode.turn_rewards.append({"turn_id": turn_id, "process_reward": process_reward})

        selected = select_best_hint(votes)
        if selected is None:
            return None

        raw_evidence_id = turn.metadata.get("raw_evidence_id")
        if self.require_raw_evidence and not raw_evidence_id:
            return None

        parsed = parse_memory_action(turn.student_action_text)
        action = teacher_action or _extract_suggested_action(selected.hint) or parsed.action
        aligned_teacher_log_probs = align_log_probs(teacher_log_probs or [], len(turn.response_ids))
        sample_id = f"opd_{session_id}_t{turn_id:04d}_{turn.memory_id}_{next(self._sample_counter)}"
        ns = next_state.to_dict() if isinstance(next_state, NextState) else dict(next_state)
        metadata = dict(turn.metadata)
        metadata.update(
            {
                "sample_id": sample_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "memory_id": turn.memory_id,
                "raw_evidence_id": raw_evidence_id,
                "hint_votes": [_vote_to_dict(vote) for vote in votes],
                "process_reward": process_reward,
            }
        )

        record = OPDSampleRecord(
            sample_id=sample_id,
            session_id=session_id,
            turn_id=turn_id,
            memory_id=turn.memory_id,
            student_prompt=turn.student_prompt,
            student_action_text=turn.student_action_text,
            next_state=ns,
            hint=selected.hint,
            teacher_action=action,
            teacher_log_probs=aligned_teacher_log_probs,
            process_reward=process_reward,
            metadata=metadata,
            prompt_ids=turn.prompt_ids,
            response_ids=turn.response_ids,
            rollout_log_probs=align_log_probs(turn.rollout_log_probs, len(turn.response_ids)),
            teacher_topk_log_probs=teacher_topk_log_probs or [],
            teacher_topk_indices=teacher_topk_indices or [],
        )
        self.samples.put(record)
        return record

    def finish_session(
        self,
        session_id: str,
        *,
        final_reward: float | None = None,
        evidence_recall: float | None = None,
        token_cost: int | None = None,
    ) -> EpisodeRecord:
        self.pending.pop(session_id, None)
        episode = self.episodes.setdefault(session_id, EpisodeRecord(episode_id=session_id))
        episode.final_reward = final_reward
        episode.evidence_recall = evidence_recall
        episode.token_cost = token_cost
        return episode

    def drain_samples(self) -> list[OPDSampleRecord]:
        drained: list[OPDSampleRecord] = []
        while True:
            try:
                drained.append(self.samples.get_nowait())
            except queue.Empty:
                break
        return drained


def _extract_suggested_action(hint: str) -> str | None:
    lower = hint.lower()
    for action in ("KEEP", "COMPRESS", "ARCHIVE", "RETRIEVE", "UPDATE", "DROP", "PIN"):
        if f"suggested_action: {action.lower()}" in lower or f"correct_action: {action.lower()}" in lower:
            return action
    return None


def _vote_to_dict(vote: HintVote | dict[str, Any]) -> dict[str, Any]:
    if isinstance(vote, HintVote):
        return {"vote_id": vote.vote_id, "score": vote.score, "hint": vote.hint, "raw": vote.raw}
    return dict(vote)


def dumps_jsonl(records: list[OPDSampleRecord]) -> str:
    """Serialize OPD records to JSONL."""

    return "\n".join(json.dumps(record.to_json_dict(), ensure_ascii=False) for record in records)


class ContextPolicyOPDProxy:
    """Optional OpenAI-compatible proxy for context-policy OPD collection.

    The core recorder above is dependency-free. This proxy is only constructed
    in deployments that have FastAPI/httpx available. It can forward
    `/v1/chat/completions` requests to an upstream policy server and record the
    returned memory action as a pending OPD turn.
    """

    def __init__(
        self,
        *,
        recorder: OPDRecorder | None = None,
        upstream_chat_url: str | None = None,
        tokenizer: Any | None = None,
    ):
        self.recorder = recorder or OPDRecorder()
        self.upstream_chat_url = upstream_chat_url
        self.tokenizer = tokenizer

    def build_app(self) -> Any:
        try:
            import httpx
            from fastapi import FastAPI, Header, HTTPException, Request
            from fastapi.responses import JSONResponse
        except ImportError as exc:  # pragma: no cover - optional deployment dependency
            raise RuntimeError("FastAPI and httpx are required for ContextPolicyOPDProxy") from exc

        app = FastAPI(title="Context Policy OPD Proxy")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz() -> dict[str, bool]:
            return {"ok": True}

        @app.post("/opd/record_turn")
        async def record_turn(request: Request) -> dict[str, Any]:
            body = await request.json()
            turn = PendingTurn(
                session_id=str(body["session_id"]),
                turn_id=int(body["turn_id"]),
                memory_id=str(body["memory_id"]),
                student_prompt=str(body["student_prompt"]),
                student_action_text=str(body["student_action_text"]),
                prompt_ids=[int(x) for x in body.get("prompt_ids", [])],
                response_ids=[int(x) for x in body.get("response_ids", [])],
                rollout_log_probs=[float(x) for x in body.get("rollout_log_probs", [])],
                metadata=dict(body.get("metadata", {})),
            )
            self.recorder.record_turn(turn)
            return {"ok": True, "pending": True}

        @app.post("/opd/next_state")
        async def next_state(request: Request) -> dict[str, Any]:
            body = await request.json()
            record = self.recorder.receive_next_state(
                session_id=str(body["session_id"]),
                turn_id=int(body["turn_id"]),
                next_state=dict(body.get("next_state", {})),
                votes=list(body.get("votes", [])),
                teacher_log_probs=[float(x) for x in body.get("teacher_log_probs", [])],
                teacher_topk_log_probs=body.get("teacher_topk_log_probs") or [],
                teacher_topk_indices=body.get("teacher_topk_indices") or [],
                teacher_action=body.get("teacher_action"),
            )
            if record is None:
                return {"accepted": False}
            return {"accepted": True, "sample": record.to_json_dict()}

        @app.post("/opd/session_done")
        async def session_done(request: Request) -> dict[str, Any]:
            body = await request.json()
            episode = self.recorder.finish_session(
                str(body["session_id"]),
                final_reward=body.get("final_reward"),
                evidence_recall=body.get("evidence_recall"),
                token_cost=body.get("token_cost"),
            )
            return {"ok": True, "episode": episode.to_json_dict()}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            x_session_id: str | None = Header(default=None),
            x_turn_id: str | None = Header(default=None),
            x_turn_type: str | None = Header(default=None),
            x_memory_id: str | None = Header(default=None),
            x_session_done: str | None = Header(default=None),
        ) -> JSONResponse:
            if not self.upstream_chat_url:
                raise HTTPException(status_code=503, detail="upstream_chat_url is not configured")

            body = await request.json()
            async with httpx.AsyncClient(timeout=None) as client:
                upstream = await client.post(self.upstream_chat_url, json=body)
                upstream.raise_for_status()
                response = upstream.json()

            turn_type = (x_turn_type or body.get("turn_type") or "main").lower()
            if turn_type == "main":
                messages = body.get("messages", [])
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message", {})
                action_text = _message_to_action_text(message)
                prompt_text = _messages_to_text(messages)
                prompt_ids = _encode_text(prompt_text, self.tokenizer)
                response_ids = _encode_text(action_text, self.tokenizer)
                session_id = x_session_id or str(body.get("session_id", "unknown"))
                turn_id = int(x_turn_id or body.get("turn_id", 0) or 0)
                memory_id = x_memory_id or str(body.get("memory_id", body.get("metadata", {}).get("memory_id", "unknown")))
                metadata = dict(body.get("metadata", {}))
                metadata.setdefault("session_id", session_id)
                metadata.setdefault("turn_id", turn_id)
                metadata.setdefault("memory_id", memory_id)
                self.recorder.record_turn(
                    PendingTurn(
                        session_id=session_id,
                        turn_id=turn_id,
                        memory_id=memory_id,
                        student_prompt=prompt_text,
                        student_action_text=action_text,
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        rollout_log_probs=_extract_response_logprobs(choice, len(response_ids)),
                        metadata=metadata,
                    )
                )

            done = str(x_session_done or body.get("session_done", "")).lower() in {"1", "true", "yes", "on"}
            if done:
                self.recorder.finish_session(x_session_id or str(body.get("session_id", "unknown")))

            return JSONResponse(content=response)

        return app


def _message_to_action_text(message: dict[str, Any]) -> str:
    if message.get("tool_calls"):
        return json.dumps(message["tool_calls"], ensure_ascii=False)
    return str(message.get("content") or "")


def _messages_to_text(messages: Any) -> str:
    if isinstance(messages, list):
        return json.dumps(messages, ensure_ascii=False)
    return str(messages)


def _encode_text(text: str, tokenizer: Any | None) -> list[int]:
    if tokenizer is not None:
        if hasattr(tokenizer, "encode"):
            return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if callable(tokenizer):
            encoded = tokenizer(text, add_special_tokens=False)
            if isinstance(encoded, dict) and "input_ids" in encoded:
                return [int(x) for x in encoded["input_ids"]]
    return list(text.encode("utf-8"))


def _extract_response_logprobs(choice: dict[str, Any], response_len: int) -> list[float]:
    content = (choice.get("logprobs") or {}).get("content")
    values: list[float] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                values.append(float(item.get("logprob", 0.0)))
    return align_log_probs(values, response_len)
