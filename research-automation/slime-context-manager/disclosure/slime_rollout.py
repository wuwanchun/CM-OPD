"""Slime-compatible rollout bridge for disclosure SFT samples."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_io.io_utils import read_jsonl
from disclosure.answer_rollout import run_answer_rollouts
from disclosure.review_sft_builder import build_review_sft_targets
from disclosure.schemas import ContextSpan
from disclosure.selection_rollout import run_selection_rollouts
from disclosure.slime_export import export_review_sft_records

logger = logging.getLogger(__name__)


@dataclass
class RolloutOutputStub:
    samples: list[list[Any]]
    metrics: dict[str, float] | None = None


@dataclass
class DisclosureSlimeRecordStub:
    prompt: str
    label: str
    metadata: dict[str, Any]
    group_index: int | None = None
    index: int | None = None
    response: str = ""
    tokens: list[int] | None = None
    response_length: int = 0
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    train_metadata: dict[str, Any] | None = None
    status: str = "PENDING"


class DisclosureRolloutBuffer:
    def __init__(self, samples: list[Any]):
        self.samples = samples
        self._cursor = 0

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "DisclosureRolloutBuffer":
        return cls([_sample_from_record(record, index=i) for i, record in enumerate(read_jsonl(path))])

    def next_groups(self, batch_size: int) -> list[list[Any]]:
        if not self.samples:
            return []
        groups = []
        for _ in range(batch_size):
            sample = self.samples[self._cursor % len(self.samples)]
            self._cursor += 1
            groups.append([sample])
        return groups


_BUFFERS: dict[str, DisclosureRolloutBuffer] = {}
_TOKENIZER = None
_SPAN_CACHE: dict[str, list[ContextSpan]] = {}
_REVIEW_RECORD_CACHE: dict[str, list[dict[str, Any]]] = {}


def generate_rollout_disclosure(args: Any, rollout_id: int, data_buffer: Any = None, evaluation: bool = False) -> Any:
    if evaluation:
        return _output([], {"rollout/disclosure_eval_noop": 1.0})

    batch_size = int(getattr(args, "rollout_batch_size", 1))
    if data_buffer is not None and hasattr(data_buffer, "get_samples"):
        groups = data_buffer.get_samples(batch_size)
        source = "slime_data_source"
    else:
        path = str(getattr(args, "disclosure_samples_path", "") or getattr(args, "prompt_data", ""))
        if not path:
            raise ValueError("generate_rollout_disclosure requires args.disclosure_samples_path, args.prompt_data, or data_buffer")
        buffer = _BUFFERS.setdefault(path, DisclosureRolloutBuffer.from_jsonl(path))
        groups = buffer.next_groups(batch_size)
        source = "jsonl_buffer"

    prepared = [[_prepare_sample_for_slime(args, sample) for sample in group] for group in groups]
    return _output(
        prepared,
        {
            "rollout/disclosure_groups": float(len(prepared)),
            "rollout/disclosure_samples": float(sum(len(group) for group in prepared)),
            "rollout/disclosure_rollout_id": float(rollout_id),
            f"rollout/disclosure_source/{source}": 1.0,
        },
    )


def generate_selection_answer_review_rollout(args: Any, rollout_id: int, data_buffer: Any = None, evaluation: bool = False) -> Any:
    """Build selection -> answer -> review SFT samples inside Slime rollout."""
    if evaluation:
        return _output([], {"rollout/selection_review_eval_noop": 1.0})

    span_path = _prompt_data_path(args)
    records = _review_records_for_rollout(args, span_path, rollout_id)
    if not records:
        raise ValueError(f"No review SFT records could be built from {span_path}")

    batch_size = int(getattr(args, "rollout_batch_size", 1))
    source_groups = data_buffer.get_samples(batch_size) if data_buffer is not None and hasattr(data_buffer, "get_samples") else []
    groups = []
    for group_offset in range(batch_size):
        source_group = source_groups[group_offset] if group_offset < len(source_groups) else [_new_sample()]
        record = records[(rollout_id * batch_size + group_offset) % len(records)]
        group = []
        for sample in source_group:
            sample.prompt = record["prompt"]
            sample.label = record.get("label", record.get("response", ""))
            sample.metadata = dict(record.get("metadata", {}))
            sample.metadata.update({"rollout_id": rollout_id, "rollout_stage": "selection_answer_review"})
            group.append(_prepare_sample_for_slime(args, sample))
        groups.append(group)

    return _output(
        groups,
        {
            "rollout/selection_review_groups": float(len(groups)),
            "rollout/selection_review_samples": float(sum(len(group) for group in groups)),
            "rollout/selection_review_records": float(len(records)),
        },
    )


def generate_selection_opd_rollout(args: Any, rollout_id: int, data_buffer: Any = None, evaluation: bool = False) -> Any:
    """Build OPD CE-fallback samples inside Slime rollout."""
    output = generate_selection_answer_review_rollout(args, rollout_id, data_buffer, evaluation=evaluation)
    for group in getattr(output, "samples", []) or []:
        for sample in group:
            sample.train_metadata = {"loss": "selection_opd_ce_fallback", "distill_mode": "ce_fallback"}
            sample.teacher_log_probs = []
            sample.metadata["distill_mode"] = "ce_fallback"
    if getattr(output, "metrics", None) is not None:
        output.metrics["rollout/selection_opd_ce_fallback"] = 1.0
    return output


def _prepare_sample_for_slime(args: Any, sample: Any) -> Any:
    prompt = str(getattr(sample, "prompt", ""))
    response = str(getattr(sample, "label", "") or getattr(sample, "response", ""))
    if not prompt:
        raise ValueError("disclosure sample is missing prompt")
    if not response:
        raise ValueError("disclosure sample is missing label/response")

    tokenizer = _get_tokenizer(args)
    if tokenizer is not None:
        tokens, response_length, loss_mask = _encode_prompt_response(
            tokenizer,
            prompt=prompt,
            response=response,
            max_length=int(getattr(args, "rollout_max_model_len", 0) or getattr(args, "max_length", 0) or 0) or None,
            max_response_length=int(getattr(args, "rollout_max_response_len", 0) or 0) or None,
        )
        sample.tokens = tokens
        sample.response_length = response_length
        sample.loss_mask = loss_mask
        if response_length <= 0 or sum(loss_mask) != response_length:
            raise ValueError("invalid disclosure response loss mask")
    else:
        logger.warning("No tokenizer available; returning disclosure sample without token fields.")

    sample.response = response
    if getattr(sample, "reward", None) is None:
        sample.reward = 1.0
    if getattr(sample, "train_metadata", None) is None:
        sample.train_metadata = {"loss": "disclosure_sft", "disclosure": True}
    if getattr(sample, "metadata", None) is None:
        sample.metadata = {}
    sample.metadata.setdefault("disclosure_source", "slime_rollout")
    _mark_completed(sample)
    return sample


def _sample_from_record(record: dict[str, Any], *, index: int) -> Any:
    prompt = str(record.get("prompt", ""))
    label = str(record.get("label", record.get("response", "")))
    metadata = dict(record.get("metadata", {}))
    try:
        from slime.utils.types import Sample  # type: ignore

        return Sample(prompt=prompt, label=label, metadata=metadata, group_index=index, index=index)
    except Exception:
        return DisclosureSlimeRecordStub(prompt=prompt, label=label, metadata=metadata, group_index=index, index=index)


def _prompt_data_path(args: Any) -> str:
    path = str(getattr(args, "prompt_data", "") or "")
    if not path:
        raise ValueError("selection/review rollout requires --prompt-data pointing to spans JSONL")
    return path.split("@[", 1)[0]


def _review_records_for_rollout(args: Any, span_path: str, rollout_id: int) -> list[dict[str, Any]]:
    token_budget = int(
        getattr(args, "disclosure_token_budget", 0)
        or os.environ.get("DISCLOSURE_TOKEN_BUDGET", "")
        or getattr(args, "token_budget", 0)
        or 4096
    )
    reserved_prompt_tokens = int(
        getattr(args, "disclosure_reserved_prompt_tokens", 0)
        or os.environ.get("DISCLOSURE_RESERVED_PROMPT_TOKENS", "")
        or getattr(args, "reserved_prompt_tokens", 0)
        or 128
    )
    allocator = str(getattr(args, "disclosure_allocator", "") or os.environ.get("DISCLOSURE_ALLOCATOR", "") or "greedy_margin_per_cost")
    policy = str(getattr(args, "selection_policy", "") or os.environ.get("SELECTION_POLICY", "") or "rule_expand_omission")
    cache_key = f"{span_path}|{policy}|{token_budget}|{reserved_prompt_tokens}|{allocator}"
    if cache_key in _REVIEW_RECORD_CACHE:
        return _REVIEW_RECORD_CACHE[cache_key]

    spans = _load_spans(span_path)
    selections = [
        rollout.to_dict()
        for rollout in run_selection_rollouts(
            spans,
            policy_name=policy,
            token_budget=token_budget,
            reserved_prompt_tokens=reserved_prompt_tokens,
            allocator=allocator,
        )
    ]
    answers = [
        answer.to_dict()
        for answer in run_answer_rollouts(
            spans,
            selections,
            token_budget=token_budget,
            reserved_prompt_tokens=reserved_prompt_tokens,
            allocator=allocator,
        )
    ]
    targets = build_review_sft_targets(spans, selections, answers)
    records = export_review_sft_records(spans, targets)
    for record in records:
        metadata = record.setdefault("metadata", {})
        metadata.setdefault("rollout_generated", True)
        metadata.setdefault("rollout_id_seed", rollout_id)
    _REVIEW_RECORD_CACHE[cache_key] = records
    return records


def _load_spans(span_path: str) -> list[ContextSpan]:
    if span_path not in _SPAN_CACHE:
        _SPAN_CACHE[span_path] = [ContextSpan.from_dict(record) for record in read_jsonl(span_path)]
    return _SPAN_CACHE[span_path]


def _new_sample() -> Any:
    try:
        from slime.utils.types import Sample  # type: ignore

        return Sample()
    except Exception:
        return DisclosureSlimeRecordStub(prompt="", label="", metadata={})


def _get_tokenizer(args: Any) -> Any | None:
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    checkpoint = str(getattr(args, "hf_checkpoint", "") or getattr(args, "model_path", "") or getattr(args, "tokenizer_path", "") or "")
    if not checkpoint:
        return None
    try:
        from slime.utils.processing_utils import load_tokenizer  # type: ignore

        _TOKENIZER = load_tokenizer(checkpoint, trust_remote_code=True)
    except Exception:
        from transformers import AutoTokenizer  # type: ignore

        _TOKENIZER = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if getattr(_TOKENIZER, "pad_token_id", None) is None and getattr(_TOKENIZER, "eos_token", None) is not None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    return _TOKENIZER


def _encode_prompt_response(
    tokenizer: Any,
    *,
    prompt: str,
    response: str,
    max_length: int | None = None,
    max_response_length: int | None = None,
) -> tuple[list[int], int, list[int]]:
    eos = getattr(tokenizer, "eos_token", None) or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + eos, add_special_tokens=False)["input_ids"]
    if max_response_length is not None and max_response_length > 0:
        response_ids = response_ids[:max_response_length]
    if max_length is not None and max_length > 0 and len(prompt_ids) + len(response_ids) > max_length:
        keep_response = min(len(response_ids), max_length)
        keep_prompt = max_length - keep_response
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        response_ids = response_ids[:keep_response]
    return prompt_ids + response_ids, len(response_ids), [1] * len(response_ids)


def _mark_completed(sample: Any) -> None:
    try:
        from slime.utils.types import Sample  # type: ignore

        sample.status = Sample.Status.COMPLETED
    except Exception:
        sample.status = "COMPLETED"


def _output(groups: list[list[Any]], metrics: dict[str, float]) -> Any:
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput  # type: ignore

        return RolloutFnTrainOutput(samples=groups, metrics=metrics)
    except Exception:
        return RolloutOutputStub(samples=groups, metrics=metrics)
