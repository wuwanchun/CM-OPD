"""Async slime-compatible rollout helpers for SDFT samples."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_io.io_utils import read_jsonl

from .sdft_sample_builder import SDFTTrainingSample

logger = logging.getLogger(__name__)


@dataclass
class RolloutOutputStub:
    samples: list[list[Any]]
    metrics: dict[str, float] | None = None


class AsyncSDFTRolloutBuffer:
    """Small deterministic async-style buffer used by slime rollout workers."""

    def __init__(self, samples: list[Any]):
        self.samples = samples
        self._cursor = 0

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AsyncSDFTRolloutBuffer":
        records = read_jsonl(path)
        if records and {"sample_id", "response", "loss_masks"}.issubset(records[0]):
            return cls([SDFTTrainingSample.from_dict(record) for record in records])
        return cls([_sample_from_prompt_label(record, index=i) for i, record in enumerate(records)])

    def next_groups(self, batch_size: int) -> list[list[Any]]:
        if not self.samples:
            return []
        groups = []
        for _ in range(batch_size):
            sample = self.samples[self._cursor % len(self.samples)]
            self._cursor += 1
            if hasattr(sample, "to_slime_sample"):
                groups.append([sample.to_slime_sample()])
            else:
                groups.append([sample])
        return groups


_BUFFERS: dict[str, AsyncSDFTRolloutBuffer] = {}
_TOKENIZER = None


def generate_rollout_sdft(args: Any, rollout_id: int, data_buffer: Any = None, evaluation: bool = False) -> Any:
    """Slime-compatible SDFT rollout function.

    This is intentionally data-driven: a preceding async hint extraction job
    writes SDFT JSONL, and the rollout function serves those corrected action
    samples to slime. When called inside THUDM/slime, `data_buffer` is the
    framework `RolloutDataSource`; we consume prompt/label records from it and
    populate the exact fields required by `sft_loss`: `tokens`,
    `response_length`, `loss_mask`, `reward`, and `status`.
    """

    if evaluation:
        return _output([], {"rollout/sdft_eval_noop": 1.0})

    batch_size = int(getattr(args, "rollout_batch_size", 1))
    if data_buffer is not None and hasattr(data_buffer, "get_samples"):
        groups = data_buffer.get_samples(batch_size)
        source = "slime_data_source"
    else:
        path = str(getattr(args, "sdft_samples_path", "") or getattr(args, "prompt_data", ""))
        if not path:
            raise ValueError("generate_rollout_sdft requires args.sdft_samples_path, args.prompt_data, or data_buffer")
        buffer = _BUFFERS.setdefault(path, AsyncSDFTRolloutBuffer.from_jsonl(path))
        groups = buffer.next_groups(batch_size)
        source = "jsonl_buffer"

    prepared_groups = [_prepare_group_for_slime(args, group) for group in groups]
    return _output(
        prepared_groups,
        {
            "rollout/sdft_groups": float(len(prepared_groups)),
            "rollout/sdft_samples": float(sum(len(group) for group in prepared_groups)),
            "rollout/sdft_rollout_id": float(rollout_id),
            f"rollout/sdft_source/{source}": 1.0,
        },
    )


def _prepare_group_for_slime(args: Any, group: list[Any]) -> list[Any]:
    return [_prepare_sample_for_slime(args, sample) for sample in group]


def _prepare_sample_for_slime(args: Any, sample: Any) -> Any:
    prompt = str(getattr(sample, "prompt", ""))
    response = _sample_response(sample)
    if not prompt:
        raise ValueError("SDFT sample is missing prompt")
    if not response:
        raise ValueError("SDFT sample is missing response/label")

    tokenizer = _get_tokenizer(args)
    if tokenizer is not None:
        input_ids, response_length, loss_mask = _encode_prompt_response(
            tokenizer,
            prompt=prompt,
            response=response,
            max_length=int(getattr(args, "rollout_max_model_len", 0) or getattr(args, "max_length", 0) or 0) or None,
            max_response_length=int(getattr(args, "rollout_max_response_len", 0) or 0) or None,
        )
        sample.tokens = input_ids
        sample.response_length = response_length
        sample.loss_mask = loss_mask
    elif not getattr(sample, "tokens", None):
        logger.warning("No tokenizer available; returning SDFT sample without token fields.")

    sample.response = response
    if getattr(sample, "reward", None) is None:
        sample.reward = 1.0
    if getattr(sample, "train_metadata", None) is None:
        sample.train_metadata = {"loss": "sdft_sft", "sdft": True}
    if getattr(sample, "metadata", None) is None:
        sample.metadata = {}
    sample.metadata.setdefault("sdft_source", "slime_async_rollout")
    _mark_completed(sample)
    return sample


def _sample_response(sample: Any) -> str:
    label = getattr(sample, "label", None)
    if label is not None:
        return str(label)
    response = getattr(sample, "response", None)
    if response is not None:
        return str(response)
    metadata = getattr(sample, "metadata", {}) or {}
    for key in ("teacher_action", "corrected_action", "response", "label"):
        if metadata.get(key) is not None:
            return str(metadata[key])
    return ""


def _get_tokenizer(args: Any) -> Any | None:
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    checkpoint = str(
        getattr(args, "hf_checkpoint", "")
        or getattr(args, "model_path", "")
        or getattr(args, "tokenizer_path", "")
        or ""
    )
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


def _sample_from_prompt_label(record: dict[str, Any], *, index: int) -> Any:
    try:
        from slime.utils.types import Sample  # type: ignore

        return Sample(
            prompt=str(record.get("prompt", "")),
            label=str(record.get("label", record.get("response", ""))),
            metadata=dict(record.get("metadata", {})),
            group_index=index,
            index=index,
        )
    except Exception:
        sample = SDFTSlimeRecordStub(
            prompt=str(record.get("prompt", "")),
            label=str(record.get("label", record.get("response", ""))),
            metadata=dict(record.get("metadata", {})),
            group_index=index,
            index=index,
        )
        return sample


@dataclass
class SDFTSlimeRecordStub:
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


def _output(groups: list[list[Any]], metrics: dict[str, float]) -> Any:
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput  # type: ignore

        return RolloutFnTrainOutput(samples=groups, metrics=metrics)
    except Exception:
        return RolloutOutputStub(samples=groups, metrics=metrics)
