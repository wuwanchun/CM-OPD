"""Async slime-compatible rollout helpers for SDFT samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.io_utils import read_jsonl

from .sdft_sample_builder import SDFTTrainingSample


@dataclass
class RolloutOutputStub:
    samples: list[list[Any]]
    metrics: dict[str, float] | None = None


class AsyncSDFTRolloutBuffer:
    """Small deterministic async-style buffer used by slime rollout workers."""

    def __init__(self, samples: list[SDFTTrainingSample]):
        self.samples = samples
        self._cursor = 0

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AsyncSDFTRolloutBuffer":
        records = read_jsonl(path)
        return cls([SDFTTrainingSample.from_dict(record) for record in records])

    def next_groups(self, batch_size: int) -> list[list[Any]]:
        if not self.samples:
            return []
        groups = []
        for _ in range(batch_size):
            sample = self.samples[self._cursor % len(self.samples)]
            self._cursor += 1
            groups.append([sample.to_slime_sample()])
        return groups


_BUFFERS: dict[str, AsyncSDFTRolloutBuffer] = {}


def generate_rollout_sdft(args: Any, rollout_id: int, data_buffer: Any = None, evaluation: bool = False) -> Any:
    """Slime-compatible SDFT rollout function.

    This is intentionally data-driven: a preceding async hint extraction job
    writes SDFT JSONL, and the rollout function serves those corrected action
    samples to slime.
    """

    if evaluation:
        return _output([], {"rollout/sdft_eval_noop": 1.0})

    path = str(getattr(args, "sdft_samples_path", "") or getattr(args, "prompt_data", ""))
    if not path:
        raise ValueError("generate_rollout_sdft requires args.sdft_samples_path or args.prompt_data")
    batch_size = int(getattr(args, "rollout_batch_size", 1))
    buffer = _BUFFERS.setdefault(path, AsyncSDFTRolloutBuffer.from_jsonl(path))
    groups = buffer.next_groups(batch_size)
    return _output(groups, {"rollout/sdft_groups": float(len(groups)), "rollout/sdft_rollout_id": float(rollout_id)})


def _output(groups: list[list[Any]], metrics: dict[str, float]) -> Any:
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput  # type: ignore

        return RolloutFnTrainOutput(samples=groups, metrics=metrics)
    except Exception:
        return RolloutOutputStub(samples=groups, metrics=metrics)
