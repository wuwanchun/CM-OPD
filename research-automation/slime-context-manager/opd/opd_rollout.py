"""Slime rollout bridge for external context-policy OPD."""

from __future__ import annotations

import atexit
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from .context_policy_opd_server import OPDRecorder


_global_worker: "AsyncOPDWorker | None" = None
_worker_lock = threading.Lock()


@dataclass
class RolloutOutputStub:
    """Fallback output when slime is not installed."""

    samples: list[list[Any]]
    metrics: dict[str, float] | None = None


class AsyncOPDWorker:
    """Background OPD sample collector."""

    def __init__(self, args: Any, data_buffer: Any = None):
        self.args = args
        self.data_buffer = data_buffer
        self.recorder = OPDRecorder(require_raw_evidence=getattr(args, "require_raw_evidence", True))
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list[Any]]] = queue.Queue(maxsize=100000)
        self.worker_thread: threading.Thread | None = None
        self._group_id = 0

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._loop, daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def _loop(self) -> None:
        while self.running:
            for record in self.recorder.drain_samples():
                self.output_queue.put((self._group_id, [record.to_slime_sample()]))
                self._group_id += 1
            time.sleep(0.05)

    def drain_groups(self, target_size: int, timeout_s: float = 30.0) -> list[list[Any]]:
        deadline = time.time() + timeout_s
        groups: list[list[Any]] = []
        while len(groups) < target_size and time.time() < deadline:
            try:
                _, group = self.output_queue.get(timeout=0.05)
                groups.append(group)
            except queue.Empty:
                continue
        return groups


def get_global_worker(args: Any, data_buffer: Any = None) -> AsyncOPDWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or _global_worker.worker_thread is None or not _global_worker.worker_thread.is_alive():
            _global_worker = AsyncOPDWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


def generate_rollout_opd(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False) -> Any:
    """Slime-compatible rollout function.

    Training data production is request/event driven. This function drains
    accepted OPD samples from the background worker.
    """

    worker = get_global_worker(args, data_buffer)
    target = int(getattr(args, "rollout_batch_size", 1))
    timeout_s = float(getattr(args, "opd_drain_timeout_s", 30.0))

    if evaluation:
        try:
            from slime.rollout.sglang_rollout import eval_rollout  # type: ignore
            from slime.utils.async_utils import run  # type: ignore

            output, _ = run(eval_rollout(args, rollout_id))
            return output
        except Exception:
            return RolloutOutputStub(samples=[], metrics={"rollout/eval_fallback": 1.0})

    groups = worker.drain_groups(target, timeout_s=timeout_s)
    metrics = {"rollout/opd_groups": float(len(groups))}
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput  # type: ignore

        return RolloutFnTrainOutput(samples=groups, metrics=metrics)
    except Exception:
        return RolloutOutputStub(samples=groups, metrics=metrics)


async def custom_rm(args: Any, sample_or_samples: Any, **_: Any) -> Any:
    """Custom reward hook that returns stored process rewards."""

    def score_one(sample: Any) -> dict[str, float]:
        reward = getattr(sample, "reward", None)
        if isinstance(reward, dict):
            return {"score": float(reward.get("score", reward.get("process_reward", 0.0)) or 0.0)}
        return {"score": float(reward or 0.0)}

    if isinstance(sample_or_samples, list):
        return [score_one(sample) for sample in sample_or_samples]
    return score_one(sample_or_samples)


atexit.register(stop_global_worker)
