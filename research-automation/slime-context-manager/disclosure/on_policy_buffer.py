"""On-policy replay buffer to prevent iterative drift in selection policy training."""

from __future__ import annotations

import json
import logging
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReplayRecord:
    """A single record in the replay buffer."""

    record_id: str
    task_id: str
    span_id: str
    source_id: str
    selection_id: str
    prompt: str
    response: str
    target_state: str
    accepted: bool
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_id": self.task_id,
            "span_id": self.span_id,
            "source_id": self.source_id,
            "selection_id": self.selection_id,
            "prompt": self.prompt,
            "response": self.response,
            "target_state": self.target_state,
            "accepted": self.accepted,
            "iteration": self.iteration,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayRecord":
        return cls(
            record_id=str(data.get("record_id", "")),
            task_id=str(data.get("task_id", "")),
            span_id=str(data.get("span_id", "")),
            source_id=str(data.get("source_id", "")),
            selection_id=str(data.get("selection_id", "")),
            prompt=str(data.get("prompt", "")),
            response=str(data.get("response", "")),
            target_state=str(data.get("target_state", data.get("target_action", ""))),
            accepted=bool(data.get("accepted", True)),
            iteration=int(data.get("iteration", 0)),
            metadata=dict(data.get("metadata", {})),
        )


class OnPolicyReplayBuffer:
    """
    Replay buffer that stores historical training samples to prevent forgetting.
    
    During iterative policy training, the buffer:
    1. Stores accepted training samples from each iteration
    2. Mixes historical samples with current on-policy samples
    3. Ensures diverse training data across iterations
    4. Prevents catastrophic forgetting of earlier learned behaviors
    """

    def __init__(
        self,
        max_size: int = 10000,
        replay_ratio: float = 0.2,
        seed: int | None = None,
    ):
        """
        Initialize the replay buffer.
        
        Args:
            max_size: Maximum number of records to store
            replay_ratio: Fraction of batch to sample from replay (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.max_size = max_size
        self.replay_ratio = replay_ratio
        self._buffer: deque[ReplayRecord] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._iteration_counts: dict[int, int] = {}
        self._rng = random.Random(seed)

    def add(self, records: list[ReplayRecord]) -> int:
        """
        Add records to the buffer.
        
        Only accepted records are stored. Records exceeding max_size
        will cause oldest records to be evicted (FIFO).
        
        Args:
            records: List of ReplayRecord to add
            
        Returns:
            Number of records actually added
        """
        added = 0
        with self._lock:
            for record in records:
                if not record.accepted:
                    continue
                self._buffer.append(record)
                self._iteration_counts[record.iteration] = self._iteration_counts.get(record.iteration, 0) + 1
                added += 1
        logger.debug(f"Added {added} records to replay buffer (size={len(self._buffer)})")
        return added

    def sample(self, batch_size: int, current_iteration: int = -1) -> list[ReplayRecord]:
        """
        Sample a mixed batch of replay and current records.
        
        Args:
            batch_size: Total number of records to sample
            current_iteration: Current iteration (records from this iteration are excluded)
            
        Returns:
            List of sampled records
        """
        with self._lock:
            if not self._buffer:
                return []

            # Filter out current iteration records for true replay
            eligible = [
                r for r in self._buffer
                if r.iteration != current_iteration
            ]
            
            if not eligible:
                return []

            # Sample based on replay ratio
            replay_size = min(int(batch_size * self.replay_ratio), len(eligible))
            sampled = self._rng.sample(eligible, replay_size)
            
            logger.debug(
                f"Sampled {len(sampled)} replay records (batch_size={batch_size}, "
                f"ratio={self.replay_ratio}, eligible={len(eligible)})"
            )
            return sampled

    def sample_all(self, batch_size: int) -> list[ReplayRecord]:
        """
        Sample from the entire buffer without iteration filtering.
        
        Args:
            batch_size: Number of records to sample
            
        Returns:
            List of sampled records
        """
        with self._lock:
            if not self._buffer:
                return []
            return self._rng.sample(list(self._buffer), min(batch_size, len(self._buffer)))

    def get_iteration_records(self, iteration: int) -> list[ReplayRecord]:
        """Get all records from a specific iteration."""
        with self._lock:
            return [r for r in self._buffer if r.iteration == iteration]

    def clear_iteration(self, iteration: int) -> int:
        """
        Remove all records from a specific iteration.
        
        Args:
            iteration: Iteration number to clear
            
        Returns:
            Number of records removed
        """
        removed = 0
        with self._lock:
            new_buffer = deque(maxlen=self.max_size)
            for record in self._buffer:
                if record.iteration != iteration:
                    new_buffer.append(record)
                else:
                    removed += 1
            self._buffer = new_buffer
            if iteration in self._iteration_counts:
                del self._iteration_counts[iteration]
        logger.debug(f"Cleared {removed} records from iteration {iteration}")
        return removed

    def save(self, path: str | Path) -> int:
        """
        Save buffer to JSONL file.
        
        Args:
            path: Output file path
            
        Returns:
            Number of records saved
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            records = [r.to_dict() for r in self._buffer]
        
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(records)} records to {path}")
        return len(records)

    def load(self, path: str | Path) -> int:
        """
        Load buffer from JSONL file.
        
        Args:
            path: Input file path
            
        Returns:
            Number of records loaded
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Replay buffer file not found: {path}")
            return 0

        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = ReplayRecord.from_dict(json.loads(line))
                    with self._lock:
                        self._buffer.append(record)
                        self._iteration_counts[record.iteration] = (
                            self._iteration_counts.get(record.iteration, 0) + 1
                        )
                    loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load record: {e}")
        
        logger.info(f"Loaded {loaded} records from {path}")
        return loaded

    def stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "size": len(self._buffer),
                "max_size": self.max_size,
                "replay_ratio": self.replay_ratio,
                "iteration_counts": dict(self._iteration_counts),
                "unique_tasks": len(set(r.task_id for r in self._buffer)),
                "unique_spans": len(set(r.span_id for r in self._buffer)),
            }

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"OnPolicyReplayBuffer(size={len(self)}, max_size={self.max_size})"


def build_replay_records_from_targets(
    targets: list[Any],
    iteration: int,
    *,
    accepted_only: bool = True,
) -> list[ReplayRecord]:
    """
    Convert ReviewSFTTarget or OnPolicyDistillationTarget to ReplayRecord.
    
    Args:
        targets: List of training targets
        iteration: Current training iteration
        accepted_only: Whether to only include accepted targets
        
    Returns:
        List of ReplayRecord
    """
    records = []
    for idx, target in enumerate(targets):
        # Check acceptance
        accepted = getattr(target, "accepted_for_training", True)
        if accepted_only and not accepted:
            continue

        # Extract fields
        record = ReplayRecord(
            record_id=f"replay_{iteration}_{idx}",
            task_id=str(getattr(target, "task_id", "")),
            span_id=str(getattr(target, "span_id", "")),
            source_id=str(getattr(target, "source_id", "")),
            selection_id=str(getattr(target, "selection_id", "")),
            prompt=str(getattr(target, "prompt", "")),
            response=str(getattr(target, "response", "")),
            target_state=str(getattr(target, "target_state", getattr(target, "target_action", ""))),
            accepted=accepted,
            iteration=iteration,
            metadata=getattr(target, "metadata", {}) or {},
        )
        records.append(record)
    
    return records


def mix_replay_with_onpolicy(
    onpolicy_records: list[ReplayRecord],
    replay_buffer: OnPolicyReplayBuffer,
    batch_size: int,
    current_iteration: int,
) -> list[ReplayRecord]:
    """
    Mix on-policy records with replay buffer samples.
    
    Args:
        onpolicy_records: Current on-policy training records
        replay_buffer: Replay buffer instance
        batch_size: Target batch size
        current_iteration: Current iteration number
        
    Returns:
        Mixed list of records
    """
    if not replay_buffer or len(replay_buffer) == 0:
        return onpolicy_records[:batch_size]

    replay_samples = replay_buffer.sample(batch_size, current_iteration)
    
    # Calculate how many on-policy records to include
    replay_count = len(replay_samples)
    onpolicy_count = min(len(onpolicy_records), batch_size - replay_count)
    
    mixed = onpolicy_records[:onpolicy_count] + replay_samples
    
    # Shuffle to avoid ordering bias
    random.shuffle(mixed)
    
    logger.debug(
        f"Mixed batch: {onpolicy_count} on-policy + {replay_count} replay = {len(mixed)} total"
    )
    
    return mixed
