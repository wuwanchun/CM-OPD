"""Dataset adapters for context-action samples."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from context_manager.schemas import ACTIONS, ContextActionSample, MemoryItem


class DatasetAdapter:
    name = "base"

    def build_samples(self, raw_path: Path | None = None, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError


class ToyAdapter(DatasetAdapter):
    name = "toy"

    def build_samples(self, raw_path: Path | None = None, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
        records = [
            _sample(
                sample_id=f"toy_{split}_001",
                split=split,
                memory_id="mem_goal",
                item_type="task_goal",
                summary="Answer the multi-hop question using exact supporting evidence.",
                token_count=42,
                gold_action="PIN",
                raw_evidence_ids=["raw_goal"],
                labels={"forbidden_actions": ["DROP"], "must_preserve": ["question"]},
            ),
            _sample(
                sample_id=f"toy_{split}_002",
                split=split,
                memory_id="mem_doc_support",
                item_type="retrieved_doc",
                summary="Person B attended University C.",
                token_count=180,
                gold_action="KEEP",
                raw_evidence_ids=["doc_2_sent_1"],
                labels={"forbidden_actions": ["DROP"], "must_preserve": ["Person B", "University C"]},
            ),
            _sample(
                sample_id=f"toy_{split}_003",
                split=split,
                memory_id="mem_long_distractor",
                item_type="retrieved_doc",
                summary="Long distractor paragraph about unrelated awards.",
                token_count=1400,
                gold_action="COMPRESS",
                raw_evidence_ids=["doc_4"],
            ),
            _sample(
                sample_id=f"toy_{split}_004",
                split=split,
                memory_id="mem_pytest",
                item_type="test_failure",
                summary="test_auth_expired_token expected 401 but got 200.",
                token_count=820,
                gold_action="KEEP",
                raw_evidence_ids=["pytest_003_line_17"],
                labels={"forbidden_actions": ["DROP"], "must_preserve": ["test_auth_expired_token", "expected 401"]},
                metadata={"has_error": True, "has_test_name": True},
            ),
            _sample(
                sample_id=f"toy_{split}_005",
                split=split,
                memory_id="mem_noise",
                item_type="tool_output",
                summary="Directory listing with no relevant files.",
                token_count=120,
                gold_action="ARCHIVE",
                raw_evidence_ids=["ls_001"],
            ),
        ]
        return records[:limit] if limit else records


class HotpotQAAdapter(DatasetAdapter):
    name = "hotpotqa"

    def build_samples(self, raw_path: Path | None = None, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
        if raw_path is None:
            raise ValueError("HotpotQAAdapter requires raw_path")
        from .io_utils import read_jsonl

        raw_records = read_jsonl(raw_path)
        out: list[dict[str, Any]] = []
        for index, item in enumerate(raw_records):
            if limit and len(out) >= limit:
                break
            question = str(item.get("question", ""))
            supporting_titles = _supporting_titles(item)
            contexts = _iter_contexts(item)
            for para_idx, (title, sentences) in enumerate(contexts):
                if limit and len(out) >= limit:
                    break
                text = " ".join(sentences)
                is_support = title in supporting_titles
                token_count = max(1, len(text.split()))
                action = "KEEP" if is_support else ("COMPRESS" if token_count > 120 else "ARCHIVE")
                sample_id = f"hotpotqa_{split}_{index:06d}_{para_idx:02d}"
                raw_id = f"{sample_id}_raw"
                out.append(
                    _sample(
                        sample_id=sample_id,
                        split=split,
                        memory_id=f"mem_{index:06d}_{para_idx:02d}",
                        item_type="retrieved_doc",
                        summary=f"{title}: {text[:220]}",
                        token_count=token_count,
                        gold_action=action,
                        raw_evidence_ids=[raw_id] if is_support else [],
                        task_goal=question,
                        task_family="hotpotqa",
                        labels={
                            "forbidden_actions": ["DROP"] if is_support else [],
                            "must_preserve": [title] if is_support else [],
                        },
                        metadata={"dataset": "hotpotqa", "title": title, "is_supporting": is_support},
                    )
                )
        return out


class SkeletonInteractiveAdapter(DatasetAdapter):
    """Adapter placeholder for ALFWorld and ScienceWorld smoke configs."""

    def __init__(self, name: str):
        self.name = name

    def build_samples(self, raw_path: Path | None = None, split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
        records = [
            _sample(
                sample_id=f"{self.name}_{split}_001",
                split=split,
                memory_id=f"{self.name}_goal",
                item_type="task_goal",
                summary=f"{self.name} task goal should stay pinned during long interaction.",
                token_count=60,
                gold_action="PIN",
                raw_evidence_ids=[f"{self.name}_goal_raw"],
                task_family=self.name,
                labels={"forbidden_actions": ["DROP"], "must_preserve": ["task goal"]},
            ),
            _sample(
                sample_id=f"{self.name}_{split}_002",
                split=split,
                memory_id=f"{self.name}_failed_action",
                item_type="debug_attempt",
                summary="Previous invalid action should be compressed to avoid repeating it.",
                token_count=140,
                gold_action="COMPRESS",
                raw_evidence_ids=[f"{self.name}_obs_raw"],
                task_family=self.name,
            ),
        ]
        return records[:limit] if limit else records


def get_adapter(name: str) -> DatasetAdapter:
    normalized = name.lower()
    if normalized == "toy":
        return ToyAdapter()
    if normalized == "hotpotqa":
        return HotpotQAAdapter()
    if normalized in {"alfworld", "scienceworld"}:
        return SkeletonInteractiveAdapter(normalized)
    raise ValueError(f"unknown dataset adapter: {name}")


def _sample(
    *,
    sample_id: str,
    split: str,
    memory_id: str,
    item_type: str,
    summary: str,
    token_count: int,
    gold_action: str,
    raw_evidence_ids: list[str],
    task_goal: str = "Manage evidence for a long-horizon task.",
    task_family: str = "toy",
    labels: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("token_count", token_count)
    memory_item = MemoryItem(
        id=memory_id,
        type=item_type,
        summary=summary,
        raw_pointer=f"archives/{split}/{memory_id}.txt",
        metadata=merged_metadata,
    )
    sample = ContextActionSample(
        sample_id=sample_id,
        episode_id=sample_id.rsplit("_", 1)[0],
        split=split,
        task_family=task_family,
        task_goal=task_goal,
        recent_context="The agent is managing memory under a fixed context budget.",
        budget=8000,
        budget_used=5600,
        memory_item=memory_item,
        candidate_actions=list(ACTIONS),
        gold_action=gold_action,
        gold_evidence_ids=raw_evidence_ids,
        labels=labels or {},
        metadata={**merged_metadata, "task_family": task_family, "covered": split != "unseen"},
    )
    return sample.to_dict()


def _supporting_titles(item: dict[str, Any]) -> set[str]:
    facts = item.get("supporting_facts", {})
    if isinstance(facts, dict):
        titles = facts.get("title", [])
        return {str(title) for title in titles}
    if isinstance(facts, list):
        return {str(row[0]) for row in facts if row}
    return set()


def _iter_contexts(item: dict[str, Any]) -> list[tuple[str, list[str]]]:
    context = item.get("context", {})
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [(str(title), [str(s) for s in sents]) for title, sents in zip(titles, sentences)]
    if isinstance(context, list):
        rows = []
        for row in context:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                rows.append((str(row[0]), [str(s) for s in row[1]]))
        return rows
    text = str(item.get("context", ""))
    title = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return [(title, [text])] if text else []
