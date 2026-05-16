"""Build FASD-GRPO records directly from Hugging Face datasets."""

from __future__ import annotations

import json
from typing import Any

from context_manager.schemas import ACTIONS

from .hindsight_relabeler import apply_hindsight_masks
from .reward_decomposer import assign_segment_and_action_rewards, compute_group_advantages
from .segmenter import assert_group_has_k_rollouts, group_actions_by_rollout, segment_rollout
from .trajectory_schema import GRPOSDFTSample, SegmentRecord, TrajectoryAction
from .grpo_sample_builder import build_training_samples


def load_hf_rows(
    *,
    dataset: str,
    config: str | None,
    split: str,
    max_rows: int,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load a small deterministic slice from a Hugging Face dataset."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required to load HF datasets") from exc

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if config:
        dataset_obj = load_dataset(dataset, config, split=split, **kwargs)
    else:
        dataset_obj = load_dataset(dataset, split=split, **kwargs)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset_obj):
        if index >= max_rows:
            break
        rows.append(dict(row))
    return rows


def build_hotpotqa_records(
    rows: list[dict[str, Any]],
    *,
    k_rollouts: int = 4,
    split: str = "validation",
    confidence_threshold: float = 0.7,
    require_raw_evidence: bool = True,
) -> tuple[list[TrajectoryAction], list[SegmentRecord], list[GRPOSDFTSample]]:
    """Convert HotpotQA rows into grouped FASD-GRPO training/eval records.

    Each HotpotQA question is one GRPO group. Candidate paragraphs become the K
    candidate rollout/action records for that task. Supporting-fact paragraphs
    receive higher episode reward and KEEP/PIN-style teacher corrections; real
    distractor paragraphs receive lower reward and ARCHIVE/COMPRESS corrections.
    """

    if k_rollouts < 2:
        raise ValueError("k_rollouts must be >= 2")

    actions: list[TrajectoryAction] = []
    for row_idx, row in enumerate(rows):
        candidates = _hotpotqa_candidates(row, row_idx=row_idx, split=split)
        if len(candidates) < k_rollouts:
            continue
        candidates.sort(key=lambda item: (not item["is_supporting"], item["title"]))
        for rollout_idx, candidate in enumerate(candidates[:k_rollouts]):
            actions.append(_candidate_to_action(row, candidate, rollout_idx=rollout_idx))

    if not actions:
        raise ValueError("no FASD-GRPO records could be built from the HF dataset")

    assign_segment_and_action_rewards(actions)
    apply_hindsight_masks(actions, min_confidence=confidence_threshold, require_raw_evidence=require_raw_evidence)
    rollouts = group_actions_by_rollout(actions)
    assert_group_has_k_rollouts(rollouts, k_rollouts)
    compute_group_advantages(rollouts)

    segments: list[SegmentRecord] = []
    for rollout in rollouts:
        segments.extend(segment_rollout(rollout))
    samples = build_training_samples(actions)
    return actions, segments, samples


def _hotpotqa_candidates(row: dict[str, Any], *, row_idx: int, split: str) -> list[dict[str, Any]]:
    supporting_titles = _supporting_titles(row)
    contexts = _iter_contexts(row)
    question = str(row.get("question", ""))
    task_id = str(row.get("id", row.get("_id", f"hotpotqa_{split}_{row_idx:06d}")))
    candidates: list[dict[str, Any]] = []
    for para_idx, (title, sentences) in enumerate(contexts):
        text = " ".join(str(sentence) for sentence in sentences)
        if not text.strip():
            continue
        is_supporting = title in supporting_titles
        raw_id = f"{task_id}_p{para_idx:02d}"
        candidates.append(
            {
                "task_id": task_id,
                "question": question,
                "title": title,
                "text": text,
                "raw_id": raw_id,
                "para_idx": para_idx,
                "is_supporting": is_supporting,
                "token_count": max(1, len(text.split())),
            }
        )
    return candidates


def _candidate_to_action(row: dict[str, Any], candidate: dict[str, Any], *, rollout_idx: int) -> TrajectoryAction:
    task_id = str(candidate["task_id"])
    title = str(candidate["title"])
    raw_id = str(candidate["raw_id"])
    token_count = int(candidate["token_count"])
    is_supporting = bool(candidate["is_supporting"])
    memory_id = f"mem_{task_id}_{candidate['para_idx']:02d}"

    student_action_name = _student_baseline_action(token_count=token_count)
    teacher_action_name = "KEEP" if is_supporting else ("COMPRESS" if token_count > 120 else "ARCHIVE")
    student_action = _action_json(student_action_name, memory_id, "HF-derived baseline action")
    teacher_action = _action_json(teacher_action_name, memory_id, "HF-derived hindsight correction")
    failure_type = "none"
    if student_action_name != teacher_action_name:
        failure_type = "missing_evidence" if is_supporting else "repeated_tool"

    episode_reward = 1.0 if is_supporting else 0.25
    action_reward = 1.0 if student_action_name == teacher_action_name else -1.0
    confidence = 0.95 if is_supporting else 0.8
    hint = (
        f"turn_id=1 memory_id={memory_id} raw_evidence_id={raw_id} "
        f"suggested_action: {teacher_action_name} "
        f"verifiable_reason: HotpotQA title '{title}' "
        f"{'is' if is_supporting else 'is not'} a supporting fact for the question."
    )

    return TrajectoryAction(
        task_id=task_id,
        rollout_id=f"{task_id}_r{rollout_idx:02d}",
        group_id=task_id,
        segment_id="seg_evidence_selection",
        turn_id=1,
        action_type="memory_action",
        student_action=student_action,
        teacher_action=teacher_action,
        episode_reward=episode_reward,
        action_reward=action_reward,
        failure_type=failure_type,
        raw_evidence_ids=[raw_id],
        hindsight_hint=hint,
        confidence=confidence,
        metadata={
            "hf_dataset": "hotpotqa/hotpot_qa",
            "question": str(row.get("question", "")),
            "title": title,
            "is_supporting": is_supporting,
            "token_count": token_count,
            "candidate_actions": list(ACTIONS),
        },
    )


def _student_baseline_action(*, token_count: int) -> str:
    if token_count > 120:
        return "COMPRESS"
    return "ARCHIVE"


def _action_json(action: str, memory_id: str, reason: str) -> str:
    return json.dumps({"action": action, "target_ids": [memory_id], "reason": reason}, ensure_ascii=False, separators=(",", ":"))


def _supporting_titles(item: dict[str, Any]) -> set[str]:
    facts = item.get("supporting_facts", {})
    if isinstance(facts, dict):
        return {str(title) for title in facts.get("title", [])}
    if isinstance(facts, list):
        return {str(row[0]) for row in facts if row}
    return set()


def _iter_contexts(item: dict[str, Any]) -> list[tuple[str, list[str]]]:
    context = item.get("context", {})
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [(str(title), [str(sentence) for sentence in sents]) for title, sents in zip(titles, sentences)]
    if isinstance(context, list):
        rows = []
        for row in context:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                rows.append((str(row[0]), [str(sentence) for sentence in row[1]]))
        return rows
    return []
