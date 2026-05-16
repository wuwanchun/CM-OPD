"""Build FASD-GRPO / CodeHER-GRPO synthetic validation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from context_manager.schemas import ContextActionSample
from data.io_utils import read_jsonl, write_jsonl
from grpo_sdft.grpo_sample_builder import build_training_samples
from grpo_sdft.hindsight_relabeler import apply_hindsight_masks
from grpo_sdft.reward_decomposer import assign_segment_and_action_rewards, compute_group_advantages
from grpo_sdft.segmenter import assert_group_has_k_rollouts, group_actions_by_rollout, segment_rollout
from grpo_sdft.trajectory_schema import TrajectoryAction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "processed" / "context_actions"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "grpo_sdft"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--k-rollouts", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--require-raw-evidence", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    records = read_jsonl(Path(args.input_dir) / f"{args.split}.jsonl")
    samples = [ContextActionSample.from_dict(record) for record in records]
    actions: list[TrajectoryAction] = []
    for idx, sample in enumerate(samples, start=1):
        actions.extend(_synthesize_k_rollouts(sample, idx=idx, k=args.k_rollouts))

    assign_segment_and_action_rewards(actions)
    apply_hindsight_masks(
        actions,
        min_confidence=args.confidence_threshold,
        require_raw_evidence=args.require_raw_evidence,
    )
    rollouts = group_actions_by_rollout(actions)
    assert_group_has_k_rollouts(rollouts, args.k_rollouts)
    compute_group_advantages(rollouts)

    # `actions` objects are mutated in place by reward/advantage helpers.
    segments = []
    for rollout in rollouts:
        segments.extend(segment_rollout(rollout))
    training_samples = build_training_samples(actions)

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / f"trajectories_{args.split}.jsonl", [action.to_dict() for action in actions])
    write_jsonl(out_dir / f"segments_{args.split}.jsonl", [segment.to_dict() for segment in segments])
    write_jsonl(out_dir / f"fasd_grpo_samples_{args.split}.jsonl", [sample.to_dict() for sample in training_samples])

    summary = {
        "split": args.split,
        "num_tasks": len(samples),
        "k_rollouts": args.k_rollouts,
        "num_actions": len(actions),
        "num_rollouts": len(rollouts),
        "num_segments": len(segments),
        "num_training_samples": len(training_samples),
        "num_sdft_samples": sum(1 for action in actions if action.loss_masks.sdft),
        "num_replay_samples": sum(1 for action in actions if action.loss_masks.replay),
        "confidence_threshold": args.confidence_threshold,
        "require_raw_evidence": args.require_raw_evidence,
    }
    summary_path = out_dir / f"summary_{args.split}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _synthesize_k_rollouts(sample: ContextActionSample, *, idx: int, k: int) -> list[TrajectoryAction]:
    if k < 2:
        raise ValueError("--k-rollouts must be >= 2 for GRPO advantage computation")
    variants = [_success_variant(sample), _missing_evidence_variant(sample), _format_error_variant(sample), _repeated_tool_variant(sample)]
    while len(variants) < k:
        variants.append(_low_confidence_variant(sample, len(variants)))

    actions: list[TrajectoryAction] = []
    task_id = sample.sample_id
    group_id = task_id
    segment_id = f"seg_{idx:03d}"
    for rollout_idx, variant in enumerate(variants[:k]):
        actions.append(
            TrajectoryAction(
                task_id=task_id,
                rollout_id=f"{task_id}_r{rollout_idx:02d}",
                group_id=group_id,
                segment_id=segment_id,
                turn_id=1,
                action_type="memory_action",
                student_action=variant["student_action"],
                teacher_action=variant["teacher_action"],
                episode_reward=float(variant["episode_reward"]),
                action_reward=float(variant["action_reward"]),
                failure_type=str(variant["failure_type"]),
                raw_evidence_ids=list(variant["raw_evidence_ids"]),
                hindsight_hint=str(variant["hindsight_hint"]),
                confidence=float(variant["confidence"]),
                metadata={
                    "source_sample_id": sample.sample_id,
                    "task_family": sample.task_family,
                    "memory_id": sample.memory_item.id,
                    "gold_action": sample.gold_action,
                    "candidate_actions": sample.candidate_actions,
                    "is_synthetic_validation": True,
                },
            )
        )
    return actions


def _action_json(action: str, memory_id: str, reason: str) -> str:
    return json.dumps({"action": action, "target_ids": [memory_id], "reason": reason}, ensure_ascii=False, separators=(",", ":"))


def _success_variant(sample: ContextActionSample) -> dict[str, object]:
    action = _action_json(sample.gold_action, sample.memory_item.id, "successful rollout kept the useful memory decision")
    return {
        "student_action": action,
        "teacher_action": action,
        "episode_reward": 1.0,
        "action_reward": 1.0,
        "failure_type": "none",
        "raw_evidence_ids": sample.gold_evidence_ids,
        "hindsight_hint": f"turn_id=1 memory_id={sample.memory_item.id} raw_evidence_id={_first_evidence(sample)} suggested_action: {sample.gold_action} verifiable_reason: successful trajectory reused this evidence.",
        "confidence": 0.95,
    }


def _missing_evidence_variant(sample: ContextActionSample) -> dict[str, object]:
    wrong = "DROP" if sample.gold_action != "DROP" else "ARCHIVE"
    return {
        "student_action": _action_json(wrong, sample.memory_item.id, "failed rollout discarded useful evidence"),
        "teacher_action": _action_json(sample.gold_action, sample.memory_item.id, "teacher correction restores evidence lifecycle action"),
        "episode_reward": 0.0,
        "action_reward": -1.0,
        "failure_type": "missing_evidence",
        "raw_evidence_ids": sample.gold_evidence_ids,
        "hindsight_hint": f"turn_id=1 memory_id={sample.memory_item.id} raw_evidence_id={_first_evidence(sample)} suggested_action: {sample.gold_action} verifiable_reason: final failure needed this raw evidence.",
        "confidence": 0.9,
    }


def _format_error_variant(sample: ContextActionSample) -> dict[str, object]:
    return {
        "student_action": "not a json memory action",
        "teacher_action": _action_json(sample.gold_action, sample.memory_item.id, "teacher correction fixes tool-call format"),
        "episode_reward": -0.25,
        "action_reward": -1.0,
        "failure_type": "format_error",
        "raw_evidence_ids": [],
        "hindsight_hint": f"turn_id=1 memory_id={sample.memory_item.id} raw_evidence_id=UNKNOWN suggested_action: {sample.gold_action} verifiable_reason: invalid memory action format prevented execution.",
        "confidence": 0.88,
    }


def _repeated_tool_variant(sample: ContextActionSample) -> dict[str, object]:
    repeated = "RETRIEVE" if sample.gold_action != "RETRIEVE" else "KEEP"
    return {
        "student_action": _action_json(repeated, sample.memory_item.id, "rollout repeated an unnecessary memory operation"),
        "teacher_action": _action_json(sample.gold_action, sample.memory_item.id, "teacher correction avoids repeated tool use"),
        "episode_reward": 0.25,
        "action_reward": -0.5,
        "failure_type": "repeated_tool",
        "raw_evidence_ids": sample.gold_evidence_ids,
        "hindsight_hint": f"turn_id=1 memory_id={sample.memory_item.id} raw_evidence_id={_first_evidence(sample)} suggested_action: {sample.gold_action} verifiable_reason: repeated memory operation wasted budget.",
        "confidence": 0.82,
    }


def _low_confidence_variant(sample: ContextActionSample, idx: int) -> dict[str, object]:
    return {
        "student_action": _action_json("KEEP", sample.memory_item.id, f"extra rollout variant {idx}"),
        "teacher_action": _action_json(sample.gold_action, sample.memory_item.id, "low confidence correction"),
        "episode_reward": 0.1,
        "action_reward": 0.0,
        "failure_type": "missing_evidence",
        "raw_evidence_ids": sample.gold_evidence_ids,
        "hindsight_hint": "low confidence synthetic correction",
        "confidence": 0.4,
    }


def _first_evidence(sample: ContextActionSample) -> str:
    return sample.gold_evidence_ids[0] if sample.gold_evidence_ids else "UNKNOWN"


if __name__ == "__main__":
    main()
