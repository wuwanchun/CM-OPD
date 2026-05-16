"""Run async SDFT hint extraction from failed HF dataset trajectories."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from data.io_utils import write_jsonl
from grpo_sdft.async_sdft_rollout import generate_rollout_sdft
from grpo_sdft.hf_dataset_builder import build_hotpotqa_records, load_hf_rows
from grpo_sdft.sdft_hint_extractor import extract_hints_async, failed_actions
from grpo_sdft.sdft_sample_builder import build_sdft_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--k-rollouts", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "sdft_async_hf"))
    parser.add_argument("--hint-concurrency", type=int, default=8)
    parser.add_argument("--votes-per-action", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.dataset != "hotpotqa/hotpot_qa":
        raise SystemExit("v1 async SDFT test supports --dataset hotpotqa/hotpot_qa")

    rows = load_hf_rows(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        max_rows=args.max_rows,
        cache_dir=args.cache_dir,
    )
    actions, segments, _ = build_hotpotqa_records(
        rows,
        k_rollouts=args.k_rollouts,
        split=args.split,
        confidence_threshold=args.min_confidence,
        require_raw_evidence=True,
    )
    failures = failed_actions(actions)
    hints = asyncio.run(
        extract_hints_async(
            failures,
            concurrency=args.hint_concurrency,
            mode="rule",
            votes_per_action=args.votes_per_action,
        )
    )
    sdft_samples = build_sdft_samples(actions, hints, min_confidence=args.min_confidence)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / f"failed_trajectories_{args.split}.jsonl", [action.to_dict() for action in failures])
    write_jsonl(output_dir / f"hindsight_hints_{args.split}.jsonl", [hint.to_dict() for hint in hints])
    write_jsonl(output_dir / f"sdft_samples_{args.split}.jsonl", [sample.to_dict() for sample in sdft_samples])
    write_jsonl(output_dir / f"sdft_slime_sft_{args.split}.jsonl", [_to_slime_sft_record(sample) for sample in sdft_samples])

    rollout_args = SimpleNamespace(
        sdft_samples_path=str(output_dir / f"sdft_samples_{args.split}.jsonl"),
        rollout_batch_size=args.rollout_batch_size,
    )
    rollout_output = generate_rollout_sdft(rollout_args, rollout_id=0, data_buffer=None, evaluation=False)
    rollout_groups = getattr(rollout_output, "samples", [])
    metrics = getattr(rollout_output, "metrics", {}) or {}

    report = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "max_rows": args.max_rows,
        "loaded_rows": len(rows),
        "k_rollouts": args.k_rollouts,
        "num_actions": len(actions),
        "num_segments": len(segments),
        "num_failed_actions": len(failures),
        "num_hints": len(hints),
        "num_sdft_samples": len(sdft_samples),
        "num_rollout_groups": len(rollout_groups),
        "rollout_metrics": metrics,
        "hint_concurrency": args.hint_concurrency,
        "votes_per_action": args.votes_per_action,
        "min_confidence": args.min_confidence,
        "status": "passed",
    }
    _validate_report(report)
    report_path = output_dir / f"sdft_async_report_{args.split}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _to_slime_sft_record(sample: object) -> dict[str, object]:
    data = sample.to_dict()
    return {
        "prompt": data["prompt"],
        "label": data["response"],
        "metadata": data["metadata"],
    }


def _validate_report(report: dict[str, object]) -> None:
    if int(report["num_failed_actions"]) <= 0:
        raise SystemExit("no failed trajectories were found for SDFT")
    if int(report["num_hints"]) <= 0:
        raise SystemExit("async hint extraction produced no hints")
    if int(report["num_sdft_samples"]) <= 0:
        raise SystemExit("SDFT sample builder produced no samples")
    if int(report["num_rollout_groups"]) <= 0:
        raise SystemExit("slime-compatible SDFT rollout produced no groups")


if __name__ == "__main__":
    main()
