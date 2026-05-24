"""Run FASD-GRPO construction/evaluation directly from a Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import write_jsonl
from grpo_sdft.hf_dataset_builder import build_hotpotqa_records, load_hf_rows
from grpo_sdft.joint_loss_config import JointLossConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--k-rollouts", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "grpo_sdft_hf"))
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    args = parser.parse_args()

    if args.dataset != "hotpotqa/hotpot_qa":
        raise SystemExit("v1 direct HF test supports --dataset hotpotqa/hotpot_qa")

    rows = load_hf_rows(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        max_rows=args.max_rows,
        cache_dir=args.cache_dir,
    )
    actions, segments, samples = build_hotpotqa_records(
        rows,
        k_rollouts=args.k_rollouts,
        split=args.split,
        confidence_threshold=args.confidence_threshold,
        require_raw_evidence=True,
    )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / f"hf_trajectories_{args.split}.jsonl", [action.to_dict() for action in actions])
    write_jsonl(output_dir / f"hf_segments_{args.split}.jsonl", [segment.to_dict() for segment in segments])
    write_jsonl(output_dir / f"hf_fasd_grpo_samples_{args.split}.jsonl", [sample.to_dict() for sample in samples])

    config = JointLossConfig()
    config.validate()
    report = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "max_rows": args.max_rows,
        "loaded_rows": len(rows),
        "k_rollouts": args.k_rollouts,
        "num_actions": len(actions),
        "num_segments": len(segments),
        "num_samples": len(samples),
        "num_groups": len({action.group_id for action in actions}),
        "num_supporting_actions": sum(1 for action in actions if action.metadata.get("is_supporting")),
        "num_sdft_samples": sum(1 for action in actions if action.loss_masks.sdft),
        "num_replay_samples": sum(1 for action in actions if action.loss_masks.replay),
        "advantage_min": min(action.advantage for action in actions),
        "advantage_max": max(action.advantage for action in actions),
        "joint_loss": config.formula(),
        "status": "passed",
    }
    report_path = output_dir / f"hf_report_{args.split}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
