"""Export context-action data to slime-compatible JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from context_manager.hf_policy import render_policy_prompt
from context_manager.schemas import ContextActionSample
from dataset_io.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "processed" / "context_actions"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "slime"))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input_dir) / f"{args.split}.jsonl")
    out = []
    for record in records:
        sample = ContextActionSample.from_dict(record)
        out.append(
            {
                "prompt": render_policy_prompt(sample),
                "label": json.dumps(
                    {
                        "action": sample.gold_action,
                        "target_ids": [sample.memory_item.id],
                        "reason": "gold context-action label",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "metadata": {
                    "sample_id": sample.sample_id,
                    "episode_id": sample.episode_id,
                    "split": sample.split,
                    "task_family": sample.task_family,
                    "memory_id": sample.memory_item.id,
                    "gold_evidence_ids": sample.gold_evidence_ids,
                    "budget": sample.budget,
                },
            }
        )

    output_path = Path(args.output_dir) / f"context_actions_{args.split}.jsonl"
    write_jsonl(output_path, out)
    print(f"Wrote {len(out)} slime samples: {output_path}")


if __name__ == "__main__":
    main()
