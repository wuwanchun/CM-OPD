"""Run selection rollouts over disclosure spans."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.schemas import ContextSpan
from disclosure.selection_rollout import run_selection_rollouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--policy", default="rule_expand_omission")
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--reserved-prompt-tokens", type=int, default=128)
    parser.add_argument("--allocator", default="greedy_margin_per_cost")
    parser.add_argument("--sglang-endpoint", default="")
    args = parser.parse_args()

    spans = [ContextSpan.from_dict(record) for record in read_jsonl(Path(args.input_dir) / f"spans_{args.split}.jsonl")]
    rollouts = run_selection_rollouts(
        spans,
        policy_name=args.policy,
        token_budget=args.token_budget,
        reserved_prompt_tokens=args.reserved_prompt_tokens,
        allocator=args.allocator,
        sglang_endpoint=args.sglang_endpoint,
    )
    output = Path(args.output_dir) / f"selections_{args.split}.jsonl"
    write_jsonl(output, [rollout.to_dict() for rollout in rollouts])
    print(f"Wrote {len(rollouts)} selection rollouts: {output}")


if __name__ == "__main__":
    main()
