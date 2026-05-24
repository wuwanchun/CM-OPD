"""Run fixed answer model or mock answers from selection records."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.answer_rollout import run_answer_rollouts
from disclosure.schemas import ContextSpan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--reserved-prompt-tokens", type=int, default=128)
    parser.add_argument("--allocator", default="greedy_margin_per_cost")
    parser.add_argument("--sglang-endpoint", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    spans = [ContextSpan.from_dict(record) for record in read_jsonl(input_dir / f"spans_{args.split}.jsonl")]
    selections = read_jsonl(input_dir / f"selections_{args.split}.jsonl")
    answers = run_answer_rollouts(
        spans,
        selections,
        token_budget=args.token_budget,
        reserved_prompt_tokens=args.reserved_prompt_tokens,
        allocator=args.allocator,
        sglang_endpoint=args.sglang_endpoint,
    )
    output = Path(args.output_dir) / f"answers_{args.split}.jsonl"
    write_jsonl(output, [answer.to_dict() for answer in answers])
    print(f"Wrote {len(answers)} answer rollouts: {output}")


if __name__ == "__main__":
    main()
