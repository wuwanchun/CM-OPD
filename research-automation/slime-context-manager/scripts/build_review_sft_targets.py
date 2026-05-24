"""Build grounded review SFT targets from selection and answer rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.review_sft_builder import build_review_sft_targets
from disclosure.schemas import ContextSpan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    spans = [ContextSpan.from_dict(record) for record in read_jsonl(input_dir / f"spans_{args.split}.jsonl")]
    selections = read_jsonl(input_dir / f"selections_{args.split}.jsonl")
    answers = read_jsonl(input_dir / f"answers_{args.split}.jsonl")
    targets = build_review_sft_targets(spans, selections, answers)
    output = Path(args.output_dir) / f"review_sft_targets_{args.split}.jsonl"
    write_jsonl(output, [target.to_dict() for target in targets])
    print(f"Wrote {len(targets)} review SFT targets: {output}")


if __name__ == "__main__":
    main()
