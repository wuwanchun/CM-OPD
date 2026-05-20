"""Export disclosure teacher targets to slime-compatible JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from data.io_utils import read_jsonl, write_jsonl
from disclosure.schemas import ContextSpan, DisclosureTarget
from disclosure.slime_sft_builder import build_slime_sft_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "slime"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--token-budget", type=int, default=4096)
    args = parser.parse_args()

    spans = [ContextSpan.from_dict(record) for record in read_jsonl(Path(args.input_dir) / f"spans_{args.split}.jsonl")]
    span_by_id = {span.span_id: span for span in spans}
    targets = [DisclosureTarget.from_dict(record) for record in read_jsonl(Path(args.input_dir) / f"teacher_targets_{args.split}.jsonl")]
    out = []
    for target in targets:
        if not target.accepted_for_training:
            continue
        span = span_by_id[target.span_id]
        out.append(build_slime_sft_record(span, target, token_budget=args.token_budget).to_dict())

    output = Path(args.output_dir) / f"disclosure_{args.split}.jsonl"
    write_jsonl(output, out)
    print(f"Wrote {len(out)} disclosure slime samples: {output}")


if __name__ == "__main__":
    main()
