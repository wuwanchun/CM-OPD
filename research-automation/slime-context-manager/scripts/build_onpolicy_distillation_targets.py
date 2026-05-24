"""Build OPD-style CE fallback targets from review SFT targets."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.opd_distillation_builder import build_onpolicy_distillation_targets
from disclosure.schemas import ContextSpan, ReviewSFTTarget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--iter", default="00")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    spans = [ContextSpan.from_dict(record) for record in read_jsonl(input_dir / f"spans_{args.split}.jsonl")]
    review_targets = [ReviewSFTTarget.from_dict(record) for record in read_jsonl(input_dir / f"review_sft_targets_{args.split}.jsonl")]
    targets = build_onpolicy_distillation_targets(spans, review_targets, distill_mode="ce_fallback")
    output = Path(args.output_dir) / f"onpolicy_targets_iter{args.iter}_{args.split}.jsonl"
    write_jsonl(output, [target.to_dict() for target in targets])
    print(f"Wrote {len(targets)} OPD CE fallback targets: {output}")


if __name__ == "__main__":
    main()
