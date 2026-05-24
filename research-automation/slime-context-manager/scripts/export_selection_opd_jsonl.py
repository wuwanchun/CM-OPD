"""Export OPD CE-fallback JSONL for inspection; training path uses Slime rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.schemas import ContextSpan, ReviewSFTTarget
from disclosure.slime_export import export_opd_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "slime"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--iter", default="00")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    spans = [ContextSpan.from_dict(record) for record in read_jsonl(input_dir / f"spans_{args.split}.jsonl")]
    targets = [ReviewSFTTarget.from_dict(record) for record in read_jsonl(input_dir / f"review_sft_targets_{args.split}.jsonl")]
    records = export_opd_records(spans, targets)
    output = Path(args.output_dir) / f"selection_opd_iter{args.iter}_{args.split}.jsonl"
    write_jsonl(output, records)
    print(f"Wrote {len(records)} OPD CE-fallback inspection records: {output}")


if __name__ == "__main__":
    main()
