"""Build deterministic teacher targets for disclosure SFT."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.schemas import ContextSpan, DisclosureDecision
from disclosure.teacher_review import heuristic_teacher_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    spans = [ContextSpan.from_dict(record) for record in read_jsonl(Path(args.input_dir) / f"spans_{args.split}.jsonl")]
    span_by_id = {span.span_id: span for span in spans}
    rollouts = read_jsonl(Path(args.input_dir) / f"rollouts_{args.split}.jsonl")
    accepted = []
    rejected = []
    for rollout in rollouts:
        for action_record in rollout.get("actions", []):
            decision = DisclosureDecision.from_dict(action_record)
            span = span_by_id.get(decision.span_id)
            if span is None:
                continue
            target = heuristic_teacher_target(span, decision, rollout_id=str(rollout.get("rollout_id", "")))
            if target.accepted_for_training:
                accepted.append(target.to_dict())
            else:
                rejected.append(target.to_dict())

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / f"teacher_targets_{args.split}.jsonl", accepted)
    write_jsonl(output_dir / f"teacher_targets_rejected_{args.split}.jsonl", rejected)
    print(f"Wrote {len(accepted)} accepted and {len(rejected)} rejected disclosure targets")


if __name__ == "__main__":
    main()
