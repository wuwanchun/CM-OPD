"""Evaluate disclosure predictions and budget allocations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from data.io_utils import read_jsonl
from disclosure.metrics import evaluate_disclosure_predictions, summarize_budget_allocations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--budget-allocations", default="")
    parser.add_argument("--output", default=str(ROOT / "runs" / "disclosure_eval" / "metrics.json"))
    args = parser.parse_args()

    targets = read_jsonl(Path(args.dataset_dir) / f"teacher_targets_{args.split}.jsonl")
    predictions = read_jsonl(args.predictions)
    metrics = evaluate_disclosure_predictions(targets, predictions)
    allocation_path = Path(args.budget_allocations) if args.budget_allocations else Path(args.dataset_dir) / f"budget_allocations_{args.split}.jsonl"
    if allocation_path.exists():
        metrics.update(summarize_budget_allocations(read_jsonl(allocation_path)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics: {output}")


if __name__ == "__main__":
    main()
