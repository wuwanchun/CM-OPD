"""Evaluate context-action predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl
from evaluation.metrics import evaluate_action_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "processed" / "context_actions"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=str(ROOT / "runs" / "metrics.json"))
    args = parser.parse_args()

    samples = read_jsonl(Path(args.dataset_dir) / f"{args.split}.jsonl")
    predictions = read_jsonl(args.predictions)
    metrics = evaluate_action_predictions(samples, predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics: {output}")


if __name__ == "__main__":
    main()
