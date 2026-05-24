"""Run rule or HF model policy over a context-action split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from context_manager.hf_policy import HFModelPolicy
from context_manager.rule_policy import RulePolicy
from context_manager.schemas import ContextActionSample
from dataset_io.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "processed" / "context_actions"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--policy", choices=["rule", "model"], default="rule")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default=str(ROOT / "runs" / "rule_policy_predictions.jsonl"))
    args = parser.parse_args()

    records = read_jsonl(Path(args.dataset_dir) / f"{args.split}.jsonl")
    samples = [ContextActionSample.from_dict(record) for record in records]
    policy = RulePolicy() if args.policy == "rule" else HFModelPolicy(args.model_path)

    predictions = []
    for sample in samples:
        start = time.perf_counter()
        pred = policy.predict(sample)
        pred.latency_ms = (time.perf_counter() - start) * 1000
        predictions.append(pred.to_dict())

    write_jsonl(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions: {args.output}")


if __name__ == "__main__":
    main()
