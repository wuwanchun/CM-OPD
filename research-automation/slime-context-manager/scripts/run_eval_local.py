"""One-command local evaluation launcher.

This is the practical entry point for evaluation. It uses local paths by
default and wraps:

1. context policy inference
2. metrics evaluation
3. visualization report generation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "processed" / "context_actions"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--policy", choices=["rule", "model"], default="rule")
    parser.add_argument("--model-path", default=str(ROOT / "models" / "qwen2_5_0_5b_instruct"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "eval_local"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = output_dir / f"{args.policy}_{args.split}_predictions.jsonl"
    metrics = output_dir / "metrics.json"
    analysis_dir = output_dir / "analysis"

    commands = [
        [
            sys.executable,
            str(ROOT / "scripts" / "run_context_policy.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--split",
            args.split,
            "--policy",
            args.policy,
            "--model-path",
            args.model_path,
            "--output",
            str(predictions),
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_predictions.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--split",
            args.split,
            "--predictions",
            str(predictions),
            "--output",
            str(metrics),
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "visualize_results.py"),
            "--metrics",
            str(metrics),
            "--output-dir",
            str(analysis_dir),
        ],
    ]
    for command in commands:
        print("+", " ".join(command))
        subprocess.run(command, check=True)

    print(f"Evaluation complete: {output_dir}")
    print(f"Metrics: {metrics}")
    print(f"Report: {analysis_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
