"""Run a local no-network smoke pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    run_dir = ROOT / "runs" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, str(ROOT / "scripts" / "download_datasets.py"), "--dataset", "toy"],
        [sys.executable, str(ROOT / "scripts" / "build_context_action_dataset.py"), "--adapter", "toy"],
        [sys.executable, str(ROOT / "scripts" / "export_slime_jsonl.py"), "--split", "train"],
        [
            sys.executable,
            str(ROOT / "scripts" / "run_context_policy.py"),
            "--policy",
            "rule",
            "--split",
            "test",
            "--output",
            str(run_dir / "rule_predictions.jsonl"),
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_predictions.py"),
            "--split",
            "test",
            "--predictions",
            str(run_dir / "rule_predictions.jsonl"),
            "--output",
            str(run_dir / "metrics.json"),
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "visualize_results.py"),
            "--metrics",
            str(run_dir / "metrics.json"),
            "--output-dir",
            str(run_dir / "analysis"),
        ],
    ]
    for command in commands:
        print("+", " ".join(command))
        subprocess.run(command, check=True)
    print(f"Smoke pipeline complete: {run_dir}")


if __name__ == "__main__":
    main()
