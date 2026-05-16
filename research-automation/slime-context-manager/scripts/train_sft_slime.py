"""Build or execute a slime SFT command for external context-policy training."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-jsonl", default=str(ROOT / "data" / "slime" / "context_actions_train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "context_policy_sft"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    command = [
        "python",
        "-m",
        "slime",
        "--hf-checkpoint",
        args.hf_checkpoint,
        "--prompt-data",
        args.train_jsonl,
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--metadata-key",
        "metadata",
        "--save",
        args.output_dir,
    ]
    print("Slime SFT command:")
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
