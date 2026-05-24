"""Build or execute a slime SFT command for disclosure policy training."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", default=str(ROOT / "models" / "qwen3-0.6B"))
    parser.add_argument("--train-jsonl", default=str(ROOT / "data" / "slime" / "disclosure_train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "disclosure_policy_slime_sft"))
    parser.add_argument("--slime-command", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.slime_command:
        rendered = args.slime_command.format(
            hf_checkpoint=args.hf_checkpoint,
            train_jsonl=args.train_jsonl,
            output_dir=args.output_dir,
            rollout_function_path="disclosure.slime_rollout.generate_rollout_disclosure",
        )
        command = shlex.split(rendered, posix=(sys.platform != "win32"))
    else:
        command = [
            "python",
            str(ROOT / "slime" / "train_async.py"),
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
            "--rollout-function-path",
            "disclosure.slime_rollout.generate_rollout_disclosure",
            "--loss-type",
            "sft_loss",
            "--save",
            args.output_dir,
        ]
    print("Disclosure slime command:")
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
