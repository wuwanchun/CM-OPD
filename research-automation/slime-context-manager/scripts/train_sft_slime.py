"""Build or execute a slime SFT command for external context-policy training."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-jsonl", default=str(ROOT / "data" / "slime" / "context_actions_train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "context_policy_sft"))
    parser.add_argument(
        "--slime-command",
        default=None,
        help="Optional cluster-specific slime launcher command. Use {hf_checkpoint}, {train_jsonl}, and {output_dir} placeholders.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.slime_command:
        rendered = args.slime_command.format(
            hf_checkpoint=args.hf_checkpoint,
            train_jsonl=args.train_jsonl,
            output_dir=args.output_dir,
        )
        command = shlex.split(rendered, posix=(sys.platform != "win32"))
    else:
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
        if not args.slime_command and importlib.util.find_spec("slime.__main__") is None:
            raise SystemExit(
                "The installed slime package has no `python -m slime` entrypoint. "
                "Pass --slime-command with your cluster launcher, or use "
                "`run_train_local.py --backend hf-smoke` for a single-GPU training smoke."
            )
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
