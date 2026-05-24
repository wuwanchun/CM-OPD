"""Launch Slime training with rollout-built selection OPD CE-fallback samples."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=os.environ.get("SLIME_PYTHON", "/root/shared-nvme/workspace/envs/slime/bin/python"))
    parser.add_argument("--hf-checkpoint", default=str(ROOT / "models" / "qwen3-0.6B"))
    parser.add_argument("--spans-jsonl", default=str(ROOT / "data" / "disclosure" / "spans_train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "selection_opd_ce_fallback"))
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--num-epoch", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["DISCLOSURE_TOKEN_BUDGET"] = str(args.token_budget)
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'slime'}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    command = [
        args.python,
        str(ROOT / "slime" / "train_async.py"),
        "--hf-checkpoint",
        args.hf_checkpoint,
        "--prompt-data",
        args.spans_jsonl,
        "--input-key",
        "question",
        "--metadata-key",
        "metadata",
        "--rollout-function-path",
        "disclosure.slime_rollout.generate_selection_opd_rollout",
        "--rollout-shuffle",
        "--num-epoch",
        str(args.num_epoch),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--global-batch-size",
        str(args.global_batch_size),
        "--loss-type",
        "sft_loss",
        "--calculate-per-token-loss",
        "--disable-compute-advantages-and-returns",
        "--debug-train-only",
        "--save",
        args.output_dir,
    ]
    print("Selection OPD CE-fallback Slime command:")
    print(" ".join(shlex.quote(part) for part in command))
    if args.execute:
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
