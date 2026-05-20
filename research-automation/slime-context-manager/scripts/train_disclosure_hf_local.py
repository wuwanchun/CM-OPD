"""Local HF SFT smoke trainer for disclosure policy records."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", default=str(ROOT / "data" / "slime" / "disclosure_train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "disclosure_policy_hf_sft_smoke"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="float32")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_hf_sft_smoke.py"),
        "--model-path",
        args.model_path,
        "--train-jsonl",
        args.train_jsonl,
        "--output-dir",
        args.output_dir,
        "--max-steps",
        str(args.max_steps),
        "--max-length",
        str(args.max_length),
        "--dtype",
        args.dtype,
    ]
    if args.save_model:
        command.append("--save-model")
    print("+", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
