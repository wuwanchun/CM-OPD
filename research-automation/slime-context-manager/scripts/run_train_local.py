"""One-command local training launcher for context-policy SFT.

This launcher uses a local Hugging Face model path by default. It prepares
slime JSONL and then either prints a slime training command or runs a tiny
single-GPU HF SFT smoke train.
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
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "processed" / "context_actions"))
    parser.add_argument("--slime-dir", default=str(ROOT / "data" / "slime"))
    parser.add_argument("--local-model-path", default=str(ROOT / "models" / "qwen2_5_0_5b_instruct"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "context_policy_sft"))
    parser.add_argument("--backend", choices=["slime", "hf-smoke"], default="slime")
    parser.add_argument("--max-steps", type=int, default=5, help="Only used by --backend hf-smoke.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="float32")
    parser.add_argument("--save-model", action="store_true", help="Only used by --backend hf-smoke.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.local_model_path)
    if not model_path.exists():
        print(
            "WARNING: local model path does not exist yet. "
            "Run `python scripts/import_hf_model.py --source-local-path C:\\path\\to\\checkpoint --local-dir models\\qwen2_5_0_5b_instruct` first, "
            "or pass --local-model-path to an existing checkpoint."
        )

    export_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_slime_jsonl.py"),
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.slime_dir,
        "--split",
        args.split,
    ]
    print("+", " ".join(export_cmd))
    subprocess.run(export_cmd, check=True)

    train_jsonl = Path(args.slime_dir) / f"context_actions_{args.split}.jsonl"
    if args.backend == "hf-smoke":
        train_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_hf_sft_smoke.py"),
            "--model-path",
            str(model_path),
            "--train-jsonl",
            str(train_jsonl),
            "--output-dir",
            args.output_dir,
            "--max-steps",
            str(args.max_steps),
            "--dtype",
            args.dtype,
        ]
        if args.save_model:
            train_cmd.append("--save-model")
    else:
        train_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_sft_slime.py"),
            "--hf-checkpoint",
            str(model_path),
            "--train-jsonl",
            str(train_jsonl),
            "--output-dir",
            args.output_dir,
        ]
        if args.execute:
            train_cmd.append("--execute")

    print("+", " ".join(train_cmd))
    if args.backend == "hf-smoke" or args.execute:
        subprocess.run(train_cmd, check=True)

    if args.backend == "slime" and not args.execute:
        print("Training command generated only. Re-run with --execute on a slime training host to start training.")
    if args.backend == "hf-smoke":
        print("HF smoke training finished. This validates local checkpoint + exported JSONL + single-GPU SFT.")


if __name__ == "__main__":
    main()
