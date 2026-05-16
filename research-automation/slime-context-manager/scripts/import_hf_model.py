"""Import or verify a Hugging Face model for context-policy deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--local-dir", default=str(ROOT / "models" / "qwen2_5_0_5b_instruct"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"DRY RUN: would download {args.model} to {args.local_dir}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface-hub is required. Install requirements.txt first.") from exc

    target = Path(args.local_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.model, local_dir=str(target), local_dir_use_symlinks=False)
    print(f"Downloaded model snapshot to {path}")


if __name__ == "__main__":
    main()
