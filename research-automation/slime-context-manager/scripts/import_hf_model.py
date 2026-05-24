"""Import or verify a local Hugging Face model for context-policy deployment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--local-dir", default=str(ROOT / "models" / "qwen2_5_0_5b_instruct"))
    parser.add_argument("--source-local-path", default=None, help="Existing local HF checkpoint directory to register or copy.")
    parser.add_argument("--copy", action="store_true", help="Copy --source-local-path into --local-dir instead of only registering it.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.source_local_path:
        source = Path(args.source_local_path).expanduser().resolve()
        target = Path(args.local_dir).expanduser().resolve()
        if args.dry_run:
            action = "copy" if args.copy else "register"
            print(f"DRY RUN: would {action} local checkpoint {source} with project path {target}")
            return
        _validate_local_checkpoint(source)
        if args.copy:
            if target.exists():
                raise SystemExit(f"Target local-dir already exists, refusing to overwrite: {target}")
            shutil.copytree(source, target)
            registered_path = target
            mode = "copied"
        else:
            target.mkdir(parents=True, exist_ok=True)
            registered_path = source
            mode = "registered"
        manifest = {
            "mode": mode,
            "source_local_path": str(source),
            "registered_model_path": str(registered_path),
            "model_hint": args.model,
        }
        manifest_path = target / "model_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Local checkpoint {mode}: {registered_path}")
        print(f"Manifest: {manifest_path}")
        return

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


def _validate_local_checkpoint(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Local model path does not exist: {path}")
    if not path.is_dir():
        raise SystemExit(f"Local model path must be a directory: {path}")
    has_config = (path / "config.json").exists()
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    has_weights = bool(list(path.glob("*.safetensors")) or list(path.glob("*.bin")) or list(path.glob("*.pt")))
    missing = []
    if not has_config:
        missing.append("config.json")
    if not has_tokenizer:
        missing.append("tokenizer file")
    if not has_weights:
        missing.append("model weights")
    if missing:
        raise SystemExit(f"Local checkpoint appears incomplete at {path}; missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
