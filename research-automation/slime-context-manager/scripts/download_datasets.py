"""Download or generate raw datasets for context-manager experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["toy", "hotpotqa", "alfworld", "scienceworld"], default="toy")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "toy":
        path = output_dir / "toy.jsonl"
        records = [
            {"id": "toy_001", "question": "Which university did Person B attend?", "context": "Person B attended University C."},
            {"id": "toy_002", "question": "Fix auth failure", "context": "test_auth_expired_token expected 401 but got 200."},
        ]
        _write_jsonl(path, records)
        print(f"Wrote toy raw dataset: {path}")
        return

    if args.dataset == "hotpotqa":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("Install requirements.txt to download HotpotQA: datasets is missing") from exc

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
        for split in ("train", "validation"):
            path = output_dir / f"hotpotqa_{split}.jsonl"
            rows = []
            for index, item in enumerate(ds[split]):
                if index >= args.max_samples:
                    break
                rows.append(dict(item))
            _write_jsonl(path, rows)
            print(f"Wrote {len(rows)} HotpotQA rows: {path}")
        return

    path = output_dir / f"{args.dataset}_manifest.json"
    path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "status": "skeleton",
                "message": "Install the official environment and run the adapter wrapper in a training machine.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote skeleton manifest: {path}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
