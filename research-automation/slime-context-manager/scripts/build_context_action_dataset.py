"""Build context-action datasets from raw sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from data.adapters import get_adapter
from data.io_utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=["toy", "hotpotqa", "alfworld", "scienceworld"], default="toy")
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed" / "context_actions"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-hf-dataset", action="store_true")
    args = parser.parse_args()

    adapter = get_adapter(args.adapter)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_to_raw = _resolve_raw_paths(args.adapter, Path(args.raw_dir))
    split_records: dict[str, list[dict]] = {}
    for split in ("train", "validation", "test", "unseen"):
        raw_path = split_to_raw.get(split)
        records = adapter.build_samples(raw_path=raw_path, split=split, limit=args.limit)
        split_records[split] = records
        write_jsonl(output_dir / f"{split}.jsonl", records)
        print(f"Wrote {len(records)} {split} samples to {output_dir / f'{split}.jsonl'}")

    if args.save_hf_dataset:
        try:
            from datasets import Dataset, DatasetDict
        except ImportError as exc:
            raise SystemExit("datasets is required for --save-hf-dataset") from exc
        dataset_dict = DatasetDict({split: Dataset.from_list(records) for split, records in split_records.items()})
        dataset_dict.save_to_disk(str(output_dir / "hf_dataset"))
        print(f"Saved HF DatasetDict to {output_dir / 'hf_dataset'}")


def _resolve_raw_paths(adapter: str, raw_dir: Path) -> dict[str, Path | None]:
    if adapter == "hotpotqa":
        return {
            "train": raw_dir / "hotpotqa_train.jsonl",
            "validation": raw_dir / "hotpotqa_validation.jsonl",
            "test": raw_dir / "hotpotqa_validation.jsonl",
            "unseen": raw_dir / "hotpotqa_validation.jsonl",
        }
    return {split: None for split in ("train", "validation", "test", "unseen")}


if __name__ == "__main__":
    main()
