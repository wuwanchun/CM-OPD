"""Build progressive disclosure span datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from data.io_utils import read_jsonl, write_jsonl
from disclosure.summary_builder import audit_summaries, span_from_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["toy", "hotpotqa"], default="toy")
    parser.add_argument("--raw-path", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-summary-words", type=int, default=32)
    args = parser.parse_args()

    if args.dataset == "toy":
        spans = _build_toy(args.split, args.max_rows, args.max_summary_words)
    else:
        raw_path = Path(args.raw_path) if args.raw_path else ROOT / "data" / "raw" / f"hotpotqa_{args.split}.jsonl"
        spans = _build_hotpotqa(raw_path, args.split, args.max_rows, args.max_summary_words)

    output_dir = Path(args.output_dir)
    records = [span.to_dict() for span in spans]
    write_jsonl(output_dir / f"spans_{args.split}.jsonl", records)
    audit = audit_summaries(spans).to_dict()
    (output_dir / f"summary_quality_{args.split}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} disclosure spans: {output_dir / f'spans_{args.split}.jsonl'}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def _build_toy(split: str, limit: int | None, max_summary_words: int) -> list[Any]:
    raw = [
        {
            "task_id": f"toy_{split}_001",
            "question": "What dosage was used after the protocol changed?",
            "texts": [
                ("doc1_sent1", "The protocol changed dosage from 5mg to 50mg after the second review."),
                ("doc2_sent1", "The clinic also updated unrelated appointment reminder templates."),
                ("doc3_sent1", "Dosage changed; exact values are documented in the trial notes."),
            ],
        },
        {
            "task_id": f"toy_{split}_002",
            "question": "Which university did Person B attend?",
            "texts": [
                ("doc4_sent1", "Person B attended University C before joining the research lab."),
                ("doc5_sent1", "Person A won several unrelated awards in 2019."),
            ],
        },
    ]
    spans = []
    for item in raw:
        for index, (source_id, text) in enumerate(item["texts"]):
            spans.append(
                span_from_raw(
                    task_id=item["task_id"],
                    span_id=f"{item['task_id']}_s{index:02d}",
                    source_id=source_id,
                    question=item["question"],
                    raw_text=text,
                    split=split,
                    metadata={"dataset": "toy"},
                    max_summary_words=max_summary_words,
                )
            )
    return spans[:limit] if limit else spans


def _build_hotpotqa(raw_path: Path, split: str, limit: int | None, max_summary_words: int) -> list[Any]:
    rows = read_jsonl(raw_path)
    spans = []
    for row_index, row in enumerate(rows):
        if limit and row_index >= limit:
            break
        question = str(row.get("question", ""))
        task_id = str(row.get("id", row.get("_id", f"hotpotqa_{split}_{row_index:06d}")))
        for context_index, (title, sentences) in enumerate(_iter_contexts(row)):
            for sent_index, sentence in enumerate(sentences):
                source_id = f"{task_id}_{context_index:02d}_{sent_index:02d}"
                spans.append(
                    span_from_raw(
                        task_id=task_id,
                        span_id=f"{source_id}",
                        source_id=source_id,
                        question=question,
                        raw_text=f"{title}: {sentence}",
                        split=split,
                        span_type="supporting_sentence" if _is_supporting(row, title, sent_index) else "candidate_sentence",
                        metadata={"dataset": "hotpotqa", "title": title, "sentence_index": sent_index},
                        max_summary_words=max_summary_words,
                    )
                )
    return spans


def _iter_contexts(item: dict[str, Any]) -> list[tuple[str, list[str]]]:
    context = item.get("context", {})
    if isinstance(context, dict):
        return [
            (str(title), [str(sentence) for sentence in sentences])
            for title, sentences in zip(context.get("title", []), context.get("sentences", []))
        ]
    if isinstance(context, list):
        rows = []
        for row in context:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                rows.append((str(row[0]), [str(sentence) for sentence in row[1]]))
        return rows
    text = str(item.get("context", ""))
    return [("context", [text])] if text else []


def _is_supporting(item: dict[str, Any], title: str, sent_index: int) -> bool:
    facts = item.get("supporting_facts", {})
    if isinstance(facts, dict):
        titles = facts.get("title", [])
        sent_ids = facts.get("sent_id", facts.get("sent_idx", []))
        return any(str(t) == title and int(s) == sent_index for t, s in zip(titles, sent_ids))
    if isinstance(facts, list):
        return any(row and str(row[0]) == title and len(row) > 1 and int(row[1]) == sent_index for row in facts)
    return False


if __name__ == "__main__":
    main()
