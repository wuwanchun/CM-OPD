#!/usr/bin/env python3
"""Build disclosure training dataset from raw documents.

This script:
1. Loads raw documents with questions
2. Generates cue-preserving summaries for each span
3. Creates disclosure training samples
4. Exports to JSONL format for Slime training

Usage:
    python scripts/build_disclosure_dataset.py --input data/raw_docs.jsonl --output data/slime/disclosure_train.jsonl
"""

import argparse
import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_summary(text: str, max_length: int = 100) -> str:
    """Generate cue-preserving summary for a span.
    
    This is a simple heuristic. In production, use a summarization model.
    """
    sentences = text.split(". ")
    if len(sentences) <= 1:
        return text[:max_length]
    
    # Take first sentence as summary
    summary = sentences[0]
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    return summary


def estimate_relevance(question: str, text: str) -> float:
    """Estimate relevance of text to question.
    
    This is a simple heuristic. In production, use a relevance model.
    """
    question_words = set(question.lower().split())
    text_words = set(text.lower().split())
    
    overlap = len(question_words & text_words)
    return min(1.0, overlap / max(len(question_words), 1))


def build_disclosure_sample(
    sample_id: str,
    question: str,
    documents: list[dict],
    gold_answer: str,
    budget_tokens: int = 4096,
) -> dict:
    """Build a disclosure training sample."""
    spans = []
    target_selection = {}
    
    for idx, doc in enumerate(documents):
        span_id = f"doc_{idx + 1}"
        raw_text = doc.get("text", "")
        summary = doc.get("summary") or generate_summary(raw_text)
        relevance = doc.get("relevance") or estimate_relevance(question, raw_text)
        
        spans.append({
            "span_id": span_id,
            "raw_text": raw_text,
            "summary": summary,
            "metadata": {"relevance": relevance},
        })
        
        # Determine target selection based on relevance
        if relevance > 0.8:
            target_selection[span_id] = "EXPAND"
        elif relevance > 0.4:
            target_selection[span_id] = "SUMMARY"
        else:
            target_selection[span_id] = "HIDDEN"
    
    return {
        "sample_id": sample_id,
        "question": question,
        "spans": spans,
        "gold_answer": gold_answer,
        "metadata": {
            "max_budget_tokens": budget_tokens,
            "target_selection": target_selection,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build disclosure training dataset")
    parser.add_argument("--input", required=True, help="Input JSONL file with raw documents")
    parser.add_argument("--output", required=True, help="Output JSONL file for training")
    parser.add_argument("--budget-tokens", type=int, default=4096, help="Token budget per sample")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            raw = json.loads(line)
            
            sample_id = raw.get("id") or str(uuid.uuid4())
            question = raw["question"]
            documents = raw.get("documents", [])
            gold_answer = raw.get("answer", "")
            
            sample = build_disclosure_sample(
                sample_id=sample_id,
                question=question,
                documents=documents,
                gold_answer=gold_answer,
                budget_tokens=args.budget_tokens,
            )
            
            f_out.write(json.dumps(sample) + "\n")
            logger.info(f"Processed sample {sample_id}")
    
    logger.info(f"Built {output_path} from {input_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
