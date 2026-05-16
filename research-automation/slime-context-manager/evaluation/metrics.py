"""Metrics for context-action predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from context_manager.schemas import ACTIONS, ContextActionSample


PRESERVE_ACTIONS = {"KEEP", "COMPRESS", "ARCHIVE", "RETRIEVE", "UPDATE", "PIN"}


def evaluate_action_predictions(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    sample_by_id = {record["sample_id"]: ContextActionSample.from_dict(record) for record in samples}
    pred_by_id = {record["sample_id"]: str(record.get("pred_action", "INVALID")).upper() for record in predictions}
    total = len(sample_by_id)
    matched = 0
    invalid = 0
    forbidden = 0
    must_preserve_violations = 0
    evidence_preserved = 0
    evidence_total = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for sample_id, sample in sample_by_id.items():
        pred = pred_by_id.get(sample_id, "MISSING")
        gold = sample.gold_action
        if pred == gold:
            matched += 1
        if pred not in ACTIONS:
            invalid += 1
        if pred in sample.labels.get("forbidden_actions", []):
            forbidden += 1
        if sample.labels.get("must_preserve") and pred == "DROP":
            must_preserve_violations += 1
        if sample.gold_evidence_ids:
            evidence_total += 1
            if pred in PRESERVE_ACTIONS:
                evidence_preserved += 1
        confusion[gold][pred] += 1

    macro_f1 = _macro_f1(confusion)
    metrics = {
        "num_samples": total,
        "action_accuracy": matched / total if total else 0.0,
        "macro_f1": macro_f1,
        "invalid_output_rate": invalid / total if total else 0.0,
        "forbidden_action_rate": forbidden / total if total else 0.0,
        "must_preserve_violation_rate": must_preserve_violations / total if total else 0.0,
        "evidence_recall": evidence_preserved / evidence_total if evidence_total else 0.0,
        "confusion": {gold: dict(counter) for gold, counter in confusion.items()},
    }
    return metrics


def _macro_f1(confusion: dict[str, Counter[str]]) -> float:
    labels = set(ACTIONS)
    scores: list[float] = []
    for label in labels:
        tp = confusion.get(label, Counter()).get(label, 0)
        fp = sum(counter.get(label, 0) for gold, counter in confusion.items() if gold != label)
        fn = sum(count for pred, count in confusion.get(label, Counter()).items() if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0
