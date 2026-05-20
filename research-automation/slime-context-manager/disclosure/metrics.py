"""Metrics for disclosure policy outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .schemas import ACTIONS, DisclosureTarget


def evaluate_disclosure_predictions(targets: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    target_by_id = {record["decision_id"]: DisclosureTarget.from_dict(record) for record in targets}
    pred_by_id = {
        str(record.get("decision_id", record.get("sample_id", ""))): str(record.get("pred_action", record.get("action", "INVALID"))).upper()
        for record in predictions
    }
    total = len(target_by_id)
    correct = 0
    invalid = 0
    expand_tp = expand_fp = expand_fn = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for decision_id, target in target_by_id.items():
        pred = pred_by_id.get(decision_id, "MISSING")
        gold = target.target_action
        if pred == gold:
            correct += 1
        if pred not in ACTIONS:
            invalid += 1
        if pred == "EXPAND" and gold == "EXPAND":
            expand_tp += 1
        elif pred == "EXPAND" and gold != "EXPAND":
            expand_fp += 1
        elif pred != "EXPAND" and gold == "EXPAND":
            expand_fn += 1
        confusion[gold][pred] += 1

    precision = expand_tp / (expand_tp + expand_fp) if expand_tp + expand_fp else 0.0
    recall = expand_tp / (expand_tp + expand_fn) if expand_tp + expand_fn else 0.0
    return {
        "num_targets": total,
        "action_accuracy": correct / total if total else 0.0,
        "invalid_output_rate": invalid / total if total else 0.0,
        "expansion_precision": precision,
        "expansion_recall": recall,
        "missed_expansion_rate": expand_fn / total if total else 0.0,
        "unnecessary_expansion_rate": expand_fp / total if total else 0.0,
        "confusion": {gold: dict(counter) for gold, counter in confusion.items()},
    }


def summarize_budget_allocations(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if not total:
        return {"num_allocations": 0, "budget_violation_rate": 0.0, "avg_token_budget_usage": 0.0}
    violations = sum(1 for record in records if record.get("budget_violation"))
    usages = [
        float(record.get("budget_used", 0)) / float(record.get("token_budget", 1) or 1)
        for record in records
    ]
    return {
        "num_allocations": total,
        "budget_violation_rate": violations / total,
        "avg_token_budget_usage": sum(usages) / total,
    }
