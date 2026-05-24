"""Disclosure rollout policies."""

from __future__ import annotations

import random

from .schemas import ContextSpan, DisclosureDecision


class DisclosurePolicy:
    name = "base"

    def decide(self, span: ContextSpan) -> DisclosureDecision:
        raise NotImplementedError


class SummaryOnlyPolicy(DisclosurePolicy):
    name = "summary_only"

    def decide(self, span: ContextSpan) -> DisclosureDecision:
        return DisclosureDecision(span.span_id, "SUMMARY", {"SUMMARY": 1.0, "EXPAND": 0.0, "HIDDEN": 0.0})


class FullRawTopKPolicy(DisclosurePolicy):
    name = "full_raw_topk"

    def decide(self, span: ContextSpan) -> DisclosureDecision:
        return DisclosureDecision(span.span_id, "EXPAND", {"EXPAND": 1.0, "SUMMARY": 0.0, "HIDDEN": 0.0})


class RuleExpandOmissionPolicy(DisclosurePolicy):
    name = "rule_expand_omission"

    def decide(self, span: ContextSpan) -> DisclosureDecision:
        text = f"{span.summary_text} {' '.join(span.summary_cues)}".lower()
        question_terms = {term.strip(" ?.,:;").lower() for term in span.question.split() if len(term) >= 4}
        has_question_cue = any(term and term in text for term in question_terms)
        if "omitted" in text or "exact" in text or any(ch.isdigit() for ch in span.raw_text) and not any(ch.isdigit() for ch in span.summary_text):
            action = "EXPAND"
        elif has_question_cue:
            action = "SUMMARY"
        else:
            action = "HIDDEN"
        return DisclosureDecision(span.span_id, action, {action: 1.0})


class RandomBudgetPerturbationPolicy(DisclosurePolicy):
    name = "random_budget_perturbation"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def decide(self, span: ContextSpan) -> DisclosureDecision:
        action = self._rng.choice(["EXPAND", "SUMMARY", "HIDDEN"])
        return DisclosureDecision(span.span_id, action, {action: 1.0})


def policy_from_name(name: str) -> DisclosurePolicy:
    normalized = name.lower()
    if normalized == "summary_only":
        return SummaryOnlyPolicy()
    if normalized == "full_raw_topk":
        return FullRawTopKPolicy()
    if normalized == "rule_expand_omission":
        return RuleExpandOmissionPolicy()
    if normalized == "random_budget_perturbation":
        return RandomBudgetPerturbationPolicy()
    raise ValueError(f"unknown disclosure policy: {name}")
