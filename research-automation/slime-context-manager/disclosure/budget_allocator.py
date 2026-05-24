"""Token budget allocation for EXPAND/SUMMARY/HIDDEN decisions."""

from __future__ import annotations

from itertools import combinations

from .schemas import ACTIONS, BudgetAllocationRecord, ContextSpan, DisclosureDecision, normalize_action


def allocate_budget(
    spans: list[ContextSpan],
    decisions: list[DisclosureDecision],
    *,
    rollout_id: str,
    token_budget: int,
    reserved_prompt_tokens: int = 0,
    allocator: str = "greedy_margin_per_cost",
) -> BudgetAllocationRecord:
    if allocator == "greedy_margin_per_cost":
        return _allocate_greedy(spans, decisions, rollout_id, token_budget, reserved_prompt_tokens)
    if allocator == "knapsack_analysis":
        return _allocate_knapsack(spans, decisions, rollout_id, token_budget, reserved_prompt_tokens)
    raise ValueError(f"unknown disclosure allocator: {allocator}")


def _allocate_greedy(
    spans: list[ContextSpan],
    decisions: list[DisclosureDecision],
    rollout_id: str,
    token_budget: int,
    reserved_prompt_tokens: int,
) -> BudgetAllocationRecord:
    span_by_id = {span.span_id: span for span in spans}
    decision_by_id = {decision.span_id: decision for decision in decisions}
    available = max(0, token_budget - reserved_prompt_tokens)
    final: dict[str, DisclosureDecision] = {}
    budget_used = reserved_prompt_tokens
    trace = []

    keep_candidates = []
    expand_candidates = []
    for span in spans:
        decision = decision_by_id.get(span.span_id, DisclosureDecision(span.span_id, "SUMMARY"))
        action = normalize_action(decision.action)
        logits = _logits(decision)
        summary_tokens = span.summary_tokens
        raw_tokens = span.raw_tokens
        upgrade_cost = max(0, raw_tokens - summary_tokens)
        margin = logits["EXPAND"] - logits["SUMMARY"]
        trace.append(
            {
                "span_id": span.span_id,
                "policy_action": action,
                "expand_logit": logits["EXPAND"],
                "summary_logit": logits["SUMMARY"],
                "hidden_logit": logits["HIDDEN"],
                "summary_tokens": summary_tokens,
                "raw_tokens": raw_tokens,
                "upgrade_cost": upgrade_cost,
                "allocation_score": margin / max(1, upgrade_cost),
            }
        )
        if action in {"SUMMARY", "EXPAND"}:
            keep_candidates.append((span, decision))
        if action == "EXPAND":
            expand_candidates.append((margin / max(1, upgrade_cost), span, decision))

    for span, decision in keep_candidates:
        if budget_used + span.summary_tokens <= token_budget:
            final[span.span_id] = DisclosureDecision(span.span_id, "SUMMARY", decision.logits, decision.score)
            budget_used += span.summary_tokens
        else:
            final[span.span_id] = DisclosureDecision(span.span_id, "HIDDEN", decision.logits, decision.score)

    for _, span, decision in sorted(expand_candidates, key=lambda item: (-item[0], item[1].span_id)):
        if final.get(span.span_id, DisclosureDecision(span.span_id, "HIDDEN")).action != "SUMMARY":
            continue
        cost = span.upgrade_cost
        if budget_used + cost <= token_budget:
            final[span.span_id] = DisclosureDecision(span.span_id, "EXPAND", decision.logits, decision.score)
            budget_used += cost

    for span in spans:
        final.setdefault(span.span_id, DisclosureDecision(span.span_id, "HIDDEN", decision_by_id.get(span.span_id, DisclosureDecision(span.span_id, "HIDDEN")).logits))

    return BudgetAllocationRecord(
        rollout_id=rollout_id,
        token_budget=token_budget,
        reserved_prompt_tokens=reserved_prompt_tokens,
        allocator="greedy_margin_per_cost",
        candidate_decisions=trace,
        final_actions=[final[span.span_id] for span in spans],
        budget_used=budget_used,
        budget_violation=budget_used > token_budget or available < 0,
    )


def _allocate_knapsack(
    spans: list[ContextSpan],
    decisions: list[DisclosureDecision],
    rollout_id: str,
    token_budget: int,
    reserved_prompt_tokens: int,
) -> BudgetAllocationRecord:
    base = _allocate_greedy(spans, decisions, rollout_id, token_budget, reserved_prompt_tokens)
    span_by_id = {span.span_id: span for span in spans}
    decision_by_id = {decision.span_id: decision for decision in decisions}
    kept = [action for action in base.final_actions if action.action in {"SUMMARY", "EXPAND"}]
    keep_cost = sum(span_by_id[action.span_id].summary_tokens for action in kept)
    remaining = max(0, token_budget - reserved_prompt_tokens - keep_cost)
    expandable = [action for action in kept if decision_by_id.get(action.span_id, action).action == "EXPAND"]

    best_ids: set[str] = set()
    best_score = float("-inf")
    for size in range(len(expandable) + 1):
        for combo in combinations(expandable, size):
            cost = sum(span_by_id[action.span_id].upgrade_cost for action in combo)
            if cost > remaining:
                continue
            score = sum(_logits(decision_by_id.get(action.span_id, action))["EXPAND"] for action in combo)
            if score > best_score:
                best_score = score
                best_ids = {action.span_id for action in combo}

    final = [
        DisclosureDecision(action.span_id, "EXPAND" if action.span_id in best_ids else ("SUMMARY" if action.action == "EXPAND" else action.action), action.logits, action.score)
        for action in base.final_actions
    ]
    budget_used = reserved_prompt_tokens + sum(
        span_by_id[action.span_id].raw_tokens if action.action == "EXPAND" else span_by_id[action.span_id].summary_tokens
        for action in final
        if action.action != "HIDDEN"
    )
    base.allocator = "knapsack_analysis"
    base.final_actions = final
    base.budget_used = budget_used
    base.budget_violation = budget_used > token_budget
    return base


def _logits(decision: DisclosureDecision) -> dict[str, float]:
    logits = {action: float(decision.logits.get(action, 0.0)) for action in ACTIONS}
    if not decision.logits:
        logits[decision.action] = max(1.0, decision.score)
    return logits
