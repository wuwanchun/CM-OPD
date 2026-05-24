"""Budget-aware allocator for disclosure selection.

Implements greedy and knapsack-based budget allocation.
"""

import logging
from typing import Optional

from .schemas import Span, SpanState

logger = logging.getLogger(__name__)


class BudgetAllocator:
    """Allocates budget across span states.
    
    Supports two allocation strategies:
    1. Greedy: Select states in score order until budget exhausted
    2. Knapsack: Optimize total value under budget constraint
    """
    
    def __init__(self, token_budget: int = 4096):
        self.token_budget = token_budget
    
    def estimate_tokens(self, span: Span, state: SpanState, tokenizer=None) -> int:
        """Estimate tokens for a span in given state.
        
        Args:
            span: The span to estimate
            state: The disclosure state
            tokenizer: Optional tokenizer for accurate count
            
        Returns:
            Estimated token count
        """
        if state == SpanState.HIDDEN:
            return 0
        elif state == SpanState.SUMMARY:
            if tokenizer:
                return len(tokenizer.encode(span.summary, add_special_tokens=False))
            return len(span.summary) // 4
        else:  # EXPAND
            return span.get_token_count(tokenizer)
    
    def greedy_allocate(
        self,
        spans: list[Span],
        state_scores: dict[str, dict[str, float]],
        tokenizer=None,
    ) -> dict[str, SpanState]:
        """Greedy allocation based on score ranking.
        
        Allocates spans to states in order of their highest score,
        respecting budget constraints.
        
        Args:
            spans: List of spans to allocate
            state_scores: Per-span scores for each state {span_id: {state: score}}
            tokenizer: Optional tokenizer
            
        Returns:
            Mapping from span_id to selected state
        """
        result = {}
        remaining_budget = self.token_budget
        
        # Create priority queue: (score, span_id, state)
        candidates = []
        for span in spans:
            span_id = span.span_id
            if span_id not in state_scores:
                result[span_id] = SpanState.HIDDEN
                continue
            
            for state_name, score in state_scores[span_id].items():
                try:
                    state = SpanState(state_name)
                except ValueError:
                    continue
                if state != SpanState.HIDDEN:
                    tokens = self.estimate_tokens(span, state, tokenizer)
                    candidates.append((score, span_id, state, tokens))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Allocate greedily
        allocated = set()
        for score, span_id, state, tokens in candidates:
            if span_id in allocated:
                continue
            if tokens <= remaining_budget:
                result[span_id] = state
                remaining_budget -= tokens
                allocated.add(span_id)
        
        # Mark remaining as HIDDEN
        for span in spans:
            if span.span_id not in result:
                result[span.span_id] = SpanState.HIDDEN
        
        return result
    
    def knapsack_allocate(
        self,
        spans: list[Span],
        state_values: dict[str, dict[str, float]],
        tokenizer=None,
    ) -> dict[str, SpanState]:
        """Knapsack-based optimal allocation.
        
        Uses dynamic programming to find optimal allocation
        that maximizes total value under budget constraint.
        
        Note: For small number of spans (< 20), uses exact DP.
        For larger sets, falls back to greedy approximation.
        
        Args:
            spans: List of spans to allocate
            state_values: Per-span value for each state
            tokenizer: Optional tokenizer
            
        Returns:
            Mapping from span_id to selected state
        """
        n_spans = len(spans)
        
        # For large sets, use greedy approximation
        if n_spans > 20:
            logger.warning(f"Large span count ({n_spans}), using greedy allocation")
            return self.greedy_allocate(spans, state_values, tokenizer)
        
        # Build item list: for each span, consider EXPAND and SUMMARY
        # HIDDEN is implicit (0 cost, 0 value)
        items = []  # (span_idx, state, cost, value)
        
        for idx, span in enumerate(spans):
            span_id = span.span_id
            for state in [SpanState.EXPAND, SpanState.SUMMARY]:
                cost = self.estimate_tokens(span, state, tokenizer)
                value = state_values.get(span_id, {}).get(state.value, 0.0)
                items.append((idx, state, cost, value))
        
        # DP: dp[budget] = (max_value, allocation)
        # For efficiency, discretize budget
        budget = self.token_budget
        granularity = max(1, budget // 1000)
        budget_steps = budget // granularity + 1
        
        # dp[j] = (max_value, list of (idx, state))
        dp = [(0.0, []) for _ in range(budget_steps)]
        
        for idx, state, cost, value in items:
            cost_steps = cost // granularity
            for j in range(budget_steps - 1, cost_steps - 1, -1):
                if dp[j - cost_steps][0] + value > dp[j][0]:
                    new_allocation = dp[j - cost_steps][1] + [(idx, state)]
                    dp[j] = (dp[j - cost_steps][0] + value, new_allocation)
        
        # Extract best allocation
        best_value, allocation = max(dp, key=lambda x: x[0])
        
        # Convert to result dict
        result = {}
        allocated_spans = set()
        for idx, state in allocation:
            span_id = spans[idx].span_id
            if span_id not in allocated_spans:
                result[span_id] = state
                allocated_spans.add(idx)
        
        # Mark remaining as HIDDEN
        for span in spans:
            if span.span_id not in result:
                result[span.span_id] = SpanState.HIDDEN
        
        return result
    
    def check_budget_constraint(
        self,
        spans: list[Span],
        states: dict[str, SpanState],
        tokenizer=None,
    ) -> tuple[bool, int]:
        """Check if allocation satisfies budget constraint.
        
        Args:
            spans: List of spans
            states: Current allocation
            tokenizer: Optional tokenizer
            
        Returns:
            Tuple of (is_valid, total_tokens)
        """
        total = 0
        span_map = {s.span_id: s for s in spans}
        
        for span_id, state in states.items():
            if span_id in span_map:
                total += self.estimate_tokens(span_map[span_id], state, tokenizer)
        
        return total <= self.token_budget, total
