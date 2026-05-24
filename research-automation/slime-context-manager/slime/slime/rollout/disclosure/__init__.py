"""Progressive Evidence Disclosure rollout module.

This module implements the selection policy training pipeline:
- Span states: EXPAND, SUMMARY, HIDDEN
- Review-SFT and OPD-style on-policy distillation
"""

from .slime_rollout import generate_rollout_disclosure
from .schemas import SpanState, DisclosureSample, SelectionResult
from .renderer import DisclosureRenderer
from .budget_allocator import BudgetAllocator

__all__ = [
    "generate_rollout_disclosure",
    "SpanState",
    "DisclosureSample",
    "SelectionResult",
    "DisclosureRenderer",
    "BudgetAllocator",
]
