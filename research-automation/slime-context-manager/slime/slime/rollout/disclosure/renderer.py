"""Disclosure renderer for building answer prompts.

Renders selected span states into context for main LLM.
"""

import logging
from typing import Optional

from .schemas import Span, SpanState, DisclosureSample, SelectionResult

logger = logging.getLogger(__name__)


class DisclosureRenderer:
    """Renders disclosure selection into answer prompt.
    
    Takes a selection result and renders the appropriate context
    based on span states (EXPAND, SUMMARY, HIDDEN).
    """
    
    TEMPLATE_EXPAND = "[Document {doc_id}]\n{content}\n\n"
    TEMPLATE_SUMMARY = "[Document {doc_id} Summary]\n{content}\n\n"
    TEMPLATE_QUESTION = "Question: {question}\n\n"
    TEMPLATE_ANSWER = "Answer:"
    
    def __init__(self, token_budget: int = 4096):
        self.token_budget = token_budget
    
    def render(
        self,
        sample: DisclosureSample,
        selection: SelectionResult,
        include_question: bool = True,
        include_answer_prompt: bool = True,
    ) -> str:
        """Render the answer prompt based on selection.
        
        Args:
            sample: The disclosure sample
            selection: Selection result with span states
            include_question: Whether to include question
            include_answer_prompt: Whether to include answer prompt
            
        Returns:
            Rendered prompt string
        """
        parts = []
        span_map = {s.span_id: s for s in sample.spans}
        
        # Render each span according to its state
        for idx, (span_id, state) in enumerate(selection.span_states.items()):
            if span_id not in span_map:
                logger.warning(f"Span {span_id} not found in sample")
                continue
            
            span = span_map[span_id]
            
            if state == SpanState.HIDDEN:
                continue
            elif state == SpanState.SUMMARY:
                parts.append(self.TEMPLATE_SUMMARY.format(
                    doc_id=idx + 1,
                    content=span.summary
                ))
            elif state == SpanState.EXPAND:
                parts.append(self.TEMPLATE_EXPAND.format(
                    doc_id=idx + 1,
                    content=span.raw_text
                ))
        
        # Add question
        if include_question:
            parts.append(self.TEMPLATE_QUESTION.format(question=sample.question))
        
        # Add answer prompt
        if include_answer_prompt:
            parts.append(self.TEMPLATE_ANSWER)
        
        return "".join(parts)
    
    def render_with_budget_check(
        self,
        sample: DisclosureSample,
        selection: SelectionResult,
        tokenizer=None,
    ) -> tuple[str, int]:
        """Render with budget verification.
        
        Returns:
            Tuple of (rendered_prompt, actual_token_count)
        """
        rendered = self.render(sample, selection)
        
        if tokenizer:
            token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        else:
            # Rough estimate
            token_count = len(rendered) // 4
        
        return rendered, token_count
    
    def render_comparison(
        self,
        sample: DisclosureSample,
        selection1: SelectionResult,
        selection2: SelectionResult,
        labels: tuple[str, str] = ("Selection A", "Selection B"),
    ) -> str:
        """Render two selections side by side for comparison.
        
        Useful for preference learning or debugging.
        """
        prompt1 = self.render(sample, selection1, include_answer_prompt=False)
        prompt2 = self.render(sample, selection2, include_answer_prompt=False)
        
        return f"=== {labels[0]} ===\n{prompt1}\n=== {labels[1]} ===\n{prompt2}"
    
    def get_context_statistics(
        self,
        sample: DisclosureSample,
        selection: SelectionResult,
    ) -> dict:
        """Get statistics about the rendered context.
        
        Returns:
            Dict with expand_count, summary_count, hidden_count, etc.
        """
        expand_count = sum(1 for s in selection.span_states.values() if s == SpanState.EXPAND)
        summary_count = sum(1 for s in selection.span_states.values() if s == SpanState.SUMMARY)
        hidden_count = sum(1 for s in selection.span_states.values() if s == SpanState.HIDDEN)
        
        span_map = {s.span_id: s for s in sample.spans}
        expand_tokens = 0
        summary_tokens = 0
        
        for span_id, state in selection.span_states.items():
            if span_id in span_map:
                span = span_map[span_id]
                if state == SpanState.EXPAND:
                    expand_tokens += len(span.raw_text) // 4
                elif state == SpanState.SUMMARY:
                    summary_tokens += len(span.summary) // 4
        
        return {
            "total_spans": len(sample.spans),
            "expand_count": expand_count,
            "summary_count": summary_count,
            "hidden_count": hidden_count,
            "expand_tokens_est": expand_tokens,
            "summary_tokens_est": summary_tokens,
            "total_tokens_est": expand_tokens + summary_tokens,
            "budget_used": selection.budget_used,
        }
