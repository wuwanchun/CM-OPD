"""Selection prompt builder for disclosure policy.

Builds prompts for the selection policy to choose span states.
"""

import json
from typing import Optional

from .schemas import Span, SpanState, DisclosureSample


class SelectionPromptBuilder:
    """Builds prompts for selection policy.
    
    The selection policy sees:
    - Question
    - Summaries of all spans
    - Budget state
    - Previous selection trace (if any)
    
    And outputs:
    - JSON with span state selections
    """
    
    SYSTEM_PROMPT = """You are a selection policy that decides which document spans to show for answering questions.

Given a question and document summaries, you must decide for each document whether to:
- EXPAND: Show the full document text (when summary is insufficient)
- SUMMARY: Show only the summary (when summary is sufficient)
- HIDDEN: Don't show the document (when irrelevant or budget constrained)

You have a token budget constraint. Prioritize showing the most relevant information.

Output your selection as a JSON object with document IDs as keys and states as values.
Example: {"doc_1": "EXPAND", "doc_2": "SUMMARY", "doc_3": "HIDDEN"}"""

    USER_TEMPLATE = """Question: {question}

Budget: {budget_used}/{budget_total} tokens

Documents:
{documents}

Select the state (EXPAND/SUMMARY/HIDDEN) for each document. Output JSON only."""

    USER_TEMPLATE_WITH_TRACE = """Question: {question}

Budget: {budget_used}/{budget_total} tokens

Documents:
{documents}

Previous selection: {previous_selection}
Answer was: {previous_answer}
Result: {previous_result}

Based on this feedback, select improved states for each document. Output JSON only."""

    def __init__(self, token_budget: int = 4096):
        self.token_budget = token_budget
    
    def build_prompt(
        self,
        sample: DisclosureSample,
        previous_selection: Optional[dict[str, SpanState]] = None,
        previous_answer: Optional[str] = None,
        previous_result: Optional[str] = None,
    ) -> str:
        """Build selection prompt for a sample.
        
        Args:
            sample: Disclosure sample with question and spans
            previous_selection: Optional previous selection (for refinement)
            previous_answer: Previous answer (if any)
            previous_result: Result of previous answer (correct/incorrect)
            
        Returns:
            Formatted prompt string
        """
        # Format documents
        docs = []
        for idx, span in enumerate(sample.spans):
            docs.append(f"[Document {idx + 1}] (ID: {span.span_id})\n{span.summary}")
        documents = "\n\n".join(docs)
        
        if previous_selection:
            # Build refinement prompt
            selection_str = json.dumps({k: v.value for k, v in previous_selection.items()})
            result_str = "CORRECT" if previous_result == "correct" else "INCORRECT"
            
            return self.USER_TEMPLATE_WITH_TRACE.format(
                question=sample.question,
                budget_used=0,
                budget_total=self.token_budget,
                documents=documents,
                previous_selection=selection_str,
                previous_answer=previous_answer or "N/A",
                previous_result=result_str,
            )
        else:
            return self.USER_TEMPLATE.format(
                question=sample.question,
                budget_used=0,
                budget_total=self.token_budget,
                documents=documents,
            )
    
    def build_messages(
        self,
        sample: DisclosureSample,
        previous_selection: Optional[dict[str, SpanState]] = None,
        previous_answer: Optional[str] = None,
        previous_result: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Build message format for chat models.
        
        Returns:
            List of message dicts with role and content
        """
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.build_prompt(
                sample, previous_selection, previous_answer, previous_result
            )},
        ]
    
    def parse_selection_response(
        self,
        response: str,
        sample: DisclosureSample,
    ) -> dict[str, SpanState]:
        """Parse model response into span states.
        
        Args:
            response: Raw model output
            sample: Original sample for span ID mapping
            
        Returns:
            Dict mapping span_id to SpanState
        """
        # Try to extract JSON from response
        try:
            # Find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: all HIDDEN
            return {span.span_id: SpanState.HIDDEN for span in sample.spans}
        
        # Convert to span states
        result = {}
        span_id_map = {f"doc_{idx + 1}": span.span_id for idx, span in enumerate(sample.spans)}
        
        for doc_key, state_str in parsed.items():
            # Map document key to span_id
            if doc_key in span_id_map:
                span_id = span_id_map[doc_key]
            else:
                span_id = doc_key  # Use as-is if already span_id
            
            try:
                result[span_id] = SpanState(state_str.upper())
            except ValueError:
                result[span_id] = SpanState.HIDDEN
        
        # Fill missing spans with HIDDEN
        for span in sample.spans:
            if span.span_id not in result:
                result[span.span_id] = SpanState.HIDDEN
        
        return result
