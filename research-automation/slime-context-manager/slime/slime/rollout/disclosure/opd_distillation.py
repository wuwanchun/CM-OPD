"""OPD-style distillation builder for Progressive Disclosure.

Based on roadmap: constructs on-policy distillation samples from
selection prompt/result + hindsight rationale.
"""

import json
import logging
from typing import Optional

from .schemas import DisclosureSample, SelectionResult, AnswerResult, ReviewTarget, SpanState

logger = logging.getLogger(__name__)


class OPDDistillationBuilder:
    """Builds OPD distillation samples from selection-answer-review pipeline.
    
    The OPD sample format:
    1. Original selection prompt (question + summaries)
    2. Original selection result (JSON states)
    3. Answer generated from selection
    4. Hindsight rationale explaining why selection was good/bad
    5. Corrected selection (optional)
    """
    
    def __init__(self, include_rationale: bool = True):
        self.include_rationale = include_rationale
    
    def build_opd_sample(
        self,
        sample: DisclosureSample,
        selection: SelectionResult,
        answer_result: AnswerResult,
        hindsight_rationale: Optional[str] = None,
        corrected_states: Optional[dict[str, SpanState]] = None,
    ) -> dict:
        """Build a single OPD distillation sample.
        
        Args:
            sample: Original disclosure sample
            selection: Selection result from policy
            answer_result: Answer result after rendering
            hindsight_rationale: Explanation of selection quality
            corrected_states: Optional corrected states (for teacher signal)
            
        Returns:
            OPD training sample dict
        """
        from .selection_prompt import SelectionPromptBuilder
        
        prompt_builder = SelectionPromptBuilder()
        original_prompt = prompt_builder.build_prompt(sample)
        
        # Build OPD training text
        parts = [original_prompt]
        
        # Add original selection
        selection_json = json.dumps(
            {k: v.value for k, v in selection.span_states.items()},
            indent=2
        )
        parts.append(f"\n\nSelection:\n{selection_json}")
        
        # Add answer and result
        parts.append(f"\n\nAnswer: {answer_result.answer}")
        parts.append(f"\nResult: {'CORRECT' if answer_result.is_correct else 'INCORRECT'}")
        
        # Add hindsight rationale
        if hindsight_rationale and self.include_rationale:
            parts.append(f"\n\nHindsight Analysis:\n{hindsight_rationale}")
        
        # Add corrected selection if available
        if corrected_states:
            corrected_json = json.dumps(
                {k: v.value for k, v in corrected_states.items()},
                indent=2
            )
            parts.append(f"\n\nImproved Selection:\n{corrected_json}")
        
        return {
            "sample_id": sample.sample_id,
            "selection_id": selection.selection_id,
            "opd_text": "".join(parts),
            "original_prompt": original_prompt,
            "original_selection": {k: v.value for k, v in selection.span_states.items()},
            "answer": answer_result.answer,
            "is_correct": answer_result.is_correct,
            "hindsight_rationale": hindsight_rationale,
            "corrected_selection": {k: v.value for k, v in corrected_states.items()} if corrected_states else None,
        }
    
    def build_review_sft_sample(
        self,
        sample: DisclosureSample,
        selection: SelectionResult,
        answer_result: AnswerResult,
        corrected_states: dict[str, SpanState],
        rationale: Optional[str] = None,
    ) -> dict:
        """Build Review-SFT sample for direct imitation learning.
        
        Format:
        [Question + Summaries + Previous Selection + Result] -> [Corrected Selection]
        """
        from .selection_prompt import SelectionPromptBuilder
        
        prompt_builder = SelectionPromptBuilder()
        
        # Build prompt with trace
        prompt = prompt_builder.build_prompt(
            sample,
            previous_selection=selection.span_states,
            previous_answer=answer_result.answer,
            previous_result="correct" if answer_result.is_correct else "incorrect",
        )
        
        # Build response
        corrected_json = json.dumps(
            {k: v.value for k, v in corrected_states.items()},
            indent=2
        )
        
        if rationale:
            response = f"{corrected_json}\n\nRationale: {rationale}"
        else:
            response = corrected_json
        
        return {
            "sample_id": sample.sample_id,
            "selection_id": selection.selection_id,
            "prompt": prompt,
            "response": response,
            "corrected_selection": {k: v.value for k, v in corrected_states.items()},
            "rationale": rationale,
        }
    
    def build_opd_training_messages(
        self,
        opd_sample: dict,
    ) -> list[dict[str, str]]:
        """Build message format for OPD training.
        
        Returns:
            List of messages for chat model
        """
        from .selection_prompt import SelectionPromptBuilder
        
        messages = [
            {"role": "system", "content": SelectionPromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": opd_sample["original_prompt"]},
        ]
        
        # Add selection and feedback
        feedback = f"Your selection was:\n{json.dumps(opd_sample['original_selection'], indent=2)}\n\n"
        feedback += f"The answer was: {opd_sample['answer']}\n"
        feedback += f"Result: {'CORRECT' if opd_sample['is_correct'] else 'INCORRECT'}\n\n"
        
        if opd_sample.get("hindsight_rationale"):
            feedback += f"Analysis:\n{opd_sample['hindsight_rationale']}\n\n"
        
        if opd_sample.get("corrected_selection"):
            feedback += f"Improved selection:\n{json.dumps(opd_sample['corrected_selection'], indent=2)}"
        
        messages.append({"role": "assistant", "content": feedback})
        
        return messages


def build_hindsight_rationale(
    sample: DisclosureSample,
    selection: SelectionResult,
    answer_result: AnswerResult,
    evidence_spans: Optional[list[str]] = None,
) -> str:
    """Generate hindsight rationale for a selection-answer pair.
    
    This is a rule-based rationale generator. In production, this would
    be generated by a reviewer model or human annotation.
    
    Args:
        sample: Original disclosure sample
        selection: Selection result
        answer_result: Answer result
        evidence_spans: List of span IDs that contained key evidence
        
    Returns:
        Hindsight rationale string
    """
    stats = {
        "expand": sum(1 for s in selection.span_states.values() if s == SpanState.EXPAND),
        "summary": sum(1 for s in selection.span_states.values() if s == SpanState.SUMMARY),
        "hidden": sum(1 for s in selection.span_states.values() if s == SpanState.HIDDEN),
    }
    
    if answer_result.is_correct:
        rationale = f"The selection was effective. "
        rationale += f"Expanded {stats['expand']} documents, showed summaries for {stats['summary']}, and hid {stats['hidden']}. "
        
        if evidence_spans:
            rationale += f"Key evidence was found in documents: {', '.join(evidence_spans)}."
    else:
        rationale = f"The selection was suboptimal. "
        rationale += f"Expanded {stats['expand']} documents, showed summaries for {stats['summary']}, and hid {stats['hidden']}. "
        
        if evidence_spans:
            rationale += f"Key evidence was in documents: {', '.join(evidence_spans)}. "
            rationale += f"Some relevant documents may have been hidden or only shown as summaries."
    
    return rationale
