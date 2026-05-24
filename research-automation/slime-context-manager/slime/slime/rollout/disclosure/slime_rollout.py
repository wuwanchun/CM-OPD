"""Progressive Disclosure rollout for Slime training.

Implements the main rollout function for disclosure policy training.
Based on roadmap: progressive-disclosure-slime-roadmap.md
"""

import json
import logging
import uuid
from typing import Optional

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

from .budget_allocator import BudgetAllocator
from .renderer import DisclosureRenderer
from .schemas import DisclosureSample, SelectionResult, Span, SpanState
from .selection_prompt import SelectionPromptBuilder

__all__ = ["generate_rollout_disclosure"]

logger = logging.getLogger(__name__)

TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def parse_disclosure_data(sample_data: dict) -> DisclosureSample:
    """Parse raw data into DisclosureSample.
    
    Expected input format:
    {
        "sample_id": "...",
        "question": "...",
        "spans": [
            {"span_id": "...", "raw_text": "...", "summary": "..."},
            ...
        ],
        "gold_answer": "...",
        "metadata": {...}
    }
    """
    spans = []
    for span_data in sample_data.get("spans", []):
        spans.append(Span(
            span_id=span_data["span_id"],
            raw_text=span_data["raw_text"],
            summary=span_data["summary"],
            metadata=span_data.get("metadata", {}),
        ))
    
    return DisclosureSample(
        sample_id=sample_data["sample_id"],
        question=sample_data["question"],
        spans=spans,
        gold_answer=sample_data.get("gold_answer", ""),
        metadata=sample_data.get("metadata", {}),
    )


def build_selection_training_prompt(
    sample: DisclosureSample,
    target_states: dict[str, SpanState],
    rationale: Optional[str] = None,
) -> str:
    """Build training prompt with target selection.
    
    Format:
    [Question + Summaries] -> [Selection JSON] [Rationale]
    """
    prompt_builder = SelectionPromptBuilder()
    prompt = prompt_builder.build_prompt(sample)
    
    # Add target selection
    target_json = json.dumps({k: v.value for k, v in target_states.items()}, indent=2)
    
    if rationale:
        return f"{prompt}\n\nSelection: {target_json}\n\nRationale: {rationale}"
    else:
        return f"{prompt}\n\nSelection: {target_json}"


def generate_rollout_disclosure(args, rollout_id, data_buffer, evaluation=False):
    """Generate rollout for Progressive Evidence Disclosure training.
    
    This function:
    1. Loads disclosure samples from the dataset
    2. Builds selection prompts with summaries
    3. Generates target selections (from oracle or previous iteration)
    4. Creates loss masks for training
    
    Args:
        args: Training arguments
        rollout_id: Current rollout iteration
        data_buffer: Buffer containing training samples
        evaluation: Whether this is evaluation mode
        
    Returns:
        List of Sample objects ready for training
    """
    assert not evaluation, "Evaluation not supported for disclosure rollout"
    assert args.rollout_global_dataset, "rollout_global_dataset is required"

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    samples = data_buffer.get_samples(args.rollout_batch_size)
    
    budget_allocator = BudgetAllocator(token_budget=getattr(args, "disclosure_budget", 4096))
    renderer = DisclosureRenderer(token_budget=getattr(args, "disclosure_budget", 4096))
    prompt_builder = SelectionPromptBuilder(token_budget=getattr(args, "disclosure_budget", 4096))

    result_samples = []
    
    for i, sample_tuple in enumerate(samples):
        (sample_data,) = sample_tuple
        
        # Parse disclosure data
        if isinstance(sample_data.prompt, str):
            # Raw JSON string
            try:
                raw_data = json.loads(sample_data.prompt)
            except json.JSONDecodeError:
                raw_data = sample_data.prompt
        else:
            raw_data = sample_data.prompt
        
        # Handle different input formats
        if isinstance(raw_data, dict) and "spans" in raw_data:
            disclosure_sample = parse_disclosure_data(raw_data)
        else:
            # Fallback: treat as simple Q&A
            disclosure_sample = DisclosureSample(
                sample_id=str(uuid.uuid4()),
                question=raw_data if isinstance(raw_data, str) else str(raw_data),
                spans=[],
                gold_answer=sample_data.metadata.get("gold_answer", "") if sample_data.metadata else "",
                metadata=sample_data.metadata or {},
            )
        
        # Build selection prompt
        selection_prompt = prompt_builder.build_prompt(disclosure_sample)
        
        # Get target selection (from metadata or oracle)
        if sample_data.metadata and "target_selection" in sample_data.metadata:
            target_states = {
                k: SpanState(v) for k, v in sample_data.metadata["target_selection"].items()
            }
        elif sample_data.metadata and "oracle_selection" in sample_data.metadata:
            target_states = {
                k: SpanState(v) for k, v in sample_data.metadata["oracle_selection"].items()
            }
        else:
            # Default: greedy allocation based on relevance scores
            if disclosure_sample.spans:
                relevance_scores = {}
                for idx, span in enumerate(disclosure_sample.spans):
                    # Simple heuristic: use metadata relevance or random
                    relevance = span.metadata.get("relevance", 0.5)
                    relevance_scores[span.span_id] = {
                        SpanState.EXPAND.value: relevance * 0.8,
                        SpanState.SUMMARY.value: relevance * 0.6,
                        SpanState.HIDDEN.value: 1.0 - relevance,
                    }
                target_states = budget_allocator.greedy_allocate(
                    disclosure_sample.spans, relevance_scores, TOKENIZER
                )
            else:
                target_states = {}
        
        # Build full training text
        rationale = sample_data.metadata.get("rationale") if sample_data.metadata else None
        full_text = build_selection_training_prompt(disclosure_sample, target_states, rationale)
        
        # Tokenize and create loss mask
        messages = [{"role": "user", "content": selection_prompt}]
        target_json = json.dumps({k: v.value for k, v in target_states.items()}, indent=2)
        if rationale:
            response = f"{target_json}\n\nRationale: {rationale}"
        else:
            response = target_json
        
        messages.append({"role": "assistant", "content": response})
        
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages)
        
        if len(token_ids) != len(loss_mask):
            raise ValueError(
                f"Mismatched token_ids/loss_mask lengths: {len(token_ids)=}, {len(loss_mask)=}"
            )
        
        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
        
        # Update sample
        sample_data.tokens = token_ids
        sample_data.response_length = response_length
        sample_data.reward = 0
        sample_data.loss_mask = loss_mask[-response_length:]
        
        # Store disclosure metadata
        sample_data.metadata = sample_data.metadata or {}
        sample_data.metadata["disclosure_sample_id"] = disclosure_sample.sample_id
        sample_data.metadata["target_states"] = {k: v.value for k, v in target_states.items()}
        
        result_samples.append(sample_data)
        
        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"disclosure_rollout::generate_rollout example: "
                f"sample_id={disclosure_sample.sample_id}, "
                f"num_spans={len(disclosure_sample.spans)}, "
                f"target_states={target_states}"
            )
            SAMPLE_PRINTED = True
    
    return result_samples
