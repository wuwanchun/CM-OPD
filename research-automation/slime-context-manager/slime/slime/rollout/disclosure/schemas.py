"""Schemas for Progressive Evidence Disclosure.

Based on roadmap: progressive-disclosure-slime-roadmap.md
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SpanState(Enum):
    """Span disclosure states for selection policy.
    
    Attributes:
        EXPAND: Show full raw span text (summary insufficient)
        SUMMARY: Show only summary (summary sufficient)
        HIDDEN: Don't include in answer prompt (irrelevant/budget)
    """
    EXPAND = "EXPAND"
    SUMMARY = "SUMMARY"
    HIDDEN = "HIDDEN"


@dataclass
class Span:
    """A single span in the document.
    
    Attributes:
        span_id: Unique identifier for this span
        raw_text: Full original text of the span
        summary: Cue-preserving summary of the span
        metadata: Additional metadata (position, source, etc.)
    """
    span_id: str
    raw_text: str
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_token_count(self, tokenizer=None) -> int:
        """Estimate token count for this span."""
        # Rough estimate: ~4 chars per token
        if tokenizer:
            return len(tokenizer.encode(self.raw_text, add_special_tokens=False))
        return len(self.raw_text) // 4


@dataclass
class DisclosureSample:
    """Sample for disclosure policy training.
    
    Represents a single training example with:
    - Question/task
    - Multiple spans with summaries
    - Ground truth answer (for training)
    - Metadata for budget and tracking
    """
    sample_id: str
    question: str
    spans: list[Span]
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_budget_state(self) -> dict[str, int]:
        """Get current budget state."""
        return {
            "total_spans": len(self.spans),
            "max_budget_tokens": self.metadata.get("max_budget_tokens", 4096),
            "used_budget_tokens": self.metadata.get("used_budget_tokens", 0),
        }


@dataclass
class SelectionResult:
    """Result of selection policy for a single sample.
    
    Attributes:
        sample_id: ID of the sample
        selection_id: Unique ID for this selection
        span_states: Mapping from span_id to selected state
        selection_scores: Raw scores for each state (for training)
        budget_used: Total tokens used in rendered context
        metadata: Additional tracking info
    """
    sample_id: str
    selection_id: str
    span_states: dict[str, SpanState]
    selection_scores: Optional[dict[str, dict[str, float]]] = None
    budget_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_training_target(self) -> dict[str, Any]:
        """Convert to Slime training format."""
        return {
            "sample_id": self.sample_id,
            "selection_id": self.selection_id,
            "span_states": {k: v.value for k, v in self.span_states.items()},
            "budget_used": self.budget_used,
        }


@dataclass
class AnswerResult:
    """Result of answer generation after selection.
    
    Attributes:
        sample_id: ID of the sample
        answer: Generated answer text
        is_correct: Whether answer matches gold answer
        metadata: Evaluation metadata
    """
    sample_id: str
    answer: str
    is_correct: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewTarget:
    """Training target from review process.
    
    Used for both Review-SFT and OPD distillation.
    """
    sample_id: str
    original_selection: SelectionResult
    answer_result: AnswerResult
    corrected_states: Optional[dict[str, SpanState]] = None  # For Review-SFT
    hindsight_rationale: Optional[str] = None  # For OPD distillation
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_sft_target(self) -> dict[str, Any]:
        """Convert to SFT training target."""
        if self.corrected_states is None:
            raise ValueError("No corrected states for SFT target")
        return {
            "sample_id": self.sample_id,
            "span_states": {k: v.value for k, v in self.corrected_states.items()},
            "rationale": self.hindsight_rationale,
        }
    
    def to_opd_target(self) -> dict[str, Any]:
        """Convert to OPD distillation target."""
        return {
            "sample_id": self.sample_id,
            "original_selection": self.original_selection.to_training_target(),
            "answer": self.answer_result.answer,
            "is_correct": self.answer_result.is_correct,
            "hindsight_rationale": self.hindsight_rationale,
        }
