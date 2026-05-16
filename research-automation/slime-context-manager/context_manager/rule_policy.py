"""Rule baseline for external memory actions."""

from __future__ import annotations

from .schemas import ContextActionSample, Prediction


class RulePolicy:
    """A deterministic baseline policy for context-action decisions."""

    def predict(self, sample: ContextActionSample) -> Prediction:
        item = sample.memory_item
        metadata = item.metadata
        item_type = item.type
        labels = sample.labels

        if item_type in {"user_constraint", "task_goal"} or metadata.get("has_user_constraint"):
            action = "PIN"
            rationale = "User constraints and task goals should remain fixed in context."
        elif labels.get("forbidden_actions") and "DROP" in labels.get("forbidden_actions", []):
            if metadata.get("has_error") or metadata.get("has_test_name") or sample.gold_evidence_ids:
                action = "KEEP"
                rationale = "Critical evidence is forbidden to drop and should stay visible."
            else:
                action = "COMPRESS"
                rationale = "Forbidden-to-drop item is preserved through compressed memory."
        elif metadata.get("has_error") or metadata.get("has_test_name") or item_type in {"test_failure", "debug_attempt"}:
            action = "KEEP"
            rationale = "Failure evidence should remain available for recovery."
        elif sample.budget_pressure >= 0.85 or int(metadata.get("token_count", 0) or 0) > 900:
            action = "COMPRESS"
            rationale = "The item is large or budget pressure is high."
        elif item_type in {"retrieved_doc", "tool_output", "code_slice"}:
            action = "ARCHIVE"
            rationale = "Store raw evidence with pointer while keeping context compact."
        else:
            action = "KEEP"
            rationale = "Default to preserving uncertain evidence."

        return Prediction(sample_id=sample.sample_id, pred_action=action, pred_rationale=rationale, confidence=1.0)
