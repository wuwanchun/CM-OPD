"""OPD (On-Policy Distillation) loss implementation for Progressive Disclosure."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import torch to avoid CUDA issues in test environments
_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch as _t
            _torch = _t
        except Exception as e:
            logger.warning(f"Failed to import torch: {e}")
            raise
    return _torch


@dataclass
class OPDLossConfig:
    """Configuration for OPD loss computation."""

    # Loss weights
    ce_weight: float = 1.0  # Cross-entropy weight for action prediction
    kl_weight: float = 0.1  # KL divergence weight for teacher distillation
    replay_weight: float = 0.2  # Weight for replay buffer loss

    # Temperature for distillation
    teacher_temperature: float = 2.0
    student_temperature: float = 1.0

    # Label smoothing
    label_smoothing: float = 0.0

    # Action states
    action_states: tuple[str, ...] = ("EXPAND", "SUMMARY", "HIDDEN")


def action_to_index(action: str) -> int:
    """Convert action string to index."""
    action_map = {"EXPAND": 0, "SUMMARY": 1, "HIDDEN": 2}
    return action_map.get(action.upper(), 1)


def index_to_action(index: int) -> str:
    """Convert index to action string."""
    actions = ["EXPAND", "SUMMARY", "HIDDEN"]
    return actions[index] if 0 <= index < len(actions) else "SUMMARY"


def compute_opd_loss(
    student_logits,
    target_actions,
    teacher_logprobs=None,
    *,
    config: OPDLossConfig | None = None,
    response_mask=None,
) -> dict:
    """
    Compute OPD loss for progressive disclosure action prediction.

    Args:
        student_logits: [batch_size, seq_len, vocab_size] or [batch_size, num_actions]
        target_actions: [batch_size] integer action indices
        teacher_logprobs: [batch_size, num_actions] teacher log probabilities (optional)
        config: Loss configuration
        response_mask: [batch_size, seq_len] mask for response tokens (optional)

    Returns:
        Dict with loss components: total_loss, ce_loss, kl_loss
    """
    torch = _get_torch()
    F = torch.nn.functional
    
    config = config or OPDLossConfig()

    # Ensure student_logits is 2D [batch, num_actions] for action classification
    if student_logits.dim() == 3:
        # Take last token logits for classification
        student_logits = student_logits[:, -1, :]

    # Map to action space (3 actions: EXPAND, SUMMARY, HIDDEN)
    num_actions = len(config.action_states)
    if student_logits.size(-1) != num_actions:
        # Project to action space if needed
        student_logits = student_logits[:, :num_actions]

    # Cross-entropy loss for action prediction
    ce_loss = F.cross_entropy(
        student_logits,
        target_actions,
        label_smoothing=config.label_smoothing,
    )

    # KL divergence loss for teacher distillation
    kl_loss = torch.tensor(0.0, device=student_logits.device)
    if teacher_logprobs is not None:
        kl_loss = compute_kl_distillation_loss(
            student_logits,
            teacher_logprobs,
            teacher_temperature=config.teacher_temperature,
            student_temperature=config.student_temperature,
        )

    # Total loss
    total_loss = (
        config.ce_weight * ce_loss +
        config.kl_weight * kl_loss
    )

    return {
        "total_loss": total_loss,
        "ce_loss": ce_loss.detach(),
        "kl_loss": kl_loss.detach() if isinstance(kl_loss, torch.Tensor) else kl_loss,
    }


def compute_kl_distillation_loss(
    student_logits,
    teacher_logprobs,
    *,
    teacher_temperature: float = 2.0,
    student_temperature: float = 1.0,
):
    """
    Compute KL divergence distillation loss.

    Args:
        student_logits: [batch_size, num_actions] student logits
        teacher_logprobs: [batch_size, num_actions] teacher log probabilities
        teacher_temperature: Temperature for teacher softmax
        student_temperature: Temperature for student softmax

    Returns:
        KL divergence loss
    """
    torch = _get_torch()
    F = torch.nn.functional
    
    # Convert to probabilities with temperature
    student_logprobs = F.log_softmax(student_logits / student_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logprobs / teacher_temperature, dim=-1)

    # KL divergence: KL(teacher || student)
    kl_div = F.kl_div(
        student_logprobs,
        teacher_probs,
        reduction="batchmean",
    )

    return kl_div * (teacher_temperature ** 2)


def compute_replay_augmented_loss(
    onpolicy_loss,
    replay_loss,
    *,
    replay_weight: float = 0.2,
):
    """
    Combine on-policy loss with replay buffer loss.

    Args:
        onpolicy_loss: Loss from current on-policy samples
        replay_loss: Loss from replay buffer samples
        replay_weight: Weight for replay loss (0.0-1.0)

    Returns:
        Combined loss
    """
    return (1 - replay_weight) * onpolicy_loss + replay_weight * replay_loss


def compute_selection_consistency_loss(
    selections: list[dict[str, Any]],
    previous_selections: list[dict[str, Any]],
    *,
    consistency_weight: float = 0.1,
):
    """
    Encourage consistency between consecutive policy iterations.

    Penalizes large changes in selection decisions for the same span
    unless justified by evidence.

    Args:
        selections: Current iteration selections
        previous_selections: Previous iteration selections
        consistency_weight: Weight for consistency penalty

    Returns:
        Consistency loss (0 if no previous selections)
    """
    torch = _get_torch()
    
    if not previous_selections:
        return torch.tensor(0.0)

    # Build mapping of span_id -> action
    prev_actions = {}
    for sel in previous_selections:
        for state in sel.get("states", []):
            span_id = state.get("span_id", "")
            action = state.get("action", "SUMMARY")
            prev_actions[span_id] = action_to_index(action)

    # Count changes
    changes = 0
    total = 0
    for sel in selections:
        for state in sel.get("states", []):
            span_id = state.get("span_id", "")
            if span_id in prev_actions:
                current_action = action_to_index(state.get("action", "SUMMARY"))
                if current_action != prev_actions[span_id]:
                    changes += 1
                total += 1

    if total == 0:
        return torch.tensor(0.0)

    change_rate = changes / total
    return torch.tensor(change_rate * consistency_weight, requires_grad=False)


def build_opd_training_batch(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int = 2048,
    max_response_length: int = 512,
) -> dict:
    """
    Build a training batch from OPD records.

    Args:
        records: List of OPD training records
        tokenizer: Tokenizer instance
        max_length: Maximum total sequence length
        max_response_length: Maximum response length

    Returns:
        Dict with input_ids, labels, attention_mask, action_labels
    """
    torch = _get_torch()
    
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    action_labels_list = []

    for record in records:
        prompt = record.get("prompt", "")
        response = record.get("response", record.get("label", ""))
        target_state = record.get("target_state", record.get("target_action", "SUMMARY"))

        # Tokenize
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

        # Truncate if needed
        if len(response_ids) > max_response_length:
            response_ids = response_ids[:max_response_length]

        total_len = len(prompt_ids) + len(response_ids)
        if total_len > max_length:
            keep_response = min(len(response_ids), max_response_length)
            keep_prompt = max_length - keep_response
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []

        # Build input_ids and labels
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        attention_mask = [1] * len(input_ids)

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
        action_labels_list.append(action_to_index(target_state))

    # Pad to same length
    max_len = max(len(ids) for ids in input_ids_list)

    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []

    for input_ids, labels, attention_mask in zip(input_ids_list, labels_list, attention_mask_list):
        pad_len = max_len - len(input_ids)
        padded_input_ids.append(input_ids + [0] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
        padded_attention_mask.append(attention_mask + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "action_labels": torch.tensor(action_labels_list, dtype=torch.long),
    }


@dataclass
class OPDLossOutput:
    """Output from OPD loss computation."""

    total_loss: Any
    ce_loss: Any
    kl_loss: Any
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_loss": self.total_loss.item() if hasattr(self.total_loss, 'item') else self.total_loss,
            "ce_loss": self.ce_loss.item() if hasattr(self.ce_loss, 'item') else self.ce_loss,
            "kl_loss": self.kl_loss.item() if hasattr(self.kl_loss, 'item') else self.kl_loss,
            "metrics": self.metrics,
        }


def compute_full_opd_loss(
    student_logits,
    batch: dict[str, Any],
    teacher_logprobs=None,
    *,
    config: OPDLossConfig | None = None,
) -> OPDLossOutput:
    """
    Compute full OPD loss with all components and metrics.

    Args:
        student_logits: Student model logits
        batch: Training batch with action_labels
        teacher_logprobs: Optional teacher log probabilities
        config: Loss configuration

    Returns:
        OPDLossOutput with loss components and metrics
    """
    torch = _get_torch()
    config = config or OPDLossConfig()
    target_actions = batch["action_labels"]

    loss_dict = compute_opd_loss(
        student_logits,
        target_actions,
        teacher_logprobs,
        config=config,
    )

    # Compute accuracy
    with torch.no_grad():
        if student_logits.dim() == 3:
            logits_2d = student_logits[:, -1, :]
        else:
            logits_2d = student_logits

        predictions = torch.argmax(logits_2d[:, :3], dim=-1)
        accuracy = (predictions == target_actions).float().mean().item()

    metrics = {
        "action_accuracy": accuracy,
        "ce_weight": config.ce_weight,
        "kl_weight": config.kl_weight,
    }

    return OPDLossOutput(
        total_loss=loss_dict["total_loss"],
        ce_loss=loss_dict["ce_loss"],
        kl_loss=loss_dict["kl_loss"],
        metrics=metrics,
    )
