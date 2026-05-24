"""Top-K distillation loss scaffold for slime.

This keeps the OpenClaw-RL compatible fields separate from the legacy
token-level OPD path:

``teacher_topk_log_probs`` and ``teacher_topk_indices``.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Callable


def compute_tail_logprob(log_probs: Any) -> Any:
    """Compute log tail mass for a row of top-K log-probs using torch."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("torch is required for top-K distillation") from exc

    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    return torch.log(-torch.expm1(log_s))


def topk_reverse_kl(student_topk_log_probs: Any, teacher_topk_log_probs: Any) -> Any:
    """Reverse KL over teacher top-K tokens plus a tail bin."""

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("torch is required for top-K distillation") from exc

    student_with_tail = torch.cat([student_topk_log_probs, compute_tail_logprob(student_topk_log_probs)], dim=-1)
    teacher_with_tail = torch.cat([teacher_topk_log_probs, compute_tail_logprob(teacher_topk_log_probs)], dim=-1)
    return F.kl_div(teacher_with_tail, student_with_tail, reduction="none", log_target=True).sum(dim=-1)


def topk_distillation_loss_function(
    args: Namespace,
    batch: dict,
    logits: Any,
    sum_of_sample_mean: Callable[[Any], Any],
) -> tuple[Any, dict[str, Any]]:
    """Slime custom loss entry point.

    This function intentionally imports slime/Megatron dependencies lazily so
    unit tests can run without a training stack.
    """

    try:
        import torch
        from megatron.core import mpu
        from slime.backends.megatron_utils.loss import get_responses
        from slime.utils.ppo_utils import compute_log_probs
    except ImportError as exc:  # pragma: no cover - only executed inside slime training
        raise RuntimeError("slime, megatron, and torch are required for top-K custom loss") from exc

    k = int(getattr(args, "distill_topk", 0))
    if k <= 0:
        zero = logits.sum() * 0.0
        return zero, {"loss": zero.detach(), "kl_loss": zero.detach()}

    teacher_topk_logprobs = batch["teacher_topk_log_probs"]
    teacher_topk_indices = batch["teacher_topk_indices"]
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    tp_group = mpu.get_tensor_model_parallel_group()

    student_rows = []
    teacher_rows = []
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    for index, (logits_chunk, _tokens_chunk) in enumerate(responses):
        teacher_lp = teacher_topk_logprobs[index].to(device=logits_chunk.device)
        teacher_idx = teacher_topk_indices[index].to(device=logits_chunk.device)
        per_k = []
        for j in range(k):
            per_k.append(compute_log_probs(logits_chunk, teacher_idx[:, j], tp_group).squeeze(-1))
        student_rows.append(torch.stack(per_k, dim=-1))
        teacher_rows.append(teacher_lp)

    student_topk = torch.cat(student_rows, dim=0)
    teacher_topk = torch.cat(teacher_rows, dim=0)
    per_token_kl = topk_reverse_kl(student_topk, teacher_topk)
    kl_loss = sum_of_sample_mean(per_token_kl)
    return kl_loss, {"loss": kl_loss.detach(), "kl_loss": kl_loss.detach()}
