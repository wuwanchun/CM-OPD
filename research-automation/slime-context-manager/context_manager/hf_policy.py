"""Hugging Face policy wrapper for context action inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from opd.action_parser import parse_memory_action

from .schemas import ContextActionSample, Prediction


def render_policy_prompt(sample: ContextActionSample) -> str:
    """Render a compact prompt for the context manager policy."""

    payload = {
        "task_goal": sample.task_goal,
        "recent_context": sample.recent_context,
        "budget": sample.budget,
        "budget_used": sample.budget_used,
        "memory_item": sample.memory_item.to_dict(),
        "candidate_actions": sample.candidate_actions,
        "labels": sample.labels,
    }
    return (
        "You are an external context manager. Choose one memory action as JSON.\n"
        "Valid actions: KEEP, COMPRESS, ARCHIVE, RETRIEVE, UPDATE, DROP, PIN.\n"
        "Return only JSON like {\"action\":\"KEEP\",\"target_ids\":[\"mem_id\"],\"reason\":\"...\"}.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


@dataclass
class HFModelPolicy:
    """Minimal Transformers-based policy wrapper."""

    model_name_or_path: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    device_map: str | None = "auto"
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("transformers and torch are required for HFModelPolicy") from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device_map,
            torch_dtype="auto",
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()

    def predict(self, sample: ContextActionSample) -> Prediction:
        prompt = render_policy_prompt(sample)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = self.temperature > 0
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = parse_memory_action(text)
        action = parsed.action if parsed.valid and parsed.action else "INVALID"
        return Prediction(
            sample_id=sample.sample_id,
            pred_action=action,
            pred_rationale=str(parsed.payload.get("reason", "")) if parsed.payload else "",
            raw_output=text,
            tokens_in=int(inputs["input_ids"].shape[-1]),
            tokens_out=int(new_tokens.shape[-1]),
        )


def policy_from_name(policy: str, **kwargs: Any) -> Any:
    if policy == "model":
        return HFModelPolicy(**kwargs)
    raise ValueError(f"unknown HF policy kind: {policy}")
