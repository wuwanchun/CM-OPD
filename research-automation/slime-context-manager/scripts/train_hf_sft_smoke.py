"""Tiny local HF SFT smoke trainer for context-action policies.

This is intentionally small: it validates that the exported slime JSONL,
local Hugging Face checkpoint, CUDA runtime, and supervised action loss can
run end-to-end on a single GPU. It is not a replacement for the full slime
Ray/Megatron training launcher.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from data.io_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", default=str(ROOT / "data" / "slime" / "context_actions_train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "context_policy_hf_sft_smoke"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="float32")
    parser.add_argument("--save-model", action="store_true", help="Save model weights; disabled by default for smoke tests.")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit("torch and transformers are required for HF SFT smoke training.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.train_jsonl)
    if not records:
        raise SystemExit(f"No training records found: {args.train_jsonl}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.to(device)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[float] = []
    started = time.time()
    max_steps = min(args.max_steps, len(records))
    for step, record in enumerate(records[:max_steps], start=1):
        batch = _encode_record(tokenizer, record, max_length=args.max_length)
        input_ids = torch.tensor([batch["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor([batch["attention_mask"]], dtype=torch.long, device=device)
        labels = torch.tensor([batch["labels"]], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite loss at step {step}: {float(loss.detach().cpu().item())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        value = float(loss.detach().cpu().item())
        losses.append(value)
        print(f"step={step} loss={value:.6f}")

    metrics = {
        "model_path": str(args.model_path),
        "train_jsonl": str(args.train_jsonl),
        "output_dir": str(output_dir),
        "num_records": len(records),
        "steps": len(losses),
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "device": str(device),
        "dtype": args.dtype,
        "elapsed_sec": round(time.time() - started, 3),
        "saved_model": bool(args.save_model),
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    if args.save_model:
        model_dir = output_dir / "model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _encode_record(tokenizer: object, record: dict[str, object], *, max_length: int) -> dict[str, list[int]]:
    prompt = str(record["prompt"])
    label = str(record["label"])
    eos = getattr(tokenizer, "eos_token", None) or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(label + eos, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > max_length:
        keep_response = min(len(response_ids), max_length)
        keep_prompt = max_length - keep_response
        kept_prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        input_ids = kept_prompt_ids + response_ids[:keep_response]
        labels = [-100] * keep_prompt + response_ids[:keep_response]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _resolve_dtype(torch: object, name: str) -> object | None:
    if name == "auto":
        return None
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


if __name__ == "__main__":
    main()
