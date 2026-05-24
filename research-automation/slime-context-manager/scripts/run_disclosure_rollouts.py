"""Run deterministic disclosure policies over context spans."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.budget_allocator import allocate_budget
from disclosure.renderer import render_prompt
from disclosure.rollout_policy import policy_from_name
from disclosure.schemas import ContextSpan, DisclosureRollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "disclosure"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--policy", choices=["summary_only", "full_raw_topk", "rule_expand_omission", "random_budget_perturbation"], default="rule_expand_omission")
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--reserved-prompt-tokens", type=int, default=128)
    parser.add_argument("--allocator", choices=["greedy_margin_per_cost", "knapsack_analysis"], default="greedy_margin_per_cost")
    args = parser.parse_args()

    spans = [ContextSpan.from_dict(record) for record in read_jsonl(Path(args.input_dir) / f"spans_{args.split}.jsonl")]
    grouped: dict[str, list[ContextSpan]] = {}
    for span in spans:
        grouped.setdefault(span.task_id, []).append(span)

    policy = policy_from_name(args.policy)
    rollouts = []
    allocations = []
    rendered = []
    for task_index, (task_id, task_spans) in enumerate(sorted(grouped.items())):
        rollout_id = f"{task_id}_{args.policy}_{task_index:03d}"
        decisions = [policy.decide(span) for span in task_spans]
        allocation = allocate_budget(
            task_spans,
            decisions,
            rollout_id=rollout_id,
            token_budget=args.token_budget,
            reserved_prompt_tokens=args.reserved_prompt_tokens,
            allocator=args.allocator,
        )
        prompt = render_prompt(task_spans[0].question, task_spans, allocation.final_actions)
        rollout = DisclosureRollout(
            rollout_id=rollout_id,
            task_id=task_id,
            policy_name=args.policy,
            token_budget=args.token_budget,
            actions=allocation.final_actions,
            rendered_prompt=prompt,
            metadata={"allocator": args.allocator},
        )
        rollouts.append(rollout.to_dict())
        allocations.append(allocation.to_dict())
        rendered.append({"rollout_id": rollout_id, "task_id": task_id, "rendered_prompt": prompt})

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / f"rollouts_{args.split}.jsonl", rollouts)
    write_jsonl(output_dir / f"budget_allocations_{args.split}.jsonl", allocations)
    write_jsonl(output_dir / f"rendered_prompts_{args.split}.jsonl", rendered)
    print(f"Wrote {len(rollouts)} disclosure rollouts to {output_dir}")


if __name__ == "__main__":
    main()
