"""Run local FASD-GRPO schema/reward/sample construction validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from dataset_io.io_utils import read_jsonl
from grpo_sdft.joint_loss_config import JointLossConfig
from grpo_sdft.trajectory_schema import GRPOSDFTSample, TrajectoryAction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--k-rollouts", type=int, default=4)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "grpo_sdft"))
    args = parser.parse_args()

    build_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_grpo_sdft_dataset.py"),
        "--split",
        args.split,
        "--k-rollouts",
        str(args.k_rollouts),
        "--output-dir",
        args.output_dir,
    ]
    print("+", " ".join(build_cmd))
    import subprocess

    subprocess.run(build_cmd, check=True)

    out_dir = Path(args.output_dir)
    trajectory_path = out_dir / f"trajectories_{args.split}.jsonl"
    sample_path = out_dir / f"fasd_grpo_samples_{args.split}.jsonl"
    trajectories = [TrajectoryAction.from_dict(record) for record in read_jsonl(trajectory_path)]
    samples = [GRPOSDFTSample(**_sample_kwargs(record)) for record in read_jsonl(sample_path)]
    summary = _validate_records(trajectories, samples, args.k_rollouts)
    config = JointLossConfig()
    config.validate()
    summary["joint_loss"] = config.formula()
    report_path = out_dir / f"validation_report_{args.split}.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def _sample_kwargs(record: dict[str, object]) -> dict[str, object]:
    from grpo_sdft.trajectory_schema import LossMasks

    data = dict(record)
    data["loss_masks"] = LossMasks.from_dict(data.get("loss_masks"))  # type: ignore[arg-type]
    return data


def _validate_records(actions: list[TrajectoryAction], samples: list[GRPOSDFTSample], k_rollouts: int) -> dict[str, object]:
    groups: dict[str, set[str]] = {}
    for action in actions:
        groups.setdefault(action.group_id, set()).add(action.rollout_id)
    bad_groups = {group_id: len(rollouts) for group_id, rollouts in groups.items() if len(rollouts) != k_rollouts}
    if bad_groups:
        raise SystemExit(f"bad rollout group sizes: {bad_groups}")

    sdft_samples = [action for action in actions if action.loss_masks.sdft]
    replay_samples = [action for action in actions if action.loss_masks.replay]
    missing_raw_sdft = [action for action in sdft_samples if not action.raw_evidence_ids]
    if missing_raw_sdft:
        raise SystemExit("SDFT samples without raw_evidence_ids were not filtered")

    if not any(action.failure_type != "none" for action in actions):
        raise SystemExit("expected at least one failed trajectory action")
    if not replay_samples:
        raise SystemExit("expected at least one successful replay sample")
    if len(samples) != len(actions):
        raise SystemExit(f"sample/action mismatch: {len(samples)} vs {len(actions)}")

    return {
        "num_actions": len(actions),
        "num_samples": len(samples),
        "num_groups": len(groups),
        "k_rollouts": k_rollouts,
        "num_sdft_samples": len(sdft_samples),
        "num_replay_samples": len(replay_samples),
        "advantages_min": min(action.advantage for action in actions),
        "advantages_max": max(action.advantage for action in actions),
        "status": "passed",
    }


if __name__ == "__main__":
    main()
