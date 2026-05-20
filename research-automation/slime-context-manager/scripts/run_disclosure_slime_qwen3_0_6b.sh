#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CHECKPOINT="${HF_CHECKPOINT:-$ROOT/models/qwen3-0.6B}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT/data/slime/disclosure_train.jsonl}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/disclosure_policy_qwen3_0_6b}"

python "$ROOT/scripts/train_disclosure_slime.py" \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --train-jsonl "$TRAIN_JSONL" \
  --output-dir "$SAVE_DIR" \
  --execute
