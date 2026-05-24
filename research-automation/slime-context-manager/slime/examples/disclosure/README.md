# Progressive Evidence Disclosure Training

This example demonstrates how to train a Progressive Evidence Disclosure policy using Slime.

## Overview

Based on the research paper "Learning When to Expand: Progressive Evidence Disclosure for Long-Context Reasoning", this implementation trains a selection policy that decides for each document span whether to:

- **EXPAND**: Show full raw text (when summary is insufficient)
- **SUMMARY**: Show only summary (when summary is sufficient)
- **HIDDEN**: Don't show in context (when irrelevant or budget-constrained)

## Quick Start

```bash
cd /root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/slime
bash examples/disclosure/run-qwen3-0.6b-disclosure.sh
```

## Training Modes

### 1. Review-SFT (Default)

Direct supervised fine-tuning from corrected selections:

```bash
TRAINING_MODE=review_sft bash examples/disclosure/run-qwen3-0.6b-disclosure.sh
```

### 2. OPD Distillation

On-policy distillation with hindsight rationale:

```bash
TRAINING_MODE=opd_distillation bash examples/disclosure/run-qwen3-0.6b-disclosure.sh
```

## Requirements

- Qwen3-0.6B model checkpoint in `/root/Qwen3-0.6B`
- Training data in JSONL format (see Data Format below)
- Ray cluster with at least 1 GPU

## Data Format

Training data should be in JSONL format with the following structure:

```json
{
  "sample_id": "unique_sample_id",
  "question": "What is the capital of France?",
  "spans": [
    {
      "span_id": "doc_1",
      "raw_text": "France is a country in Western Europe...",
      "summary": "France is in Western Europe. Capital is Paris.",
      "metadata": {"relevance": 0.95}
    }
  ],
  "gold_answer": "Paris",
  "metadata": {
    "max_budget_tokens": 4096,
    "target_selection": {"doc_1": "EXPAND"}
  }
}
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAINING_MODE` | Training mode (review_sft or opd_distillation) | review_sft |
| `DISCLOSURE_BUDGET` | Token budget for context rendering | 4096 |
| `TEACHER_MODEL_PATH` | Teacher model for OPD distillation | /root/Qwen3-4B |
| `MASTER_ADDR` | Ray master address | 127.0.0.1 |

## Key Components

- **`slime/rollout/disclosure/schemas.py`**: Data schemas for disclosure training
- **`slime/rollout/disclosure/renderer.py`**: Context renderer for answer prompts
- **`slime/rollout/disclosure/budget_allocator.py`**: Budget-aware span allocation
- **`slime/rollout/disclosure/selection_prompt.py`**: Selection prompt builder
- **`slime/rollout/disclosure/slime_rollout.py`**: Main rollout function
- **`slime/rollout/disclosure/opd_distillation.py`**: OPD distillation builder

## Output

Training checkpoints will be saved to `/root/Qwen3-0.6B_slime/`.

## Pipeline

```
Raw Data → Span Summaries → Selection Policy → Rendered Context → Answer → Review → Training Data
                                              ↑___________________________|
```

The selection policy learns from feedback to improve span state selection under budget constraints.

## References

- Roadmap: `progressive-disclosure-slime-roadmap.md`
- Paper: "Learning When to Expand: Progressive Evidence Disclosure for Long-Context Reasoning"
