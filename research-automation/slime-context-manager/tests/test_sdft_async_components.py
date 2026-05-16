import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.io_utils import write_jsonl
from grpo_sdft.async_sdft_rollout import generate_rollout_sdft
from grpo_sdft.hf_dataset_builder import build_hotpotqa_records
from grpo_sdft.sdft_hint_extractor import extract_hints_async, failed_actions
from grpo_sdft.sdft_sample_builder import build_sdft_samples


def hotpot_row():
    return {
        "id": "hpqa_sdft_1",
        "question": "Where did Person B study?",
        "context": {
            "title": ["Support A", "Support B", "Distractor C", "Distractor D"],
            "sentences": [
                ["Person B attended University C."],
                ["University C is in City D."],
                ["This paragraph is about unrelated sports awards."],
                ["A short irrelevant note."],
            ],
        },
        "supporting_facts": {"title": ["Support A", "Support B"], "sent_id": [0, 0]},
    }


class AsyncSDFTComponentTests(unittest.TestCase):
    def test_async_hint_extraction_from_failed_trajectories(self):
        actions, _, _ = build_hotpotqa_records([hotpot_row()], k_rollouts=4)
        failures = failed_actions(actions)
        hints = asyncio.run(extract_hints_async(failures, concurrency=2, votes_per_action=3))
        self.assertGreater(len(failures), 0)
        self.assertEqual(len(hints), len(failures))
        self.assertIn("raw_evidence_id", hints[0].hint)
        self.assertEqual(len(hints[0].votes), 3)
        self.assertGreaterEqual(hints[0].confidence, 0.7)

    def test_sdft_samples_and_rollout_groups(self):
        actions, _, _ = build_hotpotqa_records([hotpot_row()], k_rollouts=4)
        hints = asyncio.run(extract_hints_async(failed_actions(actions), concurrency=2))
        samples = build_sdft_samples(actions, hints)
        self.assertGreater(len(samples), 0)
        self.assertTrue(all(sample.loss_masks.sdft for sample in samples))
        self.assertIn("teacher_prompt", samples[0].metadata)
        self.assertIn("hint_votes", samples[0].metadata)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sdft.jsonl"
            write_jsonl(path, [sample.to_dict() for sample in samples])
            args = SimpleNamespace(sdft_samples_path=str(path), rollout_batch_size=2)
            output = generate_rollout_sdft(args, rollout_id=7)
        self.assertEqual(len(output.samples), 2)
        self.assertEqual(output.metrics["rollout/sdft_groups"], 2.0)


if __name__ == "__main__":
    unittest.main()
