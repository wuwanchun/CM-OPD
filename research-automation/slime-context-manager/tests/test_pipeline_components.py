import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from context_manager.rule_policy import RulePolicy
from context_manager.schemas import ContextActionSample
from data.adapters import ToyAdapter
from data.io_utils import read_jsonl, write_jsonl
from evaluation.metrics import evaluate_action_predictions


class PipelineComponentTests(unittest.TestCase):
    def test_toy_adapter_and_rule_policy(self):
        records = ToyAdapter().build_samples(split="test")
        self.assertEqual(len(records), 5)
        sample = ContextActionSample.from_dict(records[0])
        pred = RulePolicy().predict(sample)
        self.assertEqual(pred.pred_action, "PIN")

    def test_metrics_evaluate_predictions(self):
        records = ToyAdapter().build_samples(split="test")
        predictions = []
        policy = RulePolicy()
        for record in records:
            predictions.append(policy.predict(ContextActionSample.from_dict(record)).to_dict())
        metrics = evaluate_action_predictions(records, predictions)
        self.assertEqual(metrics["num_samples"], 5)
        self.assertGreaterEqual(metrics["action_accuracy"], 0.0)
        self.assertLessEqual(metrics["invalid_output_rate"], 1.0)

    def test_jsonl_roundtrip(self):
        records = ToyAdapter().build_samples(split="train", limit=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            write_jsonl(path, records)
            loaded = read_jsonl(path)
        self.assertEqual(records, loaded)


if __name__ == "__main__":
    unittest.main()
