import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.io_utils import read_jsonl, write_jsonl
from disclosure.budget_allocator import allocate_budget
from disclosure.contrastive_builder import differing_actions
from disclosure.metrics import evaluate_disclosure_predictions
from disclosure.renderer import render_prompt, render_span
from disclosure.schemas import ContextSpan, DisclosureDecision, DisclosureRollout, DisclosureTarget
from disclosure.slime_rollout import DisclosureSlimeRecordStub, _prepare_sample_for_slime
from disclosure.slime_sft_builder import build_slime_sft_record
from disclosure.summary_builder import audit_summaries, span_from_raw
from disclosure.teacher_review import heuristic_teacher_target


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1, len(str(text).split()) + 1))}


class DisclosureComponentTests(unittest.TestCase):
    def test_schema_summary_and_renderer(self):
        span = _span()
        span.validate()
        self.assertGreater(span.raw_tokens, 0)
        self.assertIn("50mg", span.raw_text)
        self.assertIn("RAW", render_span(span, "EXPAND"))
        self.assertIn("SUMMARY", render_span(span, "KEEP"))
        self.assertEqual(render_span(span, "DROP"), "")
        prompt = render_prompt(span.question, [span], [DisclosureDecision(span.span_id, "EXPAND")])
        self.assertIn("Answer:", prompt)

    def test_summary_audit(self):
        span = span_from_raw(
            task_id="task",
            span_id="s1",
            source_id="src",
            question="What dosage?",
            raw_text="Dosage changed from 5mg to 50mg.",
            split="train",
        )
        audit = audit_summaries([span]).to_dict()
        self.assertEqual(audit["num_spans"], 1)
        self.assertEqual(audit["cue_preservation_rate"], 1.0)

    def test_budget_allocator_never_exceeds_budget(self):
        spans = [
            _span(span_id="s1", raw_text="one two three four five six seven eight nine ten"),
            _span(span_id="s2", raw_text="alpha beta gamma delta epsilon zeta eta theta iota kappa"),
        ]
        decisions = [
            DisclosureDecision("s1", "EXPAND", {"EXPAND": 5.0, "KEEP": 1.0, "DROP": 0.0}),
            DisclosureDecision("s2", "EXPAND", {"EXPAND": 4.0, "KEEP": 1.0, "DROP": 0.0}),
        ]
        record = allocate_budget(spans, decisions, rollout_id="r1", token_budget=12, reserved_prompt_tokens=2)
        self.assertLessEqual(record.budget_used, 12)
        self.assertFalse(record.budget_violation)
        self.assertEqual(len(record.final_actions), 2)

    def test_knapsack_allocator_runs(self):
        spans = [_span(span_id="s1"), _span(span_id="s2")]
        decisions = [DisclosureDecision("s1", "EXPAND"), DisclosureDecision("s2", "KEEP")]
        record = allocate_budget(spans, decisions, rollout_id="r1", token_budget=100, allocator="knapsack_analysis")
        self.assertEqual(record.allocator, "knapsack_analysis")
        self.assertLessEqual(record.budget_used, 100)

    def test_teacher_target_validation_and_slime_export(self):
        span = _span()
        decision = DisclosureDecision(span.span_id, "KEEP")
        target = heuristic_teacher_target(span, decision, rollout_id="r1")
        target.validate(span)
        record = build_slime_sft_record(span, target, token_budget=4096).to_dict()
        self.assertIn("prompt", record)
        self.assertEqual(json.loads(record["label"])["action"], target.target_action)
        self.assertEqual(record["metadata"]["span_id"], span.span_id)

    def test_teacher_rejects_test_split(self):
        span = _span(split="test")
        target = DisclosureTarget(
            decision_id="d1",
            task_id=span.task_id,
            rollout_id="r1",
            span_id=span.span_id,
            source_id=span.source_id,
            student_action="KEEP",
            target_action="EXPAND",
            label_source="teacher_only",
            quoted_summary=span.summary_text,
            quoted_raw=span.raw_text,
            verifiable_reason="Needs exact value.",
            split="test",
        )
        with self.assertRaises(ValueError):
            target.validate(span)

    def test_contrastive_and_metrics(self):
        rollout_a = DisclosureRollout("r1", "task", "a", 100, [DisclosureDecision("s1", "KEEP")])
        rollout_b = DisclosureRollout("r2", "task", "b", 100, [DisclosureDecision("s1", "EXPAND")])
        self.assertEqual(len(differing_actions([rollout_a, rollout_b])), 1)
        target = heuristic_teacher_target(_span(), DisclosureDecision("s1", "KEEP"), rollout_id="r1").to_dict()
        metrics = evaluate_disclosure_predictions([target], [{"decision_id": target["decision_id"], "pred_action": target["target_action"]}])
        self.assertEqual(metrics["action_accuracy"], 1.0)

    def test_jsonl_roundtrip_for_slime_record(self):
        span = _span()
        target = heuristic_teacher_target(span, DisclosureDecision(span.span_id, "KEEP"), rollout_id="r1")
        record = build_slime_sft_record(span, target, token_budget=4096).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disclosure.jsonl"
            write_jsonl(path, [record])
            loaded = read_jsonl(path)
        self.assertEqual(loaded[0]["metadata"]["source_id"], span.source_id)

    def test_slime_rollout_prepares_response_mask(self):
        import disclosure.slime_rollout as slime_rollout

        old = slime_rollout._TOKENIZER
        slime_rollout._TOKENIZER = FakeTokenizer()
        try:
            sample = DisclosureSlimeRecordStub(
                prompt="Question: q",
                label='{"action":"EXPAND"}',
                metadata={"task_id": "task", "span_id": "s1", "source_id": "src", "token_budget": 100},
            )
            prepared = _prepare_sample_for_slime(type("Args", (), {"max_length": 128})(), sample)
        finally:
            slime_rollout._TOKENIZER = old
        self.assertGreater(prepared.response_length, 0)
        self.assertEqual(sum(prepared.loss_mask), prepared.response_length)
        self.assertIn("COMPLETED", str(prepared.status).upper())


def _span(span_id="s1", raw_text="The protocol changed dosage from 5mg to 50mg after review.", split="train"):
    return span_from_raw(
        task_id="task",
        span_id=span_id,
        source_id=f"src_{span_id}",
        question="What dosage was used after the protocol changed?",
        raw_text=raw_text,
        split=split,
        max_summary_words=5,
    )


if __name__ == "__main__":
    unittest.main()
