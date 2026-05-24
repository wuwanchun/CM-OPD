import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset_io.io_utils import read_jsonl, write_jsonl
from disclosure.budget_allocator import allocate_budget
from disclosure.contrastive_builder import differing_actions
from disclosure.metrics import evaluate_disclosure_predictions
from disclosure.renderer import render_prompt, render_span
from disclosure.schemas import ContextSpan, DisclosureDecision, DisclosureRollout, DisclosureTarget
from disclosure.slime_rollout import DisclosureSlimeRecordStub, _prepare_sample_for_slime
from disclosure.slime_sft_builder import build_slime_sft_record
from disclosure.selection_prompt import build_selection_prompt, parse_selection_response
from disclosure.slime_rollout import generate_selection_answer_review_rollout
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
        self.assertIn("EXPAND", render_span(span, "EXPAND"))
        self.assertIn("SUMMARY", render_span(span, "SUMMARY"))
        self.assertEqual(render_span(span, "HIDDEN"), "")
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
            DisclosureDecision("s1", "EXPAND", {"EXPAND": 5.0, "SUMMARY": 1.0, "HIDDEN": 0.0}),
            DisclosureDecision("s2", "EXPAND", {"EXPAND": 4.0, "SUMMARY": 1.0, "HIDDEN": 0.0}),
        ]
        record = allocate_budget(spans, decisions, rollout_id="r1", token_budget=12, reserved_prompt_tokens=2)
        self.assertLessEqual(record.budget_used, 12)
        self.assertFalse(record.budget_violation)
        self.assertEqual(len(record.final_actions), 2)

    def test_knapsack_allocator_runs(self):
        spans = [_span(span_id="s1"), _span(span_id="s2")]
        decisions = [DisclosureDecision("s1", "EXPAND"), DisclosureDecision("s2", "SUMMARY")]
        record = allocate_budget(spans, decisions, rollout_id="r1", token_budget=100, allocator="knapsack_analysis")
        self.assertEqual(record.allocator, "knapsack_analysis")
        self.assertLessEqual(record.budget_used, 100)

    def test_teacher_target_validation_and_slime_export(self):
        span = _span()
        decision = DisclosureDecision(span.span_id, "SUMMARY")
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
            student_action="SUMMARY",
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
        rollout_a = DisclosureRollout("r1", "task", "a", 100, [DisclosureDecision("s1", "SUMMARY")])
        rollout_b = DisclosureRollout("r2", "task", "b", 100, [DisclosureDecision("s1", "EXPAND")])
        self.assertEqual(len(differing_actions([rollout_a, rollout_b])), 1)
        target = heuristic_teacher_target(_span(), DisclosureDecision("s1", "SUMMARY"), rollout_id="r1").to_dict()
        metrics = evaluate_disclosure_predictions([target], [{"decision_id": target["decision_id"], "pred_action": target["target_action"]}])
        self.assertEqual(metrics["action_accuracy"], 1.0)

    def test_jsonl_roundtrip_for_slime_record(self):
        span = _span()
        target = heuristic_teacher_target(span, DisclosureDecision(span.span_id, "SUMMARY"), rollout_id="r1")
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

    def test_selection_prompt_contract_rejects_old_states(self):
        span = _span()
        prompt = build_selection_prompt([span], token_budget=128)
        self.assertIn("EXPAND, SUMMARY, or HIDDEN", prompt)
        parsed = parse_selection_response(json.dumps({span.span_id: "SUMMARY"}), [span])
        self.assertEqual(parsed[0].action, "SUMMARY")
        with self.assertRaises(ValueError):
            parse_selection_response(json.dumps({span.span_id: "KEEP"}), [span])

    def test_slime_rollout_builds_selection_review_samples(self):
        import disclosure.slime_rollout as slime_rollout

        old = slime_rollout._TOKENIZER
        slime_rollout._TOKENIZER = FakeTokenizer()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                spans_path = Path(tmp) / "spans_train.jsonl"
                write_jsonl(spans_path, [_span().to_dict()])
                args = type(
                    "Args",
                    (),
                    {
                        "prompt_data": str(spans_path),
                        "rollout_batch_size": 1,
                        "n_samples_per_prompt": 1,
                        "max_length": 256,
                        "hf_checkpoint": "",
                    },
                )()
                output = generate_selection_answer_review_rollout(args, rollout_id=0, data_buffer=None)
        finally:
            slime_rollout._TOKENIZER = old
        sample = output.samples[0][0]
        self.assertEqual(sum(sample.loss_mask), sample.response_length)
        self.assertIn("selection_id", sample.metadata)
        self.assertIn("answer_id", sample.metadata)
        self.assertIn("token_budget", sample.metadata)


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


class TestOnPolicyBuffer(unittest.TestCase):
    """Tests for on_policy_buffer module."""

    def test_replay_record_creation(self):
        from disclosure.on_policy_buffer import ReplayRecord

        record = ReplayRecord(
            record_id="test_1",
            task_id="task_1",
            span_id="span_1",
            source_id="src_1",
            selection_id="sel_1",
            prompt="Test prompt",
            response='{"action":"EXPAND"}',
            target_state="EXPAND",
            accepted=True,
            iteration=0,
        )
        self.assertEqual(record.record_id, "test_1")
        self.assertTrue(record.accepted)

        # Test roundtrip
        d = record.to_dict()
        loaded = ReplayRecord.from_dict(d)
        self.assertEqual(loaded.record_id, record.record_id)
        self.assertEqual(loaded.target_state, "EXPAND")

    def test_replay_buffer_add_and_sample(self):
        from disclosure.on_policy_buffer import OnPolicyReplayBuffer, ReplayRecord

        buffer = OnPolicyReplayBuffer(max_size=100, replay_ratio=0.5, seed=42)

        # Add records
        records = [
            ReplayRecord(
                record_id=f"rec_{i}",
                task_id=f"task_{i % 3}",
                span_id=f"span_{i}",
                source_id=f"src_{i}",
                selection_id=f"sel_{i}",
                prompt=f"Prompt {i}",
                response='{"action":"SUMMARY"}',
                target_state="SUMMARY",
                accepted=True,
                iteration=i // 5,
            )
            for i in range(10)
        ]
        added = buffer.add(records)
        self.assertEqual(added, 10)
        self.assertEqual(len(buffer), 10)

        # Sample from buffer
        sampled = buffer.sample(batch_size=4, current_iteration=-1)
        self.assertLessEqual(len(sampled), 4)

        # Sample excluding current iteration
        sampled = buffer.sample(batch_size=4, current_iteration=1)
        for s in sampled:
            self.assertNotEqual(s.iteration, 1)

    def test_replay_buffer_max_size_eviction(self):
        from disclosure.on_policy_buffer import OnPolicyReplayBuffer, ReplayRecord

        buffer = OnPolicyReplayBuffer(max_size=5, replay_ratio=0.5)

        # Add more records than max_size
        for i in range(10):
            buffer.add([
                ReplayRecord(
                    record_id=f"rec_{i}",
                    task_id="task",
                    span_id=f"span_{i}",
                    source_id="src",
                    selection_id="sel",
                    prompt="prompt",
                    response="response",
                    target_state="SUMMARY",
                    accepted=True,
                    iteration=0,
                )
            ])

        # Should be capped at max_size
        self.assertEqual(len(buffer), 5)

    def test_replay_buffer_save_load(self):
        from disclosure.on_policy_buffer import OnPolicyReplayBuffer, ReplayRecord

        import tempfile
        
        buffer = OnPolicyReplayBuffer(max_size=100)
        records = [
            ReplayRecord(
                record_id=f"rec_{i}",
                task_id=f"task_{i}",
                span_id=f"span_{i}",
                source_id="src",
                selection_id="sel",
                prompt="prompt",
                response="response",
                target_state="EXPAND",
                accepted=True,
                iteration=0,
            )
            for i in range(5)
        ]
        buffer.add(records)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay_buffer.jsonl"
            saved = buffer.save(path)
            self.assertEqual(saved, 5)

            # Load into new buffer
            new_buffer = OnPolicyReplayBuffer(max_size=100)
            loaded = new_buffer.load(path)
            self.assertEqual(loaded, 5)
            self.assertEqual(len(new_buffer), 5)

    def test_replay_buffer_stats(self):
        from disclosure.on_policy_buffer import OnPolicyReplayBuffer, ReplayRecord

        buffer = OnPolicyReplayBuffer(max_size=100, replay_ratio=0.3)
        records = [
            ReplayRecord(
                record_id=f"rec_{i}",
                task_id=f"task_{i % 2}",
                span_id=f"span_{i}",
                source_id="src",
                selection_id="sel",
                prompt="prompt",
                response="response",
                target_state="SUMMARY",
                accepted=True,
                iteration=i // 3,
            )
            for i in range(6)
        ]
        buffer.add(records)

        stats = buffer.stats()
        self.assertEqual(stats["size"], 6)
        self.assertEqual(stats["replay_ratio"], 0.3)
        self.assertIn("iteration_counts", stats)
        self.assertEqual(stats["unique_tasks"], 2)

    def test_build_replay_records_from_targets(self):
        from disclosure.on_policy_buffer import build_replay_records_from_targets
        from disclosure.schemas import ReviewSFTTarget

        targets = [
            ReviewSFTTarget(
                target_id="t1",
                task_id="task_1",
                span_id="span_1",
                source_id="src_1",
                selection_id="sel_1",
                answer_id="ans_1",
                student_state="SUMMARY",
                target_state="EXPAND",
                quoted_summary="summary text",
                verifiable_reason="needs expansion",
                token_budget=4096,
            )
        ]

        records = build_replay_records_from_targets(targets, iteration=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].iteration, 1)
        self.assertEqual(records[0].target_state, "EXPAND")

    def test_mix_replay_with_onpolicy(self):
        from disclosure.on_policy_buffer import (
            OnPolicyReplayBuffer,
            ReplayRecord,
            mix_replay_with_onpolicy,
        )

        buffer = OnPolicyReplayBuffer(max_size=100, replay_ratio=0.5, seed=42)
        
        # Add historical records
        historical = [
            ReplayRecord(
                record_id=f"hist_{i}",
                task_id="task",
                span_id=f"span_{i}",
                source_id="src",
                selection_id="sel",
                prompt="prompt",
                response="response",
                target_state="SUMMARY",
                accepted=True,
                iteration=0,
            )
            for i in range(10)
        ]
        buffer.add(historical)

        # Create on-policy records
        onpolicy = [
            ReplayRecord(
                record_id=f"onpolicy_{i}",
                task_id="task",
                span_id=f"span_new_{i}",
                source_id="src",
                selection_id="sel",
                prompt="prompt",
                response="response",
                target_state="EXPAND",
                accepted=True,
                iteration=1,
            )
            for i in range(5)
        ]

        mixed = mix_replay_with_onpolicy(onpolicy, buffer, batch_size=6, current_iteration=1)
        self.assertLessEqual(len(mixed), 6)


class TestOPDLoss(unittest.TestCase):
    """Tests for slime_opd_loss module."""

    def test_action_index_conversion(self):
        from disclosure.slime_opd_loss import action_to_index, index_to_action

        self.assertEqual(action_to_index("EXPAND"), 0)
        self.assertEqual(action_to_index("SUMMARY"), 1)
        self.assertEqual(action_to_index("HIDDEN"), 2)
        self.assertEqual(action_to_index("unknown"), 1)  # default

        self.assertEqual(index_to_action(0), "EXPAND")
        self.assertEqual(index_to_action(1), "SUMMARY")
        self.assertEqual(index_to_action(2), "HIDDEN")

    def test_compute_opd_loss_basic(self):
        import torch
        from disclosure.slime_opd_loss import compute_opd_loss, OPDLossConfig

        # Simulate student logits [batch=2, num_actions=3]
        student_logits = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]])
        target_actions = torch.tensor([0, 1])  # EXPAND, SUMMARY

        config = OPDLossConfig(ce_weight=1.0, kl_weight=0.0)
        loss_dict = compute_opd_loss(student_logits, target_actions, config=config)

        self.assertIn("total_loss", loss_dict)
        self.assertIn("ce_loss", loss_dict)
        self.assertGreater(loss_dict["ce_loss"].item(), 0)

    def test_compute_opd_loss_with_teacher(self):
        import torch
        from disclosure.slime_opd_loss import compute_opd_loss, OPDLossConfig

        student_logits = torch.tensor([[1.5, 1.0, 0.5]])
        target_actions = torch.tensor([0])
        teacher_logprobs = torch.tensor([[0.6, 0.3, 0.1]])  # Soft teacher distribution

        config = OPDLossConfig(ce_weight=1.0, kl_weight=0.5, teacher_temperature=2.0)
        loss_dict = compute_opd_loss(
            student_logits,
            target_actions,
            teacher_logprobs,
            config=config,
        )

        self.assertIn("kl_loss", loss_dict)
        self.assertGreater(loss_dict["kl_loss"].item(), 0)

    def test_compute_full_opd_loss(self):
        import torch
        from disclosure.slime_opd_loss import compute_full_opd_loss, OPDLossConfig

        student_logits = torch.tensor([[2.0, 1.0, 0.5], [0.5, 1.0, 2.0]])
        batch = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "labels": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "action_labels": torch.tensor([0, 2]),  # EXPAND, HIDDEN
        }

        config = OPDLossConfig()
        output = compute_full_opd_loss(student_logits, batch, config=config)

        self.assertIn("total_loss", output)
        self.assertIn("metrics", output)
        self.assertIn("action_accuracy", output.metrics)

    def test_build_opd_training_batch(self):
        import torch
        from disclosure.slime_opd_loss import build_opd_training_batch

        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                words = text.split()
                return {"input_ids": list(range(1, len(words) + 1))}

        records = [
            {
                "prompt": "Question: What is this?",
                "response": '{"action":"EXPAND"}',
                "target_state": "EXPAND",
            },
            {
                "prompt": "Question: Is this relevant?",
                "response": '{"action":"SUMMARY"}',
                "target_state": "SUMMARY",
            },
        ]

        batch = build_opd_training_batch(records, FakeTokenizer(), max_length=50)
        
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertIn("action_labels", batch)
        self.assertEqual(batch["action_labels"].shape[0], 2)
        self.assertEqual(batch["action_labels"][0].item(), 0)  # EXPAND
        self.assertEqual(batch["action_labels"][1].item(), 1)  # SUMMARY

    def test_replay_augmented_loss(self):
        import torch
        from disclosure.slime_opd_loss import compute_replay_augmented_loss

        onpolicy_loss = torch.tensor(1.0)
        replay_loss = torch.tensor(0.5)

        combined = compute_replay_augmented_loss(onpolicy_loss, replay_loss, replay_weight=0.3)
        expected = 0.7 * 1.0 + 0.3 * 0.5
        self.assertAlmostEqual(combined.item(), expected, places=5)
