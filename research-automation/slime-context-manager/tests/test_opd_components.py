import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opd.action_parser import parse_memory_action, render_memory_action
from opd.context_policy_opd_server import NextState, OPDRecorder, PendingTurn
from opd.hindsight_judge import HintVote, make_vote, parse_judge_result, select_best_hint


class HindsightJudgeTests(unittest.TestCase):
    def test_parse_positive_hint(self):
        raw = r"\boxed{1}\n[HINT_START]turn_id=4 memory_id=mem1 raw_evidence_id=doc1 suggested_action: KEEP verifiable_reason: needed later[HINT_END]"
        score, hint = parse_judge_result(raw)
        self.assertEqual(score, 1)
        self.assertIn("suggested_action", hint)

    def test_parse_negative_without_hint(self):
        score, hint = parse_judge_result(r"Bad action. \boxed{-1}")
        self.assertEqual(score, -1)
        self.assertEqual(hint, "")

    def test_select_longest_positive_hint(self):
        votes = [
            HintVote(0, 1, "short"),
            HintVote(1, -1, ""),
            HintVote(2, 1, "turn_id=1 memory_id=mem raw_evidence_id=raw suggested_action: KEEP verifiable_reason: useful"),
        ]
        selected = select_best_hint(votes)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.vote_id, 2)


class ActionParserTests(unittest.TestCase):
    def test_valid_json_action(self):
        parsed = parse_memory_action('{"action":"keep","target_ids":["mem1"]}')
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.action, "KEEP")
        self.assertEqual(parsed.target_ids, ["mem1"])

    def test_embedded_json_action(self):
        parsed = parse_memory_action('Action: {"tool":"memory.retrieve","memory_id":"mem2"}')
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.action, "RETRIEVE")
        self.assertEqual(parsed.target_ids, ["mem2"])

    def test_unknown_action_is_invalid(self):
        parsed = parse_memory_action('{"action":"FORGET","target_ids":["mem1"]}')
        self.assertFalse(parsed.valid)
        self.assertIn("unknown_action:FORGET", parsed.errors)

    def test_render_action(self):
        rendered = render_memory_action("pin", ["mem3"])
        self.assertEqual(rendered, '{"action":"PIN","target_ids":["mem3"]}')


class OPDSampleBuilderTests(unittest.TestCase):
    def test_sample_alignment_and_raw_evidence_requirement(self):
        recorder = OPDRecorder(require_raw_evidence=True)
        recorder.record_turn(
            PendingTurn(
                session_id="ep1",
                turn_id=4,
                memory_id="mem17",
                student_prompt="state",
                student_action_text='{"action":"ARCHIVE","target_ids":["mem17"]}',
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5],
                rollout_log_probs=[-0.1],
                metadata={"task_family": "hotpotqa", "raw_evidence_id": "doc_4_sent_2"},
            )
        )
        sample = recorder.receive_next_state(
            session_id="ep1",
            turn_id=4,
            next_state=NextState(role="env", content="missed supporting fact"),
            votes=[
                make_vote(
                    r"\boxed{1}[HINT_START]turn_id=4 memory_id=mem17 raw_evidence_id=doc_4_sent_2 suggested_action: KEEP verifiable_reason: needed later[HINT_END]"
                )
            ],
            teacher_log_probs=[-0.2, -0.3, -0.4],
        )
        self.assertIsNotNone(sample)
        self.assertEqual(sample.response_length, 2)
        self.assertEqual(sample.loss_mask, [1, 1])
        self.assertEqual(sample.rollout_log_probs, [-0.1, 0.0])
        self.assertEqual(sample.teacher_log_probs, [-0.2, -0.3])
        self.assertEqual(sample.teacher_action, "KEEP")

    def test_missing_raw_evidence_excluded_by_default(self):
        recorder = OPDRecorder(require_raw_evidence=True)
        recorder.record_turn(
            PendingTurn(
                session_id="ep1",
                turn_id=1,
                memory_id="mem1",
                student_prompt="state",
                student_action_text='{"action":"DROP","target_ids":["mem1"]}',
                prompt_ids=[1],
                response_ids=[2],
            )
        )
        sample = recorder.receive_next_state(
            session_id="ep1",
            turn_id=1,
            next_state={"role": "env", "content": "bad"},
            votes=[
                make_vote(
                    r"\boxed{1}[HINT_START]turn_id=1 memory_id=mem1 raw_evidence_id=raw suggested_action: KEEP verifiable_reason: useful[HINT_END]"
                )
            ],
        )
        self.assertIsNone(sample)


if __name__ == "__main__":
    unittest.main()
