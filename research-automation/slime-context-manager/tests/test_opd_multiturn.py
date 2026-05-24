import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opd.context_policy_opd_server import NextState, OPDRecorder, PendingTurn
from opd.hindsight_judge import make_vote


class OPDMultiTurnTests(unittest.TestCase):
    def test_finish_session_clears_pending_and_records_episode_reward(self):
        recorder = OPDRecorder(require_raw_evidence=False)
        recorder.record_turn(
            PendingTurn(
                session_id="ep9",
                turn_id=1,
                memory_id="mem1",
                student_prompt="state1",
                student_action_text='{"action":"KEEP","target_ids":["mem1"]}',
                prompt_ids=[1],
                response_ids=[2],
            )
        )
        recorder.record_turn(
            PendingTurn(
                session_id="ep9",
                turn_id=2,
                memory_id="mem2",
                student_prompt="state2",
                student_action_text='{"action":"ARCHIVE","target_ids":["mem2"]}',
                prompt_ids=[3],
                response_ids=[4],
            )
        )

        sample = recorder.receive_next_state(
            session_id="ep9",
            turn_id=1,
            next_state=NextState(role="env", content="progress"),
            votes=[
                make_vote(
                    r"\boxed{1}[HINT_START]turn_id=1 memory_id=mem1 raw_evidence_id=raw1 suggested_action: KEEP verifiable_reason: progressed[HINT_END]"
                )
            ],
            teacher_log_probs=[-0.4],
        )
        self.assertIsNotNone(sample)
        episode = recorder.finish_session("ep9", final_reward=1.0, evidence_recall=0.86, token_cost=7210)

        self.assertNotIn("ep9", recorder.pending)
        self.assertEqual(episode.final_reward, 1.0)
        self.assertEqual(episode.evidence_recall, 0.86)
        self.assertEqual(episode.token_cost, 7210)
        self.assertEqual(episode.turn_rewards[0]["turn_id"], 1)
        self.assertEqual(episode.turn_rewards[0]["process_reward"], 1.0)

    def test_negative_vote_records_process_reward_without_training_sample(self):
        recorder = OPDRecorder(require_raw_evidence=False)
        recorder.record_turn(
            PendingTurn(
                session_id="ep2",
                turn_id=1,
                memory_id="mem1",
                student_prompt="state",
                student_action_text='{"action":"DROP","target_ids":["mem1"]}',
                prompt_ids=[1],
                response_ids=[2],
            )
        )
        sample = recorder.receive_next_state(
            session_id="ep2",
            turn_id=1,
            next_state={"role": "env", "content": "failed"},
            votes=[make_vote(r"\boxed{-1}")],
        )
        self.assertIsNone(sample)
        episode = recorder.episodes["ep2"]
        self.assertEqual(episode.turn_rewards[0]["process_reward"], -1.0)


if __name__ == "__main__":
    unittest.main()
