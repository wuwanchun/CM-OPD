import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from grpo_sdft.grpo_sample_builder import build_training_sample
from grpo_sdft.hindsight_relabeler import apply_hindsight_masks
from grpo_sdft.hf_dataset_builder import build_hotpotqa_records
from grpo_sdft.reward_decomposer import assign_segment_and_action_rewards, compute_group_advantages
from grpo_sdft.segmenter import group_actions_by_rollout, segment_rollout
from grpo_sdft.trajectory_schema import TrajectoryAction


def make_action(
    rollout_idx: int,
    *,
    reward: float,
    failure_type: str = "none",
    raw=True,
    confidence=0.9,
    student="KEEP",
    teacher="KEEP",
):
    return TrajectoryAction(
        task_id="task_1",
        rollout_id=f"task_1_r{rollout_idx:02d}",
        group_id="task_1",
        segment_id="seg_001",
        turn_id=1,
        action_type="memory_action",
        student_action=student,
        teacher_action=teacher,
        episode_reward=reward,
        failure_type=failure_type,
        raw_evidence_ids=["raw_1"] if raw else [],
        hindsight_hint="turn_id=1 memory_id=mem raw_evidence_id=raw_1 suggested_action: KEEP verifiable_reason: test",
        confidence=confidence,
    )


class GRPOSDFTComponentTests(unittest.TestCase):
    def test_grpo_advantage_normalized_within_group(self):
        actions = [make_action(0, reward=1.0), make_action(1, reward=0.0, failure_type="missing_evidence", student="DROP")]
        assign_segment_and_action_rewards(actions)
        rollouts = group_actions_by_rollout(actions)
        compute_group_advantages(rollouts)
        advantages = [action.advantage for action in actions]
        self.assertGreater(advantages[0], 0)
        self.assertLess(advantages[1], 0)

    def test_segment_reward_is_assigned_to_actions(self):
        action = make_action(0, reward=1.0)
        assign_segment_and_action_rewards([action])
        rollout = group_actions_by_rollout([action])[0]
        segment = segment_rollout(rollout)[0]
        self.assertEqual(segment.segment_id, "seg_001")
        self.assertEqual(action.segment_reward, segment.segment_reward)

    def test_confidence_and_raw_evidence_gate_sdft(self):
        accepted = make_action(0, reward=0.0, failure_type="missing_evidence", student="DROP", teacher="KEEP")
        missing_raw = make_action(1, reward=0.0, failure_type="format_error", raw=False, student="bad", teacher="KEEP")
        low_confidence = make_action(
            2,
            reward=0.0,
            failure_type="missing_evidence",
            confidence=0.2,
            student="DROP",
            teacher="KEEP",
        )
        apply_hindsight_masks([accepted, missing_raw, low_confidence])
        self.assertTrue(accepted.loss_masks.sdft)
        self.assertFalse(missing_raw.loss_masks.sdft)
        self.assertFalse(low_confidence.loss_masks.sdft)

    def test_successful_trajectory_enters_replay(self):
        action = make_action(0, reward=1.0, failure_type="none", student="KEEP", teacher="KEEP")
        apply_hindsight_masks([action])
        self.assertTrue(action.loss_masks.replay)
        self.assertTrue(action.loss_masks.sdft)

    def test_training_sample_preserves_masks_and_rewards(self):
        action = make_action(0, reward=1.0)
        assign_segment_and_action_rewards([action])
        apply_hindsight_masks([action])
        sample = build_training_sample(action)
        data = sample.to_dict()
        self.assertEqual(data["group_id"], "task_1")
        self.assertIn("reward", data)
        self.assertTrue(data["loss_masks"]["replay"])

    def test_unknown_action_type_rejected(self):
        data = make_action(0, reward=1.0).to_dict()
        data["action_type"] = "unknown"
        with self.assertRaises(ValueError):
            TrajectoryAction.from_dict(data)

    def test_hotpotqa_hf_row_builds_k_rollout_group(self):
        row = {
            "id": "hpqa_1",
            "question": "Where did Person B study?",
            "context": {
                "title": ["Support A", "Distractor B", "Distractor C", "Distractor D"],
                "sentences": [
                    ["Person B attended University C."],
                    ["This paragraph is about unrelated sports awards."],
                    ["Another unrelated document."],
                    ["A short irrelevant note."],
                ],
            },
            "supporting_facts": {"title": ["Support A"], "sent_id": [0]},
        }
        actions, segments, samples = build_hotpotqa_records([row], k_rollouts=4)
        self.assertEqual(len({action.group_id for action in actions}), 1)
        self.assertEqual(len({action.rollout_id for action in actions}), 4)
        self.assertEqual(len(segments), 4)
        self.assertEqual(len(samples), 4)
        self.assertTrue(any(action.metadata["is_supporting"] for action in actions))
        self.assertTrue(any(action.loss_masks.sdft for action in actions))
        self.assertGreater(max(action.advantage for action in actions), 0)


if __name__ == "__main__":
    unittest.main()
