# External Policy Slime Training And Evaluation Plan

_Implementation plan for the external context-manager policy, with OpenClaw-RL style OPD self-training hooks._

---

## 1. Goal

This implementation targets the external policy version of the project. The main agent remains frozen. The trainable component is a lightweight context manager that chooses memory lifecycle actions:

```text
KEEP / COMPRESS / ARCHIVE / RETRIEVE / UPDATE / DROP / PIN
```

The first runnable path is `Action OPD`: collect context-policy actions, wait for next-state feedback, extract hindsight hints, and train on corrected memory actions. `Token OPD` and `Top-K OPD` are implemented as compatible extension points for slime, but GRPO is not executed in the first version.

## 2. OPD Self-Training For External Context Policy

The OPD path follows the OpenClaw-RL pattern, adapted from full agent responses to external memory actions:

```text
context state s_t
-> current external policy outputs memory action a_t
-> store prompt_ids, response_ids, rollout_log_probs, memory_id, turn_id
-> next_state arrives from env / tool / evaluator
-> judge(a_t, next_state) runs m votes
-> if positive evidence-grounded hint exists:
     append hint to original prompt
     query teacher log-probs or corrected action
     create slime-compatible Sample
   else:
     drop sample or keep process-reward-only record
```

Every accepted training sample must be evidence-grounded by default. It needs `raw_evidence_id`, `turn_id`, `memory_id`, `student_action`, `teacher_action`, and `verifiable_reason`. Samples without evidence ids are still written to diagnostics, but excluded from OPD training unless `require_raw_evidence=false`.

## 3. New Code Layout

```text
research-automation/slime-context-manager/
├── README.md
├── opd/
│   ├── __init__.py
│   ├── action_parser.py
│   ├── context_policy_opd_server.py
│   ├── hindsight_judge.py
│   ├── opd_rollout.py
│   ├── teacher_logprob.py
│   └── topk_distillation_loss.py
└── tests/
    ├── test_opd_components.py
    └── test_opd_multiturn.py
```

The package is intentionally dependency-light. Unit tests run without slime, FastAPI, SGLang, Megatron, or a GPU. When slime is installed, the rollout bridge returns slime `Sample` and `RolloutFnTrainOutput` objects; otherwise it returns local compatibility stubs.

## 4. Training Paths

| Path | Default | Purpose | Required signal |
|---|---:|---|---|
| `Action OPD` | Yes | Train JSON memory actions with CE/SFT data | `teacher_action` |
| `Token OPD` | Optional | OpenClaw-style token-level OPD | `teacher_log_probs` |
| `Top-K OPD` | Optional | SDFT/SDPO-style top-K reverse KL | `teacher_topk_log_probs`, `teacher_topk_indices` |
| `GRPO` | Future | Optimize process/final rewards | grouped rollouts and custom reward |

`Action OPD` is the low-resource default. `Token OPD` is enabled only when a teacher endpoint can return log-probs for the original action tokens. `Top-K OPD` is off by default through `distill_topk=0`.

## 5. Slime Integration Points

The implementation reserves the following slime hooks:

```text
--rollout-function-path opd.opd_rollout.generate_rollout_opd
--custom-rm-path opd.opd_rollout.custom_rm
--loss-type custom_loss
--custom-loss-function-path opd.topk_distillation_loss.topk_distillation_loss_function
```

The first version uses the rollout bridge mainly as an OPD sample queue. It can later support GRPO by grouping samples with `episode_id + turn_id` and computing reward from turn-level PRM scores plus final episode outcome.

## 6. Multi-Turn Records

Each episode keeps both turn-level and episode-level records:

```json
{
  "episode_id": "ep001",
  "turn_rewards": [
    {"turn_id": 1, "process_reward": 1.0},
    {"turn_id": 2, "process_reward": -1.0}
  ],
  "final_reward": 1.0,
  "evidence_recall": 0.86,
  "token_cost": 7210
}
```

`session_done=true` clears pending turns that never received a next state. These turns are diagnostics-only and do not become training samples.

## 7. Test And Acceptance Criteria

The smoke-test target is:

```powershell
python -m unittest discover -s research-automation/slime-context-manager/tests -v
```

Acceptance criteria:

- Hint parser handles `\boxed{1}`, `\boxed{-1}`, `\boxed{0}`, and hint blocks.
- Vote selection picks the longest valid positive hint.
- Memory action parser handles valid JSON, embedded JSON, unknown actions, and invalid JSON.
- OPD sample builder aligns `response_length`, `loss_mask`, `rollout_log_probs`, and `teacher_log_probs`.
- Samples without `raw_evidence_id` are excluded from training by default.
- Multi-turn recorder supports pending turn cleanup and episode reward summaries.

## 8. References

- OpenClaw-RL OPD README: https://github.com/Gen-Verse/OpenClaw-RL/blob/main/openclaw-opd/README.md
- OpenClaw-RL OPD server: https://github.com/Gen-Verse/OpenClaw-RL/blob/main/openclaw-opd/openclaw_opd_api_server.py
- OpenClaw-RL rollout bridge: https://github.com/Gen-Verse/OpenClaw-RL/blob/main/openclaw-opd/openclaw_opd_rollout.py
- OpenClaw-RL top-K loss: https://github.com/Gen-Verse/OpenClaw-RL/blob/main/openclaw-opd/topk_distillation_loss.py
- slime usage guide: https://github.com/THUDM/slime/blob/main/docs/en/get_started/usage.md
