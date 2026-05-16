# FASD-GRPO 长链路智能体论文蓝图

_第三条独立路线：在 FASD-Mem 的 evidence-grounded memory action 和 SDFT/OPD 的 hindsight correction 基础上，引入 GRPO 与 segment-level reward 做长轨迹信用分配。日期：2026-05-16。_

---

## 1. 不要混淆

| 维度 | FASD-Mem | 主 LLM memory-tool SDFT | FASD-GRPO / CodeHER-GRPO |
|---|---|---|---|
| 训练谁 | 外部 context manager | 主 agent backbone | 外部 policy 或主 agent，实验中必须分开报告 |
| 主优化信号 | teacher-corrected action CE | on-policy self-distillation | GRPO + SDFT + replay + KL |
| 信用分配粒度 | memory action | memory tool call | rollout / segment / action 三层 |
| 主 claim | 只换记忆管理器也能提升 | 主模型能学会 memory tool use | GRPO 排序 rollout，SDFT 修正具体步骤 |
| 资源需求 | 低 | 中 | 中高 |

本文默认标题为 `FASD-GRPO: Failure/Success-Aware Self-Distilled GRPO for Long-Horizon Agents`。Coding-agent specialization 可命名为 `CodeHER-GRPO: Hindsight Self-Distilled GRPO for Self-Evolving Coding Agents`。

## 2. 核心 Claim

Episode-level GRPO alone can rank rollouts, but it is too sparse for long-horizon agents. `FASD-GRPO` combines group-relative policy optimization with hindsight self-distillation and segment-level rewards to assign credit to concrete memory/tool decisions.

一句话贡献：

```text
GRPO 解决哪个 rollout 更好；
SDFT/OPD 解决具体哪一步该怎么改；
segment-level reward 解决长轨迹信用分配；
SFT replay 和 KL 负责稳定训练、防止能力遗忘。
```

## 3. Abstract 草稿

Long-horizon language agents often receive sparse outcome feedback after many memory, tool-use, editing, and verification decisions. Group-relative policy optimization can compare multiple rollouts for the same task, but final rewards alone provide weak credit assignment for the concrete step that caused success or failure. We propose `FASD-GRPO`, a failure- and success-aware self-distilled GRPO framework for long-horizon agents. For each task, the current policy samples multiple rollouts, receives verifier or environment feedback, and decomposes rewards into episode-, segment-, and action-level signals. A hindsight teacher observes outcomes, failure classes, and raw evidence pointers to correct key memory and tool actions. Training combines GRPO, hindsight self-distillation, behavior-cloning replay, and KL regularization. This design is intended to preserve GRPO's relative preference signal while providing dense, evidence-grounded supervision for the specific decisions that need repair. All empirical numbers are `TBD` until measured.

## 4. 方法

### 4.1 三层数据生产

```text
same task -> sample K rollouts
-> verifier / env / tests / PRM score each rollout
-> split trajectory into segments
-> assign episode_reward, segment_reward, action_reward
-> compute GRPO advantage within group
-> teacher uses outcome + raw evidence + failure class to correct key actions
-> joint train with GRPO + SDFT + replay + KL
```

### 4.2 轨迹单元

每个可训练 action record 至少包含：

```json
{
  "task_id": "task_001",
  "rollout_id": "task_001_r03",
  "group_id": "task_001",
  "segment_id": "seg_004",
  "turn_id": 17,
  "action_type": "memory_action|tool_call|edit|test",
  "student_action": "...",
  "teacher_action": "...",
  "episode_reward": 1.0,
  "segment_reward": 0.4,
  "action_reward": 1.0,
  "advantage": 0.72,
  "failure_type": "missing_evidence|wrong_edit|test_regression|repeated_tool|format_error",
  "raw_evidence_ids": ["pytest_003", "file_auth_py_21"],
  "hindsight_hint": "...",
  "confidence": 0.86,
  "loss_masks": {
    "grpo": true,
    "sdft": true,
    "replay": false
  }
}
```

### 4.3 联合目标

```text
L_total = L_GRPO
        + λ_sdft L_SDFT
        + λ_bc L_replay
        + λ_kl L_KL
```

默认解释：

| Loss | 作用 |
|---|---|
| `L_GRPO` | 用同一 task 的 K 条 rollout 计算 group-relative advantage |
| `L_SDFT` | 用 teacher correction 修正具体 memory/tool action |
| `L_replay` | 混入成功轨迹和原始能力样本，防止遗忘 |
| `L_KL` | 限制 policy 偏离 reference model |

## 5. Baselines

| 类别 | Baseline | 回答的问题 |
|---|---|---|
| SFT | Vanilla SFT | 普通监督是否足够 |
| Agent trace | Tool-call / ReAct SFT | 常规工具轨迹模仿是否足够 |
| Action | Action SFT | 静态 action 标签是否足够 |
| RL | GRPO only | 只有 rollout 相对奖励是否足够 |
| Distill | SDFT / OPD only | 只有具体修正是否足够 |
| Reward | GRPO + final reward only | 稀疏 reward 是否不足 |
| Reward | GRPO + segment reward | segment credit 是否带来增益 |
| Hybrid | GRPO + SDFT | RL + correction 是否互补 |
| Hybrid | GRPO + SDFT + replay + KL | 主方法 |
| Upper bound | Teacher direct | teacher 上限 |

所有结果表格保留 `TBD`，不得编造数字。

## 6. 实验与指标

主任务可以分为通用长链路和 coding-agent 两组：

| 任务族 | 例子 | 主要验证 |
|---|---|---|
| QA / research | HotpotQA-style multi-hop, long evidence tasks | evidence recall 与 retrieval timing |
| Embodied / science | ALFWorld, ScienceWorld | 多轮 action 和环境反馈 |
| Coding | SWE-style tasks, repo debugging, pytest feedback | edit/test/memory action credit assignment |

核心指标：

```text
task_success / pass@1
segment_success_rate
memory_action_accuracy
tool_call_validity
evidence_recall
failure_recovery_rate
test_regression_rate
token_cost
catastrophic_forgetting
teacher_correction_accept_rate
```

关键消融：

```text
w/o SDFT correction
w/o segment reward
w/o success trajectories
w/o failure trajectories
w/o raw evidence pointer
w/o confidence gating
episode reward only vs segment reward
external memory policy vs main-agent memory tool policy
```

## 7. 工程落地

新增 `grpo_sdft/` 子系统，第一阶段直接基于 HuggingFace 数据集构造 grouped rollouts、segment rewards 和 SDFT corrections；不使用 toy 本地检查作为主验收口径，也不承诺完整 slime GRPO launcher。

已规划接口：

```text
grpo_sdft/trajectory_schema.py
grpo_sdft/segmenter.py
grpo_sdft/reward_decomposer.py
grpo_sdft/hindsight_relabeler.py
grpo_sdft/grpo_sample_builder.py
grpo_sdft/joint_loss_config.py
```

脚本入口：

```text
scripts/build_grpo_sdft_dataset.py
scripts/run_fasd_grpo_local.py
scripts/run_fasd_grpo_hf_dataset.py
```

主验收入口是：

```bash
python scripts/run_fasd_grpo_hf_dataset.py \
  --dataset hotpotqa/hotpot_qa \
  --config distractor \
  --split validation \
  --max-rows 8 \
  --k-rollouts 4
```

未来 slime 接口：

```text
--rollout-function-path grpo_sdft.grpo_rollout.generate_rollout_fasd_grpo
--custom-rm-path grpo_sdft.reward_decomposer.custom_rm
--custom-loss-function-path grpo_sdft.joint_loss.compute_fasd_grpo_loss
```

## 8. Reviewer Questions

### 8.1 这和普通 GRPO 有什么区别？

普通 GRPO 只知道同组 rollout 哪个更好。`FASD-GRPO` 额外把反馈分解到 segment/action，并用 hindsight teacher 产生具体 correction，因此能处理长轨迹里的稀疏信用分配。

### 8.2 这和 SDFT/OPD 有什么区别？

SDFT/OPD 主要修正具体动作，但缺少 group-relative rollout ranking。`FASD-GRPO` 把 SDFT 作为 dense correction 信号，同时保留 GRPO 对整体任务成功率的优化。

### 8.3 会不会 teacher 泄漏答案？

训练集可以看到 verifier feedback、失败类型、工具输出和 raw evidence pointer，但 held-out test 的答案和私有测试不能进入 teacher prompt。所有 correction 必须记录 `raw_evidence_ids` 和 `verifiable_reason`。

### 8.4 为什么需要 replay 和 KL？

长链路 RL 容易过度优化 memory/tool behavior，导致普通指令跟随、普通工具调用或代码编辑能力退化。`L_replay` 和 `L_KL` 是防遗忘约束，必须报告 retention 指标。

## 9. 参考关系

| 文档 | 关系 |
|---|---|
| `external-context-manager-paper-blueprint.md` | 低资源外部 memory policy 版本 |
| `main-llm-memory-tool-policy-paper-blueprint.md` | 主模型 memory-tool SDFT 版本 |
| `self-distillation-rl-continual-learning-report.md` | SDFT/SDPO 理论来源 |
| `context-manager-rl-slime-plan.md` | memory action schema 与 slime 扩展来源 |
