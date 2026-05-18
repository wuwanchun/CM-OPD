# FASD-GRPO 长链路智能体论文蓝图

_未来 RL 扩展路线：以 progressive evidence disclosure / summary-to-raw expansion 为基础，引入 GRPO 与 segment-level reward 做长轨迹信用分配。它不是主线论文标题，也不替代 `Learning When to Expand`。日期：2026-05-16。_

---

## 1. 不要混淆

| 维度 | 主线：progressive disclosure policy | 主 LLM tool-use SFT | FASD-GRPO / CodeHER-GRPO |
|---|---|---|---|
| 训练谁 | 外部 summary-to-raw expansion policy | 主 agent backbone | 外部 policy 或主 agent，实验中必须分开报告 |
| 主优化信号 | progressive disclosure CE | tool-use SFT / OPD | GRPO + teacher-guided correction + replay + KL |
| 信用分配粒度 | disclosure action | tool call | rollout / segment / action 三层 |
| 主 claim | 只换 progressive disclosure 策略也能提升 | 主模型能学会工具使用 | GRPO 排序 rollout，teacher correction 修正具体步骤 |
| 资源需求 | 低 | 中 | 中高 |

本文默认标题为 `FASD-GRPO: Failure/Success-Aware Self-Distilled GRPO for Long-Horizon Agents`。Coding-agent specialization 可命名为 `CodeHER-GRPO: Hindsight Self-Distilled GRPO for Self-Evolving Coding Agents`。

## 2. 核心 Claim

Episode-level GRPO alone can rank rollouts, but it is too sparse for long-horizon agents. `FASD-GRPO` is a future extension that combines group-relative policy optimization with source-grounded teacher correction and segment-level rewards to assign credit to concrete disclosure/tool decisions.

一句话贡献：

```text
GRPO 解决哪个 rollout 更好；
SDFT/OPD 解决具体哪一步该怎么改；
segment-level reward 解决长轨迹信用分配；
SFT replay 和 KL 负责稳定训练、防止能力遗忘。
```

## 3. Abstract 草稿

Long-horizon language agents often receive sparse outcome feedback after many disclosure, tool-use, editing, and verification decisions. Group-relative policy optimization can compare multiple rollouts for the same task, but final rewards alone provide weak credit assignment for the concrete step that caused success or failure. `FASD-GRPO` is a future extension of progressive evidence disclosure: for each task, the current policy samples multiple summary-to-raw expansion plans, receives verifier or environment feedback, and decomposes rewards into episode-, segment-, and action-level signals. Hindsight review proposes source-grounded policy-improvement targets for suspect decisions; these targets are not treated as objective truth, and their validity is evaluated by downstream held-out performance. Training combines GRPO, teacher-guided correction, behavior-cloning replay, and KL regularization. All empirical numbers are `TBD` until measured.

## 4. 方法

### 4.1 三层数据生产

```text
same task -> sample K rollouts
-> verifier / env / tests / PRM score each rollout
-> split trajectory into segments
-> assign episode_reward, segment_reward, action_reward
-> compute GRPO advantage within group
-> teacher reviews success/failure contrast and raw evidence
-> teacher proposes step-level reflection and corrected key actions
-> source-grounding and optional counterfactual/verifier evidence audit SDFT labels
-> joint train with GRPO + SDFT + replay + KL
```

多条轨迹不是生成 reflection 的必要条件；单条失败轨迹也能被 reviewer 诊断。但如果本文主打 GRPO，`K` 条同题 rollout 的价值在于提供 group-relative contrast：成功轨迹指出哪些 visibility/tool decision 值得反事实验证，失败轨迹说明哪些 decision 可能导致证据缺失、错误工具调用或重复动作。本文将这种机制称为 `group-contrastive reflection`。

```text
same task x
-> rollout r_success gets reward 1
-> rollout r_fail gets reward 0
-> align segments / visibility decisions when possible
-> reviewer asks: which summaries did success expand to RAW that failure kept compressed or hid?
-> produce corrected visibility label for failed rollout
```

如果一个 group 全错，则 reflection 退化为 `failure-evidence reflection`，只依赖 raw evidence、verifier feedback 或 oracle supporting facts。若 group 全对，则主要进入 success replay，不强制生成 correction。无论哪种情况，teacher correction 可以进入 `L_SDFT`，但必须记录 `source_ids`、quoted evidence、label source 和 confidence，并通过 held-out evaluation 检查是否产生真实策略改进。

### 4.2 轨迹单元

每个可训练 action record 至少包含：

```json
{
  "task_id": "task_001",
  "rollout_id": "task_001_r03",
  "group_id": "task_001",
  "segment_id": "seg_004",
  "turn_id": 17,
  "action_type": "visibility|tool_call|edit|test",
  "student_action": "...",
  "teacher_action": "...",
  "episode_reward": 1.0,
  "segment_reward": 0.4,
  "action_reward": 1.0,
  "advantage": 0.72,
  "failure_type": "missing_evidence|wrong_edit|test_regression|repeated_tool|format_error",
  "source_ids": ["pytest_003", "file_auth_py_21"],
  "hindsight_hint": "...",
  "label_source": "teacher_only|counterfactual|verifier_evidence|gold_support",
  "accepted_for_sdft": true,
  "quoted_evidence": "...",
  "confidence": 0.86,
  "loss_masks": {
    "grpo": true,
    "sdft": true,
    "replay": false
  }
}
```

新增 reflection record schema：

```json
{
  "reflection_id": "ref_task001_r03_t04",
  "task_id": "task_001",
  "failed_rollout_id": "task_001_r03",
  "contrast_rollout_id": "task_001_r01",
  "first_bad_turn": 4,
  "failure_type": "missing_evidence",
  "bad_action": "KEEP_SUMMARY",
  "corrected_action": "EXPAND_TO_RAW",
  "source_ids": ["doc_4_sent_2"],
  "counterfactual_delta": null,
  "label_source": "teacher_only",
  "accepted_for_sdft": true,
  "quoted_evidence": "The exact source sentence or tool output cited by the teacher.",
  "reflection": "The failed rollout hid a supporting fact that the successful rollout showed as raw evidence.",
  "verifiable_reason": "The final answer required doc_4_sent_2, but it was hidden from the failed rollout's rendered context.",
  "confidence": 0.86
}
```

reflection 必须结构化并 evidence-grounded。泛泛的“下次注意证据”不能进入 SDFT/OPD 训练。

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
| `L_SDFT` | 用 source-grounded teacher policy-improvement targets 修正具体 disclosure/tool decision |
| `L_replay` | 混入成功轨迹和原始能力样本，防止遗忘 |
| `L_KL` | 限制 policy 偏离 reference model |

## 5. Baselines

| 类别 | Baseline | 回答的问题 |
|---|---|---|
| SFT | Vanilla SFT | 普通监督是否足够 |
| Agent trace | Tool-call / ReAct SFT | 常规工具轨迹模仿是否足够 |
| Action | Visibility SFT | 静态 visibility 标签是否足够 |
| RL | GRPO only | 只有 rollout 相对奖励是否足够 |
| Distill | SDFT / OPD only | 只有具体修正是否足够 |
| Reward | GRPO + final reward only | 稀疏 reward 是否不足 |
| Reward | GRPO + segment reward | segment credit 是否带来增益 |
| Hybrid | GRPO + SDFT | RL + correction 是否互补 |
| Hybrid | GRPO + SDFT + replay + KL | 主方法 |
| Upper bound | Teacher direct | teacher 上限 |

所有结果表格保留 `TBD`，不得编造数字。

新增 context-selection baseline：

| 类别 | Baseline | 回答的问题 |
|---|---|---|
| 检索固定 | BM25 / frozen dense top-k | 在固定 retriever 下是否仍有提升 |
| 检索路由 | RF-Mem / Recollection-only | 只是 adaptive shallow/deep retrieval 是否足够 |
| 检索后压缩 | RECOMP-extractive / abstractive | 检索结果压缩器是否足够 |
| Prompt 压缩 | LLMLingua / LongLLMLingua | 通用 prompt compressor 是否足够 |
| 反思推理 | Reflexion-style notes | reflection 只放上下文、不训练 policy 是否足够 |

检索器必须固定，不微调 retriever。本文的可学习对象是 visibility/tool policy，而不是底层 BM25、embedding encoder 或 reranker。

## 6. 实验与指标

主任务可以分为通用长链路和 coding-agent 两组：

| 任务族 | 例子 | 主要验证 |
|---|---|---|
| QA / research | HotpotQA-style multi-hop, long evidence tasks | evidence recall 与 retrieval timing |
| Embodied / science | ALFWorld, ScienceWorld | 多轮 action 和环境反馈 |
| Coding | SWE-style tasks, repo debugging, pytest feedback | edit/test/visibility credit assignment |

核心指标：

```text
task_success / pass@1
segment_success_rate
visibility_accuracy
tool_call_validity
evidence_recall
failure_recovery_rate
test_regression_rate
token_cost
catastrophic_forgetting
teacher_correction_accept_rate
expansion_precision
expansion_recall
missed_expansion_rate
accepted_sdft_label_rate
teacher_only_label_rate
```

关键消融：

```text
w/o SDFT correction
w/o segment reward
w/o success trajectories
w/o failure trajectories
w/o raw_text
w/o confidence gating
episode reward only vs segment reward
external progressive disclosure policy vs main-agent tool policy
single-failure reflection vs group-contrastive reflection
generic reflection vs evidence-grounded corrected visibility
shuffled reflection
RF-Mem retrieval features only
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

异步 SDFT 入口是：

```bash
python scripts/run_sdft_async_hf_dataset.py \
  --dataset hotpotqa/hotpot_qa \
  --config distractor \
  --split validation \
  --max-rows 8 \
  --k-rollouts 4
```

该入口只消费失败轨迹：`failed trajectory -> async hint extraction -> corrected SDFT sample -> slime-compatible rollout group`。

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

### 8.2.1 多条轨迹是不是必须？

不是。reflection 可以从单条失败轨迹生成。但多条同题 rollout 能把 reflection 从“单条失败诊断”升级为“成功/失败对比诊断”。这与 GRPO 的核心设置一致：GRPO 用 group 内 reward 做相对优势，本文额外用 group 内成功/失败差异定位具体 disclosure/tool decision。首版可以先做 `GRPO-style contrastive SDFT`，即只用 K 条采样生成对比式 correction 标签；完整 `L_GRPO` 可以作为第二阶段加入。

### 8.3 会不会 teacher 泄漏答案？

训练集可以看到 verifier feedback、失败类型、工具输出和原文证据，但 held-out test 的答案和私有测试不能进入 teacher prompt。所有 correction 必须记录 `source_ids`、quoted evidence、`verifiable_reason` 和 `label_source`。teacher target 可以训练模型，但不能在论文中表述为客观真值。

### 8.4 为什么需要 replay 和 KL？

长链路 RL 容易过度优化 visibility/tool behavior，导致普通指令跟随、普通工具调用或代码编辑能力退化。`L_replay` 和 `L_KL` 是防遗忘约束，必须报告 retention 指标。

## 9. 参考关系

| 文档 | 关系 |
|---|---|
| `external-context-manager-paper-blueprint.md` | 低资源外部 visibility policy 版本 |
| `main-llm-memory-tool-policy-paper-blueprint.md` | 主模型 tool-use 训练版本 |
| `self-distillation-rl-continual-learning-report.md` | SDFT/SDPO 理论来源 |
| `context-manager-rl-slime-plan.md` | visibility schema 与 slime 扩展来源 |
