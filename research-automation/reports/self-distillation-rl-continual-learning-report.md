# Self-Distillation 相关论文调研报告

调研日期：2026-05-16  
主题：`Reinforcement Learning via Self-Distillation` 与 `Self-Distillation Enables Continual Learning`  
目标：提炼两篇论文对“成功/失败长轨迹 -> 自我蒸馏 -> 可迁移 memory/context manager policy”的启发。

## 1. 论文清单

| 论文 | 方法名 | 状态 | 链接 | 代码 |
|---|---|---|---|---|
| Reinforcement Learning via Self-Distillation | SDPO, Self-Distillation Policy Optimization | arXiv 2026，v2 2026-02-16 | [arXiv:2601.20802](https://arxiv.org/abs/2601.20802) | [lasgroup/SDPO](https://github.com/lasgroup/SDPO) |
| Self-Distillation Enables Continual Learning | SDFT, Self-Distillation Fine-Tuning | arXiv 2026，v1 2026-01-27 | [arXiv:2601.19897](https://arxiv.org/abs/2601.19897) | [idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation) |

注意：截至本报告写作时，这两篇都按 arXiv 预印本处理，不视为已确认顶会正式发表。

## 2. 总体结论

这两篇论文共享一个关键思想：

> 同一个模型可以在不同上下文下扮演 student 和 teacher。teacher 看到额外信息，例如环境反馈、成功样例或专家 demonstration；student 只看到正常输入。训练时把 teacher 的 token/logit-level 分布蒸馏回 student，从而获得比标量 reward 或离线 SFT 更密集、更 on-policy 的学习信号。

对你的方向，最有价值的抽象是：

```text
普通 agent rollout
-> 得到成功/失败、工具反馈、环境反馈、记忆操作结果
-> 构造带额外信息的 self-teacher context
-> teacher 重新评估原轨迹或生成更好 memory action
-> distill 回 context manager policy
```

这非常适合你想做的：

```text
成功/失败长轨迹
-> 诊断 memory/context 决策点
-> 蒸馏 KEEP / COMPRESS / ARCHIVE / RETRIEVE / UPDATE / DROP / PIN
-> 后续再接 slime GRPO
```

## 3. Reinforcement Learning via Self-Distillation / SDPO

### 3.1 研究问题

RLVR，即 reinforcement learning with verifiable rewards，在代码和数学任务里常见。但传统 GRPO/PPO 类方法通常只从标量 reward 学习：

```text
pass = 1
fail = 0
```

问题是 credit assignment 太粗。一次代码失败可能包含非常有信息量的反馈：

- runtime error；
- failed unit test；
- judge explanation；
- public/private test mismatch；
- tool output；
- LLM judge critique。

SDPO 认为这些 tokenized rich feedback 不该只被折叠成一个标量，而应该转成 dense token-level learning signal。

### 3.2 核心方法

SDPO 定义了一个 self-teacher：

```text
student = 当前模型，只看 question，生成原始 attempt
self-teacher = 同一个模型，但额外看到 feedback / successful rollout / environment output
```

训练流程：

```text
1. 当前 policy 对问题采样 rollouts
2. 环境评估 rollouts，返回 reward 和 rich feedback
3. 用同一个模型构造 feedback-conditioned self-teacher
4. self-teacher 不重新生成完整答案，而是重新计算原始 attempt 的 token log-probs
5. 用 student 与 self-teacher 的 next-token distribution 差异做 distillation loss
6. 更新 policy
```

论文强调，self-teacher 的作用是 hindsight credit assignment：它看到了失败反馈后，能够判断原先 attempt 的哪些 token 更应该被鼓励或抑制。

### 3.3 与 GRPO 的差异

| 维度 | GRPO / RLVR | SDPO |
|---|---|---|
| 反馈形式 | 标量 reward | tokenized rich feedback |
| credit assignment | rollout 级或 sequence 级 | token/logit 级 |
| teacher | 无外部 teacher | 当前模型 + feedback context |
| 是否 on-policy | 是 | 是 |
| 失败样本是否有信号 | 常常为 0 advantage | 失败反馈可产生 dense signal |
| 额外成本 | 多 rollouts | 多一次 teacher log-prob 计算 |

SDPO 也可以在没有 rich feedback 的环境中工作：如果同一组 rollout 里有成功样本，就把成功 rollout 当作失败样本的 implicit feedback。

### 3.4 实验结果

论文报告了三类实验：

1. **没有 rich feedback 的 RLVR 任务**  
   例如科学问答、工具调用。SDPO 把 batch 内成功尝试作为失败尝试的反馈。论文摘要和正文报告，SDPO 相比强 GRPO baseline 有更高 aggregate final accuracy，例如 70.2% vs 66.6%。

2. **有 rich feedback 的 coding 任务**  
   在 LiveCodeBench v6 上，代码环境给出类似 LeetCode 的反馈。论文报告 SDPO 达到 48.8%，GRPO 为 41.2%，并且更快达到 GRPO 的最终准确率。

3. **Test-time self-distillation**  
   对单个困难问题做测试时自蒸馏，比 best-of-k 或多轮对话用更少 attempt 达到类似 discovery probability。

### 3.5 重要观察

1. **SDPO 会更简洁**  
   论文观察到 GRPO 有时会产生冗长、循环、表面推理；SDPO 的回答更短，同时准确率更高。这个点对 long-horizon agent 很重要：长链路能力不等于无限延长 reasoning trace。

2. **模型越强，self-teaching 越有效**  
   论文在 Qwen3 family 上做 scaling，指出 SDPO 对模型规模敏感。更强的 in-context 反馈理解能力会带来更好的 self-teacher。

3. **稳定性需要工程处理**  
   论文提到使用 regularized self-teacher，例如 EMA teacher 或 initial teacher interpolation，并采用 symmetric Jensen-Shannon divergence 改善稳定性。

4. **计算开销主要是 teacher log-prob**  
   SDPO 不需要额外采样 teacher output，只需要重新计算原 attempt 在 feedback-conditioned context 下的 log-probs，因此比“让 teacher 重新生成”更省。

## 4. Self-Distillation Enables Continual Learning / SDFT

### 4.1 研究问题

持续学习的核心问题是：模型学习新技能或新知识时，不应该破坏已有能力。

传统方案：

```text
expert demonstrations -> SFT
```

问题是 SFT 是 off-policy 的。模型训练在 expert-induced trajectories 上，但测试时会落到自己生成的状态分布里，小错误会累积，且容易 catastrophic forgetting。

SDFT 的问题定义是：

> 没有显式 reward，只有 demonstrations 时，能不能仍然做 on-policy learning？

### 4.2 核心方法

SDFT 同样使用同一个模型扮演 teacher 和 student。

```text
teacher = model(prompt + demonstration)
student = model(prompt)
```

训练流程：

```text
1. student 对 prompt 生成自己的 response
2. teacher 在 prompt + expert demonstration 条件下，对 student trajectory 给出 token-level distribution
3. 训练 student 匹配 teacher distribution
4. teacher 通常使用 EMA student 参数，并带 demonstration context
```

关键是：训练发生在 student 自己生成的轨迹上，因此是 on-policy distillation，而不是对 expert answer 做普通 SFT。

### 4.3 SDFT 与 IRL 的关系

论文把 SDFT 解释成一种 IRL-like 机制：

- expert demonstration 不直接作为 label；
- demonstration-conditioned policy 近似“更优 policy”；
- student 通过匹配这个更优 policy 的分布，相当于最大化一个由 demonstration 和 in-context learning 隐式定义的 reward。

这对你的 memory tool policy 很有启发：  
你可以不显式写出每一步 reward，而是让“带有成功/失败诊断和原始证据的 self-teacher”给出更好的 action distribution。

### 4.4 实验设置

论文测试两类持续学习：

1. **Skill Learning**  
   包括科学问答、工具调用、医学推理。

2. **Knowledge Acquisition**  
   注入 2025 年自然灾害相关的新知识，测试模型能否吸收新事实。

评估维度：

- 新任务准确率；
- 旧能力保持；
- OOD generalization；
- catastrophic forgetting。

### 4.5 关键结果

1. **SDFT 比 SFT 更少遗忘**  
   在单任务和多任务顺序学习中，SFT 学新任务会显著伤害旧能力；SDFT 能累计多个技能而不明显回退。

2. **SDFT 对新任务也更强**  
   不是只保守地保留旧能力，论文报告 SDFT 在新任务上也常超过 SFT。

3. **知识注入更像真正吸收知识**  
   在 Knowledge Acquisition 中，SDFT 对直接问题和间接问题都优于 SFT。论文报告 strict accuracy: SFT 80%，SDFT 89%；OOD accuracy: SFT 80%，SDFT 98%。

4. **模型规模很关键**  
   项目页指出 3B 模型的 ICL 太弱，SDFT 可能不如 SFT；7B 有约 4 点提升，14B 扩大到约 7 点。这意味着 self-distillation 的 teacher 质量依赖模型本身的 in-context learning 能力。

5. **可作为 RL 前置初始化**  
   论文讨论指出 SDFT 与 on-policy RL 互补：SDFT 适合没有 reward、只有 demonstrations 的场景，也可作为后续 RL fine-tuning 的初始化。

## 5. 两篇论文的对比

| 维度 | SDPO | SDFT |
|---|---|---|
| 目标 | 用 rich feedback 改善 RL credit assignment | 从 demonstrations 做 on-policy continual learning |
| 额外信息 | environment feedback、runtime error、successful rollout | expert demonstration |
| teacher context | question + feedback + original attempt | prompt + demonstration |
| student context | question | prompt |
| 训练信号 | self-teacher logit/token distribution | demonstration-conditioned teacher distribution |
| 是否需要 reward | 通常有 reward，rich feedback 更好 | 不需要显式 reward |
| 适合场景 | 代码、工具、数学、可验证任务 | 技能学习、知识注入、持续学习 |
| 对你的启发 | 用成功/失败轨迹反馈做 dense memory-action 学习 | 用成功示范/修正示范做 on-policy memory tool distillation |

## 6. 对你的 RL Context Manager 的启发

你想做的系统可以借鉴两者：

```text
SDPO: 失败反馈 / 工具反馈 / 测试反馈 -> dense credit assignment
SDFT: 成功示范 / 修正示范 -> on-policy distillation，减少遗忘
```

把它们合起来：

```text
Trajectory-to-Memory-Policy Self-Distillation
```

### 6.1 Memory Tool Self-Teacher

定义两个视角：

```text
student context:
  当前任务状态 + memory item + budget

self-teacher context:
  当前任务状态 + memory item + budget
  + 成功/失败 outcome
  + tool feedback
  + raw evidence pointer
  + 轨迹诊断
  + successful rollout / corrected action
```

student 需要输出：

```json
{
  "action": "RETRIEVE",
  "target_ids": ["mem_034"],
  "query": "raw pytest assertion and stack trace",
  "reason": "Need exact failure evidence before editing."
}
```

self-teacher 不一定要生成新 action；更稳定的方式是重新评估 student 原 action，并提供 token/logit-level 或 action-level preference signal。

### 6.2 成功轨迹如何用

借鉴 SDFT：成功轨迹是 demonstration，不要直接做离线 SFT，而应让当前 policy 在相同状态下生成自己的 memory action，再让 demonstration-conditioned teacher 修正。

流程：

```text
成功轨迹 -> 提取关键 memory decisions
当前 policy 在这些状态上重新生成 action
teacher 看到成功轨迹和 evidence，给出更优 action distribution
distill teacher -> student
```

这样避免只模仿专家路径，训练分布更接近 student 自己会犯错的状态。

### 6.3 失败轨迹如何用

借鉴 SDPO：失败轨迹里的 tool feedback 是 rich feedback。

例如：

```text
失败原因: 修改前没有回读原始函数签名
环境反馈: pytest assertion / stack trace
错误 action: COMPRESS only
更好 action: RETRIEVE raw evidence, then PIN test name
```

训练时：

```text
student = 当前 policy 原先的 memory action
self-teacher = 当前 policy + failure feedback + raw evidence
loss = distill teacher's corrected distribution
```

这比只给最终 pass/fail reward 更密集。

## 7. 推荐方法设计

### 7.1 阶段一：SDFT-style Memory Tool Distillation

不接 RL，先做 demonstrations。

数据来源：

- 成功轨迹里的关键 memory action；
- 失败轨迹经过诊断后的 corrected action；
- 人工/LLM judge 标注的少量高质量 action。

训练：

```text
student prompt = state + memory item
teacher prompt = state + memory item + demonstration/correction/evidence
loss = KL(student || teacher) 或 action-level CE
```

目标：

- 让 context manager 学会基本操作；
- 避免普通 SFT 导致的分布偏移；
- 减少对重型 GRPO 的依赖。

### 7.2 阶段二：SDPO-style Rich Feedback Training

接入环境反馈。

适用环境：

- HotpotQA：supporting fact 是否找回；
- ALFWorld：success / invalid action / repeated action；
- ScienceWorld：score、实验状态、失败动作；
- coding：pytest、lint、runtime error、CI 反馈。

训练：

```text
policy 采样 memory tool calls
环境执行 agent episode
获得 rich feedback
self-teacher 看到 feedback 后重新评价原 memory tool call
distill 回 policy
```

目标：

- 解决长链路 credit assignment；
- 让失败样本也产生 dense signal；
- 比纯 GRPO 更省样本。

### 7.3 阶段三：slime-GRPO 结合

SDFT/SDPO 可以作为 GRPO 前置。

```text
Rule policy
-> SDFT-style self-distillation
-> SDPO-style rich feedback self-distillation
-> slime GRPO fine-tuning
```

这样 GRPO 不从随机 memory policy 开始，而从已经知道基本保留/压缩/回读规则的 policy 开始。

## 8. 与 slime 训练计划的衔接

### 8.1 数据格式

在已有 `context_action` schema 上增加 teacher context：

```json
{
  "sample_id": "act_ep001_t017_mem034",
  "student_prompt": "state + memory item + budget",
  "teacher_prompt": "state + memory item + budget + outcome + feedback + evidence + correction",
  "student_action": "COMPRESS",
  "teacher_action": "RETRIEVE",
  "feedback": {
    "outcome": "failure",
    "failure_type": "missing_evidence",
    "raw_evidence_ids": ["tool_pytest_003"]
  },
  "gold_action": "RETRIEVE"
}
```

### 8.2 训练选择

低资源优先：

```text
action-level SFT / DPO
```

中等资源：

```text
teacher logits distillation
```

后期：

```text
slime custom rollout + custom reward + GRPO
```

### 8.3 自定义 reward

如果接 SDPO-style rich feedback，reward 不只看任务成功：

```text
task_success
evidence_recall
symbol_recall
retrieve_precision
memory_pollution
repeated_tool_loop
token_budget
```

### 8.4 自定义 rollout

slime rollout 中需要保存：

```text
student memory action
environment feedback
teacher prompt
raw evidence ids
success/failure outcome
```

这样后续可以同时做：

- GRPO；
- SDPO-style distillation；
- failure diagnosis dataset generation。

## 9. 风险与注意事项

### 9.1 Self-teacher 可能不可靠

两篇论文都依赖模型本身的 in-context learning。SDFT 项目页明确指出，3B 模型 ICL 太弱时 SDFT 可能不如 SFT。因此：

- 0.5B/1.5B 模型不适合直接做 self-teacher；
- 可以用 7B/14B/API model 做 teacher；
- student 可以是更小的 context policy。

### 9.2 Teacher context 不能泄漏测试答案

对于 HotpotQA / coding 评测，teacher prompt 可以看到 failure feedback 和 raw evidence，但不能直接看到 held-out test answer。否则会变成数据泄漏。

### 9.3 失败诊断必须 evidence-grounded

不能允许 teacher 泛泛反思。每条 correction 必须包含：

```text
turn_id
memory_id
raw_evidence_id
bad_action
correct_action
verifiable_reason
```

### 9.4 on-policy 数据成本仍然存在

SDPO/SDFT 都强调 on-policy，所以需要当前 policy 采样。低资源下建议先做：

```text
少量 episode
规则 policy bootstrap
action-level distillation
再逐步扩展
```

## 10. 适合你的落地方案

我建议把两篇论文改造成你的路线：

```text
Failure/Success-Aware Self-Distillation for Memory Tool Policies
```

最小闭环：

1. 用规则 context manager 跑 HotpotQA / ALFWorld 小样本。
2. 保存成功/失败轨迹、memory actions、raw evidence。
3. 用 teacher prompt 生成 corrected memory tool calls。
4. 训练一个轻量 memory action policy。
5. 用同一套 HF evaluator 比较 rule vs distilled policy。
6. 有资源后接 slime GRPO。

实验 baseline：

```text
Rule policy
Action SFT
Offline distillation
On-policy self-distillation
On-policy self-distillation + utility rerank
slime GRPO
```

核心指标：

```text
task_success_rate
memory_action_accuracy
evidence_recall
symbol_recall
retrieve_precision
token_reduction
failure_recovery_rate
catastrophic_forgetting_across_tasks
```

## 11. 最终判断

`Reinforcement Learning via Self-Distillation` 给你的关键启发是：失败反馈不应该只变成 0 reward，而应该通过 self-teacher 转成 dense credit assignment。

`Self-Distillation Enables Continual Learning` 给你的关键启发是：如果只有成功示范或修正示范，不要直接 SFT；让当前 policy 在自己的状态分布上学习 demonstration-conditioned teacher，能减少分布偏移和遗忘。

两者合起来，正好支撑你的研究设想：

```text
成功轨迹提供 demonstration
失败轨迹提供 rich feedback
raw evidence 提供可验证 grounding
self-teacher 生成 memory tool supervision
student 学会可迁移的 context manager policy
```

这条路线比直接做大规模 GRPO 更适合低资源场景，也更容易形成清晰的论文贡献。

## 12. 参考来源

- [Reinforcement Learning via Self-Distillation, arXiv:2601.20802](https://arxiv.org/abs/2601.20802)
- [Reinforcement Learning via Self-Distillation, HTML](https://arxiv.org/html/2601.20802)
- [lasgroup/SDPO](https://github.com/lasgroup/SDPO)
- [Self-Distillation Enables Continual Learning, arXiv:2601.19897](https://arxiv.org/abs/2601.19897)
- [Self-Distillation Enables Continual Learning project page](https://self-distillation.github.io/SDFT)
- [idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation)

