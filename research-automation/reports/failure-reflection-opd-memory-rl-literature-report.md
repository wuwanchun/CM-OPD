# Failure Reflection + OPD for Memory/Context RL: Literature Report

Date: 2026-05-18

## 1. Executive Takeaway

你的新想法可以被整理成一个更清晰的论文主张：

> Long-horizon agent RL should not only optimize scalar trajectory rewards. Failed trajectories should be reviewed by a policy/teacher model into evidence-grounded reflections, and those reflections should be converted into targeted OPD/SDFT supervision for memory/context/tool decisions.

这条逻辑有两组文献支撑。

第一组是 memory/context management 文献。早期系统通常把记忆做成外部模块：摘要、检索、更新、遗忘、分层存储、图结构链接。更新通常是 rule/prompt/system policy，而不是 RL action。最新一批工作开始把 memory operation 显式放进 agent action space，并用 PPO/GRPO 或 step-wise GRPO 训练，例如 Memory-R1、AgeMem、ContextBudget、Memory-as-Action。

第二组是 reflection / self-distillation / OPD 文献。Reflexion 证明失败反馈可以转成语言反思并存入 episodic memory；Agent-R、AgentHER、OpenClaw-RL、SDPO、RLSD 进一步把失败反馈变成训练信号。核心共同点是：标量 reward 只能告诉模型“这条轨迹好/坏”，但 reflection/OPD 能告诉模型“哪一步错、为什么错、应该怎么改”。

因此，你的方法最自然的定位不是单纯 GRPO，也不是单纯 memory compression，而是：

```text
Failure-aware trajectory review
-> evidence-grounded reflection
-> memory/tool action correction
-> OPD/SDFT dense supervision
-> optional GRPO for rollout-level ranking
```

## 2. Research Question 1: Memory Compression/Extraction Operations Are Represented How?

### 2.1 Non-RL Memory Systems: Memory as External State

早期 agent memory 系统一般不训练主模型，而是把 memory 当外部状态或数据库。

| Paper | Memory operation form | Training form | Relevance |
|---|---|---|---|
| MemoryBank | retrieve, update, summarize, forget/reinforce | mostly system/prompt/rule driven | 说明长期记忆通常包括召回、更新、遗忘和摘要。 |
| MemGPT | move information between memory tiers; archival/recall/core memory | no RL mainline; OS-style virtual context management | 说明 memory 可以被建模成类似操作系统的分页/迁移。 |
| A-MEM | add note, generate attributes, link related memories, evolve old memories | agentic memory construction, not GRPO | 说明 memory 不一定是固定槽位，也可以是动态知识网络。 |
| Reflexion | write reflection into episodic memory | no weight update; verbal RL via memory | 说明失败反馈可以先变成可复用文字记忆。 |

关键形式：

```text
raw interaction / observation
-> memory extraction
-> structured memory note / summary / reflection / embedding
-> retrieval into future context
```

这些方法的问题是：memory operation 通常由规则、prompt 或外部模块决定，缺少端到端策略学习；如果 summary-only 失真，后续模型很难追溯 raw evidence。

### 2.2 RL Memory Systems: Memory as Action or Tool Call

新一批论文把 memory operation 放进策略动作空间。

| Paper | Operation set | RL/GRPO form | Important detail |
|---|---|---|---|
| Memory-R1 | ADD, UPDATE, DELETE, NOOP; separate answer-side retrieval/use | PPO and GRPO outcome-driven RL | Memory Manager learns structured memory edits; Answer Agent selects relevant entries. |
| AgeMem | store, retrieve, update, summarize, discard as tool actions | three-stage progressive RL; step-wise GRPO | Explicitly targets sparse/discontinuous reward caused by memory operations. |
| ContextBudget/BACM-RL | decide when/how much history to compress under token budget | curriculum RL over budget-aware context decisions | Compress amount becomes a sequential decision, not a static heuristic. |
| Memory-as-Action / MemAct | explicit working-memory editing actions | SFT initialization + RL phase | Notes that memory edits fracture the ordinary prefix trajectory assumption. |

在 GRPO 训练里，memory/context 操作通常有三种表示：

1. **Tool-call action**  
   模型输出结构化工具调用，例如：

   ```json
   {"action": "UPDATE", "target": "mem_12", "content": "..."}
   ```

   GRPO 把完整 rollout 作为 group 内样本，最终 reward 或 step reward 回传给产生这些 tool calls 的 token。

2. **Discrete memory action label**  
   训练一个外部 context manager：

   ```text
   state_t -> KEEP / COMPRESS / ARCHIVE / RETRIEVE / UPDATE / DROP / PIN
   ```

   这更适合低资源，因为训练对象可以是小模型或 classifier，而不是主 LLM。

3. **Compression ratio / budget decision**  
   不是选具体 memory id，而是决定压缩触发点和压缩量：

   ```text
   budget_state, history_state -> compress now? how much? which span?
   ```

   ContextBudget 属于这一类。

对你的启发：如果你的目标是“可迁移 RL context manager”，版本 A 最干净的 action schema 是：

```text
OBSERVE(state_t)
-> memory_action_t in {KEEP, COMPRESS, ARCHIVE, RETRIEVE, UPDATE, DROP, PIN}
-> evidence pointers: raw_evidence_ids
-> downstream task outcome
```

如果你的目标是训练主 agent，版本 B/FASD-GRPO 则把这些 action 变成主 LLM 的 tool calls。

## 3. Research Question 2: Why Reflection/OPD Helps Beyond Scalar Reward?

### 3.1 Scalar RL Problem: Reward Is Too Sparse

RLVR/GRPO 的强项是相对排序：同一个任务采样 K 条 rollout，谁通过测试、谁拿高分，group 内归一化 advantage 后更新策略。

但长链路 agent 的失败通常不是“整个策略都错”，而是某一步 memory/tool decision 错：

```text
turn 4: dropped evidence
turn 9: retrieved wrong document
turn 14: compressed away variable constraint
turn 20: final answer fails
```

最终 reward = 0 只告诉模型失败了，不告诉它 turn 4 才是关键失败点。

SDPO 直接把这个问题表述为 RLVR 的 credit-assignment bottleneck：许多 verifiable environments 有 runtime errors、judge feedback 等 rich textual feedback，但普通 RLVR 只用 scalar outcome reward。SDPO 的做法是让模型在 feedback-conditioned context 下作为 self-teacher，把 tokenized feedback 转成 dense learning signal。

### 3.2 Reflection Line: Failure -> Verbal Review -> Better Future Behavior

Reflexion 是这条线的经典起点。它不更新权重，而是让 agent 根据任务反馈写 verbal reflection，并把 reflection 放入 episodic memory。这个机制证明了：失败反馈可以先变成语言化策略修正，而不只是 reward。

Self-Refine 说明同一模型可以生成 feedback 并迭代改写输出；它不训练模型，但证明了“模型 review 自己输出”在推理时有效。

Agent-R 更接近你的训练目标：它用 iterative self-training 训练 agent 学会 reflection。关键点是模型要定位 failed trajectory 的 first error step，再从那里恢复正确路径。这和你的 memory action correction 很像：

```text
failed trajectory
-> locate first bad memory/tool decision
-> generate reflection/correction
-> train policy to revise earlier
```

### 3.3 OPD/SDFT Line: Reflection -> Label/Dense Token Signal

OpenClaw-RL 的 OPD 实现给出了工程化模板：

```text
student action a_t
-> next_state / user reply / tool output
-> hindsight judge extracts hint
-> append hint to teacher context
-> compute teacher log-probs on original action tokens
-> submit OPD sample to slime
```

它把 next-state signal 拆成两类：

```text
evaluative signal: scalar reward / PRM score
directive signal: what should have been different
```

这正好对应你的论点：

```text
GRPO answers: which rollout is better?
OPD/reflection answers: which action should change, and how?
```

RLSD 进一步提醒一个风险：如果 teacher 使用 privileged information 直接提供学习信号，可能信息泄漏和训练不稳定。因此更稳的组合是：

```text
RLVR/GRPO gives reliable update direction from environment correctness
self-distillation gives token-level magnitude / fine-grained localization
```

## 4. How to Update Your Paper Logic

你现在可以把 FASD-GRPO 的方法部分改成三层信号，而不是“GRPO + SDFT”平铺。

### 4.1 New Method Name

建议主名：

```text
FASD-GRPO: Failure-Aware Self-Distilled GRPO for Long-Horizon Memory Agents
```

如果强调 coding agent：

```text
CodeHER-GRPO: Hindsight Reflection and Self-Distilled GRPO for Coding Agents
```

### 4.2 New Core Claim

旧 claim：

```text
GRPO + SDFT improves long-horizon memory/tool policy.
```

新 claim：

```text
Long-horizon agent failures contain reusable diagnostic information. 
FASD-GRPO uses policy-model trajectory review to convert failed trajectories into evidence-grounded reflections, then distills these reflections into targeted memory/tool action updates while retaining GRPO for rollout-level preference optimization.
```

### 4.3 Training Signal Decomposition

```text
same task -> K rollouts
-> verifier/env/test gives episode reward
-> policy reviewer inspects failed rollouts
-> reflection object:
     failure_type
     first_bad_turn
     bad_memory_action
     raw_evidence_ids
     corrected_memory_action
     verifiable_reason
-> OPD/SDFT sample
-> joint train:
     L_total = L_GRPO + lambda_reflect L_OPD + lambda_bc L_replay + lambda_kl L_KL
```

### 4.4 Reflection Object Schema

最重要的是 reflection 不能是泛泛的“下次注意”。建议强制结构化：

```json
{
  "trajectory_id": "task_001_r03",
  "failure_type": "missing_evidence",
  "first_bad_turn": 4,
  "bad_action": {
    "type": "memory_action",
    "name": "DROP",
    "target_ids": ["doc_4_sent_2"]
  },
  "corrected_action": {
    "type": "memory_action",
    "name": "PIN",
    "target_ids": ["doc_4_sent_2"]
  },
  "raw_evidence_ids": ["doc_4_sent_2", "final_eval_missing_fact"],
  "reflection": "The policy dropped a supporting fact needed for the final comparison.",
  "verifiable_reason": "The final answer required doc_4_sent_2, but it was absent from retrieved context.",
  "confidence": 0.86
}
```

### 4.5 OPD/SDFT Targets

你可以做两个版本。

**Action-level SDFT, lower resource**

```text
input: state_t + trajectory reflection
target: corrected memory action JSON
loss: CE only on corrected action tokens
```

优点是简单、稳定、适合 0.5B/1.5B context policy。

**Token-level OPD, closer to OpenClaw-RL**

```text
input: original prompt + hindsight reflection
target: original student action tokens
signal: teacher_log_prob - student_log_prob
loss: PPO-style clipped token-level OPD or top-k reverse-KL
```

优点是更“on-policy”，能保留当前 policy 的状态分布；缺点是工程和稳定性更难。

## 5. Baseline Implications

你的实验应该把 baselines 拆成三类。

### 5.1 Memory/Context Baselines

```text
Sliding window
Summary-only memory
RAG top-k
Rule context manager
MemGPT-style tiered memory
A-MEM-style graph memory
Memory-R1/AgeMem-style learned memory action, if reproducible
```

### 5.2 RL Baselines

```text
SFT only
GRPO final reward only
GRPO + segment reward
GRPO + process reward
```

### 5.3 Reflection/OPD Baselines

```text
Reflection only in context, no training
Failed trajectory SFT without structured reflection
Action-level SDFT from reflection
Token-level OPD from reflection
GRPO + action-level SDFT
GRPO + token-level OPD
GRPO + reflection + replay + KL
```

关键对照：

```text
generic reflection vs evidence-grounded reflection
reflection without raw evidence pointer vs with raw evidence pointer
teacher correction without confidence gating vs with confidence gating
episode-level reward only vs first-bad-turn localized reflection
```

## 6. Recommended Positioning for Your Paper

### 6.1 What Is Novel Enough

单独说“用失败轨迹做反思”不够新，因为 Reflexion、Agent-R、AgentHER、OpenClaw-RL 都有相关思想。

更强的新意应该是：

```text
Use policy-model trajectory review to convert failed long-context trajectories into evidence-grounded memory-operation corrections, then combine GRPO rollout ranking with OPD/SDFT action-level correction for transferable context management.
```

这把你的工作和已有论文区分开：

| Prior line | What they do | Your sharper angle |
|---|---|---|
| Reflexion | reflection stored in memory; no weight update | reflection becomes training label for memory action. |
| Agent-R | train reflection/recovery using MCTS paths | specialize reflection to memory/context lifecycle decisions. |
| AgentHER | relabel failed trajectories into new goals/data | relabel bad memory actions into corrected memory actions with raw evidence. |
| SDPO/RLSD | distill rich feedback/token-level signals | feedback is structured trajectory review over memory/tool operations. |
| Memory-R1/AgeMem | train memory operations with RL | add reflection/OPD to solve sparse reward and credit assignment. |

### 6.2 Best First Experiment

不要一开始做完整 coding agent。先做一个低资源、归因干净的实验：

```text
Dataset: HotpotQA / MuSiQue-style multi-hop QA or ALFWorld/SciWorld
Agent: fixed base LLM
Memory policy: small context manager
Failure source: missing evidence, wrong retrieve, over-compression
Reviewer: policy model/API model creates structured reflection
Training: action-level SDFT first
Later: add GRPO final/segment reward
```

最小可发论文实验：

```text
Rule manager
Action SFT on successful memory decisions
GRPO final reward only
Failure-reflection SDFT
GRPO + failure-reflection SDFT
```

核心指标：

```text
task_success
evidence_recall
first_bad_turn_detection_accuracy
memory_action_accuracy
retrieve_precision
compression_ratio
token_cost
reflection_accept_rate
memory_pollution_rate
failure_recovery_rate
```

## 7. Key Papers and How to Use Them

### Memory / Context Management

1. **MemoryBank: Enhancing Large Language Models with Long-Term Memory**  
   Use for: long-term memory with recall, update, forgetting/reinforcement, user/personality adaptation.

2. **MemGPT: Towards LLMs as Operating Systems**  
   Use for: virtual context management and tiered memory as an OS analogy.

3. **A-MEM: Agentic Memory for LLM Agents**  
   Use for: memory as dynamic notes/links/evolution rather than static vector retrieval; NeurIPS 2025.

4. **Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via RL**  
   Use for: explicit ADD/UPDATE/DELETE/NOOP operations trained by PPO/GRPO.

5. **Agentic Memory / AgeMem**  
   Use for: unified LTM/STM management as tool actions and step-wise GRPO.

6. **ContextBudget / BACM-RL**  
   Use for: context compression as sequential budget-aware RL.

7. **Memory-as-Action / MemAct**  
   Use for: working memory editing as learnable actions; note trajectory-fracture issue.

### Reflection / Failure Learning / OPD

1. **Reflexion: Language Agents with Verbal Reinforcement Learning**  
   Use for: failed trial feedback -> verbal reflection -> episodic memory.

2. **Self-Refine**  
   Use for: model-as-reviewer/refiner without extra training.

3. **Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training**  
   Use for: failed trajectory first-error localization and reflection training.

4. **AgentHER**  
   Use for: failed trajectories can be relabeled into useful SFT/DPO/ShareGPT data.

5. **Reinforcement Learning via Self-Distillation / SDPO**  
   Use for: scalar RLVR has credit assignment bottleneck; tokenized feedback gives dense self-distillation signal.

6. **OpenClaw-RL**  
   Use for: next-state signal split into evaluative reward and directive OPD hint; asynchronous training infrastructure.

7. **Self-Distilled RLVR / RLSD**  
   Use for: combine RLVR direction with self-distillation magnitude; warns about privileged teacher leakage.

## 8. Concrete Update to Your Blueprint

Add a new subsection to FASD-GRPO:

```text
Policy-Model Trajectory Review

Given a failed trajectory tau = (s_1, a_1, o_1, ..., s_T, a_T, o_T), a reviewer policy R inspects the trajectory, environment feedback, and raw evidence. R emits a structured reflection z containing failure_type, first_bad_turn, raw_evidence_ids, bad_action, corrected_action, and verifiable_reason. We filter z by confidence and evidence grounding. The corrected action becomes an action-level SDFT target or token-level OPD teacher context.
```

Then define loss:

```text
L_total =
  L_GRPO(episode/segment reward)
  + lambda_opd L_OPD(reflection-conditioned teacher)
  + lambda_reflect L_action_CE(corrected memory action)
  + lambda_replay L_success_replay
  + lambda_kl L_KL(reference policy)
```

Most important reviewer question you must answer:

```text
Does reflection actually add targeted credit assignment, or is the gain just more data?
```

Ablation to answer it:

```text
same failed trajectories + generic reflection
same failed trajectories + shuffled reflection
same failed trajectories + evidence-grounded reflection
same failed trajectories + evidence-grounded reflection but no corrected_action
```

If only evidence-grounded reflection with corrected_action wins, your claim is much stronger.

## Sources

- [Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://arxiv.org/abs/2508.19828)
- [Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents](https://arxiv.org/abs/2601.01885)
- [ContextBudget: Budget-Aware Context Management for Long-Horizon Search Agents](https://arxiv.org/abs/2604.01664)
- [Memory-as-Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://openreview.net/pdf?id=ddGsiaISXg)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
- [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training](https://arxiv.org/abs/2501.11425)
- [AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling](https://arxiv.org/abs/2603.21357)
- [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
- [OpenClaw-RL: Train Any Agent Simply by Talking](https://huggingface.co/papers/2603.10165)
- [OpenClaw-RL OPD README](https://github.com/Gen-Verse/OpenClaw-RL/blob/main/openclaw-opd/README.md)
- [Self-Distilled RLVR](https://arxiv.org/abs/2604.03128)
