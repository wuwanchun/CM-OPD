# Slime Context Manager 中文说明

本目录用于实现 **外部 context manager / memory policy** 的训练与评测流程。这里不训练主 agent，不更新主 LLM 参数；训练对象是一个轻量的外部上下文管理策略，用来决定长链路任务中证据的生命周期。

对应英文说明见：[README.md](README.md)

## 核心目标

外部 policy 的输入是当前任务状态、memory item、上下文预算和已有 memory inventory；输出是一个 memory action：

```text
KEEP / COMPRESS / ARCHIVE / RETRIEVE / UPDATE / DROP / PIN
```

主 agent 仍然负责推理、回答、调用任务工具。外部 context manager 只负责决定：

```text
哪些证据保留在 working context
哪些证据压缩成摘要
哪些证据归档但保留 raw pointer
什么时候回读原文
哪些内容需要固定或禁止删除
```

## OPD 自训练流程

当前实现参考 OpenClaw-RL 的 OPD 思路，但把训练对象从完整 agent response 改成 memory action：

```text
student memory action
-> next-state feedback
-> hindsight hint
-> teacher correction 或 teacher log-probs
-> slime-compatible sample
```

具体步骤：

1. 当前外部 policy 输出 memory action。
2. 系统暂存该 turn 的 `prompt_ids`、`response_ids`、`rollout_log_probs`、`memory_id` 和 `turn_id`。
3. 下一轮状态到来后，例如环境反馈、工具返回、最终 evaluator 反馈，触发 hindsight judge。
4. judge 输出 `\boxed{1}`、`\boxed{0}` 或 `\boxed{-1}`，正样本需要包含 `[HINT_START]...[HINT_END]`。
5. 如果 hint 有效且包含 `raw_evidence_id`，则构造 OPD 训练样本。
6. 样本可以进入 Action OPD、Token OPD 或未来的 Top-K OPD / GRPO 流程。

## 文件结构

```text
research-automation/slime-context-manager/
├── README.md
├── README.zh-CN.md
├── opd/
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

关键模块：

| 模块 | 作用 |
|---|---|
| `action_parser.py` | 解析 memory action JSON，处理非法 action |
| `hindsight_judge.py` | 解析 judge 输出、选择最佳 hindsight hint |
| `context_policy_opd_server.py` | OPD recorder、多轮 pending turn、可选 OpenAI-compatible proxy |
| `teacher_logprob.py` | 构造带 hint 的 teacher prompt，并对齐 teacher log-probs |
| `opd_rollout.py` | slime rollout bridge 和 custom reward hook |
| `topk_distillation_loss.py` | 预留 Top-K OPD reverse-KL loss |

## OPD 模式

| 模式 | 当前状态 | 说明 |
|---|---|---|
| Action OPD | 已实现骨架 | teacher 生成 corrected memory action，用于 action-level CE / SFT |
| Token OPD | 已预留接口 | teacher 对原始 action tokens 计算 log-probs |
| Top-K OPD | 已预留 loss | 使用 teacher top-K 分布做 reverse-KL 蒸馏，默认关闭 |
| GRPO | 后续扩展 | 已保留 turn-level reward 和 episode-level record |

第一阶段默认使用 **Action OPD**。它资源成本最低，也最适合先验证外部 policy 是否能学会更好的记忆动作。

## 多轮对话与过程奖励

每个 episode 使用 `session_id`，每个 memory decision 使用 `turn_id`。当前 turn 的 action 会先进入 pending 状态，直到下一状态到来后再判断是否生成 OPD 样本。

同时保留过程奖励字段：

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

后续如果接 GRPO，可以使用：

```text
reward = final_reward
       + alpha * sum(process_reward_t)
       + beta * evidence_recall
       - gamma * token_cost
```

## Slime 接入口

后续接入 slime 时预留以下入口：

```text
--rollout-function-path opd.opd_rollout.generate_rollout_opd
--custom-rm-path opd.opd_rollout.custom_rm
--loss-type custom_loss
--custom-loss-function-path opd.topk_distillation_loss.topk_distillation_loss_function
```

首版不执行 GRPO。`opd_rollout.py` 主要负责收集 OPD 样本并返回 slime-compatible sample。

## 快速测试

在仓库根目录运行：

```powershell
python -m unittest discover -s research-automation/slime-context-manager/tests -v
```

当前测试覆盖：

- hindsight hint parser
- 多 vote hint selection
- memory action parser
- OPD sample builder
- `response_length / loss_mask / teacher_log_probs` 对齐
- 无 `raw_evidence_id` 样本默认不进入训练
- 多轮 pending turn 清理
- episode-level reward record

## 当前边界

## 完整本地 Smoke Pipeline

无需联网的 toy pipeline：

```powershell
python research-automation/slime-context-manager/scripts/run_smoke_pipeline.py
```

它会依次完成：

```text
生成 toy raw dataset
-> 构造 context_action 数据
-> 导出 slime JSONL
-> 运行 rule policy
-> 生成 prediction JSONL
-> 计算 metrics JSON
-> 生成 SVG 图表和 analysis_report.md
```

输出目录：

```text
research-automation/slime-context-manager/runs/smoke/
```

## 常用命令

进入项目目录：

```powershell
cd research-automation/slime-context-manager
```

安装默认依赖：

```powershell
pip install -r requirements.txt
```

生成 toy 数据：

```powershell
python scripts/download_datasets.py --dataset toy
python scripts/build_context_action_dataset.py --adapter toy
```

下载并处理 HotpotQA：

```powershell
python scripts/download_datasets.py --dataset hotpotqa --max-samples 200
python scripts/build_context_action_dataset.py --adapter hotpotqa --limit 500 --save-hf-dataset
```

导出 slime 训练数据：

```powershell
python scripts/export_slime_jsonl.py --split train
```

HF 模型导入 dry run：

```powershell
python scripts/import_hf_model.py --model Qwen/Qwen2.5-0.5B-Instruct --dry-run
```

部署 rule policy：

```powershell
python scripts/serve_policy.py --policy rule --host 127.0.0.1 --port 8088
```

运行 rule baseline 推理和评测：

```powershell
python scripts/run_context_policy.py --policy rule --split test --output runs/rule_predictions.jsonl
python scripts/evaluate_predictions.py --split test --predictions runs/rule_predictions.jsonl --output runs/metrics.json
python scripts/visualize_results.py --metrics runs/metrics.json --output-dir runs/analysis
```

生成 slime SFT 命令：

```powershell
python scripts/train_sft_slime.py --train-jsonl data/slime/context_actions_train.jsonl
```

## 依赖文件

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 本地数据处理、HF 模型导入、部署、评测、可视化 |
| `requirements-dev.txt` | 测试和开发工具 |
| `requirements-slime.txt` | 可选 heavy training stack，训练机上再装 |

## 当前边界

当前实现已经从单纯 OPD 骨架扩展为可 smoke test 的训练-评测项目。它已经具备：

- 外部 policy OPD 数据结构
- 多轮 session 管理
- Action OPD 样本构造
- Token OPD / Top-K OPD 扩展接口
- GRPO 过程奖励字段预留
- toy / HotpotQA 数据处理入口
- HF 模型导入 dry run 和模型 policy wrapper
- rule policy 部署与推理脚本
- slime JSONL 导出与训练命令生成
- metrics 评测和 SVG/Markdown 可视化分析

还未实现的部分：

- ALFWorld / ScienceWorld 真实环境 rollout wrapper
- 真实 slime GPU 训练验证
- Token OPD 连接真实 teacher log-prob server
- GRPO 训练执行

这些会在下一阶段接到当前 OPD 骨架上。
