# Slime Progressive Evidence Disclosure 中文说明

本目录用于实现 **progressive evidence disclosure / summary-to-raw expansion** 的训练与评测流程。这里不训练主 agent，不更新主 LLM 参数，也不主张完整长期记忆系统；训练对象是一个轻量外部 policy，用来判断摘要是否足够，或是否需要展开原文。

对应英文说明见：[README.md](README.md)

## 核心目标

外部 policy 的输入是任务状态、候选片段的 cue-preserving summary、上下文预算和当前渲染状态；输出是三类 progressive disclosure 动作：

```text
HIDE
KEEP_SUMMARY
EXPAND_TO_RAW
```

主 agent 仍然负责推理、回答、调用任务工具。外部 selector 只负责决定：

```text
哪些 summary 当前不放入 prompt
哪些 summary 保持摘要视图即可
哪些 summary 信息不足，需要展开 RAW 原文
```

## Teacher-Guided 训练流程

当前实现保留 OpenClaw-RL 风格 OPD/SDFT 接口，但论文主线不是“新的蒸馏算法”。主线是 progressive evidence disclosure；teacher target 是基于 outcome 的 policy-improvement signal，不是客观真值。

```text
所有候选片段先生成 cue-preserving summary
-> 当前 policy 决定 HIDE / KEEP_SUMMARY / EXPAND_TO_RAW
-> 固定主模型回答
-> evaluator 返回答案对错 / 证据指标
-> teacher 分块审查 question + summary + outcome，必要时查看局部 raw
-> teacher 给出 HIDE / KEEP_SUMMARY / EXPAND_TO_RAW target
-> 训练下一轮 disclosure policy
-> fixed held-out split 验证是否真实提升
```

teacher 的优势不来自一次性读取完整长文本，而来自：

```text
分块审查：逐个 span 看 question + summary + outcome
结果特权：训练集可看 gold answer / failed answer / verifier feedback
对比特权：同题成功/失败 rollout 的差异帮助定位关键 span
```

每条 teacher target 必须包含 `source_id`、`quoted_summary`、必要时的 `quoted_raw` 和 `verifiable_reason`。`teacher_only` 样本可以训练，但必须和 counterfactual / verifier / gold-support 标签分开统计。

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
| `action_parser.py` | 解析 selector JSON，处理非法 target |
| `hindsight_judge.py` | 解析 judge 输出、选择最佳 hindsight hint |
| `context_policy_opd_server.py` | OPD recorder、多轮 pending turn、可选 OpenAI-compatible proxy |
| `teacher_logprob.py` | 构造带 hint 的 teacher prompt，并对齐 teacher log-probs |
| `opd_rollout.py` | slime rollout bridge 和 custom reward hook |
| `topk_distillation_loss.py` | 预留 Top-K OPD reverse-KL loss |

## 训练模式

| 模式 | 当前状态 | 说明 |
|---|---|---|
| Progressive Disclosure CE | 主路径 | 训练 `HIDE/KEEP_SUMMARY/EXPAND_TO_RAW` 动作 |
| Visibility OPD | 兼容路径 | 保留旧版三态 visibility JSON 标签 |
| Token OPD | 已预留接口 | teacher 对原始 selector tokens 计算 log-probs |
| Top-K OPD | 已预留 loss | 使用 teacher top-K 分布做 reverse-KL 蒸馏，默认关闭 |
| GRPO | 后续扩展 | 已保留 turn-level reward 和 episode-level record |

第一阶段默认使用 **Progressive Disclosure CE**。它资源成本最低，也最适合验证外部 policy 是否能学会什么时候 summary 不够、需要展开 RAW。

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
- progressive disclosure action parser
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

FASD-GRPO / CodeHER-GRPO 的主测试入口直接读取 HuggingFace 数据集，不再以 toy/smoke 作为验收口径：

```powershell
python scripts/run_fasd_grpo_hf_dataset.py --dataset hotpotqa/hotpot_qa --config distractor --split validation --max-rows 8 --k-rollouts 4
```

输出包括 grouped rollout records、segment records 和 FASD-GRPO samples，默认写入：

```text
data/grpo_sdft_hf/
```

失败轨迹的异步 SDFT 入口：

```powershell
python scripts/run_sdft_async_hf_dataset.py --dataset hotpotqa/hotpot_qa --config distractor --split validation --max-rows 8 --k-rollouts 4
```

这条链路会筛选失败 selector decisions，用 `asyncio` 并发提取 hindsight hints，生成 source-grounded correction samples，并验证 slime-compatible rollout batch。

基于生成的 correction 数据，在单卡上实际跑训练：

```powershell
python scripts/train_sdft_hf_local.py --model-path C:\path\to\local\checkpoint --train-jsonl data\sdft_async_hf\sdft_slime_sft_validation.jsonl --output-dir checkpoints\context_policy_sdft --max-steps 16 --dtype float32
```

这个训练脚本只对 corrected selector target 的 response tokens 计算 loss，默认用 `float32` 保证单卡小规模训练稳定。当前机器上的 slime 包如果只有 rollout 模块、没有通用 `python -m slime` 训练入口，可以先用这个入口完成本地训练闭环，再把 `grpo_sdft.async_sdft_rollout.generate_rollout_sdft` 接到机器上的 slime launcher。

在远端单卡机器上走 THUDM/slime async 训练入口：

```bash
cd /root/CM-OPD/research-automation/slime-context-manager
CONDA_PREFIX=/root/miniconda3/envs/slime bash scripts/run_sdft_slime_async_qwen3_0_6b.sh
```

这个 launcher 会把 `grpo_sdft.async_sdft_rollout.generate_rollout_sdft` 作为 `--rollout-function-path` 接入 slime，读取 `prompt/label/metadata` SDFT JSONL，补齐 `tokens/response_length/loss_mask`，并使用 slime 原生 `sft_loss` 训练。如果 Megatron torch-dist checkpoint 不存在，会先从本地 HF checkpoint 转换。root 文件系统空间紧张时，可以覆盖 `TORCH_DIST_LOAD` 或 `SAVE_DIR`。

导出 slime 训练数据：

```powershell
python scripts/export_slime_jsonl.py --split train
```

HF 模型导入 dry run：

```powershell
python scripts/import_hf_model.py --source-local-path C:\path\to\local\checkpoint --local-dir models/qwen2_5_0_5b_instruct --dry-run
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

实际启动入口推荐用这两个脚本：

```powershell
# 评测入口：默认使用本地数据路径，输出到 runs/eval_local
python scripts/run_eval_local.py --policy rule --split test

# 训练入口：默认使用本地模型路径 models/qwen2_5_0_5b_instruct
python scripts/run_train_local.py --local-model-path models/qwen2_5_0_5b_instruct

# 单卡真实训练 smoke：仍然使用导出的 slime JSONL，但用 Transformers 跑少量 SFT step；默认 float32 更稳
python scripts/run_train_local.py --backend hf-smoke --local-model-path C:\path\to\local\checkpoint --output-dir checkpoints\context_policy_hf_sft_smoke
```

`hf-smoke` 用来验证本地 checkpoint、CUDA、导出的 slime JSONL、response-token loss 是否能在单卡上真实跑通。完整 slime/Ray/Megatron 训练仍然是生产路径；如果当前安装的 slime 包没有 `python -m slime` 入口，先用 `hf-smoke` 做 3090 单卡训练闭环，再把 `opd.generate_rollout_opd` 接到机器上的 slime 集群启动脚本。训练 smoke 遇到非有限 loss 会直接失败。

HF 模型导入也走本地路径。默认不联网下载，只登记你机器上已有的本地 checkpoint：

```powershell
python scripts/import_hf_model.py --source-local-path C:\path\to\local\checkpoint --local-dir models/qwen2_5_0_5b_instruct
```

如果你希望复制一份到项目目录，再加 `--copy`：

```powershell
python scripts/import_hf_model.py --source-local-path C:\path\to\local\checkpoint --local-dir models/qwen2_5_0_5b_instruct --copy
```

如果训练机已经有本地 checkpoint，可以直接传：

```powershell
python scripts/run_train_local.py --local-model-path C:\path\to\local\checkpoint
python scripts/run_train_local.py --backend hf-smoke --local-model-path C:\path\to\local\checkpoint
python scripts/run_eval_local.py --policy model --model-path C:\path\to\local\checkpoint
```

## 依赖文件

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 本地数据处理、HF 模型导入、部署、评测、可视化 |
| `requirements-dev.txt` | 测试和开发工具 |
| `requirements-slime.txt` | 可选 heavy training stack，训练机上再装 |

## 当前边界

当前实现已经从单纯 OPD 骨架扩展为可运行的训练-评测项目。它已经具备：

- 外部 progressive disclosure policy 数据结构
- 多轮 session 管理
- progressive disclosure / visibility 样本构造
- Token OPD / Top-K OPD 扩展接口
- GRPO 过程奖励字段预留
- toy / HotpotQA 数据处理入口
- 本地 HF checkpoint 登记/复制和模型 policy wrapper
- rule policy 部署与推理脚本
- slime JSONL 导出与训练命令生成
- metrics 评测和 SVG/Markdown 可视化分析

还未实现的部分：

- ALFWorld / ScienceWorld 真实环境 rollout wrapper
- 真实 slime GPU 训练验证
- Token OPD 连接真实 teacher log-prob server
- GRPO 训练执行

这些会在下一阶段接到当前 progressive disclosure 训练骨架上。
