# 自动化科研流程：长上下文压缩与 Coding Agent

本项目把 `deep-research-report.md` 的结论落成一套可执行的科研流程，目标是从“阅读前沿论文”稳定推进到“原型实现、基准评测、复现记录、路线决策”。

核心判断：

- 先做独立式、代码感知、可插拔的上下文压缩模块。
- 同时建设可回溯证据层：原文归档、索引、指针、回读。
- 先用 LongCodeZip / ACON 风格方法验证工程收益，再评估 ContextBudget / Context-Folding / Memex(RL) 风格 RL 升级。

## 目录

- `research-automation/pipeline.md`：端到端自动化科研流程。
- `research-automation/config.yaml`：研究主题、阶段、指标和节奏配置。
- `research-automation/backlog.md`：按阶段拆解的研究任务池。
- `research-automation/templates/`：论文卡片、实验卡片、评测报告模板。
- `research-automation/scripts/new_research_run.py`：生成一次研究迭代工作目录。

## 快速开始

创建一次新的研究迭代：

```powershell
python .\research-automation\scripts\new_research_run.py --topic "LongCodeZip baseline"
```

脚本会在 `research-automation/runs/` 下生成日期化目录，并复制需要填写的模板。

推荐节奏：

1. 每天新增或更新 1-3 张论文卡片。
2. 每周完成 1 个可运行实验或复现实验。
3. 每两周更新一次评测矩阵和路线决策。
4. 每个阶段结束时，用 `eval-report.md` 模板沉淀结论。

