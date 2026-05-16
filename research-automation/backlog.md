# Research Backlog

## P0

- [ ] 建立论文卡片库：LongCodeZip、ACON、Context-Folding、ContextBudget、Memex(RL)、LoCoEval。
- [ ] 定义第一版压缩对象 schema：`code_slice`、`dialogue_turn`、`tool_output`、`test_failure`、`patch_context`。
- [ ] 实现 LongCodeZip 风格的训练自由基线：函数级粗筛 + block 级细筛。
- [ ] 设计证据指针格式：文件路径、行号、run id、archive id、原始文本 hash。
- [ ] 建立第一版评测集：10 个代码 QA / repo 切片任务。

## P1

- [ ] 复现 ACON 风格 guideline 压缩：自然语言规则 + 小模型/LLM 压缩器。
- [ ] 建立工具输出压缩器：测试失败、堆栈、lint、diff。
- [ ] 增加 evidence recall 与 symbol recall 自动评分脚本。
- [ ] 设计 summary-only vs indexed-memory 对照实验。
- [ ] 记录 20 个压缩失败案例并做失败归因。

## P2

- [ ] 接入 LoCoEval 或构造 LoCoEval 风格 repo conversation 子集。
- [ ] 尝试 ContextBudget 风格预算信号：紧/中/宽三档上下文预算。
- [ ] 设计 RL action space：keep、drop、compress、archive、retrieve、fold。
- [ ] 构造可用于偏好优化或 GRPO 的过程奖励。
- [ ] 评估是否需要进入 Memex(RL) 风格索引化经验记忆。

## 决策记录

| 日期 | 决策 | 依据 | 后续动作 |
|---|---|---|---|
| 2026-05-15 | 先做独立式代码感知压缩 + 可回溯证据层 | 调研报告建议该路线工程成功率最高 | 建立流程、模板和第一版 backlog |

