# ICLR 2026 论文报告：Evoking User Memory / RF-Mem

论文：Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval  
会议：ICLR 2026 Poster  
作者：Yingyi Zhang, Junyi Li, Wenlin Zhang, Pengyue Jia, Xianneng Li, Yichao Wang, Derong Xu, Yi Wen, Huifeng Guo, Yong Liu, Xiangyu Zhao  
核心方法：RF-Mem, Recollection-Familiarity Memory Retrieval  
报告日期：2026-05-18

## 1. 一句话总结

这篇论文提出 RF-Mem：一个不微调检索器、不训练 LLM 的 personalized memory retrieval 框架。它把用户记忆检索分成两条路径：

```text
Familiarity path: 高置信、低不确定时，直接 one-shot dense top-K 检索。
Recollection path: 低置信或高不确定时，启动多轮 retrieve-cluster-mix，逐步重构证据链。
```

它的核心价值在于：不是简单地“检索更多”，而是根据 query 对 memory 的熟悉度信号，决定是否值得进入更慢但更深的 recollection 检索。

## 2. 论文要解决的问题

个性化 LLM 需要利用用户历史、偏好、上下文和长期交互记录。已有方法主要有两个极端：

```text
Full context:
把用户全部历史塞进 prompt。
问题：成本高、不可扩展，1M token 级别直接超上下文。

One-shot dense retrieval:
用 query embedding 对 memory embedding 做一次 top-K 检索。
问题：只捕捉表层相似，容易漏掉需要跨片段重构的个人经历、偏好变化和长尾证据。
```

RF-Mem 的判断是：personal memory retrieval 和普通 RAG 不一样。普通 RAG 多是从客观知识库找事实，而 user memory retrieval 要找的是某个用户过去说过、做过、偏好过、变化过的东西。很多问题不是靠一个相似片段就能解决，而需要“想起来一串相关经历”。

## 3. 认知科学动机

论文借用人类记忆的 dual-process theory：

```text
Familiarity:
快速、粗粒度、直觉式识别。
比如看到一个人觉得“我认识他”。

Recollection:
慢速、细节化、链式重构。
比如回忆这个人是谁、在哪里见过、当时发生了什么。
```

对应到 agent memory：

```text
Familiarity = 普通 embedding top-K 检索
Recollection = 多轮证据扩展和上下文重构
```

这给你的研究一个很好的理论支点：memory retrieval 不一定总是一个静态 top-K 操作，它可以是一个有路径选择的过程。

## 4. RF-Mem 方法结构

RF-Mem 有四个阶段：

```text
user query q
-> probe retrieval
-> familiarity signal estimation
-> choose Familiarity or Recollection path
-> retrieved memory text
-> LLM generation / evaluation
```

### 4.1 Familiarity signal

给定用户记忆库：

```text
M = {m_1, ..., m_M}
```

每个 memory fragment 被编码成 embedding：

```text
z_i = phi(m_i)
```

query 被编码成：

```text
x_t = phi(q)
```

先做一次 probe retrieval，拿到 top-K 相似度分数：

```text
s_i = cosine(x_t, z_i)
```

然后计算两个信号：

```text
mean score: s_bar
entropy: H(p)
```

其中 entropy 来自 softmax-normalized retrieval score distribution。直觉是：

```text
mean score 高：
query 和某些记忆明显相关，熟悉感强。

entropy 低：
相关证据集中在少数候选上，检索比较确定。

entropy 高：
候选分布分散，不确定，需要更深的 recollection。
```

### 4.2 路径选择规则

论文里的策略是：

```text
if s_bar >= theta_high:
    use Familiarity
elif s_bar <= theta_low:
    use Recollection
else:
    if H(p) <= tau:
        use Familiarity
    else:
        use Recollection
```

也就是说：

```text
高平均相似度 -> 直接 top-K
低平均相似度 -> 启动回忆
中间区域 -> 看 entropy
```

这不是 RL policy，而是一个基于分数和熵的 heuristic router。论文附录也尝试了 learned gate，但主方法仍是手工阈值。

### 4.3 Familiarity path

Familiarity path 就是标准 dense retrieval：

```text
C_t = TopK({(m_i, cosine(q, m_i))})
```

优点：

```text
快
稳定
延迟低
适合直接事实、强关键词匹配、明显偏好检索
```

缺点：

```text
只能抓表层匹配
可能漏掉分散在多段记忆里的证据
```

### 4.4 Recollection path

当 Familiarity signal 不足时，RF-Mem 启动多轮 recollection。

每轮流程：

```text
current query embedding x^(r)
-> retrieve top-N candidates
-> KMeans cluster candidate embeddings into B clusters
-> compute each cluster centroid g_b^(r)
-> alpha-mix current query and centroid
-> form recollect query x_b^(r+1)
-> retrieve again
```

核心公式可以理解成：

```text
x_b^(r+1) = norm(alpha * x^(r) + (1 - alpha) * g_b^(r) + x_t)
```

其中：

```text
x_t: 原始 query embedding，作为 residual 保留原问题语义
g_b: 当前候选簇的中心，代表一条可能的记忆线索
alpha: 控制保留原 query vs 探索新线索的比例
B: beam width，保留多少条 recollection branch
F: fanout，每轮扩大多少候选
R: 最大 recollection 轮数
```

最后把多轮拿到的 candidate memories 合并去重，再截断到 top-K。

## 5. 这篇论文的“检索”到底怎么做？需要微调吗？

不需要微调检索器。

论文强调它是 retrieval-only comparison，并且所有方法共享相同 memory vectors。它使用的是 frozen embedding retrievers，比如：

```text
multi-qa-MiniLM-L6-cos-v1
all-MiniLM-L6-v2
all-mpnet-base-v2
BAAI/bge-base-en-v1.5
```

RF-Mem 训练-free，主要改的是：

```text
retrieval strategy / routing
```

不是：

```text
retriever backbone
LLM backbone
memory encoder
```

这对你的研究很重要，因为它说明 ICLR 级别 memory retrieval 论文也会把 retriever 固定住，专注证明 memory retrieval/control strategy 的价值。

## 6. 数据集

论文用了三个 personalized memory benchmark。

### 6.1 PersonaMem

PersonaMem 是主实验数据集，面向个性化长记忆问答/生成。

它包含多轮用户-LLM 交互历史，覆盖不同现实任务和用户偏好。论文构建了三种 memory corpus scale：

```text
32K memory corpus
128K memory corpus
1M memory corpus
```

任务类型包括：

```text
Revisit Reasons
Track Evolution
Latest Preferences
Aligned Recommendations
New Scenarios
Shared Facts
New Ideas
```

这个数据集很适合测：

```text
用户偏好是否被记住
偏好变化是否被追踪
历史原因是否能回忆
新场景下是否能迁移旧偏好
```

### 6.2 PersonaBench

PersonaBench 更偏 retrieval evaluation。它包含私有用户文档和查询，用来评估 personal memory pieces 是否被正确取回。

论文在 PersonaBench 上主要看：

```text
Recall@5
Recall@10
```

### 6.3 LongMemEval

LongMemEval 是 long-term interactive memory benchmark，论文使用：

```text
LongMemEval-S
LongMemEval-M
```

它的问题需要在大量用户-AI 对话 session 中找回相关事实。

论文报告的规模：

```text
LongMemEval-S: 500 questions, about 25,112 session-level memories
LongMemEval-M: 500 questions, about 250,948 session-level memories
```

## 7. Baseline 设置

论文 baseline 非常克制，主要是 retrieval-only：

```text
Zero Memory:
不使用用户记忆。

Full Context:
把全部用户历史塞进上下文。

Dense Retrieval / Familiarity:
标准 one-shot embedding top-K。

Recollection-only:
总是走 recollection path，不做 adaptive routing。

RF-Mem:
根据 mean score + entropy 自适应切换。
```

这里有一个关键实验设计原则：

```text
所有方法使用同一套 memory vectors。
```

也就是说，论文没有把提升归因给更强 encoder，而是归因给 retrieval path selection 和 recollection expansion。

## 8. 实验结果

### 8.1 PersonaMem 生成准确率

论文在 PersonaMem 32K / 128K / 1M 上比较个性化回答 accuracy。

关键结果：

```text
32K:
Full Context overall = 0.6129
Dense Retrieval overall = 0.5908
Recollection-only overall = 0.6214
RF-Mem overall = 0.6350

128K:
Full Context overall = 0.3231
Dense Retrieval overall = 0.5259
Recollection-only overall = 0.5288
RF-Mem overall = 0.5394

1M:
Full Context = out-of-context
Dense Retrieval overall = 0.4518
Recollection-only overall = 0.4544
RF-Mem overall = 0.4589
```

解读：

```text
1. 小规模 32K 时，Full Context 还可用，但 RF-Mem 更省 token 且更准。
2. 128K 时，Full Context 明显退化。
3. 1M 时，Full Context 直接不可用。
4. Dense Retrieval 快，但对复杂记忆重构不够。
5. Recollection-only 更深，但不是总是更好。
6. RF-Mem 的优势来自“该快时快，该深时深”。
```

### 8.2 Retrieval recall

在 LongMemEval 上，RF-Mem 在 MiniLM / MPNet / BGE 三种 retriever backbone 下都比较稳定。

以 BGE 为例：

```text
LongMemEval-S:
Familiarity Recall@5 = 0.7924
Recollection Recall@5 = 0.8162
RF-Mem Recall@5 = 0.8186

LongMemEval-M:
Familiarity Recall@5 = 0.4964
Recollection Recall@5 = 0.5131
RF-Mem Recall@5 = 0.5155
```

RF-Mem 的特点不是每个指标都大幅领先，而是：

```text
更稳
延迟低于 always-on recollection
性能高于 one-shot familiarity
```

### 8.3 Latency / token

RF-Mem 在 PersonaMem 中 retrieval latency 接近 dense retrieval，但比 recollection-only 更低。

例如 32K：

```text
Dense Retrieval: 3.14ms, 3515.9 tokens
Recollection-only: 7.09ms, 3711.1 tokens
RF-Mem: 5.09ms, 3566.6 tokens
```

结论是：

```text
RF-Mem 用少量额外延迟换取更好的 recall/accuracy。
```

## 9. 论文贡献

### 9.1 贡献一：把 personalized memory retrieval 建模成双路径系统

它不是简单地说“top-K 不够，所以多检索几轮”，而是提出：

```text
不是所有 query 都需要深检索。
只有不熟悉 / 不确定时才需要 recollection。
```

### 9.2 贡献二：用 mean score + entropy 做轻量 router

这个 router 的优点是：

```text
不需要训练
不需要标注
不需要 LLM 判断
只依赖 retrieval score distribution
```

这让它很容易作为你的 baseline 或模块。

### 9.3 贡献三：embedding-space recollection

Recollection path 不调用 LLM 生成新 query，而是在 embedding space 里做：

```text
retrieve
cluster
centroid
alpha-mix
retrieve again
```

这比 LLM query expansion 更便宜，也更可控。

### 9.4 贡献四：RF-Mem 可以叠加到其他 memory index 上

论文实验显示 RF-Mem 可以和 MemoryBank summary index、HyDE query expansion、Search-o1 iterative retrieval 组合。

这意味着 RF-Mem 是一个 online retrieval layer，而不是完整 memory system。

## 10. 局限性

### 10.1 它不解决 memory 写入

RF-Mem 假设 memory bank 已经存在。它研究的是：

```text
如何从已有 memory 中取回
```

不是：

```text
哪些内容应该写入 memory
怎么更新 memory
怎么删除污染 memory
```

### 10.2 它不是训练方法

RF-Mem 是 training-free retrieval strategy，不是 RL，也不是 SDFT/OPD。

所以它不能直接解决：

```text
失败轨迹怎么变成训练监督
模型怎么学会何时 KEEP / COMPRESS / ARCHIVE / RETRIEVE
```

### 10.3 它对 embedding retriever 质量仍敏感

虽然论文测试了 MiniLM / MPNet / BGE，但 RF-Mem 仍然建立在 embedding similarity 上。如果底层 embedding 完全无法捕捉语义，recollection 也只能在错误空间里扩展。

### 10.4 Recollection 不一定总赢

论文自己的结果显示，always-on Recollection 不总是最佳。简单 query 走 Recollection 可能引入噪声和额外延迟。

这恰恰支持它的 adaptive router。

## 11. 对你的研究的启发

### 11.1 RF-Mem 可以作为强 baseline

你的方法如果研究 context manager / memory policy，可以把 RF-Mem 作为 retrieval baseline：

```text
Dense Retrieval
Recollection-only
RF-Mem
Rule context manager
Ours reflection-OPD context manager
```

尤其适合 LoCoMo / LongMemEval / PersonaBench 这种 user memory 场景。

### 11.2 你可以借用它的 familiarity signal 作为 state feature

你的 context manager 输入里可以加入：

```json
{
  "mean_retrieval_score": 0.54,
  "retrieval_entropy": 0.18,
  "topk_score_gap": 0.07,
  "retrieval_mode": "familiarity|recollection"
}
```

这能帮助 policy 判断：

```text
当前是不是熟悉问题？
是否需要回读更多 memory？
是否需要深检索？
是否要压缩当前 evidence？
```

### 11.3 你的研究可以补 RF-Mem 没做的部分

RF-Mem 管的是：

```text
retrieval path selection
```

你的方法可以管：

```text
memory lifecycle control
KEEP / COMPRESS / ARCHIVE / RETRIEVE / UPDATE / DROP / PIN
```

也就是说，RF-Mem 决定“怎么找”，你的 context manager 决定“哪些证据值得被保留、压缩、归档、回读，以及失败后怎么修正”。

### 11.4 Reflection/OPD 可以学习 RF-Mem 的 router

RF-Mem 主要是 heuristic gate。你可以进一步做：

```text
失败轨迹：
Dense retrieval 没找回 supporting memory
-> reviewer 发现应触发 recollection
-> corrected action = RETRIEVE_DEEP / RECOLLECT
-> OPD/SDFT 训练 context policy
```

这会形成一个很自然的扩展：

```text
RF-Mem: hand-crafted uncertainty gate
Ours: failure-reflection-trained memory operation policy
```

## 12. 建议你如何把它写进论文

可以在 related work 里这样定位：

```text
RF-Mem demonstrates that personalized memory retrieval benefits from adaptive switching between fast one-shot familiarity retrieval and slower stepwise recollection. However, RF-Mem remains a training-free retrieval-layer method and assumes a pre-existing memory bank. In contrast, our method learns a memory operation policy that controls the lifecycle of evidence, including keeping, compressing, archiving, retrieving, updating, and dropping memories. Moreover, failed trajectories are reviewed into evidence-grounded reflections and distilled into action-level corrections, enabling targeted credit assignment beyond retrieval uncertainty heuristics.
```

实验上可以这样设计：

```text
Fixed embedding retriever
Dense Retrieval
RF-Mem
RF-Mem + rule lifecycle manager
Ours reflection-OPD lifecycle manager
Ours + RF-Mem retrieval layer
```

这样你既尊重了 ICLR 2026 强 baseline，又能清楚说明你的增量。

## 13. 适合作为你系统模块的最小实现

可以直接实现一个 RF-Mem style retriever：

```python
def route(scores, theta_low, theta_high, tau):
    mean_score = scores.mean()
    probs = softmax(lambda_ * (scores - scores.max()))
    entropy = -(probs * log(probs)).sum()

    if mean_score >= theta_high:
        return "familiarity"
    if mean_score <= theta_low:
        return "recollection"
    return "familiarity" if entropy <= tau else "recollection"
```

Recollection path：

```text
for r in range(R):
    candidates = topN(query_embedding)
    clusters = KMeans(candidates, B)
    for centroid in clusters:
        recollect_query = norm(alpha * current_query + (1-alpha) * centroid + original_query)
        retrieve again
```

你的 memory action policy 可以把它包装成工具：

```json
{
  "action": "RETRIEVE",
  "mode": "familiarity|recollection",
  "target": "user_memory",
  "budget": 10
}
```

## 14. 最终判断

RF-Mem 是你当前方向里非常值得借鉴的一篇 ICLR 2026 论文，但它和你的研究目标有明确区别：

```text
RF-Mem:
训练-free，解决 retrieval strategy。

你的方向:
训练 context/memory policy，解决 memory lifecycle 和失败信用分配。
```

它最适合作为：

```text
1. 强 retrieval baseline
2. memory policy 的 state feature
3. recollection-style RETRIEVE action 的实现工具
4. 证明“复杂 memory 操作不一定需要微调 retriever”的论文依据
```

## Sources

- OpenReview: https://openreview.net/forum?id=f7p0F2X6XN
- Paper PDF: https://openreview.net/pdf/299e700211a41d6876a8d790ce8ee11b530df900.pdf
- arXiv: https://arxiv.org/abs/2603.09250
