# 多 Agent 辩论与协作

> Du et al.（ICML 2024，"Society of Minds"）运行 N 个模型实例独立提出答案，再通过 R 轮互相批评收敛。提升事实正确性、规则遵循和推理能力。稀疏拓扑在 token 成本上优于全网状。

**类型：** 概念学习 + 动手实现
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 05（Self-Refine 和 CRITIC）
**时间：** 约 60 分钟

## 学习目标

- 解释辩论协议：N 个提议者、R 轮，在共享答案上收敛。
- 描述辩论为何能提升事实正确性、规则遵循和推理。
- 解释稀疏拓扑：不是每个辩手都需要看到其他所有辩手。
- 用标准库实现基于脚本化 LLM 的辩论，含全网状和稀疏变体；测量 token 成本与准确率。

## 问题背景

Self-Refine（第 5 课）是一个模型自我批评——有群体思维风险。CRITIC（第 5 课）用外部工具为批评提供事实依据——但外部工具并非随时可用。辩论引入第三种模式：多个实例、交叉批评、以分歧收敛。

## 核心概念

### Society of Minds（Du et al.，ICML 2024）

- N 个模型实例独立对同一问题提出答案。
- 经过 R 轮，每轮每个模型阅读其他人的提议并批评。
- 模型根据批评更新自己的答案。
- R 轮后返回收敛后的答案。

原始实验因成本限制使用 N=3、R=2。在难题（MMLU、GSM8K、国际象棋走法有效性、传记生成）上，更多 Agent 和更多轮次能提升准确率。

跨模型组合优于单模型辩论：ChatGPT + Bard 共同 > 任一单独表现。

### 稀疏拓扑

"Improving Multi-Agent Debate with Sparse Communication Topology"（arXiv:2406.11776，2024–2025）表明全网状辩论并非总是最优。稀疏拓扑（星形、环形、中心辐射）可以用更低的 token 成本达到相同准确率。每个辩手只看到部分同伴。

启示：

- 全网状 N=5、R=3 = 5 × 3 = 15 个提议，每个读 4 个同伴 = 60 次批评操作。
- 星形 N=5、R=3（一个中心 + 4 个辐射节点）= 15 个提议，辐射节点只读中心 = 12 次批评操作。

### 辩论何时有效

- **事实正确性。** N 个独立提议，交叉检查减少幻觉。
- **规则遵循。** 国际象棋走法有效性——一个模型漏掉规则，其他模型可以发现。
- **开放性推理。** 多种框架收敛到正确答案。

### 辩论何时有害

- **延迟敏感的 UX。** N × R 串行轮次带来的延迟可能无法承受。
- **成本敏感的规模化。** 每个问题消耗 N × R 个 token。
- **简单的事实查询。** 一次查询比五次辩论更便宜。

### 2026 年实践落地

- **Anthropic orchestrator-workers**（第 12 课）— 带综合步骤的辩论变体。
- **LangGraph supervisor**（第 13 课）— 中央路由 + 专家 Agent 可以将辩论实现为一个节点。
- **OpenAI Agents SDK**（第 16 课）— Agent 之间来回交接进行迭代批评。
- **多 Agent 评估** — 辩论 + 评估器-优化器配对获取评估信号。

### 这个模式的常见误区

- **收敛崩塌。** 所有 Agent 收敛到第一个错误答案。用强制分歧轮次缓解。
- **中心节点失效。** 星形拓扑中，坏的中心节点会污染所有人。轮换或使用多个中心。
- **Prompt 同质化。** 所有 Agent 用同一个 Prompt；产生相同的答案。使用多样 Prompt 和/或不同模型。

## 动手实现

`code/main.py` 实现标准库辩论：

- `Debater` 类（带每个辩手观点漂移的脚本化 LLM）。
- `FullMeshDebate` 和 `SparseDebate` 运行器。
- 三个问题：一个事实型、一个规则型、一个推理型。
- 指标：收敛答案、收敛轮数、总批评操作数。

运行：

```
python3 code/main.py
```

输出：每个协议的准确率和成本；稀疏方案在 2/3 问题上的准确率与全网状持平，成本更低。

## 用现成库

- **Anthropic orchestrator-workers** 用于简单的 2–3 个 Worker 辩论。
- **LangGraph** 用于带检查点的有状态多轮辩论。
- **自建** 用于研究或特殊正确性保证。

## 产出

`outputs/skill-debate.md` 脚手架一个多 Agent 辩论系统，可配置拓扑、N、R 和收敛规则。

## 练习

1. 实现"强制分歧"规则：第一轮每个辩手必须产生不同的提议。测量对收敛速度的影响。
2. 添加置信度加权聚合：辩手返回（答案，置信度）；聚合器按置信度加权。有帮助吗？
3. 将一个"Agent"替换为持不同观点的脚本化 LLM。异质性能提升准确率吗？
4. 在你的三个问题上测量全网状与稀疏的 token 成本。绘制成本 vs 准确率图。
5. 阅读 Society of Minds 论文。将你的玩具扩展到 N=5、R=3。什么会出问题？什么会改善？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Debate | "多 Agent 批评" | N 个提议者，R 轮交叉批评，在答案上收敛 |
| Full mesh | "全互联" | 每轮每个辩手阅读所有同伴 |
| Sparse topology | "受限视野" | 辩手只阅读部分同伴 |
| Hub-and-spoke | "星形拓扑" | 一个中心辩手，N-1 个辐射节点只读中心 |
| Convergence | "共识" | 辩手在共享答案上收敛 |
| Society of Minds | "Du et al. 辩论论文" | ICML 2024 多 Agent 辩论方法 |

## 延伸阅读

- [Du et al.，Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 经典多 Agent 辩论
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — 稀疏拓扑结果
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers 作为辩论变体
- [Madaan et al.，Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 单模型自我批评对比