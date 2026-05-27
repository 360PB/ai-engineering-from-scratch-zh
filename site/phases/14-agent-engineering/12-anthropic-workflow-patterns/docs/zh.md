# Anthropic 的工作流模式：简单优于复杂

> Schluntz 与 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）与 Agent（动态工具调用）。五种工作流模式覆盖了大多数场景。从直接调用 API 出发。只有在步骤无法预测时才引入 Agent。

**类型：** 学习 + 动手实现
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（Agent 循环）
**时长：** 约 60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：提示链、路由、并行化、编排器-工作者、评估器-优化器。
- 解释 Agent 与工作流的区别，以及各自的工程成本。
- 判断何时选工作流、何时选 Agent（以及反过来）。
- 用标准库对一个脚本化 LLM 实现全部五种模式。

## 问题所在

团队在只需要一次函数调用的问题上，也会去用多 Agent 框架。成本是真实存在的：框架增加了层级，掩盖了提示词，隐藏了控制流，引入了过早的复杂性。Schluntz 和 Zhang 2024 年 12 月的文章是业界最被引用的反思：先从简单的做起，只有当复杂性证明了自己的价值时才引入。

## 核心概念

### 工作流与 Agent 的区别

- **工作流。** 通过预定义的代码路径编排 LLM 和工具。工程师拥有这张图。
- **Agent。** LLM 动态指挥自己的工具并自主决定下一步。模型拥有这张图。

两者各有其用。工作流更便宜、更快、更容易调试。Agent 解锁了开放性问题，但也让失败模式更难推理。

### 增强型 LLM

所有五种模式的基石：一个 LLM 具备三种能力——搜索（检索）、工具（行动）、记忆（持久化）。任何 API 调用都可以使用这些能力。

### 五种模式

1. **提示链。** 第 1 次调用的输出是第 2 次调用的输入。适用于任务有清晰线性分解的场景。步骤之间可以有可选的程序化关卡。
2. **路由。** 一个分类器 LLM 决定调用哪个下游 LLM 或工具。适用于分类上不同的输入需要不同处理（一线支持 vs 退款 vs 缺陷 vs 销售）。
3. **并行化。** 同时运行 N 个 LLM 调用，聚合结果。两种形态：分段（不同分块）和投票（相同提示，N 次运行，取多数/综合）。
4. **编排器-工作者。** 一个编排器 LLM 动态决定运行哪些工作者（也是 LLM）并综合它们的输出。类似 Agent 循环，但编排器不会无限循环。
5. **评估器-优化器。** 一个 LLM 提出答案，另一个 LLM 评估。迭代直到评估通过。这是 Self-Refine（第 05 课）的泛化。

### 工作流优于 Agent 的场景

- **可预测的任务。** 如果你能枚举步骤，就应该枚举。
- **成本受限的任务。** 工作流有有限的步数上限；Agent 可能螺旋上升。
- **合规受限的任务。** 审计员想读这张图，而不是从轨迹中推断。

### Agent 优于工作流的场景

- **开放性研究。** 下一步依赖于上一步返回了什么。
- **长度可变的任务。** 从几分钟到几小时的工作，步数未知。
- **新领域。** 你还不知道正确的工作流——先探索，再固化。

### 上下文工程配套

"AI Agent 的有效上下文工程"（Anthropic，2025）将相邻学科形式化：200k 窗口是一笔预算，而非容器。何时包含、何时压缩、何时让上下文增长。详细内容在第 14 阶段关于上下文压缩的课程中（本课程重新编号前的第 06 课）。

## 动手实现

`code/main.py` 针对一个 `ScriptedLLM` 实现了全部五种工作流模式：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` — N 次运行，聚合。
- `orchestrator_workers(task, workers)` — 编排器挑选工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环直到通过。

运行：

```
python3 code/main.py
```

每种模式打印其执行跟踪。每个模式的代码行数约 10-15 行；用框架的成本以千行计。

## 用现成库

- 大多数任务直接调用 API。
- 只有当模式真正需要持久化状态（LangGraph）、Actor 模型并发（AutoGen v0.4）或角色模板化（CrewAI）时才引入框架。
- 当你想拥有 Claude Code 的 Agent 结构但不想重建时，选用 Claude Agent SDK。

## 产出

`outputs/skill-workflow-picker.md` 根据任务描述选择正确的模式，包括决策依据和工作流不足时重构为 Agent 的路径。

## 练习

1. 实现带置信度阈值的路由。低于阈值 -> 升级给人工。在一线支持场景下阈值设在哪里合适？
2. 给 `parallel_vote` 增加超时。一个调用挂起时会发生什么？如何处理缺失的投票？
3. 将 `evaluator_optimizer` 改为保留型：跨迭代保留前 2 名输出，这样后期的好结果不会被后期的坏结果覆盖。
4. 将提示链与路由结合：路由器从三条链中选一条。比较 token 成本与单一大提示词方案。
5. 选一个你的生产功能。画出工作流图。数步骤。Agent 真的更好吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 工作流 | "预定义流程" | 工程师拥有的 LLM 和工具调用图 |
| Agent | "自主 AI" | 模型拥有的图；动态工具指挥 |
| 增强型 LLM | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；原子单位 |
| 提示链 | "顺序调用" | 第 N 次调用的输出是第 N+1 次调用的输入 |
| 路由 | "分类器分发" | 选择由哪条链/模型处理输入 |
| 并行化 | "扇出" | N 个并发调用；按分段或投票聚合 |
| 编排器-工作者 | "调度 Agent" | 编排器 LLM 动态挑选专家 LLM |
| 评估器-优化器 | "提议者 + 裁判" | 迭代直到评估通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic，Building Effective Agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式
- [Anthropic，Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 配套学科
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 何时状态图证明了自己的成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 编排器-工作者模式的产品化