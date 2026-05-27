# 编排模式：Supervisor、Swarm、分层

> 2026 年各框架反复出现四种编排模式：supervisor-worker、swarm / 点对点、分层、辩论。Anthropic 的指导："关键在于为你的需求构建合适的系统。" 从简单开始；只有当单一 Agent 加五个工作流模式不够用时才引入拓扑。

**类型：** 概念学习 + 动手实现
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 25（多 Agent 辩论）
**时间：** 约 60 分钟

## 学习目标

- 说出四种反复出现的编排模式及其各自适用场景。
- 描述 2026 年 LangChain 的建议：基于工具调用的监督 vs supervisor 库。
- 解释 Anthropic 的"构建合适的系统"规则及如何用它把关拓扑选型。
- 用标准库针对同一脚本化 LLM 实现四种模式。

## 问题背景

团队在还没必要时就开始追求"多 Agent"。四种模式在各框架中反复出现；一旦能说出它们的名字，就能选对正确的那个——或者完全不需要拓扑。

## 核心概念

### Supervisor-worker

- 中央路由 LLM 分发到专家 Agent。
- 决策：返回自身、交给专家、终止。
- 专家之间不直接通信；所有路由经过 supervisor。

框架：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI 分层流程。

**2026 年 LangChain 建议：** 通过直接工具调用而非 `create_supervisor` 来实现监督。更精细的上下文工程控制——你决定每个专家能看到什么。

### Swarm / 点对点

- Agent 通过共享工具面直接交接。
- 无中央路由。
- 延迟低于 supervisor（跳数更少）。
- 更难推理（无单一控制点）。

框架：LangGraph swarm 拓扑、OpenAI Agents SDK handoff（当所有 Agent 都可以互相交接时）。

### 分层（Hierarchical）

- Supervisor 管理子-supervisor，子-supervisor 管理 Worker。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中为嵌套 Crew。
- 可扩展到大规模 Agent 群体，代价是运维复杂度。

适用场景：当单一 supervisor 的上下文预算无法容纳所有专家的描述时。

### 辩论

- 并行提议者 + 迭代交叉批评（第 25 课）。
- 严格说不是编排——更像是验证——但出现在框架的拓扑选项中。

### CrewAI Crew vs Flow

CrewAI 明确了两种部署模式：

- **Flow** 用于确定性事件驱动自动化（生产推荐起步点）。
- **Crew** 用于自主角色协作。

这与上述四种模式是正交的，但映射到拓扑：CrewAI Flow 通常是 supervisor 或分层；Crew 通常是带 LLM 路由器的 supervisor 形状。

### Anthropic 的指导

"LLM 领域的成功不在于构建最复杂的系统，而在于为你的需求构建合适的系统。"

决策顺序：

1. 单一 Agent + 工作流模式（第 12 课）—— 从这里开始。
2. Supervisor-worker — 有 2–4 个专家时考虑。
3. Swarm — 当延迟比推理清晰度更重要时。
4. 分层 — 仅当 supervisor 上下文预算不足时。
5. 辩论 — 当准确率比成本更重要时。

### 这个模式的常见误区

- **拓扑优先思维。** 在还没明确"多 Agent 解决什么问题"之前就说"我们需要多 Agent"。
- **Swarm 中来回交接。** A -> B -> A -> B。用跳数计数器防止。
- **虚假分层。** 三层是因为"企业级"；实际上只有两个团队。扁平化。

## 动手实现

`code/main.py` 用标准库针对脚本化 LLM 实现四种模式：

- `Supervisor` — 中央路由器。
- `Swarm` — 带直接交接的点对点。
- `Hierarchical` — 多级 Supervisor。
- `Debate` — 并行提议者 + 批评。

每个模式处理相同的三个意图任务（退款 / 缺陷 / 销售）。追踪形状不同。

运行：

```
python3 code/main.py
```

输出：每种模式的追踪 + 操作计数。Supervisor 最清晰；Swarm 最短；Hierarchical 最深；Debate 最贵。

## 用现成库

- **LangGraph** 用于 supervisor 和分层（嵌套子图）。
- **OpenAI Agents SDK** 用于作为工具的交接（supervisor 形状）。
- **CrewAI Flow** 用于生产确定性场景。
- **自建** 用于辩论或需要精确控制时。

## 产出

`outputs/skill-orchestration-picker.md` 选一个拓扑并实现它。

## 练习

1. 去掉路由器，将 supervisor-worker 转换为 swarm。什么坏了？什么改善了？
2. 给 swarm 添加跳数计数器：超过 3 次交接则拒绝。能捕获 A->B->A 来回吗？
3. 为 12 个专家领域构建两级分层系统。如果不嵌套，上下文预算在哪里失效？
4. 在生产形状的工作负载上对四种模式做性能分析。哪个指标（延迟、成本、准确率、可调试性）谁胜出？
5. 阅读 Anthropic 的"Building Effective Agents"文章。将你的每个生产流程映射到四种模式之一。有无法干净映射的吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Supervisor-worker | "路由器 + 专家" | 中央 LLM 分发到专家；专家之间不通信 |
| Swarm | "点对点" | 通过共享工具直接交接；无中央路由 |
| Hierarchical | "Supervisor 的 Supervisor" | 用于大规模群体的嵌套子图 |
| Debate | "提议者 + 批评" | 并行提议者，交叉批评（第 25 课） |
| Tool-call-based supervision | "无库的 Supervisor" | 通过直接工具调用实现 supervisor 以获得上下文控制 |
| Crew | "自主团队" | CrewAI 的角色协作模式 |
| Flow | "确定性工作流" | CrewAI 的事件驱动生产模式 |

## 延伸阅读

- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 五种模式 + Agent vs 工作流
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor、swarm、分层
- [CrewAI 文档](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al.，Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 辩论模式