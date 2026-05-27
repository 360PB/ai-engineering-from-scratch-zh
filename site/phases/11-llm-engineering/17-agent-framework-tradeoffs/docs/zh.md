# Agent 框架权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架卖同一个演示（研究 Agent 写报告）并隐藏同一个 bug（状态 schema 与编排层冲突）。选择其抽象匹配你的问题形状的框架；其他都是你写两次的胶水。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 11 · 09（Function Calling），Phase 11 · 16（LangGraph）
**时间：** 约 45 分钟

## 问题背景

你有一个需要多个 LLM 调用的任务。也许是研究工作流（计划、搜索、摘要、引证）。也许是代码审查流水线（解析 diff、批评、打补丁、验证）。也许是多轮助理，预订航班、写邮件和提交费用报告。你选一个框架。

三天后，你发现框架的抽象泄露了。CrewAI 给你角色但当"研究员"需要将结构化计划交给"作家"时它跟你打架。AutoGen 给你 Agent 之间的聊天但没有一等状态所以你的检查点是对话日志的 pickle。LangGraph 给你状态图但在你知道 Agent 会做什么之前强迫你命名每个转换。Agno 给你单 Agent 原语但当你想扇出到三个并发 worker 时尖叫。

解决方法不是"选最好的框架"。是将框架的核心抽象匹配到你的问题形状。本课画出那张地图。

## 核心概念

![Agent 框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

2026 年四个主导框架。它们的核心理念不一样。

| 框架 | 核心抽象 | 最佳匹配 | 最差匹配 |
|------|---------|---------|---------|
| **LangGraph** | `StateGraph` — 类型状态、节点、条件边、检查点器 | 需要显式状态和人工介入中断的工作流；需要时间旅行调试的生产 Agent | 拓扑未知的松散、角色驱动头脑风暴 |
| **CrewAI** | `Crew` — 角色（目标、背景故事）、任务、流程（顺序或层级） | 带短线性/层级计划的角色扮演或人格驱动工作流 | crew 轮次历史之外有状态的东西；复杂分支 |
| **AutoGen** | `ConversableAgent` 对 — 两个或更多 Agent 相互聊天直到退出条件 | 多 Agent *对话*（师生、提议者-批评者、演员-审查者），其中思维从聊天中涌现 | 已知 DAG 的确定性工作流；跨重启需要持久状态的任何东西 |
| **Agno** | `Agent` — 单个 LLM + 工具 + 记忆，可组合成团队 | 快速构建的单 Agent 和轻量级团队；强多模态和内置存储驱动 | 带自定义 reducer 的深度显式分支图 |

### "抽象"实际意味着什么

框架的核心理念是你在白板上画出架构时画的东西。

- **LangGraph** → 你画一个图。节点是步骤，边是转换，每个点的状态对象是类型的。思维模型是状态机。
- **CrewAI** → 你画一个组织图。每个角色有工作描述，经理路由任务。思维模型是小专家团队。
- **AutoGen** → 你画一个 Slack DM。两个 Agent 互相发消息；如果需要调解人第三个加入。思维模型是聊天。
- **Agno** → 你画一个带工具悬挂在上面的单一框。把框并排得到团队。思维模型是"带电池的 Agent"。

### 状态问题

状态是大多数框架选择在生产中崩溃的地方。

- **LangGraph。** 类型状态（`TypedDict` 或 pydantic 模型）、每字段 reducer、一等检查点器（SQLite/Postgres/Redis）。恢复、中断和时间旅行是免费的。*（参见 Phase 11 · 16。）*
- **CrewAI。** 状态通过 `context` 字段在任务间作为字符串流动，或通过 `output_pydantic` 结构化。开箱即用无持久化每 crew 存储；如果 crew 必须跨重启存活，你得自己加上。
- **AutoGen。** 状态是聊天历史和任何用户定义的 `context`。对话记录持久化；任意工作流状态不持久化除非你写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 附到 `Agent` —— 对话会话和用户记忆自动持久化。不是完整图检查点器；是会话存储。

### 分支问题

每个非平凡 Agent 都会分支。谁决定分支很重要。

- **LangGraph** —— 你决定，通过条件边。路由是带命名分支的 Python 函数。分支在编译图中是一等的；检查点器记录走了哪个分支。
- **CrewAI** —— 经理在层级模式中决定；在顺序模式中你在构建时决定。路由隐含在任务列表中；在经理 prompt 之外没有一等"if"。
- **AutoGen** —— Agent 通过聊天决定。分支从谁下一句发言中涌现。`GroupChatManager` 选择下一个发言人；你可以手写 `speaker_selection_method` 但默认是 LLM 驱动的。
- **Agno** —— Agent 通过下一步调用哪个工具决定。团队有协调员/路由器/协作者模式；超出该范围的分支是开发者责任。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任何 OTel 导出器的 OpenTelemetry。通过 LangSmith 是一等选项；Langfuse/Phoenix 也有适配器。每个节点转换是一个追踪 span；检查点兼作可回放的追踪。
- **CrewAI** —— 自 2025 年底起一等 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** —— 通过 `autogen-core` 的 OpenTelemetry 集成；AgentOps 和 Opik 有连接器。追踪粒度是每 Agent 消息，非每节点。
- **Agno** —— 内置 `monitoring=True` 标志加 OpenTelemetry 导出器；与 Langfuse 对话追踪紧密集成。

### 成本和延迟

所有四个框架添加每调用开销（框架逻辑、验证、序列化）。大致递增顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由框架做了多少额外 LLM 路由主导。CrewAI 的层级经理花 token 决定谁下一步；AutoGen 的 `GroupChatManager` 同样。LangGraph 只在你写 `llm.invoke` 的地方花 token。Agno 的单 Agent 路径薄。

当每次运行成本重要时，优先显式路由（LangGraph 边、AutoGen `speaker_selection_method`）而非 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可以适配。crew 间通过 `allow_delegation=True` 委托。
- **AutoGen** → `FunctionTool` 包装任何 Python 可调用对象；有 MCP 适配器。与 AG2 生态系统紧密耦合用于 Agent 到 Agent 模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可以在 Agent 和团队间共享。

## 技能

> 你能用一句话解释为什么一个给定框架对给定 Agent 问题是对的。

预建清单：

1. **画形状。** 这是图（类型状态、命名转换）？角色扮演（专家互相交接）？聊天（Agent 聊到完）？带工具的单 Agent？
2. **决定谁分支。** 开发者决定分支 → LangGraph。经理 Agent 决定 → CrewAI 层级。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复？时间旅行？中途运行的人类中断？如果是，LangGraph 是默认值；Agno 会话覆盖对话作用域状态。
4. **检查成本预算。** LLM 选择的路由每轮额外花 token。如果 Agent 每天运行数千次，优先显式路由。
5. **预算框架开销。** 每个框架都是另一个依赖。如果任务是两个 LLM 调用加一个工具，写 30 行纯 Python；没有框架比没有框架更便宜。

在你能画图、组织图、聊天或 Agent 框之前拒绝伸手要框架。拒绝选择迫使你为实际需要的东西而战的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|---------|---------|------|
| 带类型状态、人类批准、长期运行的 Workflow DAG | LangGraph | 一等状态、检查点器、中断、时间旅行 |
| 带不同角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | CrewAI 中角色每任务表达便宜；分支复杂时用 LangGraph 扩展 |
| 提议者-批评者或师生对话 | AutoGen | 两个 Agent 聊天是其原生形状 |
| 带工具和会话、记忆的单 Agent | Agno | 最薄设置，内置存储和记忆 |
| 带 reducer 的数千并行扇出 | LangGraph + `Send` | 一等并行分派原语唯一 |
| 快速原型，无框架承诺 | 纯 Python + 提供商 SDK | 没有框架是最快的框架 |

## 练习

1. **简单。** 用相同任务——"研究 Anthropic 总部，写 200 字简报，引用来源"——用 LangGraph（四个节点：计划、搜索、写、引用）和 CrewAI（三个角色：研究员、作家、编辑）实现它。报告每次运行的 token 成本和代码行数。

2. **中等。** 用 AutoGen（研究员 ↔ 作家聊天，编辑通过 `GroupChat` 加入）和 Agno（带 `search_tools` 和 `write_tools` 的单个 Agent 加会话存储）构建相同任务。排名四个实现在 (a) 每次运行成本、(b) 崩溃后恢复能力、(c) 在写步骤前注入人类批准的能力。

3. **困难。** 构建决策树脚本 `pick_framework.py`，接收简短问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回带一句话理由的推荐。用你自己设计的六个案例验证它。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 编排 | "Agent 如何协调" | 决定哪个节点/角色/Agent 接下来运行的那层 |
| 持久状态 | "重启后恢复 | 跨进程死亡存活的状态，附到检查点或会话存储 |
| LLM 选择的路由 | "让模型决定" | 规划 LLM 每轮选择下一步；灵活但每决策花 token |
| 显式路由 | "开发者决定 | Python 函数或静态边选择下一步；便宜且可审计 |
| Crew | "CrewAI 团队 | 角色 + 任务 + 流程（顺序或层级）绑定到单个 runnable |
| GroupChat | "AutoGen 的多 Agent 聊天 | N 个 Agent 之间托管对话，带发言人选择器 |
| Team (Agno) | "Agno 多 Agent | 跨一组 Agent 的路由/协调/协作模式 |
| StateGraph | "LangGraph 的图 | 类型状态、节点、条件边、检查点器原语 |

## 扩展阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— StateGraph、检查点器、中断、时间旅行
- [CrewAI 文档](https://docs.crewai.com/) —— Crews、Flows、Agents、Tasks、Processes
- [AutoGen 文档](https://microsoft.github.io/autogen/) —— ConversableAgent、GroupChat、teams、tools
- [Agno 文档](https://docs.agno.com/) —— Agent、Team、Workflow、storage、memory
- [Anthropic — 构建有效 Agent (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 框架无关的模式库（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个框架打扮的原语
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) —— AutoGen 设计论文
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) —— CrewAI 风格 persona 栈建立的角色扮演基础
- Phase 11 · 16（LangGraph）—— 本课对比的框架
- Phase 11 · 19（Reflexion）—— 干净映射到 LangGraph 但尴尬映射到 CrewAI 的模式
- Phase 11 · 22（生产可观测性）—— 如何检测你选择的任何框架