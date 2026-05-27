# LangGraph：有状态图与持久化执行

> LangGraph 是 2026 年低阶有状态编排的事实标准。Agent 是一个状态机；节点是函数；边是转移；状态不可变，每步后做检查点保存。从任何故障点精确恢复，继续运行。

**类型：** 学习 + 动手实现
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（Agent 循环）、Phase 14 · 12（工作流模式）
**时间：** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：不可变状态 + 函数节点 + 条件边 + 每步检查点保存。
- 说出文档强调的四大能力：持久化执行、流式输出、人类介入、全面记忆。
- 解释 LangGraph 支持的三种编排拓扑：主管（Supervisor）、点对点（Swarm）、层级（Hierarchical，嵌套子图）。
- 实现一个标准库状态图，含不可变状态、条件边和检查点/恢复周期。

## 问题背景

Agent 和工作流面临同一个问题：当一个 40 步的运行在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。二流的状态模型迫使操作人员在假设全新运行的库上自己hack 重试逻辑。

LangGraph 的设计回答：状态是一等公民的typed 对象，变更显式化，每节点后检查点持久化。恢复只需一次 `load_state(session_id)` 调用。

## 核心概念

### 图的结构

一个图由以下部分定义：

- **状态类型。** 一个 typed dict（或 Pydantic 模型），每个节点都读取和修改它。
- **节点。** 纯函数 `(state) -> state_update`。返回后更新被合并进状态。
- **边。** 节点之间的直接转移或条件转移。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个 Agent 有 `classify`、`refund`、`bug`、`sales`、`done` 节点——一个路由工作流被建模为图。

### 持久化执行

每节点返回后，运行时将状态序列化并写入检查点存储（SQLite、Postgres、Redis、自定义）。在第 N 步失败时，运行时可以 `resume(session_id)`，从第 N+1 步精确恢复状态。

LangGraph 文档明确提到了这在生产中发挥作用的用户：Klarna、Uber、J.P. Morgan。关键不在图的结构本身，而在于图的结构加上检查点保存使故障恢复变得廉价。

### 流式输出

每个节点可以产生部分输出。图向调用方流式发送每节点增量事件，使 UI 在图运行过程中实时更新。

### 人类介入循环

在节点之间检查和修改状态。实现方式：在关键节点前暂停、将状态呈现给人类、接受修改、恢复执行。检查点存储使这变得容易，因为状态已经是序列化好的。

### 记忆

短期（在一次运行内——对话历史保存在状态中）和长期（跨运行——通过检查点存储持久化，加上独立的长期存储）。LangGraph 通过工具集成外部记忆系统（Mem0、自定义）。

### 三种拓扑

1. **主管。** 中央路由器 LLM 向专家子 Agent 分发任务。`create_supervisor()` 在 `langgraph-supervisor` 中（不过 LangChain 团队在 2026 年推荐直接通过工具调用来做，以便更好地控制上下文）。
2. **Swarm / 点对点。** Agent 通过共享工具表面直接交接。没有中央路由器。
3. **层级。** 主管管理子主管，实现为嵌套子图。

### 这个模式会出问题的地方

- **检查点粒度太小。** 只对对话轮次做检查点保存会导致工具状态和记忆写入无法恢复。必须序列化完整状态。
- **节点行为不确定性。** 恢复假设节点输入产生相同的状态更新。随机种子、系统时间、外部 API 的结果必须被捕获。
- **过度使用条件边。** 每个边都是条件的图本质上是一个无法理清的状态机。优先使用线性链，只在必要时做分支。

## 动手实现

`code/main.py` 实现了一个标准库有状态图：

- `State` — 含 `messages`、`step`、`route`、`output`、`human_approval` 的 typed dict。
- `Node` — 接收状态并返回更新字典的可调用对象。
- `StateGraph` — 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存中的伪实现）——每节点后序列化状态；`load(session_id)` 恢复状态。
- 一个演示图：classify -> branch(refund / bug / sales) -> 人工门控 -> send。

运行：

```
python3 code/main.py
```

执行跟踪展示了第一次运行在人工门控处失败、持久化、然后恢复产生最终输出的全过程。

## 用现成库

- **LangGraph** — 事实标准，生产可用。用 `create_react_agent`、`create_supervisor`，或构建自己的图。
- **AutoGen v0.4**（第 14 课）— 高并发场景下的 Actor 模型替代方案。
- **Claude Agent SDK**（第 17 课）— 带内置会话存储的托管运行环境。
- **自研** — 当你需要精确控制状态结构或检查点后端时。

## 产出

`outputs/skill-state-graph.md` 生成任意目标运行时中的 LangGraph 风格状态图，含检查点保存和恢复逻辑。

## 练习

1. 从 `classify` 添加一条到 `end` 的条件边，当分类置信度低于阈值时触发。人工设置 `route` 后恢复运行。
2. 将类 SQLite 的伪实现换成真实的 SQLite 检查点存储。测量每步序列化的额外开销。
3. 实现并行边：两个节点同时运行，通过自定义归约器合并。不可变状态在这里带来了什么好处？
4. 读取 `langgraph-supervisor` 参考文档。将玩具实现迁移到 `create_supervisor`。对比两者的执行跟踪形状。
5. 添加流式输出：每个节点在运行时产生部分状态。到达时打印增量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| State graph | "Agent 即状态机" | 类型化状态 + 节点 + 边 + 归约器 |
| Checkpointer | "持久化后端" | 每节点后序列化状态；支持恢复 |
| Reducer | "状态合并器" | 将当前状态与节点更新合并的函数 |
| Conditional edge | "分支" | 由状态函数决定的边 |
| Subgraph | "嵌套图" | 作为另一图中的节点使用的图 |
| Durable execution | "故障恢复" | 从最后一个成功节点重启，状态完全一致 |
| Supervisor | "路由器 LLM" | 专家子 Agent 的中央调度器 |
| Swarm | "点对点 Agent" | Agent 通过共享工具交接；无中央路由器 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — 主管模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — Actor 模型替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储与子 Agent