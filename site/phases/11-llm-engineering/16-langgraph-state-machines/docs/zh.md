# LangGraph — Agent 的状态机

> 手写的 ReAct 循环是 `while True`。用 LangGraph 写的 ReAct 循环是一个你可以检查点化、中断、分支和时间旅行的图。Agent 没变。周围的线束变了。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 11 · 09（Function Calling），Phase 11 · 14（Model Context Protocol）
**时间：** 约 75 分钟

## 问题背景

你发布了一个 function-calling Agent。它工作了三个轮次，然后出了问题：模型尝试一个返回 500 的工具、用户中途改变主意，或者 Agent 决定在没有人类签字的情况下退款订单。`while True:` 循环没有钩子。你无法暂停它，无法回退，也无法分支出"如果模型选择了另一个工具会怎样"。一旦过了演示阶段，Agent 变成了一个要么成要么败的黑箱。

下一步一旦看到就显而易见。Agent 已经是状态机——系统 prompt 加消息历史加待处理工具调用加下一个动作。把状态机显式化：节点代表"模型思考"、"工具运行"、"人类批准"，边代表它们之间的条件转换。一旦图显式化，线束免费获得四样东西：检查点化（步骤之间保存状态）、中断（暂停等人类）、流式传输（流 token 和中间事件）和时间旅行（回退到先前状态并尝试不同分支）。

LangGraph 是提供这个抽象的库。它不是 LangChain 意义上的 Agent 框架（"这是一个 AgentExecutor，祝你好运"）。它是一个带一等状态、一等持久化和一等中断的图运行时。Agent 循环是你画的东西，不是手写的东西。

## 核心概念

![LangGraph StateGraph：节点、边和检查点](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三样东西。

1. **State。** 一个类型字典（TypedDict 或 Pydantic 模型），流经整个图。每个节点接收完整状态并返回部分更新，LangGraph 使用 *reducer* 合并每个字段——列表用 `operator.add` 累积，默认覆盖。
2. **Nodes。** Python 函数 `state -> partial_state`。每个是离散步骤："调用模型"、"运行工具"、"摘要"。
3. **Edges。** 节点之间的转换。静态边去一个地方。条件边取路由函数 `state -> next_node_name`，让图可以在模型输出上分支。

编译图。Compile 绑定拓扑，附加检查点器（可选但生产必需），返回 runnable。用初始状态和 `thread_id` 调用它。执行的每一步都持久化一个检查点，键为 `(thread_id, checkpoint_id)`。

### 四个超能力

**检查点化。** 每个节点转换将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。通过用相同 `thread_id` 再次调用图来恢复。图从暂停处继续。

**中断。** 用 `interrupt_before=["human_review"]` 标记节点，执行在该节点运行前停止。状态持久化。你的 API 用"等待批准"回复用户。后续用 `Command(resume=...)` 的相同 `thread_id` 请求恢复执行。

**流式传输。** `graph.stream(state, mode="updates")` 在发生时产生状态增量。`mode="messages"` 在模型节点内流 LLM token。`mode="values"` 产生完整快照。你选择在你的 UI 上显示什么。

**时间旅行。** `graph.get_state_history(thread_id)` 返回完整检查点日志。将任何先前 `checkpoint_id` 传给 `graph.invoke` 并从该点分叉。这对调试（"如果模型选择了工具 B 会怎样？"）和回放生产追踪进行回归测试很棒。

### Reducer 是关键

每个状态字段有一个 reducer。大多数默认值没问题——新值覆盖旧的。但消息列表需要 `operator.add`，让新消息附加而非替换。并行边通过 reducer 合并更新。如果两个节点都更新 `messages` 而你忘了 `Annotated[list, add_messages]`，第二个静默获胜，你丢失半轮。Reducer 是库中唯一微妙的东西；做对了，其余都组合。

### 四个节点的 ReAct 图

生产 ReAct Agent 是四个节点和两条边：

1. `agent` —— 用当前消息历史调用 LLM。返回 assistant 消息（可能包含 tool_calls）。
2. `tools` —— 执行最后 assistant 消息中的任何 tool_calls，将工具结果作为 tool 消息附加。
3. 从 `agent` 的条件边，如果最后消息有 tool_calls 路由到 `tools`，否则到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就这样。你获得完整 ReAct 循环（Thought → Action → Observation → Thought → …）带检查点化、中断和流式传输，约 40 行代码。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让一个节点分派并行子图。例如：Agent 决定同时查询三个检索器。每个 `Send` 生成目标节点的并行执行；它们的输出通过状态 reducer 合并。这就是 LangGraph 表达 orchestrator-workers 模式的方式，无需线程原语。

### 子图

一个编译图可以是另一个图中的节点。外层图看到一个节点；内层图有自己的状态和自己的检查点。团队构建 supervisor-worker Agent 的方式：supervisor 图将用户意图路由到域作用域 worker 子图。

## 构建

### 第一步：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让消息列表累积而非覆盖的 reducer。忘记它是最常见的 LangGraph bug。

### 第二步：用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("查找 Anthropic 总部地址")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新是一个字典 `{node_name: state_delta}`。你的前端可以将这些流式传输到 UI，让用户看到"agent 正在思考… 调用 search_web… 获得结果… 回答中"。

### 第三步：添加人工介入中断

标记一个节点，使其在运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 每次工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("删除生产数据库")]}, config)
# state["__interrupt__"] 已设置。检查提议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写一条拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("被人工审查员阻止。")]})
```

状态、检查点和线程都在中断期间持久化。除了执行期间，没有什么在内存中。

### 第四步：时间旅行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从先前检查点分叉
target = history[3].config  # 三步之前
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从那点向前重放
```

传递 `None` 作为输入从给定检查点重放；传递一个值作为对该检查点状态的更新再恢复。这让你无需重跑整个对话就能重现糟糕的 Agent 运行。

### 第五步：为生产换检查点器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

提供 SQLite、Redis 和 Postgres。`MemorySaver` 用于测试。任何跨重启持久化的东西都需要真实存储。

## 技能

> 你将 Agent 构建为图，而非 `while True` 循环。

在求助于 LangGraph 之前，花 60 秒设计：

1. **命名节点。** 每个离散决策或副作用动作是一个节点。"Agent 思考"、"工具运行"、"审查员批准"、"响应流式传输。"如果列不出来，任务还不是 Agent 形状的。
2. **声明状态。** 带每个列表字段 reducer 的最小 TypedDict。不要把所有东西塞进 `messages`；将任务特定字段（工作的 `plan`、预算计数器 `budget`、检索文档列表 `retrieved_docs`）提升到顶层。
3. **画边。** 静态的除非下一步取决于模型输出。每个条件边需要一个带命名分支的路由函数。
4. **提前选择检查点器。** 测试用 `MemorySaver`，其他用 Postgres/Redis/SQLite。不要不带检查点器上线——无检查点意味着无恢复、无中断、无时间旅行。
5. **在工具运行前决定中断，非之后。** 批准放在副作用节点进入边上，这样你可以在危害前取消；验证放在模型出边上，这样你可以廉价地拒绝坏调用。
6. **默认流式传输。** UI 用 `mode="updates"`，模型节点内 token 级流式传输用 `mode="messages"`，评估期间完整快照用 `mode="values"`。

拒绝发货没有检查点器的 LangGraph Agent。拒绝发货在副作用之后中断的。拒绝发货没有 `add_messages` 作为 reducer 的 `messages` 字段。

## 练习

1. **简单。** 用计算器工具和网络搜索工具实现上述四节点 ReAct 图。验证 `list(app.get_state_history(config))` 对两轮对话返回至少四个检查点。

2. **中等。** 添加一个在 `agent` 前运行的 `planner` 节点，写入结构化 `plan: list[str]` 到状态。让 `agent` 标记计划步骤完成。如果 `plan` 在检查点恢复中丢失则测试失败（错误的 reducer）。

3. **困难。** 构建一个在三个子图（`researcher`、`writer`、`reviewer`）之间路由的 supervisor 图，使用 `Send`。每个子图有自己的状态和检查点。在外层图加 `interrupt_before=["writer"]` 让人类批准研究简报。确认从先前检查点时间旅行只重跑分叉分支。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| StateGraph | "LangGraph 图" | 你在 compile 前添加节点和边的构建器对象 |
| Reducer | "字段如何合并" | 节点返回该字段更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 附加 |
| Thread | "会话 ID" | 一个 `thread_id` 字符串，作用域为一个会话的所有检查点 |
| Checkpoint | "暂停的状态" | 节点转换后完整图状态的持久化快照，键为 `(thread_id, checkpoint_id)` |
| Interrupt | "暂停等人类" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复 |
| 时间旅行 | "从先前步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放 |
| Send | "并行子图分派" | 节点可以返回的构造函数，生成 N 个目标节点并行执行的并行化 |
| 子图 | "作为节点的编译图" | 在另一个图中作为节点使用的编译 StateGraph；保留自己的状态作用域 |

## 扩展阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— StateGraph、reducer、检查点器和中断的权威参考
- [LangGraph 概念：状态、reducer、检查点器](https://langchain-ai.github.io/langgraph/concepts/low_level/) —— 本课使用的思维模型，直接来自源头
- [LangGraph 持久化和检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) —— Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的细节
- [LangGraph 人工介入](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) —— `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和编辑状态模式
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个 LangGraph Agent 实现的模式；读它以获取推理追踪的理由
- [Anthropic — 构建有效 Agent (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 优先选择哪种图形状（chain、router、orchestrator-workers、evaluator-optimizer）及何时
- Phase 11 · 09（Function Calling）—— 每个 LangGraph Agent 节点重用的工具调用原语
- Phase 11 · 14（Model Context Protocol）—— 插入 LangGraph `ToolNode` 的外部工具发现，通过 MCP 适配器
- Phase 11 · 17（Agent 框架权衡）—— 何时选 LangGraph 而非 CrewAI、AutoGen 或 Agno