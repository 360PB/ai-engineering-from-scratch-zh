# 交接和例程——无状态编排

> OpenAI 的 Swarm（2024 年 10 月）将多智能体编排蒸馏为两个原语：**例程**（指令 + 工具作为系统提示）和**交接**（返回另一个 Agent 的工具）。没有状态机，没有分支 DSL——LLM 通过调用正确的交接工具来路由。OpenAI Agents SDK（2025 年 3 月）是生产继承者。Swarm 本身保持最干净的概念参考——其整个源代码只有几百行。模式是病毒式的，因为 API 面大致是"agent = prompt + tools；handoff = 返回 agent 的函数。"限制：无状态，所以内存是调用方的问题。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** Phase 16 · 04（原语模型）
**时间：** 约60分钟

## 问题

每个多智能体框架要你学习其 DSL：LangGraph 节点和边、CrewAI crews 和 tasks、AutoGen GroupChat 和 managers。DSL 是真实抽象，但它们让东西感觉比需要的更重。

Swarm 推向相反方向：使用模型已有工具调用能力。交接变成工具调用。编排器是当前持有对话的智能体。状态机隐式在智能体的系统提示中。

## 概念

### 两个原语

**例程。** 定义智能体角色和可用工具的系统提示。想成作用域指令集："你是一个分诊智能体；如果用户询问退款，交给退款智能体。"

**交接。** 智能体可调用返回新 Agent 对象的工具。Swarm 运行时检测 Agent 返回值并在下一轮切换活动智能体。

这就是全部抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm 看到 Agent 返回 → 切换活动智能体

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊智能体的系统提示使其根据用户消息选择正确交接。LLM 的工具调用做路由。

### 为什么病毒式

- **小 API。** 两个概念。
- **使用模型已有能力。** 工具调用已经跨提供商生产成熟。
- **无状态机负担。** 你不描述图；智能体提示描述谁交接给谁。

### 无状态权衡

Swarm 在运行之间明确无状态。框架在运行期间保持消息历史，但不持久化任何东西。内存、连续性、长运行任务——都是调用方的问题。

在生产中（OpenAI Agents SDK，2025 年 3 月）这是主要变化之一：SDK 添加内置会话管理、guardrails 和 tracing，同时保持交接原语。

### 何时 Swarm/交接适合

- **分诊模式。** 前线智能体路由用户到专家。
- **基于技能的交接。** "如果任务需要代码，调用编码员；如果需要研究，调用研究员。"
- **短、有界对话。** 客户支持、FAQ 到工单、简单工作流。

### 何时 Swarm 困难

- **长会话共享内存。** 交接将对话状态重置为新智能体提示加历史。没有跨智能体持久状态除非调用方管理内存。
- **并行执行。** 交接一次一个——活动智能体切换。在 Swarm 运行中编排多个需要调用方编排。
- **审计和重放。** 无状态运行难以精确重放；LLM 的交接选择不确定。

### OpenAI Agents SDK（2025 年 3 月）

生产继承者添加：
- **会话状态。** 跨运行持久线程。
- **Guardrails。** 输入/输出验证钩子。
- **追踪。** 每个工具调用和交接都记录。
- **交接过滤器。** 控制交接时什么上下文转移。

交接原语存活；生产人体工程学围绕它添加。

### Swarm vs GroupChat

都使用 LLM 驱动路由，但不同在*谁选下一个*：

- GroupChat：选择器（函数或 LLM）从外部从组中选择下一个发言者。
- Swarm：当前智能体通过调用交接工具选择其后继者。

Swarm 是"智能体决定下一个"；GroupChat 是"管理者决定下一个"。Swarm 的决定在活动智能体的工具调用中；GroupChat 的在 `GroupChatManager`。

## 构建

`code/main.py` 从零实现 Swarm：Agent dataclass，交接机制（工具返回 Agent）和检测智能体切换的运行循环。

演示：分诊智能体路由到退款、销售或支持专家。每个专家有其自己的工具。运行循环打印每次交接。

运行：

```
python3 code/main.py
```

## 使用

`outputs/skill-handoff-designer.md` 设计给定任务的交接拓扑：存在哪些智能体、可以调用哪些交接、什么上下文转移。

## 交付

清单：

- **交接日志。** 每次交接写跟踪事件含 from-agent、to-agent、上下文快照。
- **上下文转移规则。** 决定什么在交接上移动：完整历史（昂贵）、最后 N 条消息或摘要。
- **交接 guardrail。** 交接到具有不同工具权限的专家必须认证——不然提示注入可强制不想要的交接。
- **循环检测。** 两个智能体来回交接是常见失败；用简单最后 K 环检测。
- **后备智能体。** 如果交接目标不存在，返回安全默认值。

## 练习

1. 运行 `code/main.py`，分诊到退款智能体。确认第二轮的活动智能体是退款。
2. 添加循环检测规则：如果相同两个智能体连续交接 3 次，强制退出。设计后备。
3. 阅读 OpenAI Agents SDK 交接过滤器文档。实现"交接摘要"版本：传出智能体在传入智能体接管前将上下文压缩为要点摘要。
4. 比较 Swarm 交接与 GroupChatManager 选择器。哪个模式使提示注入更糟，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 做出的一个显式设计决策，OpenAI Agents SDK 改变了或保留了。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 例程 | "智能体提示" | 系统提示 + 工具列表。定义角色和可用交接。 |
| 交接 | "转移到另一个智能体" | 活动智能体可调用的工具，返回新 Agent。运行时切换活动智能体。 |
| 无状态 | "运行间无内存" | Swarm 不持久化任何东西；内存是调用方责任。 |
| 活动智能体 | "现在谁在说话" | 当前持有对话的智能体。交接改变这个。 |
| 上下文转移 | "交接上什么移动" | 历史规则：完整、上次 N，或摘要。 |
| 交接循环 | "智能体乒乓球" | 两个智能体不断互相交接的失败模式。 |
| OpenAI Agents SDK | "生产 Swarm" | 2025 年 3 月继承者；在交接原语上添加会话和 tracing。 |
| 交接过滤器 | "转移上的门控" | SDK 功能在交接边界检查和修改上下文。 |

## 延伸阅读

- [OpenAI cookbook——编排智能体：例程和交接](https://developers.openai.com/cookbook/examples/orchestrating_agents)——参考阐述
- [OpenAI Swarm 仓库](https://github.com/openai/swarm)——原始实现，作为概念参考保留
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/)——带会话和 tracing 的生产继承者
- [Anthropic 交接-in-Claude 笔记](https://docs.anthropic.com/en/docs/claude-code)——Claude Code 子智能体如何使用类似交接模式通过 `Task`
