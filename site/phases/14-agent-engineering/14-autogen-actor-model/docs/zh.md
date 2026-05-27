# AutoGen v0.4：Actor 模型与 Agent 框架

> AutoGen v0.4（微软研究院，2025 年 1 月）以 Actor 模型重新设计了 Agent 编排。异步消息交换、事件驱动 Agent、故障隔离、原生并发。框架现处于维护模式，Microsoft Agent Framework（2025 年 10 月公开预览）成为后继者。

**类型：** 学习 + 动手实现
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（Agent 循环）、Phase 14 · 12（工作流模式）
**时间：** 约 75 分钟

## 学习目标

- 描述 Actor 模型：Agent 即 Actor，消息是唯一 IPC，故障按 Actor 隔离。
- 说出 AutoGen v0.4 的三层 API——Core、AgentChat、Extensions——各自的用途。
- 解释消息投递与处理解耦为何带来故障隔离和原生并发。
- 用 Python 标准库实现一个 Actor 运行时，并将双 Agent 代码审查流程迁移到该运行时上。

## 问题背景

大多数 Agent 框架是同步的：一个 Agent 产生，一个 Agent 消耗，发生在调用栈里。故障时栈崩溃。并发是后来硬加上去的。分布式部署需要重写。

AutoGen v0.4 的回答：Actor 模型。每个 Agent 是一个 Actor，有私有收件箱。消息是唯一的交互方式。运行时将投递和处理解耦。一个 Actor 崩溃不影响另一个。并发是原生特性。分布式只是换个传输层。

## 核心概念

### Actor

一个 Actor 拥有：

- 一个私有状态（外部永远不能直接访问）。
- 一个收件箱（消息队列）。
- 一个处理器：`receive(message) -> effects`，效果可以是"回复"、"发消息给其他 Actor"、"孵化新 Actor"、"更新状态"、"停止自己"。

两个 Actor 不能共享内存。它们只能发消息。

### AutoGen v0.4 的三层 API

1. **Core。** 低阶 Actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换、事件驱动。
2. **AgentChat。** 任务驱动的高阶 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成层——OpenAI、Anthropic、Azure、工具、记忆。

### 解耦为何重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 同步阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 把消息放入 agent_b 的收件箱后立即返回。运行时稍后投递。三大效果：

- **故障隔离。** Agent B 崩溃不会崩溃 Agent A——运行时在 B 的处理器中捕获故障并决定处理方式（记录、重试、死信）。
- **原生并发。** 同一时间可以有多条消息在飞行中；Actor 并发处理收件箱中的消息。
- **天然支持分布式。** 收件箱 + 传输是同一套抽象，无论 Actor 在进程内还是另一台主机上。

### 拓扑

- **RoundRobinGroupChat。** Agent 按固定顺序轮流。
- **SelectorGroupChat。** 选择器 Agent 根据对话上下文决定下一个处理者。
- **Magentic-One。** 参考多 Agent 团队，用于网页浏览、代码执行、文件处理。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息发出一个 span；工具调用按 2026 年 OTel GenAI 语义约定（Phase 14 · 23）携带 `gen_ai.*` 属性。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 稳定，可用于研究和原型。微软已将活跃开发转移到 Microsoft Agent Framework（2025 年 10 月 1 日公开预览；1.0 GA 目标 2026 年 Q1 末）。AutoGen 的模式可以平滑迁移——Actor 模型才是耐久的核心思想。

## 动手实现

`code/main.py` 用标准库实现了一个 Actor 运行时：

- `Message` — 含 `sender`、`recipient`、`topic`、`body` 的类型化载荷。
- `Actor` — 含 `receive(message, runtime)` 的抽象基类。
- `Runtime` — 事件循环，含共享队列、投递、故障隔离。
- 一个双 Actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 执行检查清单；它们交换消息直到达成共识。

运行：

```
python3 code/main.py
```

执行跟踪展示了消息投递、模拟的单一 Actor 故障不影响另一个、以及在共同结论上收敛。

## 用现成库

- **AutoGen v0.4/v0.7**（维护中）——研究和原型、稳定的多 Agent 模式。
- **Microsoft Agent Framework**（公开预览）——前进方向；在全新 API 中贯彻相同的 Actor 模型思想。
- **LangGraph swarm 拓扑**（第 13 课）——通过共享工具交接实现的类似模式。
- **自研 Actor 运行时**——当你需要特定传输层时（NATS、RabbitMQ、gRPC）。

## 产出

`outputs/skill-actor-runtime.md` 生成一个最小 Actor 运行时，以及针对给定多 Agent 任务团队模板（轮询或选择器）。

## 练习

1. 添加死信队列：当处理器抛出异常时，将失败消息停车等待人工检查。在你的玩具实现中 DLQ 多久触发一次？
2. 实现 `SelectorGroupChat`：一个选择器 Actor 根据对话状态决定谁处理下一条消息。
3. 添加分布式传输：将进程内队列换成 JSON-over-HTTP 服务器，使 Actor 可以在独立进程中运行。
4. 为每条消息接入 OTel span（或无操作替代品）。按第 23 课的要求输出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 读取 AutoGen v0.4 的架构文章。将你的玩具迁移到真实的 `autogen_core` API。你跳过了哪些在生产中重要的事？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Actor | "Agent" | 私有状态 + 收件箱 + 处理器；无共享内存 |
| Message | "事件" | 类型化载荷；Actor 之间交互的唯一方式 |
| Inbox | "邮箱" | 每个 Actor 的待处理消息队列 |
| Runtime | "Agent 宿主" | 路由消息、隔离故障的事件循环 |
| Topic | "通道" | Actor 之间的命名发布-订阅路由 |
| Fault isolation | "让它崩溃" | 一个 Actor 失败不会导致其他崩溃 |
| RoundRobinGroupChat | "固定轮转团队" | Agent 按顺序轮流 |
| SelectorGroupChat | "上下文路由团队" | 选择器决定下一个处理者 |
| Magentic-One | "参考团队" | 用于网页 + 代码 + 文件的多 Agent 小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 图形态替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 span