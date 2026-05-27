# OpenAI Agents SDK：交接、护栏、追踪

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多 Agent 框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。交接是名为 `transfer_to_<agent>` 的工具。护栏在输入或输出上触发。追踪默认开启。

**类型：** 学习 + 动手实现
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（Agent 循环）、Phase 14 · 06（工具使用）
**时间：** 约 75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释交接：为何建模为工具、模型看到的名称形态、上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` vs 阻塞模式。
- 用标准库实现含交接 + 护栏 + span 风格追踪的运行时。

## 问题背景

不能干净利落地委托的 Agent 最终会把所有东西塞进一个提示词。没有护栏的 Agent 会泄露 PII、产生违规输出或无限循环。OpenAI 的 SDK 将使多 Agent 工作变得可驾驭的三个原语成文化。

## 核心概念

### 五个原语

1. **Agent。** LLM + 指令 + 工具 + 交接。
2. **Handoff。** 委托给另一个 Agent。在模型看来表示为一个名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 输入验证（仅首个 Agent）、输出验证（仅末个 Agent）或工具调用验证（按每个函数工具）。
4. **Session。** 跨轮次自动保存对话历史。
5. **Tracing。** LLM 生成、工具调用、交接、护栏的内置 span。

### 交接即工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它通知运行时：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 将其折叠成一条）。
2. 用目标 Agent 的指令初始化它。
3. 用目标 Agent 继续运行。

这是主管模式（第 13 课 / 第 28 课）的产品化版本。

### 护栏

三种口味：

- **输入护栏。** 在首个 Agent 的输入上运行。在任何 LLM 调用之前拒绝不安全的或超出范围的请求。
- **输出护栏。** 在末个 Agent 的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具护栏。** 按函数工具运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。护栏 LLM 与主 LLM 并行运行。尾部延迟更低。如果触发，主 LLM 的工作被丢弃（Token 浪费）。
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。如果触发，不在主调用上浪费 Token。

触发线抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每次 LLM 生成、工具调用、交接和护栏都发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 退出。`add_trace_processor(processor)` 将 span 扇出到你自己的后端，与 OpenAI 的并行。

### Session

`Session` 在后端（SQLite、Redis、自定义）存储对话历史。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 这个模式会出问题的地方

- **交接漂移。** Agent A 交接给 Agent B，Agent B 又交接回 Agent A。加一个跳转计数器。
- **护栏绕过。** 工具护栏只在函数工具上触发；内置工具（文件读取器、网络获取）需要单独策略。
- **过度追踪。** Span 中的敏感内容。配合 OTel GenAI 内容捕获规则（第 23 课）使用——外部存储，按 ID 引用。

## 动手实现

`code/main.py` 用标准库实现了 SDK 的形态：

- `Agent`、`FunctionTool`、`Handoff`（作为带交接语义的函数工具）。
- 带输入/输出/工具护栏、交接分发和跳转计数器的 `Runner`。
- 一个简单的 span 发射器，展示跟踪形状。
- 一个分诊 Agent，根据用户查询交接给计费或支持；护栏在一个输入上触发。

运行：

```
python3 code/main.py
```

执行跟踪展示了两次成功的交接、一次输入护栏触发，以及与真实 SDK 发出的 span 树形结构镜像。

## 用现成库

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品。
- **LangGraph**（第 13 课）当你需要显式状态和持久化恢复时。
- **自研** 当你需要精确控制（语音、多提供商、联邦部署）时。

## 产出

`outputs/skill-agents-sdk-scaffold.md` 脚手架一个 Agents SDK 应用，含分诊 Agent、交接、输入/输出/工具护栏、会话存储和跟踪处理器。

## 练习

1. 添加交接跳转计数器：N 次转移后拒绝。追踪该行为。
2. 实现 `nest_handoff_history` 作为选项——在转移前将之前消息折叠成一条摘要。
3. 写一个阻塞输出护栏。对比会触发它的提示词与通过的提示词的延迟。
4. 将 `add_trace_processor` 接入 JSON 记录器。它每个 span 发出的形状是什么？
5. 读取 SDK 文档。将标准库玩具迁移到 `openai-agents-python`。你建模错了什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent | "LLM + 指令" | SDK 中的 Agent 类型；拥有工具和交接 |
| Handoff | "转移" | 模型调用来委托给另一个 Agent 的工具 |
| Guardrail | "策略检查" | 对输入 / 输出 / 工具调用的验证 |
| Tripwire | "护栏触发" | 护栏拒绝时抛出的异常 |
| Session | "历史存储" | 跨运行持久化的对话记忆 |
| Tracing | "Span" | 对 LLM + 工具 + 交接 + 护栏的内置可观测性 |
| Blocking guardrail | "顺序检查" | 护栏先运行；触发时不在 Token 上浪费 |
| Parallel guardrail | "并发检查" | 护栏并行运行；延迟更低，触发时 Token 浪费 |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)：原语、交接、护栏、追踪
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：Claude 口味的对等物
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：何时需要交接
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：Agents SDK span 所映射的标准