# OpenTelemetry GenAI —— 端到端追踪工具调用

> Agent 调用五个工具、三个 MCP 服务器和两个子 Agent。你需要跨所有这些的单一追踪。OpenTelemetry GenAI 语义约定（v1.37 及以上中的稳定属性）是 2026 年标准，由 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课命名必需属性，走查跨度层次结构（Agent → LLM → 工具），并发货一个你可以插入任何 OTel 导出器的标准库跨度发出器。

**类型：** 构建
**语言：** Python（标准库、OTel 跨度发出器）
**前置要求：** Phase 13 · 07（MCP 服务器）、Phase 13 · 08（MCP 客户端）
**时间：** 约 75 分钟

## 学习目标

- 为 LLM 跨度命名必需的 OTel GenAI 属性和工具执行跨度。
- 构建覆盖 Agent 循环、LLM 调用、工具调用和 MCP 客户端调度的追踪层次结构。
- 决定捕获什么内容（选择加入）vs 编辑（默认）。
- 发出跨度到本地收集器（Jaeger、Langfuse），无需重写工具代码。

## 问题

2026 年 2 月的一个调试：用户报告"我的 Agent 有时 30 秒回复，有时 3 秒。"无追踪。日志显示 LLM 调用，但不显示工具调度、MCP 服务器往返、子 Agent。你猜测。最终发现：一个 MCP 服务器偶尔在冷启动上挂起。

没有端到端追踪，你无法找到这个。OTel GenAI 修复了它。

约定在 2025-2026 年 OpenTelemetry 语义约定组下确定。它们定义了稳定属性名，因此 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都解析相同的跨度。一次检测；运往任何后端。

## 概念

### 跨度层次结构

```
agent.invoke_agent  (顶层，INTERNAL 跨度)
 ├── llm.chat       (CLIENT 跨度)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT 跨度)
 ├── llm.chat       (CLIENT 跨度)
 └── subagent.invoke (INTERNAL)
```

整个东西嵌套在一个 trace id 下。Span id 链接父子关系。

### 必需属性

根据 2025-2026 年语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 提供商响应 id 用于关联。

工具跨度：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 特定调用 id。
- `gen_ai.tool.description` — 工具描述（可选）。

Agent 跨度：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度种类

- `SpanKind.CLIENT` 用于跨进程边界的调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL` 用于 Agent 自己的循环步骤和工具执行。

### 选择加入的内容捕获

默认情况下，跨度携带指标和时序——而非提示或补全。大载荷和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定 content-capture 环境变量以包含内容。在生产中启用前仔细审查。

### 跨度上的事件

可以添加 token 级事件作为跨度事件：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在跨度内按时序排列，用于详细重放。

### 导出器

OTel 跨度导出到：

- **Jaeger / Tempo。** 开源，本地。
- **Langfuse。** LLM 可观测性特定；可视化 token 使用。
- **Arize Phoenix。** Eval + 追踪结合。
- **Datadog。** 商业；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列导向；查询友好。

全部说 OTLP，即线路格式。你的代码不关心。

### 跨 MCP 的传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头注入请求。Streamable HTTP 支持标准头。Stdio 原生不携带 HTTP 头；规范 2026 年路线图讨论在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在那之前：在每个请求的 `_meta` 中手动包含 traceparent。服务器记录 trace id。

### 指标

除了跨度，GenAI 语义约定定义指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

用这些构建不需要每调用详情的仪表板。

### AgentOps 层

AgentOps（2024 年成立）专精 GenAI 可观测性。它包装流行框架（LangGraph、Pydantic AI、CrewAI）以自动发出 OTel 跨度。如果你的栈使用支持的框架很有用；否则使用手动检测。

## 使用它

`code/main.py` 向 stdout 发出 OTel 形状跨度（OTLP-JSON 类似格式），用于调用 LLM、调度两个工具并进行一次 MCP 往返的 Agent。无真实导出器——本课专注于跨度形状和属性集。将输出粘贴到 OTLP 兼容查看器或直接阅读。

要注意的点：

- Trace id 在所有跨度间共享。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量打开它。

## 发布它

本课生成 `outputs/skill-otel-genai-instrumentation.md`。给定 Agent 代码库，该 Skill 生成检测计划：哪里添加跨度、填充哪些属性、目标哪些导出器。

## 练习

1. 运行 `code/main.py`。计算跨度数量并识别哪些是 CLIENT vs INTERNAL。

2. 打开内容捕获（环境变量）并确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration` 并作为直方图样本每个调用发出。

4. 将父 Agent 跨度中的 traceparent 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器会看到相同的 trace id。

5. 阅读 OTel GenAI 语义约定规范。找出语义约定中列出而本课代码未发出的属性。添加它。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| OTel | "OpenTelemetry" | 追踪、指标、日志的开放标准 |
| GenAI semconv | "GenAI 语义约定" | LLM / 工具 / Agent 跨度的稳定属性名 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span（跨度） | "计时操作" | 具有开始、结束和属性的工作单元 |
| Trace（追踪） | "跨跨度血统" | 共享 trace id 的跨度树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 跨度方向提示 |
| OTLP | "OpenTelemetry 线路协议" | 导出器的线路格式 |
| Opt-in content（选择加入内容） | "提示/补全捕获" | 默认关闭；环境变量启用 |
| traceparent | "W3C 头" | 跨服务传播追踪上下文 |
| Exporter（导出器） | "后端特定发送器" | 将跨度发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI 跨度、指标和事件的规范约定
- [OpenTelemetry — GenAI 跨度](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行跨度属性列表
- [OpenTelemetry — GenAI Agent 跨度](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — Agent 级 `invoke_agent` 跨度
- [open-telemetry/semantic-conventions — GenAI 跨度](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 上的真相来源
- [Datadog — LLM OTel 语义约定](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成走查