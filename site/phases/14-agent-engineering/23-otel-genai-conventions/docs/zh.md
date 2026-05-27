# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 Agent 遥测的标准模式。跨度名称、属性和内容捕获规则在各厂商间趋于统一，使 Agent 追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中的含义一致。

**类型：** 概念学习 + 动手实现
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 24（可观测性平台）
**时间：** 约 60 分钟

## 学习目标

- 列出 GenAI 跨度类别：model/client、agent、tool。
- 区分 `invoke_agent` 的 CLIENT 和 INTERNAL 跨度及各自适用场景。
- 列出顶级 GenAI 属性：provider 名称、请求模型、数据源 ID。
- 解释内容捕获契约：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用推荐方案。

## 问题背景

每个厂商自创跨度名称。运维团队被迫为每个框架单独构建仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一套全生态统一遵循的标准来解决这个问题。

## 核心概念

### 跨度类别

1. **Model / client 跨度。** 覆盖原始 LLM 调用。由 provider SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **Agent 跨度。** `create_agent`（Agent 构建时）和 `invoke_agent`（Agent 执行时）。
3. **Tool 跨度。** 每次工具调用一个；通过父子关系连接到 Agent 跨度。

### Agent 跨度命名

- 跨度名称：`invoke_agent {gen_ai.agent.name}`（若命名了），否则回退为 `invoke_agent`。
- 跨度类型：
  - **CLIENT** — 远程 Agent 服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 进程内 Agent 框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 实际解析到的模型（可能因路由与请求模型不同）。
- `gen_ai.agent.name` — Agent 标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — RAG 场景：咨询了哪个语料库或存储。

各厂商有专属约定：Anthropic、Azure AI Inference、AWS Bedrock、OpenAI。

### 内容捕获

默认规则：插桩默认**不应**捕获输入/输出。通过以下属性选择性地捕获：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐生产模式：将内容存到外部（S3、自建日志存储），跨度上只记录引用（指针 ID，而非文本）。这正是第 27 课内容投毒防御在可观测性领域的落地。

### 稳定性

截至 2026 年 3 月，大多数约定仍为实验阶段。通过以下方式接入稳定的预览版：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM 可观测性模式。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式的常见误区

- **将完整 prompt 捕获进跨度。** PII、密钥、客户数据进入追踪，运维可读。存到外部。
- **没有 `gen_ai.provider.name`。** 多 provider 仪表盘因缺少归属信息而无法工作。
- **跨度没有父链接。** 孤立的工具跨度。始终传播上下文。
- **未设置稳定性 opt-in。** 属性可能在后端升级时被重命名。

## 动手实现

`code/main.py` 实现符合 GenAI 约定的标准库跨度发射器：

- 带 GenAI 属性 schema 的 `Span`。
- 带 `start_span` 和嵌套上下文的 `Tracer`。
- 一个脚本化 Agent 运行，发射：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的跨度、LLM 调用的 `chat` 跨度。
- 一种内容捕获模式：将 prompt 存到外部，跨度属性仅记录 ID。

运行：

```
python3 code/main.py
```

输出：一棵包含所有必需 GenAI 属性的跨度树，以及一个展示 opt-in 内容引用的"外部存储"。

## 用现成库

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）— 自动插桩整个生态。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；从 GenAI 属性构建仪表盘。
- **自托管** — 用 GenAI 处理器运行 OTel Collector。

## 产出

`outputs/skill-otel-genai.md` 将 OTel GenAI 跨度接入现有 Agent，含内容捕获默认值和外部引用存储。

## 练习

1. 给第 1 课的 ReAct 循环添加 `invoke_agent`（INTERNAL）+ 每个工具的跨度。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：prompt 存 SQLite，跨度属性仅携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将它接入第 9 课的 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证 collector 不会重命名你的属性。
5. 构建仪表盘：仅靠 GenAI 属性展示"哪种模型对应哪种工具错误"。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GenAI SIG | "OpenTelemetry GenAI 组" | 定义 schema 的 OTel 工作组 |
| invoke_agent | "Agent 跨度" | 代表一次 Agent 运行的跨度名称 |
| CLIENT span | "远程调用" | 调用远程 Agent 服务的跨度 |
| INTERNAL span | "进程内" | 进程内 Agent 运行的跨度 |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 来源" | 检索命中了哪个语料库/存储 |
| Content capture | "Prompt 日志" | 选择性捕获消息；生产环境存外部 |
| Stability opt-in | "预览模式" | 用于固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范文档
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认发出 GenAI 跨度
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel 跨度
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context 传播