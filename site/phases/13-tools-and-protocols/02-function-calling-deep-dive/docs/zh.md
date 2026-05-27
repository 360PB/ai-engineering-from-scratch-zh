# 函数调用深度解析 —— OpenAI、Anthropic、Gemini

> 2024 年，三个前沿提供商在相同的工具调用循环上收敛，然后在其他所有方面分化。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 id 关联。本课将三者并排对比，这样针对一个提供商编写的代码在移植到另一个提供商时不会出问题。

**类型：** 构建
**语言：** Python（标准库、schema 转换器）
**前置要求：** Phase 13 · 01（工具接口）
**时间：** 约 75 分钟

## 学习目标

- 说明 OpenAI、Anthropic 和 Gemini 函数调用载荷之间的三个形状差异（声明、调用、结果）。
- 将一个工具声明转换为三个提供商的格式，并预测严格模式约束的不同之处。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬性限制（工具数量、schema 深度、参数长度）以及超出限制时各自发出的错误签名。

## 问题

函数调用请求的形状因提供商而异。2026 年生产栈的三个具体例子：

**OpenAI Chat Completions / Responses API。** 传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是必须解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码强制 schema 合规。

**Anthropic Messages API。** 传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 形式返回。`input` 已经是解析后的对象（不是字符串）。你回复一个新的 `user` 消息，其中包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应到达为 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`，其中 `id` 在 Gemini 3 及以上版本中唯一，用于并行调用关联。你回复 `{functionResponse: {name, id, response}}`。

相同的循环。不同的字段名、不同的嵌套、不同的字符串 vs 对象约定、不同的关联机制。一个在 OpenAI 上编写天气 Agent 的团队移植到 Anthropic 需要两天，再移植到 Gemini 又需要一天——这还只是管道工作。

本课构建一个转换器，将三种格式统一为一种规范工具声明，并在边缘路由。Phase 13 · 17 将同一模式泛化为 LLM 网关。

## 概念

### 共同结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制某个特定工具、禁止工具，或让模型决定。
3. **调用发出。** 结构化输出命名工具和参数。
4. **调用 id。** 将响应与正确调用关联（对并行很重要）。
5. **结果注入。** 将结果绑定回调用的消息或块。

### 形状差异，按字段逐项

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明包装 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` 块 | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化 JSON | 已解析对象 | 已解析对象 |
| Id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic 生成） | UUID（Gemini 3+） |
| 结果块 | role `tool`，`tool_call_id` | 带 `tool_result` 的 `user`，`tool_use_id` | 带匹配 `id` 的 `functionResponse` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 即 schema（始终强制） | 请求级别的 `responseSchema` |

### 你会实际碰到的限制

- **OpenAI。** 每次请求 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。严格模式要求无未解析 `$ref`，无带重叠的 `oneOf`/`anyOf`/`allOf`，每个属性都列在 `required` 中。
- **Anthropic。** 每次请求 64 个工具。Schema 深度实际无界但实际限制 10。无严格模式标志；schema 是合约，模型通常遵守。
- **Gemini。** 每次请求 64 个函数。并行调用的唯一 id 自 Gemini 3 起。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有分歧）。

### `tool_choice` 行为

三种模式，每个提供商都支持，但名称不同。

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须至少调用一个工具。
- **None。** 模型不得调用工具。

加一个每个提供商独有的模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use` 标志区分单调用 vs 多调用。
- **Gemini。** `mode: "VALIDATED"` 将每个响应路由通过 schema 验证器，不管模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一个 assistant 消息中发出多个调用。你全部运行它们，然后用包含每个 `tool_call_id` 一个条目的批量工具角色消息回复。Anthropic 历史上做单调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起默认）启用多调用。Gemini 2 允许并行调用但没有稳定 id；Gemini 3 添加了 UUID，使乱序响应可以干净地关联。

### 流式传输

三个提供商都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐步到达。你积累直到 `finish_reason: "tool_calls"`。
- **Anthropic。** Block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带 `functionCallId` 的块，使多个并行调用可以交错。

Phase 13 · 03 深入讲解并行 + 流式重组。本课专注于声明和单调用形状。

### 错误和修复

无效参数错误也各不相同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码期间发生；无效 JSON 不可能，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务器端验证。
- **Gemini。** OpenAPI 3.0 怪癖：对象字段上的 `enum` 被静默忽略；自己验证。

### 转换器模式

代码中的规范工具声明是这样的（你选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三个提供商的形状。`code/main.py` 中的 harness 正是这样做的，然后用每个提供商的响应形状往返一个假工具调用。不需要网络——本课教的是形状，不是 HTTP。

生产团队将此转换器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 发货一个网关，在三个提供商任一前面暴露 OpenAI 形状的 API。

## 使用它

`code/main.py` 定义了一个规范 `Tool` dataclass 和三个发出 OpenAI、Anthropic 和 Gemini 声明 JSON 的转换器。然后它解析每个形状的手动构造提供商响应到同一规范调用对象，证明语义在表面之下是相同的。运行它并并排对比三个声明。

要注意的点：

- 三个声明块只在包装和字段名上不同。
- 三个响应块在调用所在位置不同（顶级 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从所有三个响应形状中提取 `{id, name, args}`。

## 发布它

本课生成 `outputs/skill-provider-portability-audit.md`。给定针对一个提供商的函数调用集成，该 Skill 生成可移植性审计：该集成依赖哪些提供商限制、哪些字段需要重命名、移植到其他提供商时哪些会出问题。

## 练习

1. 运行 `code/main.py` 并验证三个提供商声明 JSON 都序列化了相同的底层 `Tool` 对象。修改规范工具添加一个 enum 参数，并确认只有 Gemini 转换器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生不支持；注意这种不对称。

3. 实现 `tool_choice` 转换：将规范 `ToolChoice(mode="force", tool_name="x")` 映射到所有三个提供商形状。然后映射 `mode="any"` 和 `mode="none"`。检查本课的对比表。

4. 选择三个提供商之一，从头到尾阅读其函数调用指南。在其 schema 规范中找到其他两个不支持的一个字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 写一个测试向量：一个参数违反声明 schema 的工具调用。用每个提供商的验证器（用第 01 课的标准库验证器作为代理即可）运行它，并记录哪些错误触发。记录你会在生产中用于严格性的提供商。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Function calling（函数调用） | "工具使用" | 用于发出结构化工具调用的提供商级 API |
| Tool declaration（工具声明） | "工具规范" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制/禁止" | Auto / required / none / specific-name 模式 |
| Strict mode（严格模式） | "Schema 强制执行" | 强制解码匹配 schema 的 OpenAI 标志 |
| `tool_use` 块 | "Anthropic 的调用形状" | 带 id、name、input 的内联内容块 |
| `functionCall` 部分 | "Gemini 的调用形状" | `parts[]` 中包含 name、args 和 id 的条目 |
| Arguments-as-string（字符串参数） | "字符串化 JSON" | OpenAI 将参数作为 JSON 字符串返回，而非对象 |
| Parallel tool calls（并行工具调用） | "单轮扇出" | 一个 assistant 消息中的多个工具调用 |
| Refusal（拒绝） | "模型拒绝" | 严格模式独有的拒绝块，而非调用 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用类似 JSON Schema 的方言，有细微差异 |

## 延伸阅读

- [OpenAI — 函数调用指南](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的规范参考
- [Anthropic — 工具使用概述](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 id 和 OpenAPI 子集
- [Vertex AI — 函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业表面
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制执行详情