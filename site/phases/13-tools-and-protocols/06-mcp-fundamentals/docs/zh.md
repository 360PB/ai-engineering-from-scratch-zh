# MCP 基础 —— 原语、生命周期、JSON-RPC 基础

> 在 MCP 出现之前，每个集成都是定制方案。模型上下文协议（Model Context Protocol）由 Anthropic 于 2024 年 11 月首次发布，现由 Linux 基金会 Agentic AI Foundation 托管，将发现和调用标准化，使任何客户端都能与任何服务器通信。2025-11-25 规范定义了六个原语（三个服务器端、三个客户端端）、三阶段生命周期和 JSON-RPC 2.0 线路格式。掌握这些，Phase 13 的其余 MCP 章节就是阅读练习。

**类型：** 学习
**语言：** Python（标准库、JSON-RPC 解析器）
**前置要求：** Phase 13 · 01 到 05（工具接口和函数调用）
**时间：** 约 45 分钟

## 学习目标

- 命名所有六个 MCP 原语（服务器端的 tools、resources、prompts；客户端的 roots、sampling、elicitation），并为每个给出一种用例。
- 走查三阶段生命周期（初始化、运行、关闭），说明每个阶段谁发送哪条消息。
- 解析并发出 JSON-RPC 2.0 请求、响应和通知信封。
- 解释 `initialize` 时能力协商是什么，没有它会出什么错。

## 问题

在 MCP 之前，每个使用工具的 Agent 都有自己的协议。Cursor 有 MCP 形状但不兼容的工具系统。Claude Desktop 用了另一个。VS Code 的 Copilot 扩展是第三个。构建了"Postgres 查询"工具的团队将同一工具写了三次，每次对着不同宿主的 API。要复用就得复制代码。

结果是定制集成的寒武纪大爆发，生态系统速度遇到天花板。

MCP 通过标准化线路格式修复了这个问题。单个 MCP 服务器可以在每个 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ 个客户端。每月 1.1 亿次 SDK 下载。10,000+ 个公共服务器。Linux 基金会于 2025 年 12 月在新成立的 Agentic AI Foundation 下接管了托管工作。

本 phase 使用的规范版本是 **2025-11-25**。它添加了异步任务（SEP-1686）、URL 模式elicitation（SEP-1036）、带工具的采样（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 资源指示器语义。Phase 13 · 09 到 16 涵盖这些扩展。本课停在基础上。

## 概念

### 三个服务器端原语

1. **Tools。** 可调用操作。与 Phase 13 · 01 相同的四步循环。
2. **Resources。** 暴露的数据。只读内容，通过 URI 寻址：`file:///path`、`db://query/...`、自定义方案。
3. **Prompts。** 可复用模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **Roots。** 服务器可访问的 URI 集合。客户端声明它们；服务器遵守它们。
5. **Sampling。** 服务器请求客户端的模型执行补全。实现服务器托管的 Agent 循环，无需服务器端 API 密钥。
6. **Elicitation。** 服务器在飞行中请求客户端用户输入结构化内容。表单或 URL（SEP-1036）。

MCP 中的每个功能恰好属于这六个原语之一。Phase 13 · 10 到 14 逐一深入讲解。

### 线路格式：JSON-RPC 2.0

每条消息都是一个 JSON 对象，带这些字段：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — 无 `id`，不期望响应。

基础规范有约 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：初始化。**

客户端发送带其 `capabilities` 和 `clientInfo` 的 `initialize`。服务器以自己的 `capabilities`、`serverInfo` 和它说的规范版本响应。客户端在消化响应后发送 `notifications/initialized`。从此时起，任意一方可以按协商的能力发送请求。

**阶段 2：运行。**

双向。客户端调用 `tools/list` 发现，然后 `tools/call` 调用。服务器如果声明了该能力，可以发送 `sampling/createMessage`。服务器的工具集变更时可以发送 `notifications/tools/list_changed`。用户变更根范围时客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：关闭。**

任意一方关闭传输。无 MCP 结构化关闭方法；传输（stdio 或 Streamable HTTP，Phase 13 · 09）携带连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 是合约。服务器示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态系统漂移的原因。不支持采样的客户端仍然是有效的 MCP 客户端；不调用 `sampling` 的服务器仍然是有效的 MCP 服务器。他们只是不在该功能上合作。

### 结构化内容和错误形状

`tools/call` 返回类型化块的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 在该列表中添加了 MCP Apps（`ui://` 交互式 UI）。

错误使用 JSON-RPC 错误码。规范定义的补充：`-32002` "资源未找到"、`-32603` "内部错误"，加上 MCP 特定错误数据作为 `error.data`。

### 客户端能力 vs 工具调用详情

一个常见混淆：`capabilities.tools` 是客户端是否支持工具列表变更通知。客户端是否会调用特定工具是运行时选择，由其模型驱动，不是能力标志。能力标志是规范级合约。模型的选择是正交的。

### 为什么是 JSON-RPC 而非 REST？

JSON-RPC 2.0（2010）是一种轻量级双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（采样、通知），所以 JSON-RPC 的对称请求/响应形状是自然选择。JSON-RPC 还可以干净地组合在 stdio 和 WebSocket/Streamable HTTP 上，而无需重新发明 HTTP 的请求形状。

## 使用它

`code/main.py` 发货一个最小化 JSON-RPC 2.0 解析器和发出器，然后手动走查 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。无真实传输；只有消息形状。与延伸阅读中的规范对比，验证每个信封。

要注意的点：

- `initialize` 双向声明能力；响应有 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应的 `content` 是 `{type, text}` 块的数组。

## 发布它

本课生成 `outputs/skill-mcp-handshake-tracer.md`。给定 MCP 客户端-服务器交互的 pcap 风格记录，该 Skill 为每条消息标注属于哪个原语、哪个生命周期阶段、依赖哪个能力。

## 练习

1. 运行 `code/main.py`。找出能力协商发生的行，并描述如果服务器没有声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时运行的 `tools/call` 进行时发出它，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范——整个文档约 80 页。找出大多数服务器不需要的能力标志。提示：它与资源订阅有关。

4. 在纸上勾勒一个假设"定时任务"功能属于哪个原语。（提示：服务器希望客户端在调度时间调用它。六个原语今天都不适合。）MCP 的 2026 路线图有一个该功能的草案 SEP。

5. 从 GitHub 上的开源 MCP 服务器解析一个会话日志。统计请求 vs 响应 vs 通知消息。计算生命周期 vs 运行阶段流量的比例。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| MCP | "模型上下文协议" | 模型到工具发现和调用的开放协议 |
| Server primitive（服务器原语） | "服务器暴露什么" | tools（操作）、resources（数据）、prompts（模板） |
| Client primitive（客户端原语） | "服务器可用的客户端功能" | roots（作用域）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线路格式" | 对称请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一对消息；服务器和客户端声明它们支持的功能 |
| `tools/list` | "发现" | 客户端向服务器询问其当前工具集 |
| `tools/call` | "调用" | 客户端请求服务器用参数执行工具 |
| `notifications/*_changed` | "变更事件" | 服务器告诉客户端其原语列表已变更 |
| Content block（内容块） | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "规范演进提案" | 命名草案提案（如 SEP-1686 异步任务） |

## 延伸阅读

- [Model Context Protocol — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 规范文档
- [Model Context Protocol — 架构概念](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — 介绍模型上下文协议](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布帖
- [MCP 博客 — MCP 一周年](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一年回顾和 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 规范更新](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 摘要