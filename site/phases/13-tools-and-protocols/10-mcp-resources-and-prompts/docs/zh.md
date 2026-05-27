# MCP Resources 和 Prompts —— 超越工具的上下文暴露

> 工具占据了 90% 的 MCP 关注度。另外两个服务器原语解决不同问题。Resources 暴露数据供读取；prompts 暴露可复用模板作为斜杠命令。许多服务器应该用 resources 代替将读取包装在工具中，用 prompts 代替在客户端提示中硬编码工作流。本课命名决策规则并走查 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建
**语言：** Python（标准库、resource + prompt 处理程序）
**前置要求：** Phase 13 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 为给定领域决定将能力暴露为工具、resource 还是 prompt。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 和带参数模板的 `prompts/get`。
- 识别宿主何时将 prompts 作为斜杠命令 vs 自动注入上下文展示。

## 问题

笔记应用的简单 MCP 服务器将所有内容暴露为工具：`notes_read`、`notes_list`、`notes_search`。这将每个数据访问包装在模型驱动的工具调用中。结果：

- 模型必须决定是否对每个可能受益于上下文的查询调用 `notes_read`。
- 只读内容无法订阅或流式传输到宿主侧面板。
- 客户端 UI（Claude Desktop 的 resource 附加面板、Cursor 的"包含文件"选择器）无法展示数据。

正确的划分：暴露数据为 resource，暴露变更或计算操作作为工具，暴露可复用多步工作流作为 prompt。每个原语有其 UX 功能和访问模式。

## 概念

### 工具 vs resources vs prompts —— 决策规则

| 能力 | 原语 |
|------|------|
| 用户想要搜索、过滤或转换数据 | tool |
| 用户想要宿主将此数据作为上下文包含 | resource |
| 用户想要可重运行的模板工作流 | prompt |

准则：如果模型在每个相关查询中调用它会受益，那就是工具。如果用户会从将其附加到会话中受益，那就是 resource。如果整个多步工作流是用户想要重用的单元，那就是 prompt。

### Resources

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义方案）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制。二进制使用 base64 编码字符串加 `mimeType` 的 `blob` 字段。

### Resource 订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源变更时服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：笔记服务器的 resources 是磁盘上的文件；文件监视器触发更新通知；当在宿主外部编辑时 Claude Desktop 重新将文件拉入上下文。

### Resource 模板（2025-11-25 新增）

`resourceTemplates` 让你暴露参数化 URI 模式：`notes://{id}`，`id` 作为补全目标。客户端可以在 resource 选择器中自动补全 id。

### Prompts

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

Prompt 是一个模板，填充为宿主喂给模型的消息列表。例如，`code_review` prompt 接受 `file_path` 参数并返回一个三消息序列：系统消息、包含文件体的用户消息，以及带推理模板的助手启动消息。

### 宿主和 prompts

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中将 prompts 展示为斜杠命令。用户输入 `/code_review` 并从表单中选择参数。服务器的 prompt 是"用户快捷方式"和"发给模型的完整提示"之间的合约。

并非每个客户端都支持 prompts——检查能力协商。声明了 prompt 能力但客户端不支持 prompt 的服务器只是不会显示斜杠命令。

### "列表变更"通知

Resources 和 prompts 在集合变更时都发出 `notifications/list_changed`。刚导入 20 个新笔记的笔记服务器发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 获取新增内容。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，加 `blob` 字段。
MCP Apps（第 14 课）：`text/html;profile=mcp-app`，在 `ui://` URI 中。

### 动态 resources

Resource URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新的五个笔记。`db://query/users/active` 可以执行参数化查询。服务器可以动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须稳定。如果计算是一次性的，URI 应包含时间戳或随机数以避免客户端缓存过期。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获取服务器推送。不支持订阅的预订阅客户端或宿主通过重新读取进行轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪个。

订阅成本：服务器上每个会话的状态（谁订阅了什么）。保持订阅集有界；断开连接的客户端应超时。

### Prompts vs 系统提示

MCP 中的 prompts 不是系统提示。宿主自己的系统提示（其自身操作指令）和 MCP prompts（用户调用的服务器提供模板）并存。一个行为良好的客户端永远不会让服务器 prompt 覆盖其自己的系统提示；它将它们分层。

## 使用它

`code/main.py` 用以下内容扩展第 07 课的笔记服务器：

- 每个笔记的 resources（`notes://note-1` 等）带 `resources/subscribe` 支持。
- 一个 `review_note` prompt，渲染为三消息模板。
- 文件监视器模拟，在笔记修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态 resource，总是返回最新的五个笔记。

运行演示查看完整流程。

## 发布它

本课生成 `outputs/skill-primitive-splitter.md`。给定一个拟议的 MCP 服务器，该 Skill 将每个能力分类为 tool / resource / prompt 并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始 resource 列表，然后触发笔记编辑并验证 `notifications/resources/updated` 事件触发。

2. 添加 `resources/list_changed` 发出器：当创建新笔记时，发送通知以便客户端重新发现。

3. 为 GitHub MCP 服务器设计三个 prompts：`summarize_pr`、`triage_issue`、`release_notes`。每个带参数 schema。Prompt 体应可运行而无需进一步编辑。

4. 取第 07 课服务器中的一个现有工具，分类它应该保持为工具还是拆分为 resource 加工具对。用一句话为你的选择辩护。

5. 阅读规范的 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中很少填充但规范支持的字段。提示：看 resource content 上的 `_meta`。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Resource | "暴露的数据" | 宿主可读取的 URI 可寻址内容 |
| Resource URI | "数据指针" | 带方案前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视变更" | 客户端选择加入的服务器推送特定 URI 更新 |
| `notifications/resources/updated` | "资源已更新" | 向客户端发出信号：订阅的资源有新内容 |
| Resource template | "参数化 URI" | 带宿主选择器补全提示的 URI 模式 |
| Prompt | "斜杠命令模板" | 带参数槽的命名多消息模板 |
| Prompt arguments | "模板输入" | 宿主在渲染前收集的类型化参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充的消息列表 |
| Content block | "类型化块" | `{type: text \| image \| resource \| ui_resource}` |
| Slash-command UX | "用户快捷方式" | 宿主将 prompts 展示为以 `/` 开头的命令 |

## 延伸阅读

- [MCP — 概念：Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URI、订阅和模板
- [MCP — 概念：Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt 模板和斜杠命令集成
- [MCP — 服务器 resources 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — `resources/*` 消息的完整参考
- [MCP — 服务器 prompts 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — `prompts/*` 消息的完整参考
- [MCP — 协议信息站：resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 扩展官方文档的社区指南