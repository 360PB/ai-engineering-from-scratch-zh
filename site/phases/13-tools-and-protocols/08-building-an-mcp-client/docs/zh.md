# 构建 MCP 客户端 —— 发现、调用、会话管理

> 大多数 MCP 内容是服务器教程，对客户端挥挥手就略过了。客户端代码才是艰难编排所在：进程派生、能力协商、跨多服务器工具列表合并、采样回调、重连和命名空间冲突解析。本课构建一个多服务器客户端，将三个不同 MCP 服务器提升为一个扁平工具命名空间供模型使用。

**类型：** 构建
**语言：** Python（标准库、多服务器 MCP 客户端）
**前置要求：** Phase 13 · 07（构建 MCP 服务器）
**时间：** 约 75 分钟

## 学习目标

- 将 MCP 服务器作为子进程派生，完成 `initialize` 并发送 `notifications/initialized`。
- 维护每服务器会话状态（能力、工具列表、上次看到的通知 id）。
- 跨多个服务器合并工具列表到一个命名空间并处理冲突。
- 将工具调用路由到拥有它的服务器并重组响应。

## 问题

真实的 Agent 宿主（Claude Desktop、Cursor、Goose、Gemini CLI）同时加载多个 MCP 服务器。用户可能同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的工作：

1. 派生每个服务器。
2. 独立握手每个。
3. 对每个调用 `tools/list` 并扁平化结果。
4. 当模型发出 `notes_search` 时，在合并命名空间中查找并路由到正确服务器。
5. 处理来自任何服务器的通知（`tools/list_changed`）而不阻塞。
6. 传输失败时重连。

手写所有这些就是"玩具"和"可服务"的分水岭。官方 SDK 包装了这些，但心智模型必须是你自己的。

## 概念

### 子进程派生

`subprocess.Popen` 带 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式进行逐行读取。每个服务器是一个进程；客户端每个服务器持有一个 `Popen` 句柄。

### 每服务器会话状态

每个服务器的 `Session` 对象持有：

- `process` — Popen 句柄。
- `capabilities` — 服务器在 `initialize` 时声明的内容。
- `tools` — 上次 `tools/list` 结果。
- `pending` — 请求 id 到等待响应的 promise/future 的映射。

请求本质是异步的；在服务器 A 处于中间调用时发送到服务器 B 的 `tools/call` 不得阻塞。要么用带队列的线程，要么用 asyncio。

### 合并命名空间

当客户端看到聚合工具列表时，名称可能冲突。两个服务器可能都暴露 `search`。客户端有三个选择：

1. **按服务器名加前缀。** `notes/search`、`files/search`。清晰但难看。
2. **静默先到。** 后来的服务器 `search` 覆盖早的。有风险；隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器；通知用户。对安全敏感宿主最安全。

Claude Desktop 使用按服务器加前缀。Cursor 使用冲突拒绝并给出清晰错误。VS Code MCP 也采用按服务器加前缀。

### 路由

合并后，分发表将 `tool_name -> session` 映射。模型按名称发出调用；客户端找到会话并将 `tools/call` 消息写入该服务器的 stdin，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可以发送 `sampling/createMessage` 请求客户端运行其 LLM。客户端必须：

1. 在采样解析之前阻止对该服务器的进一步请求，或如果其实现支持并发则管道化。
2. 调用其 LLM 提供商。
3. 将响应发回服务器。

第 11 课涵盖端到端采样。本课存根它以保持完整性。

### 通知处理

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果资源在使用中则重新读取。通知不得产生响应——不要尝试 ack 它们。

一个常见客户端 bug：在 `tools/call` 上阻塞读取循环，而通知坐在流中。使用后台读取器线程将每条消息推入队列；主线程出队并分发。

### 重连

传输可能失败：服务器崩溃、操作系统杀死进程、stdio 管道断裂。客户端检测 stdout 上的 EOF 并将会话视为死亡。选项：

- 静默重启服务器并重新握手。对纯只读服务器可以。
- 向用户暴露失败。对有状态服务器和用户可见会话可以。

Phase 13 · 09 涵盖 Streamable HTTP 重连语义；stdio 更简单。

### Keepalive 和会话 id

Streamable HTTP 使用 `Mcp-Session-Id` 头。Stdio 没有会话 id——进程身份就是会话。Keepalive ping 是可选的；stdio 管道在非活跃时不会断裂。

## 使用它

`code/main.py` 派生三个模拟 MCP 服务器作为子进程，独立握手每个，合并它们的工具列表，并将工具调用路由到正确的一个。"服务器"实际上是运行 toy 响应器的其他 Python 进程（无真实 LLM）。运行它可以看到：

- 三次初始化，每次有各自的能力集。
- 三次 `tools/list` 结果合并为一个 7 工具的命名空间。
- 基于工具名的路由决策。
- 通过命名空间前缀防止一个冲突。

要注意的点：

- `Session` dataclass 干净地持有每服务器状态。
- 后台读取器线程出队 stdout 上的每一行而不阻塞主线程。
- 分发表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明相同名称时，后来的一个用前缀重命名。

## 发布它

本课生成 `outputs/skill-mcp-client-harness.md`。给定 MCP 服务器的声明式列表（名称、命令、参数），该 Skill 生成派生它们、合并工具列表并提供路由函数的 harness，带冲突解决。

## 练习

1. 运行 `code/main.py` 并观察服务器派生日志。用 SIGTERM 终止其中一个模拟服务器进程，观察客户端如何检测 EOF 并将该会话标记为死亡。

2. 实现命名空间前缀。当两个服务器暴露 `search` 时，将第二个重命名为 `<server>/search`。更新分发表并验证工具调用正确路由。

3. 添加连接池风格的重连退避：连续失败时指数退避，上限 30 秒，三次失败后向用户发出通知。

4. 勾勒一个支持 100 个并发 MCP 服务器的客户端。什么数据结构取代简单分发表？（提示：前缀命名空间的 trie，加上每服务器工具数指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 包装了 `stdio_client` 和 `ClientSession`。代码应从约 200 行缩减到约 40 行，同时保留多服务器路由。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| MCP client（MCP 客户端） | "Agent 宿主" | 派生服务器并编排工具调用的进程 |
| Session（会话） | "每服务器状态" | 能力、工具列表和待处理请求的簿记 |
| Merged namespace（合并命名空间） | "一个工具列表" | 跨所有活动服务器的扁平工具名集合 |
| Namespace collision（命名空间冲突） | "两个服务器同名工具" | 客户端必须前缀、拒绝或先到处理重复 |
| Routing（路由） | "谁接收这个调用？" | 从工具名到拥有服务器的调度 |
| Background reader（后台读取器） | "非阻塞 stdout" | 将服务器 stdout 排入队列的线程或任务 |
| Sampling callback（采样回调） | "LLM 即服务" | 客户端对服务器 `sampling/createMessage` 的处理程序 |
| `notifications/*_changed` | "原语变更" | 客户端必须重新发现或重新读取的信号 |
| Reconnection policy（重连策略） | "服务器死时" | 传输失败时的重启语义 |
| Stdio session（stdio 会话） | "进程 = 会话" | 无会话 id；子进程生命周期就是会话 |

## 延伸阅读

- [Model Context Protocol — 客户端规范](https://modelcontextprotocol.io/specification/2025-11-25/client) — 规范客户端行为
- [MCP — 快速入门客户端指南](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello world 客户端教程
- [MCP Python SDK — 客户端模块](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — 客户端](https://github.com/modelcontextprotocol/typescript-sdk) — TS 并行
- [VS Code — 扩展中的 MCP](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中多路复用多个 MCP 服务器