# 构建 MCP 服务器 —— Python 和 TypeScript SDK

> 大多数 MCP 教程只展示 stdio 的 hello world。真实服务器暴露工具加资源加 prompts，处理能力协商，发出结构化错误，在不同 SDK 间行为一致。本课从头构建一个笔记服务器：标准库 stdio 传输、JSON-RPC 分发、三个服务器原语，以及当你进阶时可以插入 Python SDK 的 FastMCP 或 TypeScript SDK 的纯函数风格。

**类型：** 构建
**语言：** Python（标准库、stdio MCP 服务器）
**前置要求：** Phase 13 · 06（MCP 基础）
**时间：** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写分发循环，从 stdin 读取 JSON-RPC 消息并写入 stdout 响应。
- 按 JSON-RPC 2.0 规范和 MCP 附加码发出结构化错误响应。
- 将标准库实现升级到 FastMCP（Python SDK）或 TypeScript SDK，无需重写工具逻辑。

## 问题

在使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要干净的本地服务器。本地意味着 stdio：服务器由客户端作为子进程派生，消息通过 stdin/stdout 换行符分隔流动。

2025-11-25 规范规定 stdio 消息编码为带显式 `\n` 分隔符的 JSON 对象。这里没有 SSE；SSE 是旧的远程模式，2026 年中期正在移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用；Keboola 于 2026 年 4 月 1 日）。对于 stdio，每行一个 JSON 对象是完整的线路格式。

笔记服务器是很好的形状，因为它涵盖了所有三个服务器原语。工具做变更（`notes_create`）。资源暴露数据（`notes://{id}`）。Prompts 附带模板（`review_note`）。本课的形状泛化到任何领域。

## 概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 信封的内容。调试日志输出到 stderr。
- 每个请求必须用携带相同 `id` 的响应匹配。
- 通知不得产生响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的内容。客户端依赖能力集来门控功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目有 `name`、`description`、`inputSchema`。`tools/call` 接受 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是类型化的。最常见的有：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形状。协议级错误（未知方法、错误参数）是 JSON-RPC 错误。工具级错误（调用有效但工具失败）作为 `{content: [...], isError: true}` 返回。这让模型在其上下文中看到失败。

### 实现 resources

Resources 设计上是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义方案如 `notes://`。

当你将数据作为 resource 而非工具暴露时：

- 模型不会"调用"它；客户端可以在用户请求时将其注入上下文。
- 订阅让服务器在资源变更时推送更新（Phase 13 · 10）。
- Phase 13 · 14 用 `ui://` 扩展此为交互式 resources。

### 实现 prompts

Prompts 是带命名参数的模板。宿主将其作为斜杠命令展示。`review_note` prompt 可能接受一个 `note_id` 参数并生成多消息提示模板，客户端喂给其模型。

### Stdio 传输细节

- 换行符分隔 JSON。无长度前缀分帧。
- 不要缓冲。每次写入后 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时干净退出。
- 不要静默处理 SIGPIPE；记录并退出。

### 注解

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` — 纯只读，安全重试。
- `destructiveHint: true` — 不可逆副作用；客户端应确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### 升级路径

`code/main.py` 中的标准库服务器约 180 行。FastMCP（Python）将相同逻辑折叠为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等效形状。当你准备好时，升级路径是插入式的；概念（能力、分发、内容块）是相同的。

## 使用它

`code/main.py` 是一个完整的 stdio 笔记 MCP 服务器，纯标准库。它处理 `initialize`、三个工具的 `tools/list` 和 `tools/call`（`notes_list`、`notes_search`、`notes_create`）、每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` prompt。你可以通过管道输入 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

要注意的点：

- 分发器是一个按方法名键控的 `dict[str, Callable]`。
- 每个工具执行器返回内容块列表，而非裸字符串。
- 执行器抛出时设置 `isError: true`。

## 发布它

本课生成 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该 Skill 用正确的 tools / resources / prompts 划分来搭建 MCP 服务器，并提供 SDK 升级路径。

## 练习

1. 运行 `code/main.py` 并用手动构建的 JSON-RPC 消息驱动它。执行 `notes_create`，然后 `resources/read` 取回新笔记。

2. 添加一个带 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会弹出确认对话框（这需要真实宿主；Claude Desktop 可以）。

3. 实现 `resources/subscribe`，使服务器在笔记修改时推送 `notifications/resources/updated`。添加一个 keepalive 任务。

4. 将服务器移植到 FastMCP。Python 文件应缩减到 80 行以下。线路行为必须相同；用相同的 JSON-RPC 测试 harness 验证。

5. 阅读规范的 `server/tools` 部分，找出本课服务器未实现的一个工具定义字段。（提示：有几个；选一个并添加它。）

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| MCP server（MCP 服务器） | "暴露工具的东西" | 通过 stdio 或 HTTP 说 MCP JSON-RPC 的进程 |
| stdio transport（stdio 传输） | "子进程模型" | 服务器由客户端派生；通过 stdin/stdout 通信 |
| Dispatcher（分发器） | "方法路由器" | JSON-RPC 方法名到处理函数 的映射 |
| Content block（内容块） | "工具结果块" | 工具响应 `content` 数组中的类型化元素 |
| `isError` | "工具级失败" | 发出工具失败信号；与 JSON-RPC 错误区分 |
| Annotations（注解） | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 在 MCP 协议之上的装饰器风格高级框架 |
| Resource URI | "可寻址数据" | `file://`、`db://` 或标识资源的自定义方案 |
| Prompt template（Prompt 模板） | "斜杠命令简介" | 服务器提供的模板，带宿主 UI 的参数槽 |
| Capability declaration（能力声明） | "功能开关" | `initialize` 中声明的每个原语标志 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 并行 TS 实现
- [FastMCP — 服务器框架](https://gofastmcp.com/) — MCP 服务器的装饰器风格 Python API
- [MCP — 快速入门服务器指南](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — 服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考