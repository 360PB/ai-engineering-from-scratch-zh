# 异步任务（SEP-1686）——"调用即发，获取稍后"的长时运行工作

> 真实 Agent 工作需要分钟到小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞 UI。SEP-1686，在 2025-11-25 合并，添加了一个任务原语：任何请求都可以被增强为任务，结果可以稍后获取或通过状态通知流式传输。漂移风险提示：任务在 2026 年 H1 期间是实验性的；SDK 表面仍在围绕规范设计。

**类型：** 构建
**语言：** Python（标准库、异步任务状态机）
**前置要求：** Phase 13 · 07（MCP 服务器）、Phase 13 · 09（传输）
**时间：** 约 75 分钟

## 学习目标

- 识别何时将工具从同步提升为任务增强型（服务器端工作超过 30 秒）。
- 走查任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，使崩溃不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题

`generate_report` 工具运行多分钟提取管道。同步模型下的选项：

1. 保持连接开放三分钟。远程传输会断开；客户端超时；UI 冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。破坏 MCP 统一性。
3. 发射后忘记；无结果。

都不好。SEP-1686 添加第四种：任务增强。任何请求（通常是 `tools/call`）都可以标记为任务。服务器立即返回任务 id。客户端轮询 `tasks/status`，完成后获取 `tasks/result`。服务器端状态在重启后存活。

## 概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定）将请求变为任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器保留状态的承诺；ttl 后任务结果被丢弃。

### 每工具选择加入

工具注解可以声明任务支持：

- `taskSupport: "forbidden"` — 此工具始终同步运行。快速工具安全。
- `taskSupport: "optional"` — 客户端可以请求任务增强。
- `taskSupport: "required"` — 客户端必须使用任务增强。

`generate_report` 工具应为 `required`。`notes_search` 工具应为 `forbidden`。

### 状态

```
working  -> input_required -> working  (通过 elicitation 循环)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是仅追加的：一旦 `completed`、`failed` 或 `cancelled`，任务就是终态。

### 方法

- `tasks/status {taskId}` — 返回当前状态和进度提示。
- `tasks/result {taskId}` — 阻塞或未完成则返回 404。
- `tasks/cancel {taskId}` — 幂等；终态忽略。
- `tasks/list` — 可选；枚举活动中和最近完成的任务。

### 流状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

流式而非轮询的客户端获得更好的 UX。轮询始终作为最小表面受支持。

### 持久状态

规范要求声明任务支持的服务器持久化状态。崩溃不应丢失 ttl 内的完成结果。存储从 SQLite 到 Redis 到文件系统不等。第 13 课 harness 使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器尝试停止（检查执行器协作取消）。如果已经是终态，请求是一个空操作。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将任何进程死亡且状态为 `working` 的任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 为其 ttl 保留 `completed` / `failed` / `cancelled`。

### 异步任务加采样

任务本身可以调用 `sampling/createMessage`。这是长期运行研究任务的工作方式：服务器的任务线程根据需要采样客户端的模型，而客户端的 UI 将任务显示为 `working` 并带有定期进度更新。

### 为什么这是实验性的

SEP-1686 在 2025-11-25 发布，但更广泛的路线图指出三个开放问题：持久订阅原语、子任务（父子任务关系）和结果-TTL 标准化。预计规范在 2026 年会演进。生产代码应仅将任务视为常见情况下稳定，并为子任务的未来 SDK 更改设置防护。

## 使用它

`code/main.py` 实现了一个持久化任务存储（文件系统支持）和一个在后台线程中运行的 `generate_report` 工具。客户端调用工具，立即获得任务 id，在工作线程更新进度时轮询 `tasks/status`，完成后获取 `tasks/result`。取消有效；崩溃恢复通过杀死工作线程并重新加载状态来模拟。

要注意的点：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `progress` 字段；轮询显示它前进。
- 来自客户端的取消设置一个事件；工作线程检查并提前退出。
- "崩溃"时状态重载将进行中的任务标记为 `failed`，`CRASH_RECOVERY`。

## 发布它

本课生成 `outputs/skill-task-store-designer.md`。给定一个长期运行工具（研究、构建、导出），该 Skill 设计任务存储（状态形状、ttl、持久性），选择正确的 taskSupport 标志，并勾勒进度通知。

## 练习

1. 运行 `code/main.py`。启动 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中途添加 `tasks/cancel` 调用。验证工作线程遵守它，状态变为 `cancelled`。

3. 模拟崩溃恢复：杀死工作线程，重启加载器，观察 `CRASH_RECOVERY` 失败模式。

4. 将存储扩展到 SQLite。持久性优势相同；查询选项打开（列出会话 X 的所有任务）。

5. 阅读 2026 年 MCP 路线图帖子。找出最可能在明年影响 SDK API 设计的与任务相关的开放问题。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Task（任务） | "长期运行工具调用" | 用 `_meta.task` 增强以异步执行的请求 |
| SEP-1686 | "任务规范" | 在 2025-11-25 添加任务的规范演进提案 |
| `_meta.task` | "任务信封" | 包含 id、状态、ttl 的每请求元数据 |
| taskSupport | "工具标志" | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | "轮询方法" | 获取当前状态和可选进度提示 |
| `tasks/result` | "获取结果" | 返回完成的有效载荷，未完成则返回 404 |
| `tasks/cancel` | "停止它" | 幂等取消请求 |
| ttl | "保留预算" | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | "状态推送" | 服务器发起的状态变更事件 |
| Durable store（持久存储） | "崩溃安全状态" | 文件系统 / SQLite / Redis 持久层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 问题](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案和完整讨论
- [WorkOS — AI Agent 工作流中的 MCP 异步任务](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带理由的设计走查
- [DeepWiki — MCP 任务系统和异步操作](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制和状态机
- [FastMCP — 任务](https://gofastmcp.com/servers/tasks) — SDK 级任务实现模式
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 开放问题和 2026 年优先级，包括子任务