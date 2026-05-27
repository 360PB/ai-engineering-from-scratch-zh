# Claude Agent SDK：子 Agent 与会话存储

> Claude Agent SDK 是 Claude Code 工具链的库形式版本。内建工具、用于上下文隔离的子 Agent、钩子、W3C 跟踪传播、会话存储 parity。Claude Managed Agents 是面向长时间运行的异步工作的托管替代方案。

**类型：** 学习 + 动手实现
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（Agent 循环）、Phase 14 · 10（技能库）
**时间：** 约 75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）和 Claude Agent SDK（工具链形态）的区别。
- 描述子 Agent——并行化和上下文隔离——以及何时使用它们。
- 说出 Python SDK 的会话存储表面（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 用标准库实现一个含内建工具、子 Agent 孵化（隔离上下文）、生命周期钩子和会话存储的工具链。

## 问题背景

原始 LLM API 只给你一次往返。生产 Agent 需要工具执行、MCP 服务器、生命周期钩子、子 Agent 孵化、会话持久化、跟踪传播。Claude Agent SDK 将这些形态作为库提供——Claude Code 使用的工具链，向自定义 Agent 开放。

## 核心概念

### Client SDK vs Agent SDK

- **Client SDK（`anthropic`）。** 原始 Messages API。你拥有循环、工具、状态。
- **Agent SDK（`claude-agent-sdk`）。** 内建工具执行、MCP 连接、钩子、子 Agent 孵化、会话存储。Claude Code 的循环作为库。

### 内建工具

SDK 开箱提供 10+ 个工具：文件读写、Shell、grep、glob、网络获取等。自定义工具通过标准工具 schema 接口注册。

### 子 Agent

Anthropic 记录了两个用途：

1. **并行化。** 并发运行独立工作。"为这 20 个模块各找一个测试文件"是 20 个并行子 Agent 任务。
2. **上下文隔离。** 子 Agent 使用自己的上下文窗口；只有结果返回给编排器。编排器的预算被保留。

Python SDK 最新增：`list_subagents()`、`get_subagent_messages()` 用于读取子 Agent 转录本。

### 会话存储

与 TypeScript 协议 parity：

- `append(session_id, message)` — 添加一轮。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举。
- `delete(session_id)` — 级联删除子 Agent 会话。
- `list_subkeys(session_id)` — 列出子 Agent 键。

`--session-mirror`（CLI 标志）在流式传输时将转录本镜像到外部文件，用于调试。

### 钩子

可注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 门控或审计工具调用。
- `SessionStart`、`SessionEnd` — 设置和清理。
- `UserPromptSubmit` — 在模型看到用户输入之前对其采取行动。
- `PreCompact` — 在上下文压缩之前运行。
- `Stop` — Agent 退出时清理。
- `Notification` — 侧通道告警。

钩子是 pro-workflow（第 14 课课程参考）及相关系统添加切面行为的方式。

### W3C trace context

调用方上活跃的 OTel span 通过 W3C trace context 头传播到 CLI 子进程。整个多进程跟踪在你的后端显示为一条跟踪。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。长时间运行的异步工作、内建提示词缓存、内建压缩。用托管基础设施换取控制权。

### 这个模式会出问题的地方

- **子 Agent 过度孵化。** 为 100 个小任务孵化 100 个子 Agent。开除外占主导。并行批处理代替。
- **钩子蔓延。** 每个团队都加钩子；启动时间膨胀。每个季度审查钩子。
- **会话膨胀。** 会话累积；大小增长。使用 `list_sessions` + 过期策略。

## 动手实现

`code/main.py` 用标准库实现了 SDK 的形态：

- `Tool`、`ToolRegistry`，含内建的 `read_file`、`write_file`、`list_dir`。
- `Subagent` — 私有上下文、隔离运行、结果返回。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示：主 Agent 并行孵化 3 个子 Agent（各自隔离）、聚合结果、持久化会话。

运行：

```
python3 code/main.py
```

执行跟踪展示了子 Agent 上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 用现成库

- **Claude Agent SDK** 用于需要 Claude Code 工具链形态的 Claude 优先产品。
- **Claude Managed Agents** 用于托管长时间运行的异步工作。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的对等方案。
- **LangGraph + 自定义工具** 当你需要图形态状态机时。

## 产出

`outputs/skill-claude-agent-scaffold.md` 脚手架一个 Claude Agent SDK 应用，含子 Agent、钩子、会话存储、MCP 服务器接入和 W3C 跟踪传播。

## 练习

1. 添加一个子 Agent 孵化器，将 20 个任务批处理为每组 5 个并行子 Agent。测量编排器上下文大小 vs 每个任务一个的情况。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用限速（每会话每分钟 5 次）。追踪该行为。
3. 将 `list_subkeys` 接入以渲染子 Agent 树。深度嵌套是什么样子？
4. 将玩具迁移到真实的 `claude-agent-sdk` Python 包。工具注册发生了什么变化？
5. 读取 Claude Managed Agents 文档。什么时候从自托管切换到托管？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent SDK | "Claude Code 即库" | 工具链形态：工具、MCP、钩子、子 Agent、会话存储 |
| Subagent | "子 Agent" | 独立上下文、自己的预算；结果向上冒泡 |
| Session store | "对话数据库" | 持久化、加载、列举、删除轮次，含子 Agent 级联 |
| Hook | "生命周期回调" | 工具前/后、会话、提示词提交、压缩、停止 |
| W3C trace context | "跨进程跟踪" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管工具链" | Anthropic 托管长时间运行的异步工作 |
| `--session-mirror` | "转录本镜像" | 将会话轮次写入外部文件，在流式传输时实时写入 |
| MCP server | "工具表面" | 接入 Agent 的外部工具/资源源 |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对等方案