# 顶点项目 —— 构建完整工具生态系统

> Phase 13 教了每个组件。本顶点项目将它们连线成一个生产形状的系统：一个带工具 + 资源 + prompts + 任务 + UI 的 MCP 服务器、边缘的 OAuth 2.1、RBAC 网关、多服务器客户端、A2A 子 Agent 调用、到收集器的 OTel 追踪、CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 包。完成后你可以捍卫每个架构选择。

**类型：** 构建
**语言：** Python（标准库、端到端生态系统 harness）
**前置要求：** Phase 13 · 01 到 21
**时间：** 约 120 分钟

## 学习目标

- 组合一个暴露工具、资源、prompts 和带 `ui://` 应用的任务的 MCP 服务器。
- 用强制执行 RBAC 和固定哈希的 OAuth 2.1 网关为服务器加前缀。
- 编写一个端到端用 OTel GenAI 属性追踪的多服务器客户端。
- 将部分工作负载委托给 A2A 子 Agent；验证不透明性被保留。
- 用 AGENTS.md + SKILL.md 打包整个栈，以便其他 Agent 可以驱动它。

## 问题

交付"研究和报告"系统：

- 用户问："总结 2026 年关于 Agent 协议引用最多的三篇 arXiv 论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要委托给专业写作 Agent；汇总结果；将交互式报告渲染为 MCP Apps `ui://` resource；将每步记录到 OTel。

Phase 13 的所有原语都出现了。这不是玩具——2026 年 Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPT）和第三方发布的生产研究助手系统具有完全相同的形状。

## 概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP 工具: arxiv_search (纯)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP 任务: generate_report (长)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A 调用: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI 跨度
```

### 追踪层次结构

```
agent.invoke_agent
 ├── llm.chat (启动)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── 任务转换（不透明内部）
 ├── mcp.call -> tools/call generate_report (任务增强)
 │    └── tasks/status 轮询
 │    └── tasks/result (完成，返回 ui:// resource)
 └── llm.chat (最终综合)
```

一个 trace id。每个跨度都有正确的 `gen_ai.*` 属性。

### 安全姿态

- OAuth 2.1 + PKCE，资源指示器将受众固定到网关。
- 网关持有上游凭证；用户永远看不到它们。
- RBAC：`alice` 有 `research:read`、`research:write`，可以调用所有工具。`bob` 有 `research:read`，不能调用 `generate_report`。
- 固定描述清单：丢弃工具哈希变更的任何服务器。
- 二选一原则审计：没有工具组合不可信输入、敏感数据和影响型操作。

### 渲染

最终的 `generate_report` 任务返回内容块加 `ui://report/current` resource。客户端的宿主（Claude Desktop 等）在沙盒 iframe 中渲染交互式仪表板。仪表板包含排序论文列表、引用计数，以及一个按钮，调用 `host.callTool('summarize_paper', {arxiv_id})` 用于用户点击的任何论文。

### 打包

整个东西作为以下发货：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶层工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                      # A2A Agent
  gateway/
    config.yaml                 # RBAC + 固定清单
```

用户用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动系统。

### 每节 Phase 13 课程的贡献

| 课程 | 顶点项目使用 |
|------|------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、schema、检查 |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + prompts |
| 11-14 | 采样、roots + elicitation、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子 Agent 委托 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 使用它

`code/main.py` 将前面课程的模式缝合到一个可运行演示中。全部标准库，全部进程内，因此你可以从头到尾阅读。它运行研究和报告场景的完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、`generate_report` 作为任务、A2A 调用 writer、`ui://` resource 返回、OTel 跨度发出。

要注意的点：

- 跨每个跳的单一 trace id。
- 网关策略阻止第二个用户写入。
- 任务生命周期经过 working → completed 并返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器不透明。
- AGENTS.md 和 SKILL.md 是其他 Agent 再现工作流所需的唯一文件。

## 发布它

本课生成 `outputs/skill-ecosystem-blueprint.md`。给定产品需求（研究、摘要、自动化），该 Skill 生成完整架构：哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、哪些打包。

## 练习

1. 运行 `code/main.py`。注意单一 trace id 和跨度如何嵌套。计算演示触及 Phase 13 的多少个原语。

2. 扩展演示：添加第二个后端 MCP 服务器（如 `bibliography`）并确认网关将其工具合并到同一命名空间。

3. 用运行在子进程上的真实 A2A writer Agent 替换假 A2A writer Agent。使用第 19 课 harness。

4. 在编排器和 LLM 之间的路由网关中添加 PII 编辑步骤。确认用户查询中的电子邮件被清除。

5. 为将维护此系统的队友写一个 AGENTS.md。它应该用不到五分钟读完，并给他们驱动顶点项目（在 Cursor 或 Codex 中）所需的一切。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Capstone（顶点项目） | "Phase-13 集成演示" | 使用每个原语的端到端系统 |
| Research and report（研究和报告） | "场景" | 搜索、摘要、渲染模式 |
| Ecosystem（生态系统） | "所有组件在一起" | 服务器 + 客户端 + 网关 + 子 Agent + 遥测 + 打包 |
| Trace hierarchy（追踪层次结构） | "单一 trace id" | 每个跳的跨度共享 trace；通过 span id 的父子 |
| Gateway-issued token（网关发行令牌） | "传递认证" | 客户端只看到网关的令牌；网关持有上游凭证 |
| Merged namespace（合并命名空间） | "所有工具在一个扁平列表中" | 在网关合并多服务器，冲突时加前缀 |
| Opacity boundary（不透明边界） | "A2A 调用隐藏内部" | 子 Agent 的推理对编排器不可见 |
| Three-layer stack（三层栈） | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| Defense-in-depth（纵深防御） | "多层安全" | 固定哈希、OAuth、RBAC、二选一原则、审计日志 |
| Spec compliance matrix（规范合规矩阵） | "我们发货的规范要求" | 将交付物映射到 2025-11-25 要求的检查清单 |

## 延伸阅读

- [MCP — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 合并参考
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议走向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范追踪约定
- [Anthropic — Claude Agent SDK 概述](https://code.claude.com/docs/en/agent-sdk/overview) — 生产 Agent 运行时模式