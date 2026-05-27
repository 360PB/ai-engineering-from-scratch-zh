# MCP 网关和注册表 —— 企业控制平面

> 企业不能让每个开发人员安装任意 MCP 服务器。网关集中管理认证、RBAC、审计、速率限制、缓存和工具投毒检测，然后作为单一 MCP 端点暴露合并的工具表面。官方 MCP 注册表（Anthropic + GitHub + PulseMCP + Microsoft，命名空间验证）是规范上游。本课命名网关适合的位置，走查最小化实现，并概述 2026 年供应商格局。

**类型：** 学习
**语言：** Python（标准库、最小化网关）
**前置要求：** Phase 13 · 15（工具投毒）、Phase 13 · 16（OAuth 2.1）
**时间：** 约 45 分钟

## 学习目标

- 解释 MCP 网关在哪里（介于 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现五个网关职责：认证、RBAC、审计、速率限制、策略。
- 在网关层强制执行固定工具哈希清单。
- 区分官方 MCP 注册表和元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

财富 500 强企业有 30 个批准 MCP 服务器、5000 名开发人员、合规和审计要求，以及想要集中策略的安全团队。让每个开发人员在 IDE 中安装任意服务器是不可行的。

网关模式：

1. 网关作为单个 Streamable HTTP 端点运行，开发人员连接到它。
2. 网关持有每个后端 MCP 服务器的凭证。
3. 每个开发人员请求通过网关自己的 OAuth 进行认证和作用域映射。
4. 网关路由调用到后端服务器，应用策略。
5. 所有调用都记录以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway——2025-2026 年都发布了网关或网关功能。

与此同时，官方 MCP 注册表作为规范上游启动：策划的、命名空间验证的、反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）聚合来自多个来源的服务器。

## 概念

### 五个网关职责

1. **认证。** OAuth 2.1 识别开发人员；映射到用户角色。
2. **RBAC。** 每用户策略：哪些服务器、哪些工具、哪些作用域。
3. **审计。** 每个调用都记录谁、何时、什么、结果。
4. **速率限制。** 每用户 / 每工具 / 每服务器上限以防止滥用。
5. **策略。** 拒绝投毒描述、执行二选一原则、编辑 PII。

### 网关作为单一端点

对开发人员来说，网关看起来像一个 MCP 服务器。内部路由到 N 个后端。会话 id（Phase 13 · 09）在边界处重写。

### 凭证保险库

开发人员永远看不到后端令牌。网关持有它们（或代理到这样做的身份提供商）。在网关上拥有 `notes:read` 的开发人员可能经由网关自己的后端凭证传递访问笔记 MCP 服务器——但仅在将传递访问绑定的策略下。

### 网关层工具哈希固定

网关持有批准工具描述清单（SHA256 哈希）。在发现时，它获取每个后端的 `tools/list`，将哈希与清单比较，并移除描述已变更的任何工具。这是 Phase 13 · 15 的拉地毯防御的集中应用。

### 策略即代码

高级网关用 OPA/Rego、Kyverno 或 Styra 表达策略。像"用户 `alice` 只能在 org `acme` 的仓库上调用 `github.open_pr`"这样的规则被声明性编码。简单网关使用手写 Python。两种形状都有效。

### 会话感知路由

当用户的会话包含服务器混合时，网关多路复用：开发者的单一 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的通知通过网关路由到开发者的会话。

### 命名空间合并

网关合并所有后端的工具命名空间，通常带冲突前缀。`github.open_pr`、`notes.search`。这使路由明确无歧。

### 注册表

- **官方 MCP 注册表（`registry.modelcontextprotocol.io`）。** 在 Anthropic、GitHub、PulseMCP、Microsoft 托管下启动。命名空间验证（反向 DNS：`io.github.user/server`）。预过滤基本质量。
- **Glama。** 搜索导向的元注册表，聚合多个来源。
- **MCPMarket。** 商业导向目录，带供应商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 在其 LobeChat 应用中的 UI 集成注册表。

企业网关默认从官方注册表拉取，允许管理员策划的元注册表添加，并拒绝任何未固定的内容。

### 反向 DNS 命名

官方注册表要求公共服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间防止抢注，使信任委托更清晰。

### 供应商调查，2026 年 4 月

| 供应商 | 优势 |
|--------|------|
| Cloudflare MCP Portals | 边缘托管；OAuth 集成；免费层 |
| Kong AI Gateway | K8s 原生；细粒度策略；日志输出到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | DevOps 导向；指标优先 |
| MintMCP | 开发者平台导向 |
| Envoy AI Gateway | 开源；可自定义过滤器 |

Phase 17（生产基础设施）深入讲网关运营。

## 使用它

`code/main.py` 用约 150 行发货一个最小化网关：通过假 Bearer 令牌认证用户，持有每用户 RBAC 策略，路由请求到两个后端 MCP 服务器，将每个调用写入审计日志，强制速率限制，并拒绝描述哈希与固定清单不匹配的任何后端工具。

要注意的点：

- `RBAC` dict 按 `user_id` 键控，带允许的 `server_tool` 条目。
- `AUDIT_LOG` 是仅追加的事件列表。
- 速率限制对每个用户使用令牌桶。
- 固定清单是 `server::tool -> hash` 的 dict。

## 发布它

本课生成 `outputs/skill-gateway-bootstrap.md`。给定企业 MCP 计划（用户、后端、合规），该 Skill 生成网关配置规范。

## 练习

1. 运行 `code/main.py`。作为允许用户调用；然后作为不允许用户；然后超过速率限制。验证所有三种流程。

2. 添加在返回给客户端之前编辑结果中 PII 的策略。使用简单正则表达式传递来编辑 SSN 形状字符串；注意差距（电子邮件、电话号码）。

3. 将审计日志扩展为发出 OpenTelemetry GenAI 跨度。Phase 13 · 20 涵盖确切属性。

4. 为 50 名开发人员和五个后端（notes、github、postgres、jira、slack）的团队设计 RBAC 策略。谁在每个上获得只读？谁获得写权限？

5. 从头到尾阅读 Cloudflare 企业 MCP 帖子。找出 Cloudflare 发货而此标准库网关没有的一个功能。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Gateway（网关） | "MCP 代理" | 集中服务器位于客户端和后端之间 |
| Credential vaulting（凭证保险库） | "后端令牌留在服务器端" | 开发人员永远看不到上游令牌 |
| Session-aware routing（会话感知路由） | "多后端会话" | 网关多路复用每个开发者会话的 N 个后端会话 |
| Tool-hash pinning（工具哈希固定） | "已批准清单" | 每个批准工具描述的 SHA256；集中阻止拉地毯 |
| RBAC | "每用户策略" | 工具和服务器的基于角色的访问控制 |
| Policy-as-code（策略即代码） | "声明性规则" | 在网关强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| Audit log（审计日志） | "谁、什么、何时" | 用于合规的仅追加事件日志 |
| Rate limit（速率限制） | "每用户令牌桶" | 每分钟上限以防止滥用 |
| Official MCP Registry（官方 MCP 注册表） | "规范上游" | `registry.modelcontextprotocol.io`，命名空间验证 |
| Reverse-DNS naming（反向 DNS 命名） | "注册表命名空间" | `io.github.user/server` 约定 |

## 延伸阅读

- [官方 MCP 注册表](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间验证
- [Cloudflare — 企业 MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和策略的网关模式
- [agentic-community — MCP 网关注册表](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry — 什么是 MCP 网关？](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能比较文章
- [IBM — MCP 上下文锻造](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关