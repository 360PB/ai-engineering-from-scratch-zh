# Capstone 13 — 带注册中心与治理的 MCP 服务器

> Model Context Protocol 不再是未来，而是2026年的默认工具使用规范。Anthropic、OpenAI、Google 和每个主流 IDE 都内置了 MCP 客户端。Pinterest 发布了其内部 MCP 服务器生态。AAIF Registry 在 `.well-known` 规范化了能力元数据。AWS ECS 发布了无状态部署参考。Block 的 goose-agent 将同一协议放入托管助手。2026年的生产形态：StreamableHTTP 传输、OAuth 2.1 作用域、OPA 策略门控，以及让平台团队发现、验证和启用服务器的注册中心。从零构建端到端。

**类型：** 毕业项目
**语言：** Python（服务器，通过 FastMCP）或 TypeScript（@modelcontextprotocol/sdk），Go（注册中心服务）
**前置知识：** Phase 11（LLM工程）、Phase 13（工具与MCP）、Phase 14（Agent工程）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P11 · P13 · P14 · P17 · P18
**时长：** 25小时

## 问题

MCP 成为工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 和每个托管 Agent 现在都消费 MCP 服务器。生产挑战不在创作服务器（FastMCP 让这很容易），而是大规模部署企业级需求：每租户 OAuth 作用域、破坏性工具的 OPA 策略、StreamableHTTP 无状态扩展、用于发现的注册中心、每工具调用审计日志。Pinterest 内部 MCP 生态和 AAIF Registry 规范设定了2026年的标准。

你将构建一个暴露10个内部工具（Postgres 只读、S3 列表、Jira、Linear、Datadog 等）的 MCP 服务器、一个用于平台发现的注册中心 UI，以及破坏性工具的人工审批门控。负载测试演示 StreamableHTTP 水平扩展。审计追踪满足企业安全评审。

## 核心概念

MCP 2026 修订版强制 StreamableHTTP 为默认传输。与早期的 stdio 加 SSE 形态不同，StreamableHTTP 默认无状态：单一 HTTP 端点接受 JSON-RPC 请求，流式响应，支持长连接的notifications。无状态意味着可在负载均衡器后水平扩展。

授权是带每工具作用域的 OAuth 2.1。Token 携带 `jira:read`、`s3:list`、`postgres:query:readonly` 等作用域。MCP 服务器在工具调用时检查作用域，而非会话启动时。对于高风险工具，服务器拒绝任何在过去 N 分钟内未通过 Slack 审批卡将作用域升级为 `approved:by:human` 的调用。

注册中心是独立服务。每个 MCP 服务器暴露 `.well-known/mcp-capabilities` 文档，含工具清单、传输 URL、认证要求。注册中心轮询、验证和索引。平台团队使用注册中心 UI 查看哪些工具可用、需要哪些作用域、哪个团队拥有它们。

## 架构

```
MCP 客户端（Claude Code、Cursor 3、...）
          |
          v
StreamableHTTP over HTTPS（JSON-RPC + 流式）
          |
          v
MCP 服务器（FastMCP）在负载均衡器后
          |
     +------+------+---------+----------+------------+
     v             v         v          v            v
Postgres      S3 列表    Jira       Linear     Datadog
（只读）      （分页）    （读）     （读）     （查询）
          |
     +------+-------------+
     v                    v
OPA 策略门控          破坏性工具 MCP（独立服务器）
                        |
                        v
                   人工审批 via Slack
                        |
                        v
                   审计日志（只追加，每租户）

  注册中心服务
     |
     v  GET /.well-known/mcp-capabilities 从每个服务器
     v
     UI：搜索 / 验证 / 启用禁用 / 所有权
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输：StreamableHTTP over HTTPS（无状态）
- 认证：OAuth 2.1，通过 SPIFFE / SPIRE 的工作负载身份
- 策略：OPA / Rego 规则 per 工具；每个请求的策略决策服务
- 注册中心：自托管，消费 `.well-known/mcp-capabilities` 清单
- 人工审批：破坏性工具的 Slack 交互消息
- 部署：AWS ECS Fargate 或 Fly.io，每租户一个服务器或共享带租户范围
- 审计：结构化 JSONL 每租户桶，含每次调用的溯源

## 动手实现

1. **工具表面。** 暴露10个内部工具：Postgres 只读查询、S3 列出对象、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 在岗查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 只读。每个工具有类型 schema 和作用域标签。

2. **FastMCP 服务器。** 挂载工具。配置 StreamableHTTP 传输。添加 OAuth token 检查和作用域强制的中间件。

3. **OPA 策略。** 每个工具的 Rego 策略：哪些作用域允许调用、哪些 PII 脱敏适用、哪些 payload 大小上限适用。每个工具调用调用决策服务。

4. **注册中心服务。** 单独的 Go 或 TS 服务，从注册的服务器轮询 `.well-known/mcp-capabilities`，用 JSON Schema 验证，并暴露 list / search / validate / enable-disable UI。

5. **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`：工具列表、认证要求、传输 URL、所有者团队、SLO。

6. **破坏性工具分离。** 改变状态的工具（Jira 创建、Linear 创建、Postgres 写）在第二个 MCP 服务器上，更严格的认证流程：token 必须在过去15分钟内通过 Slack 卡将作用域升级为 `approved:by:human`。

7. **审计日志。** 每租户只追加 JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 做 PII 脱敏。

8. **负载测试。** 100个并发客户端在 StreamableHTTP 上。添加第二个副本演示水平扩展；展示负载均衡器无会话粘性地重新分配。

9. **一致性测试。** 用官方 MCP 一致性套件在两个服务器上运行。通过所有必选部分。

## 用现成库

```bash
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[注册中心]  能力验证通过：postgres.readonly v1.2
[策略]     作用域 postgres:query:readonly 存在；允许
[审计]     记录：user=u42 tool=postgres.readonly outcome=ok
响应:      { "result": { "rows": [[1]] } }
```

## 产出

`outputs/skill-mcp-server.md` 描述交付物。生产级 MCP 服务器 + 注册中心 + 审计层，面向内部工具，带 OAuth 2.1 作用域和 OPA 门控。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | 作用域强制、每个工具的 OPA 覆盖、密钥卫生 |
| 20 | 可观测性 | 带 PII 脱敏的每工具调用审计日志 |
| 20 | 规模 | 100 客户端负载测试水平扩展演示 |
| 15 | 注册中心 UX | 发现 / 验证 / 启用禁用工作流 |
| **100** | | |

## 练习

1. 添加新工具（Confluence 搜索）。通过注册中心验证流程发布它，不触动核心服务器。

2. 编写一个 OPA 策略，对包含名为 `email`、`ssn` 或 `phone` 列的 Postgres 查询结果脱敏。用探测查询练习。

3. 基准测试 StreamableHTTP vs stdio 的本地延迟。报告 per-call p50/p95。

4. 实现每租户配额：每个工具每租户每分钟最多 N 次调用。通过第二个 OPA 规则强制执行。

5. 运行 MCP 一致性套件来自 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 并修复每个失败。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| StreamableHTTP | "2026 MCP 传输" | 无状态 HTTP + 流式；替代 SSE + stdio 用于网络服务器 |
| Capability manifest | "Well-known 文档" | `.well-known/mcp-capabilities` 含工具列表、认证、传输 URL |
| OPA / Rego | "策略引擎" | Open Policy Agent 用于基于外部规则授权工具调用 |
| Scope elevation | "人类审批" | 通过 Slack 审批授予的短期作用域，破坏性工具需要 |
| Registry | "工具发现" | 从 MCP 服务器能力清单索引的服务 |
| Workload identity | "SPIFFE / SPIRE" | 用于 OAuth token 发行的加密服务身份 |
| Conformance suite | "规范测试" | StreamableHTTP + 工具清单正确性的官方 MCP 测试套件 |

## 扩展阅读

- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册中心
- [AAIF MCP Registry 规范](https://github.com/modelcontextprotocol/registry) — 2026年注册中心规范
- [AWS ECS 参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产部署
- [Pinterest 内部 MCP 生态](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署
- [Block `goose` MCP 用法](https://block.github.io/goose/) — 参考 Agent 消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架
- [Open Policy Agent](https://www.openpolicyagent.org/) — 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份参考