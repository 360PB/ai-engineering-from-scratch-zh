# MCP 安全 II —— OAuth 2.1、资源指示器、增量作用域

> 远程 MCP 服务器需要授权，而不仅仅是认证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）对齐。SEP-835 添加了增量作用域同意，带 403 WWW-Authenticate 的升级授权。本课实现升级流程作为状态机，让你看到每一跳。

**类型：** 构建
**语言：** Python（标准库、OAuth 状态机模拟器）
**前置要求：** Phase 13 · 09（传输）、Phase 13 · 15（安全 I）
**时间：** 约 75 分钟

## 学习目标

- 区分资源服务器和授权服务器职责。
- 走查 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混乱代表攻击。
- 实现升级授权：服务器用 WWW-Authenticate 响应 403 并请求更高作用域；客户端重新提示用户同意并重试。

## 问题

早期 MCP（2025 年之前）发布的远程服务器使用临时 API 密钥，甚至没有认证。2025-11-25 规范用完整 OAuth 2.1 profile 填补了差距。

三个真实需求：

- **普通远程服务器。** 用户安装访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 加 PKCE 是正确的形状。
- **作用域升级。** 被授予 `notes:read` 的笔记服务器后来可能需要 `notes:write` 来执行特定操作。升级（SEP-835）不是重做整个流程，而是请求额外作用域。
- **防止混乱代表。** 客户端持有一个针对服务器 A 的令牌受众范围令牌。服务器 A 是恶意的，试图将令牌呈现给服务器 B。资源指示器（RFC 8707）将令牌固定到其预期受众。

OAuth 2.1 不是新东西。新的是 MCP 的 profile：特定必需流程（仅授权码 + PKCE；无隐式，默认无客户端凭证）、每个令牌请求上强制资源指示器，以及发布受保护资源元数据以便客户端知道去哪里。

## 概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres，随你）。
- **授权服务器。** 发行令牌。可以与资源服务器是同一服务，也可以是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的 profile 中，资源服务器和授权服务器可以是同一主机，但应用 URL 区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器根据存储的挑战验证验证器的哈希并发行访问令牌。
6. 客户端在每个到资源服务器的请求上使用令牌：`Authorization: Bearer ...`。

PKCE 防止授权码拦截攻击。资源指示器防止令牌在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器在 `.well-known/oauth-protected-resource` 发布文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少配置——客户端只需资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数将令牌的预期受众固定。发出的令牌包含 `aud: "https://notes.example.com"`。接收到此令牌的其他 MCP 服务器检查 `aud` 并拒绝。

### 作用域模型

作用域是空格分隔的字符串。常见 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理能力（谨慎使用）
- `profile:read` 用于身份

作用域选择应遵循最小权限：现在需要什么就请求什么，需要更多时升级。

### 升级授权（SEP-835）

用户授予 `notes:read`。他们后来要求 Agent 删除一个笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 insufficient_scope 错误，用额外作用域的同意对话框提示用户，对其执行迷你 OAuth 流程，用新令牌重试请求。

### 令牌受众验证

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这阻止跨服务器令牌复用。

### 短期令牌和轮换

访问令牌应该是短期（默认 1 小时）。刷新令牌每次刷新都轮换。客户端在后台处理静默刷新。

### 无令牌传递

采样服务器（Phase 13 · 11）不得将客户端的令牌传递到其他服务。采样请求是边界。

### 防止混乱代表

令牌绑定到 `aud`。客户端绑定到 `client_id`。每个请求根据两者验证。规范明确禁止旧的"传递令牌"模式，该模式在 MCP 前的远程工具生态系统中很常见。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 发布其元数据。授权服务器可以获取客户端的元数据文档以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关和 OAuth

Phase 13 · 17 展示企业网关如何处理 OAuth：网关持有上游服务器凭证，发给客户端的令牌是网关发行的，上游令牌永不离开网关。这翻转了信任模型——用户一次向网关认证；网关处理 N 个服务器授权。

## 使用它

`code/main.py` 模拟完整 OAuth 2.1 升级流程作为状态机。它实现：

- PKCE code-verifier / challenge 生成。
- 带资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- `insufficient_scope` 上的升级。

本课没有 HTTP 服务器；状态机在内存中运行，因此你可以追踪每一跳。Phase 13 · 17 的网关课将其连接到实际传输。

## 发布它

本课生成 `outputs/skill-oauth-scope-planner.md`。给定带工具的远程 MCP 服务器，该 Skill 设计作用域集、固定规则和升级策略。

## 练习

1. 运行 `code/main.py`。追踪两个作用域的升级流程。注意升级时哪些跳重复。

2. 添加刷新令牌轮换：每次刷新发行新刷新令牌并使旧令牌失效。模拟被盗刷新令牌在轮换后被使用并确认它失败。

3. 使用 stdlib http.server 将受保护资源元数据端点实现为真实 HTTP 响应。镜像第 09 课的 /mcp 端点。

4. 为 GitHub MCP 服务器设计作用域层次结构：读取仓库、写入 PR、批准 PR、合并 PR、管理。每个级别之间使用升级。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 使用方式与 RFC 示例不同的字段。（提示：它涉及 `scopes_supported`。）

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| OAuth 2.1 | "现代 OAuth" | 强制 PKCE 并禁止隐式流的合并 RFC |
| PKCE | "持有证明" | 代码验证器 + 挑战击败授权码拦截 |
| Resource indicator（资源指示器） | "令牌受众" | RFC 8707 `resource` 参数将令牌固定到一台服务器 |
| Protected-resource metadata（受保护资源元数据） | "发现文档" | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization（升级授权） | "增量同意" | SEP-835 按需添加作用域的流程 |
| `insufficient_scope` | "带 WWW-Authenticate 的 403" | 服务器信号重新同意更大作用域 |
| Confused deputy（混乱代表） | "跨服务令牌复用" | 受信任持有者不当转发令牌的攻击 |
| Short-lived token（短期令牌） | "访问令牌 TTL" | 快速过期的 Bearer；刷新令牌续期 |
| Scope hierarchy（作用域层次结构） | "最小权限栈" | 带级别之间升级的渐进作用域集 |
| Client ID metadata（客户端 ID 元数据） | "客户端发现文档" | 客户端发布其自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 规范 MCP OAuth profile
- [den.dev — MCP 11 月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更走查
- [RFC 8707 — OAuth 2.0 资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定 RFC
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1、PKCE 和 AI 授权的未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实际升级流程走查