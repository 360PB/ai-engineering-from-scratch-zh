# MCP 生产级认证 —— DCR、JWKS 轮换、Audience 绑定令牌

> 第 16 课在内存中实现了 OAuth 2.1 状态机。到了 2026 年，你交付给真实组织的每一个 MCP 服务器都会接入生产认证：动态客户端注册（RFC 7591）、授权服务器元数据发现（RFC 8414）、不中断凌晨 3 点令牌验证的 JWKS 轮换，以及拒绝混乱委托复用的 audience 绑定令牌。本课将所有这些通过 iii 原语连接起来——`iii.registerTrigger` 用于 HTTP 和 cron，`iii.registerFunction` 用于认证逻辑，`state::set/get` 用于缓存密钥——使认证面可观测、可重启、可重放，与引擎中的其他工作负载一致。

**类型：** 构建
**语言：** Python（stdlib，iii 原语在本课环境中为模拟）
**前置要求：** Phase 13 · 16（OAuth 2.1 状态机）、Phase 13 · 17（网关）
**时间：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可自主注册。
- 使用 cron 触发器缓存和轮换 JWKS 密钥，确保签名验证在密钥滚动期间不中断。
- 使用 RFC 8707 资源指示符将令牌绑定到单一 MCP 资源，并拒绝混乱委托复用。
- 将每个端点和后台作业连接为 iii 原语——HTTP 触发器、cron 触发器、命名函数和 `state::*` 读写——使单次重启即可重建认证面。
- 读取 IdP 能力矩阵，当 IdP 无法满足 MCP 认证配置时拒绝部署。

## 问题

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境有三个内存模拟器看不到的操作缺口。

第一个缺口是注册。真实组织运行着数百个 MCP 服务器和数千个 MCP 客户端。运维人员不会手动为每个 Cursor 用户注册为 OAuth 客户端。RFC 7591 动态客户端注册让客户端向授权服务器 `POST /register`，即时获得 `client_id`（以及可选的 `client_secret`）。服务器在其 RFC 8414 元数据中发布 `registration_endpoint`；客户端无需带外配置即可发现它。

第二个缺口是密钥轮换。JWT 验证依赖于授权服务器的签名密钥，以 JSON Web 密钥集（JWKS）形式发布。授权服务器按计划轮换这些密钥（通常每小时一次，在事件响应时可能更快）。在启动时一次性获取 JWKS 的 MCP 服务器，在轮换窗口前验证正常——之后所有请求失败，直到重启。生产环境将 JWKS 作为缓存值处理，并配置一个刷新作业，在上一个密钥过期前覆盖缓存，外加缓存未命中时的同步回退获取（处理令牌签名密钥比缓存更新的情况）。

第三个缺口是 audience 绑定。第 16 课引入了 RFC 8707 资源指示符。在生产环境中，该指示符成为每个请求上的硬性声明检查。MCP 服务器将 `token.aud` 与其自身的规范资源 URL 进行比较，拒绝不匹配项并返回 HTTP 401。这是防止上游 MCP 服务器（或持有本应发给某个服务器的令牌的恶意客户端）将该令牌重放给同一信任网状中的另一个服务器的唯一防御手段。

本课将每个缺口都作为 iii 原语处理。元数据文档是一个返回函数输出的 HTTP 触发器。JWKS 轮换是一个调用 `auth::rotate-jwks` 的 cron 触发器，后者写入 `state::set("auth/jwks/<issuer>", ...)`。JWT 验证是一个其他方通过 `iii.trigger("auth::validate-jwt", token)` 调用的函数。MCP 服务器本身只是一个在分发前调用验证的 HTTP 触发器。重启引擎：触发器注册表重建；状态保留；认证面无需手动协调即可运行。

## 核心概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定了 MCP 资源 URL 的客户端链式发现：RFC 9728 的 `oauth-protected-resource`（资源服务器文档）指明 issuer，然后本 RFC 的 `oauth-authorization-server` 指明每个端点。客户端永不硬编码授权 URL。

在信任 IdP 托管 MCP 之前验证的契约：

- `code_challenge_methods_supported` 必须包含 `S256`（RFC 7636 的 PKCE）。
- `grant_types_supported` 必须包含 `authorization_code` 并拒绝 `password` 和 `implicit`。
- `registration_endpoint` 必须存在（RFC 7591 支持）。
- `response_types_supported` 必须恰好为 `["code"]`（OAuth 2.1）。

如果任何一项缺失，MCP 服务器拒绝针对此 IdP 部署。部署清单是错的，不是代码。

### RFC 9728（回顾）— 受保护资源元数据

第 16 课覆盖了 RFC 9728。生产中的差异：此文档是客户端查找此 MCP 服务器所信任的授权服务器的唯一位置。一个 MCP 服务器可能接受来自多个 IdP 的令牌（一个给员工，一个给合作伙伴）。RFC 9728 声明该集合；RFC 8414 记录每个 IdP 支持的内容。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — 动态客户端注册

没有 DCR，每个 MCP 客户端（Cursor、Claude Desktop、自定义 agent）都需要与 IdP 管理员进行带外交换。使用 DCR，客户端发送：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP 客户端的正确默认值。它们只获得 `client_id`——没有可泄露的 `client_secret`。PKCE 为公共客户端提供所需的存在证明。

三个生产陷阱：

- 注册端点必须按源 IP 做速率限制。否则恶意行为者可以脚本化百万级虚假注册，耗尽 `client_id` 命名空间。iii 使这变得简单：注册 HTTP 触发器在分派给注册器之前调用 `auth::rate-limit` 函数。
- 某些企业 IdP 要求 `software_statement`（为客户端担保的签名 JWT）。本课的模拟跳过此字段；生产环境需要验证步骤，拒绝来自 localhost 以外重定向 URI 的无签名注册。
- `registration_access_token` 必须以哈希形式存储，而非明文。一旦此令牌被盗，攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示符

第 16 课建立了基本形态。生产规则：每个令牌请求必须包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 与其自身资源 URL 匹配。如果 MCP 服务器可通过 `https://notes.example.com/mcp` 访问，规范 URL 是 `https://notes.example.com`——排除路径部分，这样单个服务器可在单一 audience 下托管多个路径。

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制性的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器拒绝任何没有验证因子或验证因子哈希值与存储的 challenge 不匹配的令牌请求。

### MCP Spec 2025-11-25 认证配置文件

MCP 规范（2025-11-25）对 MCP 服务器授权层必须执行的操作有精确规定：

- 在 `/.well-known/oauth-protected-resource`（RFC 9728）上发布。
- 仅通过 `Authorization: Bearer ...` 接受令牌。
- 每次请求验证 `aud`、`iss`、`exp` 和所需 scopes。
- 每个 401 和 403 都以携带 `Bearer error=...` 的 `WWW-Authenticate` 响应，包括适用的 `scope=` 和 `resource=` 参数。
- 拒绝 `aud` 与规范资源不匹配的令牌。
- 拒绝 `iss` 不在受保护资源元数据 `authorization_servers` 列表中的令牌。

OAuth 2.1 草案是底层；RFC 8414/7591/8707/9728 + RFC 7636 是表面；MCP 规范是配置文件。

### IdP 能力矩阵

并非每个 IdP 都支持完整的 MCP 配置。以下矩阵记录了截至 2025-11-25 规范的事实性能力声明。它是**部署门槛**，不是推荐。

| IdP 类别 | RFC 8414 元数据 | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|
| 自托管（Keycloak） | 是 | 是 | 是（自 24.x 起） | 是 | 本课的参考 IdP；支持每个 RFC 端到端。 |
| 企业 SSO（Microsoft Entra ID） | 是 | 是（高级层） | 是 | 是 | DCR 可用性因租户层级而异；部署前在目标租户中验证。 |
| 企业 SSO（Okta） | 是 | 是（Okta CIC / Auth0） | 是 | 是 | DCR 可在 Auth0（现为 Okta CIC）上使用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（通用） | 视情况 | 很少 | 很少 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；不要依赖 DCR。仅用作身份源，在其上叠加你自己的 MCP 感知授权服务器。 |
| 自定义/自研 | 视情况 | 视情况 | 视情况 | 视情况 | 如果你自己实现，要实现完整配置。省略上述四个 RFC 中的任何一个都会破坏 MCP 认证契约。 |

部署清单的拒绝规则：如果所选 IdP 未返回 `registration_endpoint` 且未在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器拒绝启动。没有降级模式。

### 使用 iii 进行 JWKS 轮换模式

生产故障模式是过期的 JWKS 缓存。通过 cron 触发器和 `state::*` 缓存解决：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，cron 触发器调用 `auth::rotate-jwks`，后者获取 `<issuer>/.well-known/jwks.json` 并写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从 `state::get` 读取。当令牌的 `kid` 在缓存中缺失时，触发同步 `auth::rotate-jwks` 调用作为回退。这同时处理两种情况：计划轮换（cron）和密钥重叠窗口（同步回退）。

状态形态：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两个密钥是稳态。授权服务器通过在退役前一个（`k_2026_03`）之前引入下一个密钥（`k_2026_04`）来轮换，这样在旧密钥下签发的令牌在其过期前仍然有效。缓存保存并集；验证器按 `kid` 选取。

### iii 原语连接（本课的核心部分）

五个原语组成认证面：

```python
# 1. RFC 8414 元数据文档
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 动态客户端注册
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT 验证作为可调用函数（资源服务器触发它）
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. 用于增量 scope 的步进发行（SEP-835 from L16）
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron 驱动的 JWKS 轮换
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP 服务器本身从不直接调用验证。它做的是：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接调用就是 iii 的核心价值。明天你可以将验证器替换为同时查询两个 IdP 的扇出，或者添加一个 span 发射器，或者缓存正向验证。MCP 服务器无需改变。

### 通过 audience 绑定演练混乱委托

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都向同一个授权服务器注册。服务器 A 被攻陷。攻击者获取用户的 notes 令牌并将其重放至服务器 B。

服务器 B 的验证器：

1. 解码 JWT，按 `kid` 从 JWKS 获取，验证签名。
2. 根据其受保护资源元数据的 `authorization_servers` 检查 `iss`。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401 并附上 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

在协议层，audience 声明是抵御此攻击的唯一防御手段。为性能跳过它是生产中最常见的错误；验证器必须在每个请求上运行，而非仅在会话开始时运行。

### 故障模式

- **过期的 JWKS。** 密钥轮换后验证器拒绝有效令牌。修复方法是上述 cron+回退模式。永远不要在没有刷新作业的情况下缓存 JWKS。
- **缺失的 `aud` 声明。** 某些 IdP 默认省略 `aud`，除非令牌请求中存在 `resource`。验证器必须拒绝缺失 `aud` 的令牌，而非将缺失视为通配符。
- **Scope 升级竞态。** 同一用户的两个并发步进流程可能同时成功，并产生两个作用域不同的访问令牌。验证器必须使用请求上呈现的令牌，而非查找"用户当前 scope"——这会产生 TOCTOU 窗口。
- **注册令牌被盗。** 泄露的 `registration_access_token` 让攻击者可以重写重定向 URI。在静态存储时对这些做哈希处理；要求客户端在每次更新时呈现明文；在可疑时轮换。
- **`iss` 未绑定。** 接受任何 `iss` 的验证器让攻击者可以架设自己的授权服务器，为目标 audience 注册客户端并签发令牌。受保护资源元数据的 `authorization_servers` 列表是允许列表；强制执行它。

## 使用

`code/main.py` 通过 stdlib Python 和一个模拟 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 和 `state::set/get` 的小型 `iii_mock` 注册表来演练完整的生产流程。流程：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 上发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，发现注册端点。
3. MCP 客户端向 `/register` 发送请求（RFC 7591）并获得 `client_id`。
4. MCP 客户端运行带 PKCE 保护的授权码流程（RFC 7636）以及 `resource` 指示符（RFC 8707）。
5. MCP 客户端使用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器触发 `auth::validate-jwt`，后者从 `state::get` 读取 JWKS。
7. cron 触发器触发 `auth::rotate-jwks`，替换状态中的 JWKS。
8. 下一个调用在无需重启的情况下根据新密钥进行验证。
9. 针对不同 MCP 资源的混乱委托尝试返回 401 并附上 audience 不匹配。

这里的模拟 JWT 使用带共享密钥的 HS256（使本课仅用 stdlib 即可运行）。生产使用 RS256 或 EdDSA 以及上述 JWKS 模式；否则验证逻辑完全相同。

## 交付

本课产出 `outputs/skill-mcp-auth-iii.md`。给定一个 MCP 服务器配置和 IdP 能力集，该 skill 生成要注册的 iii 原语、JWKS 轮换计划、scope 映射，以及当 IdP 不支持完整 RFC 配置时应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。跟踪 9 步流程。注意 `state::get` 在 `auth::rotate-jwks` 覆盖它之前立即返回过期数据的方式，以及下一个请求现在如何根据新密钥进行验证。

2. 在受保护资源元数据的 `authorization_servers` 列表中添加一个新的 IdP。签发一个由新 IdP 签名的令牌，并确认验证器接受它。签发一个由未列出 IdP 签名的令牌，并确认验证器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 将 `auth::rate-limit` 实现为 iii 函数，并在注册 HTTP 触发器内部注册器运行之前调用它。使用存储在 `state::set("auth/ratelimit/<ip>", ...)` 中的按源 IP 的令牌桶。

4. 阅读 RFC 7591 并找出本课 `/register` 处理程序未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI 方案。）

5. 阅读 MCP 规范 2025-11-25 授权部分。找出本课验证器当前未发出的关于 `WWW-Authenticate` 头的一个规范性要求。添加它。

## 关键术语

| 术语 | 别人怎么说 | 实际含义 |
|------|----------------|------------------------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程 |
| JWKS | "JWT 验证的公钥" | JSON Web 密钥集，从 `jwks_uri` 获取，按 `kid` 索引 |
| 资源指示符 | "Audience 参数" | RFC 8707 `resource` 参数将令牌绑定到一个服务器 |
| `aud` 声明 | "Audience" | JWT 声明，验证器将其与规范资源 URL 比较 |
| 混乱委托 | "令牌重放" | 为服务器 A 签发的令牌被呈现给服务器 B 的攻击 |
| `iss` 允许列表 | "可信授权服务器" | 受保护资源元数据 `authorization_servers` 中列出的集合 |
| 密钥轮换 | "滚动 JWKS" | 带重叠窗口的签名密钥定期替换 |
| 公共客户端 | "原生或浏览器客户端" | 没有 `client_secret` 的 OAuth 客户端；PKCE 弥补 |
| `WWW-Authenticate` | "401/403 响应头" | 携带 `Bearer error=...` 指令，驱动客户端恢复 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课实现 MCP 认证配置
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端存在证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience 绑定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 统一的 OAuth 底层