# 安全 — 密钥、API 密钥轮换、审计日志、护栏

> 通过集中式 vault（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥蔓延。绝不把凭证存在配置文件、VCS 中的 env 文件、电子表格里。用 IAM 角色替代静态密钥；CI/CD 用 OIDC。2026 年方案是 AI 网关模式：应用 → 网关 → 模型供应商，网关在运行时从 vault 拉凭证。在 vault 中轮换，所有应用分钟级 pickup——无需重新部署，无需"谁有新密钥"的 Slack 消息。轮换策略 ≤90 天；每次提交用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短效 token、设备态势。PII 清洗用实体识别在转发前遮盖 PHI/PII；一致性 token 化（Mesh 方法）将敏感值映射到稳定占位符，这样 LLM 保留代码/关系语义。网络出口：LLM 服务在专用 VPC/VNet 子网，仅允许清单 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站。2026 年事件驱动因素：Vercel 供应链攻击通过被攻击 CI/CD 凭证渗透了数千客户部署的 env vars。

**类型：** 精读
**语言：** Python（标准库，玩具级 PII 清洗器 + 审计日志写入器）
**前置要求：** Phase 17 · 19（AI 网关）、Phase 17 · 13（可观测性）
**时长：** 约 60 分钟

## 学习目标

- 枚举四个密钥管理反模式（VCS 中的配置文件、硬编码 env、电子表格、静态密钥）并说出替代方案。
- 解释 AI 网关从 vault 拉取模式作为 2026 年生产标准。
- 实现带一致性 token 化（相同值 → 相同占位符）的 PII 清洗器，以便语义保留。
- 说出 2026 年 Vercel 供应链事件及它对 CI/CD 凭证卫生的教训。

## 背景问题

实习生提交了含 API 密钥的 `.env`。很快删了。密钥已在 git 历史——GitGuardian 扫描捕获，轮换流程是"Slack 全员，更新 40 个配置文件，重新部署所有服务。"8 小时后，一半服务已上线，一半在等部署窗口。

另外，用户提示词里有"我的社保号是 123-45-6789。"提示词发给 OpenAI。你有 BAA 但内部政策是转发前遮盖 PII。你没做。

另外，你的 EKS 集群上 LLM pod 可以到达任何互联网主机。有人通过 DNS 查询向攻击者控制的域名泄露数据。没拦。

LLM 服务安全必须覆盖所有三个向量。Vault 支撑凭证。PII 清洗。网络出口过滤。审计日志。

## 核心概念

### 集中式 vault + IAM 角色拉取

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一真相来源。

**IAM 角色**：应用/网关通过其 IAM 身份认证，不是静态密钥。Vault 返回令牌生命周期内的密钥。

**AI 网关模式**：网关在请求时从 vault 拉取 `OPENAI_API_KEY`。在 vault 中轮换；下次请求获取新密钥。无需重新部署。

### 轮换策略 ≤90 天

所有 API 密钥、vault 根令牌、CI/CD 凭证。尽可能自动化轮换。手动轮换要记录和追踪。

### 密钥扫描

- **TruffleHog** — commit 上正则 + 熵检测。
- **GitGuardian** — 商业，高准确率。
- **Gitleaks** — OSS，在 CI 中跑。

每次提交跑。检测到新密钥则 block PR。

### 零信任姿态

- 所有账户要求 MFA。
- 通过 SAML/OIDC 的 SSO。
- RBAC（基于角色）或 ABAC（基于属性）做细粒度访问控制。
- 短效 token（小时，不是天）。
- 设备态势——仅 corp 设备加磁盘加密。

### PII / PHI 清洗

在提示词离开你基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业产品）。
2. 遮盖匹配的实体：`"我的社保号是 123-45-6789"` → `"我的社保号是 [SSN_TOKEN_A3F]"`。
3. 一致性 token 化（Mesh 方法）：相同值映射到相同占位符，这样 LLM 保留关系。
4. 可选：LLM 响应的反向映射。

静态正则过滤器捕获基本模式；NER 捕获更多。两者都用。

### 输入 + 输出护栏

输入：阻止已知越狱、禁止话题；按用户限流。

输出：正则清洗泄露密钥（API 密钥格式、邮件格式在拒绝上下文中）、分类器检测策略违规。

### 网络出口白名单

LLM 服务在专用子网：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量 DB 端点、vault 端点。
- 其他：一律丢弃。
- DNS 通过仅允许解析器（避免 DNS 隧道泄露）。

### 审计日志

每条 LLM 调用的不可变日志：
- 时间戳。
- 用户 / 租户。
- 提示词哈希（不为隐私存原始提示词）。
- 模型 + 版本。
- Token 数。
- 成本。
- 响应哈希。
- 任何护栏触发。

按监管要求保留（SOC 2 一年，HIPAA 六年）。

### 2026 年 Vercel 事件

供应链攻击：被攻击 CI/CD 凭证渗透了数千客户部署的 env vars。教训：CI/CD 凭证等于生产凭证。存 vault。范围窄。激进轮换。

### 必须记住的数字

- 轮换策略：≤90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证被攻 → 数以千计客户 env vars 泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 用现成库

`code/main.py` 实现带一致性 token 化的玩具 PII 清洗器和仅追加审计日志。

## 产出

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围和当前状态，计划 vault 迁移、洗涤器、出口、审计日志。

## 练习

1. 运行 `code/main.py`。发送两条引用相同社保号的提示词。确认两条得到相同占位符。
2. 为 vLLM-on-EKS 部署设计网络出口策略，调用 OpenAI + Anthropic + Weaviate。
3. 你在 git 历史中发现一个密钥（2 年前）。正确响应是什么——轮换密钥、清洗历史还是两者？说明理由。
4. 审计日志每天增长 10 GB。设计保留层级（热 30d、暖 12mo、冷 6yr）。
5. 论证反向 token 化（将真实值替换回 LLM 响应）是否值得复杂性，还是保持占位符可见。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Vault | "密钥库" | 集中式凭证管理服务 |
| IAM 角色 | "基于身份认证" | 应用承担的角色；返回短效凭证 |
| CI/CD 用 OIDC | "云颁发令牌" | CI 中无静态密钥——通过 OIDC 身份 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时密钥检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII 清洗 | "数据遮盖" | 移除或 token 化敏感实体 |
| 一致性 token 化 | "稳定占位符" | 相同值每次映射到相同 token |
| Mesh 方法 | "Mesh token 化" | 保留语义的 token 化模式 |
| 出口白名单 | "出站允许清单" | 仅允许的可达域名 |
| 审计日志 | "不可变历史" | 用于合规的仅追加记录 |

## 扩展阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测和匿名化。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)