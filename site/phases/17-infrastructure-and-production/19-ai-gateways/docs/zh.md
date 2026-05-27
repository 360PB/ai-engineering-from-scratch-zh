# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于应用和模型供应商之间。核心功能：供应商路由、回退、重试、限流、密钥引用、可观测性、护栏。2026 年市场分布：**LiteLLM** 是 MIT OSS，100+ 供应商，OpenAI 兼容，但在约 2000 RPS 时崩溃（8 GB 内存，社区基准测试中级联故障）；最适合 Python、<500 RPS、开发/原型。**Portkey** 定位控制面（护栏、PII 删除、越狱检测、审计轨迹），2026 年 3 月 Apache 2.0 开源，延迟开销 20-40 ms，$49/月生产层。**Kong AI Gateway** 基于 Kong Gateway 构建——Kong 在相同 12 CPU 上的基准：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/模型/月（Plus 层最多 5 个）。适合已在用 Kong、>500 RPS 的企业。**Bifrost**（Maxim AI）——自动重试配可配置退避，OpenAI 429 时回退到 Anthropic。**Cloudflare / Vercel AI Gateways** ——托管，零运维，基本重试。数据驻留决定自托管决策；Portkey 和 Kong 居中，OSS + 可选托管。

**类型：** 精读
**语言：** Python（标准库，玩具级网关路由模拟器）
**前置要求：** Phase 17 · 01（托管 LLM 平台）、Phase 17 · 16（模型路由）
**时长：** 约 60 分钟

## 学习目标

- 枚举六个核心网关功能（路由、回退、重试、限流、密钥、可观测性、护栏）。
- 映射四个 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）到各自的规模上限和场景。
- 引用 Kong 基准（比 Portkey 快 228%，比 LiteLLM 快 859%），并解释这对 >500 RPS 的意义。
- 根据数据驻留和运维预算选择自托管 vs 托管。

## 背景问题

你的产品调用 OpenAI、Anthropic 和一个自托管 Llama。每个供应商有不同的 SDK、错误模型、限流和认证方式。你想要故障转移（OpenAI 429 时试 Anthropic）、统一凭证存储、统一可观测性和每租户限流。

在应用层重复造这个轮子会让每个服务耦合到每个供应商。网关层把它合并为一个进程、一个 API（通常 OpenAI 兼容），然后扇出到供应商。

## 核心概念

### 六个核心功能

1. **供应商路由** — OpenAI、Anthropic、Gemini、自托管等都在一个 API 后面。
2. **回退** — 429、5xx 或质量失败时，换地方重试。
3. **重试** — 指数退避，有界重试次数。
4. **限流** — 按租户、按 key、按模型。
5. **密钥引用** — 运行时从 vault 拉凭证（永不进应用）。
6. **可观测性** — OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。
7. **护栏** — PII 删除、越狱检测、允许话题过滤器。

### LiteLLM — MIT OSS，Python

- 100+ 供应商，OpenAI 兼容，router 配置，基本可观测性。
- Kong 基准在 2000 RPS 时崩溃；8 GB 内存占用，持续负载下级联故障。
- 最佳场景：Python 应用、<500 RPS、开发/预发网关、实验性路由。
- 成本：OSS 免费；云版有免费层。

### Portkey — 控制面定位

- 截至 2026 年 3 月 Apache 2.0 开源。护栏、PII 删除、越狱检测、审计轨迹。
- 每请求延迟开销 20-40 ms。
- 生产层 $49/月，含保留和 SLA。
- 最佳场景：受监管行业需要护栏 + 可观测性打包。

### Kong AI Gateway — 规模化方案

- 基于 Kong Gateway（成熟 API 网关产品，lua+OpenResty）。
- Kong 自测在等效 12-CPU 上：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 层最多 5 个。
- 最佳场景：已在用 Kong；>1000 RPS；愿意付费授权。

### Bifrost（Maxim AI）

- 自动重试配可配置退避。
- OpenAI 429 时回退到 Anthropic 是经典配方。
- 新进入者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管，零运维。基本重试和可观测性。
- 最佳场景：Cloudflare/Vercel 上边缘服务的 JavaScript 应用。
- 在护栏和限流上比 Kong/Portkey 有限。

### 自托管 vs 托管

数据驻留是决定因素。医疗和金融默认自托管（LiteLLM 或 Portkey OSS 或 Kong）。消费品默认托管（Cloudflare AI Gateway）或中层（Portkey 托管）。混合：受监管租户自托管，其他托管。

### 延迟预算

- LiteLLM：通常 5-15 ms 开销。
- Portkey：20-40 ms 开销。
- Kong：3-8 ms 开销。
- Cloudflare/Vercel：边缘优势 1-3 ms 开销。

网关延迟直接加到 TTFT。TTFT P99 < 100 ms SLA 用 Kong 或 Cloudflare。< 500 ms 都可以。

### 限流语义很重要

简单 token-bucket 在中等规模下 OK。多租户需要滑动窗口 + 突发配额 + 每租户分层。LiteLLM 用 token-bucket；Kong 用滑动窗口；Portkey 用分层。

### 网关 + 可观测性 + 路由 组合

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）是生产中的同一层。选一个覆盖全部的工具，或仔细连接：大多数 2026 年部署组合 Helicone（可观测性）或 Portkey（护栏）加 Kong（规模）做角色拆分。

### 必须记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8 GB 内存。
- Portkey：20-40 ms 开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 层最多 5 个。
- Cloudflare/Vercel：边缘 1-3 ms 开销。

## 用现成库

`code/main.py` 模拟跨 3 个供应商在 429/5xx 注入下的网关路由回退。报告延迟、重试率和回退命中率。

## 产出

本课产出 `outputs/skill-gateway-picker.md`。给定规模、运维姿态、合规要求和延迟预算，选出网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管的回退。在 5% 供应商错误率下预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，基线 300 ms。哪个网关在预算内？
3. 医疗客户要求自托管 + PII 删除 + 审计。选 Portkey OSS 还是 Kong？
4. 比较 LiteLLM vs Kong：在什么 RPS 上限时团队应该迁移？
5. 为多租户 SaaS 设计限流策略：免费层、试用层、付费层。用 token-bucket 还是滑动窗口？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 网关 | "API 代理" | 位于应用和供应商之间的进程 |
| LiteLLM | "MIT 的那个" | Python OSS，100+ 供应商，2000 RPS 时崩溃 |
| Portkey | "护栏网关" | 控制面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "规模化那个" | 基于 Kong Gateway，基准领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 回退配方 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署托管网关，零运维 |
| PII 删除 | "数据清洗" | 发送模型前用正则 + NER 遮盖 |
| 越狱检测 | "提示词注入护栏" | 用户输入上的分类器 |
| 审计轨迹 | "合规日志" | 每条 LLM 调用的不可变记录 |
| Token-bucket | "简单限流" | 基于补给的限流器 |
| 滑动窗口 | "精确限流" | 时间窗口限流器；公平性更好 |

## 扩展阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)