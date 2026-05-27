# LLM 路由层 —— LiteLLM、OpenRouter、Portkey

> 提供商锁定是昂贵的。不同的工具调用工作负载适合不同的模型。路由网关提供统一的 API 表面、重试、故障转移、成本追踪和护栏。2026 年主导的三种类型：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课命名决策标准并走查标准库路由网关。

**类型：** 学习
**语言：** Python（标准库、路由 + 故障转移 + 成本追踪器）
**前置要求：** Phase 13 · 02（函数调用）、Phase 13 · 17（网关）
**时间：** 约 45 分钟

## 学习目标

- 区分自托管、托管和生产级路由选项。
- 实现一个回退链，按定义优先级顺序在提供商故障时重试。
- 跨提供商追踪每请求成本和 token 使用。
- 为给定生产约束在 LiteLLM、OpenRouter 和 Portkey 之间做出决定。

## 问题

提供商路由重要的场景：

1. **成本。** Claude Sonnet 花费是 Haiku 的 3 倍。对于分类任务，Haiku 足够了；对于综合任务，Sonnet 值得。逐请求路由。

2. **故障转移。** OpenAI 有一个糟糕的小时。每个请求失败。你想自动回退到 Anthropic，无需重新部署。

3. **延迟。** 实时聊天 UI 需要快速的首 token 时间。批处理摘要器不需要。按延迟 SLA 路由。

4. **合规。** EU 用户必须留在 EU 区域。按区域路由。

5. **实验。** 在同一工作负载上 A/B 测试两个模型。按测试桶路由。

每个集成都手写这些是重复的。路由网关提供单一 OpenAI 兼容 API 并处理其余部分。

## 概念

### OpenAI 兼容代理形状

每个人都用 OpenAI 形状。路由网关暴露 `/v1/chat/completions`，接受 OpenAI schema，并在内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何。客户端不关心。

### 模型别名

你的代码说 `our_smart_model` 而非 `claude-3-5-sonnet-20251022`。网关将别名映射到真实模型。当 Anthropic 发布 Claude 4，你端侧更改别名；你的代码不动。

### 回退链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义这个。重试计入预算，因此回退级联不会爆炸成本。

### 语义缓存

相同或接近相同的提示命中缓存而非提供商。在重复 Agent 循环上的节省可达 30% 到 60%。Key 基于嵌入；接近的提示共享缓存槽。

### 护栏

网关级：

- **PII 编辑。** 发送提示前的正则或 ML 传递。
- **策略违规。** 拒绝含禁止内容的提示。
- **输出过滤器。** 清除补全中的泄漏。

Portkey 和 Kong 都发布了有观点的护栏。LiteLLM 将它们留为可选。

### 每密钥速率限制

一个 API 密钥 = 一个团队。每密钥预算防止一个团队消耗共享配额。大多数网关支持。

### 自托管 vs 托管权衡

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产） |
|------|------------------|-------------------|----------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 设置 | 部署代理 | 注册 | 两者都行 |
| 提供商 | 100+ | 300+ | 100+ |
| 计费 | 你自己的密钥 | OpenRouter 积分 | 你自己的密钥 |
| 可观测性 | OpenTelemetry | 仪表板 | 完整 OTel + PII 编辑 |
| 最适合 | 想要完全控制的数据主权团队 | 想要单一订阅且无基础设施 | 需要开箱即用的护栏和合规 |

当你有 SRE 团队且想要数据主权时 LiteLLM 胜出。当你想单一订阅且无基础设施时 OpenRouter 胜出。当你需要护栏和合规时 Portkey 胜出。

### 成本追踪

每个请求携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以每模型每 token 价格（网关维护的价格表拉取）。按用户 / 团队 / 项目聚合。

### MCP 加路由

网关可以同时路由 LLM 调用和 MCP 采样请求。当采样请求的 modelPreferences 偏好特定模型时，网关翻译到正确后端。这是 Phase 13 · 17（MCP 网关）和本课路由网关有时合并为一项服务的地方。

### 路由策略

- **静态优先级。** 列表中第一个；错误时回退。
- **负载均衡。** 轮询或加权。
- **成本感知。** 选择满足延迟/质量的最便宜模型。
- **延迟感知。** 选择过去 N 分钟内最快的模型。
- **任务感知。** 提示分类器将编码路由到一个模型，将摘要路由到另一个。

## 使用它

`code/main.py` 用约 150 行实现路由网关：接受 OpenAI 形状请求，翻译到每提供商存根，运行优先级回退链，追踪每请求成本，并在输入上应用 PII 编辑传递。运行三个场景：正常请求、触发回退的主提供商中断、被编辑捕获的 PII 泄漏。

要注意的点：

- `ROUTES` dict：别名 -> 按优先顺序排列的具体提供商列表。
- 回退循环在 5xx 上重试。
- 成本追踪器将 token 使用乘以每模型费率。
- PII 编辑器在转发前清除 SSN 形状模式。

## 发布它

本课生成 `outputs/skill-routing-config-designer.md`。给定工作负载配置文件（延迟、成本、合规），该 Skill 选择 LiteLLM / OpenRouter / Portkey 并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发中断场景；确认回退落在第二个提供商，成本正确归属。

2. 添加语义缓存：提示的 SHA256 是查找键；缓存命中即时返回。在重复调用上测量成本节省。

3. 添加提示分类器，将"code ..."提示路由到偏好智能的别名，将"summarize ..."提示路由到偏好速度的别名。

4. 设计每团队预算：每个团队有月度消费上限；一旦达到上限网关拒绝请求。选择执行粒度（每请求还是窗口）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 文档。说出每个发货而其他两个没有的一个功能。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Routing gateway（路由网关） | "LLM 代理" | 位于多个提供商前面的单一 API 表面层 |
| OpenAI-compatible（OpenAI 兼容） | "说 OpenAI schema" | 接受 `/v1/chat/completions` 形状，翻译到任何后端 |
| Model alias（模型别名） | "our_smart_model" | 代码中的名称映射到具体模型 |
| Fallback chain（回退链） | "重试列表" | 失败时尝试的有序提供商列表 |
| Semantic caching（语义缓存） | "提示嵌入缓存" | Key 是提示的嵌入；近重复共享缓存命中 |
| Guardrails（护栏） | "输入/输出过滤器" | 编辑 PII，拒绝策略违规 |
| Per-key rate limit（每密钥速率限制） | "团队预算" | 配额范围到 API 密钥 |
| Cost tracking（成本追踪） | "每请求支出" | 聚合 token 使用 x 每模型价格 |
| LiteLLM | "开放代理" | 可自托管的开源路由网关 |
| OpenRouter | "托管 SaaS" | 带积分计费的托管网关 |
| Portkey | "生产选项" | 内置护栏的开源 + 托管 |

## 延伸阅读

- [LiteLLM — 文档](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — 快速入门](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — 文档](https://portkey.ai/docs) — 带护栏的生产路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南
- [Relayplane — 2026 LLM 网关比较](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调查