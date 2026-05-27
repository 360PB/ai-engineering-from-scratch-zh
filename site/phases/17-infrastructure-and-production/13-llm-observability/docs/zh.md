# LLM 可观测性栈选型

> 2026 年可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示词管理、会话回放打包。网关/插桩工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注遥测。Langfuse 是 MIT 许可核心，OSS 平衡良好（云版每月 50K 事件免费）。Phoenix 是 OpenTelemetry 原生，采用 Elastic License 2.0——出色的漂移/RAG 可视化，不是持久化生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，号称比单体式可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 生态领先，$39/用户/月，企业版才可自托管。Helicone 是代理模式，15-30 分钟配置，10 万请求/月免费，但深度不如 Agent traces。常见生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**类型：** 精读
**语言：** Python（标准库，玩具级 trace 采样模拟器）
**前置要求：** Phase 17 · 08（推理指标）、Phase 14（Agent 工程）
**时长：** 约 60 分钟

## 学习目标

- 区分开发平台（打包评估 + 提示词 + 会话）和网关/遥测工具（traces + 指标）。
- 映射六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）到各自的许可、定价和最佳场景。
- 解释 OpenTelemetry 粘合模式——用网关工具配合独立评估平台的方案。
- 说出 2026 年成本差异点（Arize AX 的零拷贝方案 vs 单体式摄入），并说明约 100 倍乘数。

## 背景问题

你上线了一个 LLM 功能。能用了。但你对提示词失败、工具循环、延迟回归、成本尖峰或提示词缓存命中率毫无可见性。你 Google"LLM 可观测性"，出来八个工具，都声称用三种不同价格解决同一问题。

它们的解决的问题并不相同。LangSmith 回答"这个 LangGraph 运行为什么失败？"Phoenix 回答"我的 RAG 流水线是否漂移？"Helicone 回答"哪个应用在烧 token？"Langfuse 回答"我能全量自托管吗？"不同工具，不同受众。

选型涉及四个轴：技术栈（LangChain？原生 SDK？多供应商？）、许可容忍度（仅 MIT？Elastic 可以？商业许可？）、预算（免费层？$100/月？$1000/月？）和自托管（必须？能接受？不考虑？）。

## 核心概念

### 两大类别

**开发平台**将可观测性与评估、提示词管理、数据集版本控制、会话回放打包。你做实验，看哪个提示词有效，数据集回归测试新提示词对旧赢家。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**只负责插桩推理调用——提示词、响应、token 数、延迟、模型、成本。Helicone、Singoz、OpenLLMetry、Phoenix。极简主义。可通过 OpenTelemetry 与独立评估工具组合。

### Langfuse — OSS 平衡

- 核心 Apache / MIT 许可；Docker 自托管。
- 云版免费层：每月 50K 事件。付费：$29/月团队版。
- 评估、提示词管理、traces、数据集。四个开发平台功能覆盖合理。
- 最佳场景：想要 LangSmith 级别功能但必须自托管或使用 OSS 许可。

### Phoenix（Arize）— 遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管简单。
- RAG 和漂移可视化出色。embedding 空间散点图作为一等公民。
- 不是设计为持久化生产后端——主要是开发阶段可观测性。
- 最佳场景：RAG 流水线开发、漂移调试，与独立网关组合用于生产。

### Arize AX — 规模化方案

- 商业产品。通过 Iceberg/Parquet 零拷贝数据湖集成。
- 号称比单体式可观测性（Datadog 级别）在大规模下便宜约 100 倍。数学：你把 traces 存在你自己的 Parquet on S3；Arize 直接读。
- 最佳场景：每天 >1000 万 traces、已有数据湖、想要 LLM 专用仪表板但不想付 Datadog 价。

### LangSmith — LangChain/LangGraph 优先

- 商业，$39/用户/月。仅企业版可自托管。
- LangChain 和 LangGraph 栈上同类最佳。如果不在两者上，紧迫感较弱。
- 最佳场景：团队已押注 LangChain，愿意付费。

### Helicone — 代理式最小可行

- 15-30 分钟配置，只需把 `OPENAI_API_BASE` 换成 Helicone 代理。
- MIT 许可；每月 10 万请求免费，付费 $20/月+。
- 含故障转移、缓存、限流——本身也充当网关。
- Agent/多步 traces 深度不足。
- 最佳场景：快速上手、单技术栈应用、需要一个网关加可观测性合一。

### Opik（Comet）— OSS 开发平台

- Apache 2.0，完全开源。
- 特性集与 Langfuse 相似，有 Comet 传承。
- 最佳场景：已在 Comet 上的 ML 团队，想在同一界面用 LLM 可观测性。

### SigNoz — OpenTelemetry 优先的全栈 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 加上 LLM 调用。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年底发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。正在浮现的生产模式：

1. 每次 LLM 调用发出带 GenAI 约定的 OTel。
2. 路由到网关（Helicone / Portkey）做日常使用。
3. 双发到评估平台（Phoenix / Langfuse）做回归检测。
4. 归档到数据湖（Iceberg）供 Arize AX 或 DuckDB 长期分析。

### 陷阱：在错误层插桩

在 agent 框架内部插桩（如加 LangSmith traces）会让你耦合到那个框架。在 HTTP/OpenAI-SDK 层插桩（通过 OpenLLMetry 或你的网关）是可移植的。

### 采样——你留不住所有数据

每天 >100 万请求时，完整 trace 保留成本超过 LLM 调用本身。按规则采样：错误 100%、高成本 100%、成功 5%。聚合永远保留；raw 数据保留长尾部分。

### 必须记住的数字

- Langfuse 云版免费：每月 50K 事件。
- LangSmith：$39/用户/月。
- Helicone 免费：每月 10 万请求。
- Arize AX 号称：大规摸下比单体式便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 用现成库

`code/main.py` 模拟每天 100 万 traces 跨保留策略（全量摄入、采样、采样+错误）下的表现。报告每种策略的存储成本和丢失内容。

## 产出

本课产出 `outputs/skill-observability-stack.md`。给定技术栈、规模、预算和许可姿态，选出工具或工具组合。

## 练习

1. 你的团队用 LangChain，想要 OSS 自托管可观测性。选 Langfuse 或 Opik 并说明理由。
2. 每天 500 万 traces，Datadog 报价 $15 万/月。计算 Arize AX 的平衡点。
3. 设计一个组织应强制在每次 LLM 调用上使用的 OpenTelemetry GenAI 属性集。
4. 论证 Phoenix 单独是否够用于生产。什么时候不够？
5. Helicone 有 20ms 代理开销。在 P99 TTFT 300 ms 下，这可接受吗？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OpenLLMetry | "LLM 的 OTel" | LLM 的开源 OpenTelemetry 插桩库 |
| GenAI 约定 | "OTel 属性" | LLM 调用的标准 OTel 属性名 |
| LangSmith | "LangChain 可观测性" | 商业平台，打包在 LangChain 生态 |
| Langfuse | "OSS LangSmith" | MIT OSS，功能集相似 |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生开发/评估平台 |
| Arize AX | "规模化可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | HTTP 代理收集 LLM 遥测 + 网关功能 |
| Opik | "Comet LLM" | Comet 出品的 Apache 2.0 OSS 开发平台 |
| 会话回放 | "trace 重放" | 完整 Agent 会话含工具调用重放 |
| 评估 | "离线测试" | 在标签数据集上跑候选模型/提示词 |

## 扩展阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)