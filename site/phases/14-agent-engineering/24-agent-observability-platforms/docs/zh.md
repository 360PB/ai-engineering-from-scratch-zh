# Agent 可观测性平台：Langfuse、Phoenix、Opik

> 2026 年三大开源 Agent 可观测性平台。Langfuse（MIT）— 月均 600 万+ SDK 安装，追踪 + Prompt 管理 + 评估 + 会话回放。Arize Phoenix（Elastic 2.0）— 深度 Agent 特化评估、RAG 相关性、OpenInference 自动插桩。Comet Opik（Apache 2.0）— 自动 Prompt 优化、护栏、LLM-Judge 幻觉检测。

**类型：** 概念学习
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 23（OTel GenAI）
**时间：** 约 45 分钟

## 学习目标

- 说出三大开源 Agent 可观测性平台及其许可证。
- 区分各自最强项：Langfuse（Prompt 管理 + 会话）、Phoenix（RAG + 自动插桩）、Opik（优化 + 护栏）。
- 解释为何 2026 年有 89% 的组织已建立 Agent 可观测性。
- 实现一个标准库追踪转仪表盘流水线，含 LLM-Judge 评估。

## 问题背景

OTel GenAI（第 23 课）给了你 schema。但你仍需要一个平台来摄入跨度、运行评估、存储 Prompt 版本、并呈现回归。三个竞争者各自强调了生命周期的不同部分。

## 核心概念

### Langfuse（MIT）

- 月均 600 万+ SDK 安装，GitHub 1.9 万+ 星。
- 功能：追踪、带版本控制的 Prompt 管理 +  Playground、评估（LLM-as-judge、用户反馈、自定义）、会话回放。
- 2025 年 6 月：原商业模块（LLM-as-a-judge、标注队列、Prompt 实验、Playground）以 MIT 协议开源。
- 最适合：紧耦合 Prompt 管理环路的端到端可观测性。

### Arize Phoenix（Elastic License 2.0）

- 更深度的 Agent 特化评估：追踪聚类、异常检测、RAG 检索相关性。
- 原生 OpenInference 自动插桩。
- 与托管服务 Arize AX 配合使用。
- 无 Prompt 版本控制——定位为漂移/行为回归工具，与更广泛的平台协同。
- 最适合：RAG 相关性、行为漂移、异常检测。

### Comet Opik（Apache 2.0）

- 通过 A/B 实验自动优化 Prompt。
- 护栏（PII 脱敏、主题约束）。
- LLM-Judge 幻觉检测。
- Comet 自测：Opik 日志 + 评估耗时 23.44s，Langfuse 耗时 327.15s（约 14 倍差距）—— 厂商数据仅供参考。
- 最适合：优化循环、自动实验、护栏执行。

### 行业数据

根据 Maxim（2026 年现场分析）：89% 的组织已建立 Agent 可观测性；质量问题是最主要的生产障碍（32% 的受访者提及）。

### 选型参考

| 需求 | 选哪个 |
|------|--------|
| 带 Prompt 管理的全能方案 | Langfuse |
| 深度 RAG 评估 + 漂移检测 | Phoenix |
| 自动优化 + 护栏 | Opik |
| 开放许可证，无 ELv2 约束 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 任一——均导出 OTel |

### 这个模式的常见误区

- **没有评估策略。** 追踪而无评估只是昂贵的日志。
- **自建 LLM-Judge 无基础。** CRITIC 模式（第 5 课）适用——Judge 需要外部工具进行事实核查。
- **Prompt 版本未与追踪关联。** 生产回归时，无法定位到引发问题的 Prompt。

## 动手实现

`code/main.py` 实现标准库追踪收集器 + LLM-Judge 评估器：

- 摄入 GenAI 格式的跨度。
- 按会话分组，标记失败运行（护栏触发、低置信度评估）。
- 一个脚本化 LLM-Judge，根据评分标准对 Agent 回复打分。
- 类似仪表盘的摘要：失败率、主要失败原因、评估分分布。

运行：

```
python3 code/main.py
```

输出：每个会话的评估分和失败分类，与 Langfuse/Phoenix/Opik 展示的内容一致。

## 用现成库

- **Langfuse** 自托管或云端；通过 OTel 或 SDK 接入。
- **Arize Phoenix** 自托管；OpenInference 自动插桩。
- **Comet Opik** 自托管或云端；自动优化循环。
- **Datadog LLM Observability** 适合已用 Datadog 的运维 + ML 混合团队。

## 产出

`outputs/skill-obs-platform-wiring.md` 选一个平台，将追踪 + 评估 + Prompt 版本接入现有 Agent。

## 练习

1. 导出一周 OTel 追踪到 Langfuse 云（免费版）。哪些会话失败了？为什么？
2. 为你的领域编写 LLM-Judge 评分标准（事实正确性、语气、范围合规）。在 50 条追踪上测试。
3. 对比 Langfuse Prompt 版本控制与 Phoenix 的追踪聚类。哪个能更快定位问题？
4. 阅读 Opik 的护栏文档。将 PII 脱敏护栏接入一个 Agent 运行。
5. 在你的语料上对三者做基准测试。忽略厂商发布的数据；自己测。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Tracing | "跨度收集器" | 摄入 OTel / SDK 跨度；按会话索引 |
| Prompt management | "Prompt CMS" | 与追踪关联的版本化 Prompt |
| LLM-as-judge | "自动评估" | 用独立 LLM 按评分标准对 Agent 输出打分 |
| Session replay | "追踪回放" | 回放历史运行用于调试 |
| RAG relevancy | "检索质量" | 检索到的上下文是否匹配查询 |
| Trace clustering | "行为分组" | 对相似运行聚类，用于漂移检测 |
| Guardrail enforcement | "日志时策略检查" | 对记录内容做 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) — 追踪、评估、Prompt 管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) — 自动插桩、漂移检测
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者共同消耗的 schema