# Capstone 11 — LLM 可观测性与评估仪表盘

> Langfuse 转为开源。Arize Phoenix 发布了2026年 GenAI semconv 映射。Helicone 和 Braintrust 都加倍投入了每用户成本归属。Traceloop 的 OpenLLMetry 成为事实标准 SDK 插桩。生产形态是 ClickHouse 存 trace，Postgres 存元数据，Next.js 做 UI，一小批评估作业（DeepEval、RAGAS、LLM-judge）在采样 trace 上跑。从零构建一个自托管版本，至少接入四个 SDK 家族，演示在5分钟内捕获注入的回归。

**类型：** 毕业项目
**语言：** TypeScript（UI），Python / TypeScript（接入 + 评估），SQL（ClickHouse）
**前置知识：** Phase 11（LLM工程）、Phase 13（工具）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P11 · P13 · P17 · P18
**时长：** 25小时

## 问题

2026年每个在生产流量上跑 AI 模型的团队都会在模型旁边维护一个可观测平面。成本归属。幻觉检测。漂移监控。越狱信号。SLO 仪表盘。PII 泄露告警。开源参考——Langfuse、Phoenix、OpenLLMetry——汇聚到 OpenTelemetry GenAI 语义约定作为接入模式。你现在可以用一个 SDK 插桩 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，产出兼容的 span。

你将构建一个自托管仪表盘，至少从四个 SDK 家族接入 trace，在采样 trace 上跑一组评估作业，检测漂移并告警。测量标准：给定一个故意注入的回归（一个开始泄露 PII 的提示），仪表盘在5分钟内捕获它并触发告警。

## 核心概念

接入是 OTLP HTTP。SDK 产出 GenAI-semconv span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 落入 ClickHouse 做列式分析；元数据（用户、会话、应用）落入 Postgres。

评估作为批处理作业在采样 trace 上运行。DeepEval 评分 faithfulness、toxicity 和 answer relevance。RAGAS 在 trace 带有检索上下文时评分检索指标。自定义 LLM-judge 运行领域特定检查（PII 泄露、策略外回复）。评估运行将评估 span 写回同一 ClickHouse，链接到父 trace。

漂移检测监控随时间推移的嵌入空间分布（prompt 嵌入上的 PSI 或 KL 散度）和评估分数趋势。告警喂给 Prometheus Alertmanager 然后到 Slack / PagerDuty。UI 是 Next.js 15 + Recharts。

## 架构

```
生产应用：
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  带 GenAI semconv 的 OpenTelemetry SDK
       |
       v  OTLP HTTP
  collector（接入、采样、分发）
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 归档
   （spans）     （元数据）  （原始事件）
       |
       +---> 评估作业（DeepEval, RAGAS, LLM-judge）
       |     采样或全量 trace
       |     写评估 span 回写
       |
       +---> 漂移检测器（prompt 嵌入上的 PSI / KL）
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 仪表盘（Recharts）
```

## 技术栈

- 接入：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- Collector：OpenTelemetry Collector，带尾采样处理器（控制成本）
- 存储：ClickHouse 存 spans，Postgres 存元数据，S3 归档原始事件
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix 评估包、自定义 LLM-judge
- 漂移：句子 Transformer 汇集 prompt 嵌入上的 PSI / KL，每周
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 动手实现

1. **Collector 配置。** OpenTelemetry Collector 带 OTLP HTTP 接收器，尾采样器保留 100% 错误 trace 和 10% 成功 trace，导出器到 ClickHouse 和 S3。

2. **ClickHouse schema。** `spans` 表列镜像 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，加 JSON bag 存长 payload。按 user_id 和 app_id 加二级索引。

3. **SDK 覆盖测试。** 用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）写一个小客户端应用，用 OpenLLMetry 自动插桩。验证每个产出规范的 GenAI span 落入 ClickHouse。

4. **评估作业。** 定时作业读取最近15分钟采样 trace，跑 DeepEval faithfulness、toxicity 和 answer relevance。输出是链接到父 trace 的评估 span。

5. **自定义 LLM-judge。** PII 泄露裁判：给定回复，调用 guard LLM 对 PII 泄露可能性评分。高分回复进入分类队列。

6. **漂移检测。** 每周作业计算本周汇集 prompt 嵌入与过去4周基线之间的 PSI。如果 PSI 高于阈值，告警。

7. **仪表盘。** Next.js 15 页面：概览（spans/sec、cost/user、p95 latency）、trace（搜索 + 瀑布）、评估（faithfulness 趋势、toxicity）、漂移（PSI 随时间）、告警。

8. **告警链。** Prometheus 导出器读取评估分数聚合和延迟百分位；Alertmanager 路由到 Slack 告警和 PagerDuty 严重违规。

9. **回归探测。** 注入 bug：被评估的聊天机器人在1%的时间里开始泄露假 SSN。测量 MTTR：从 bug 部署到 Slack 告警。

## 用现成库

```bash
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  接收了1条 trace，3个 spans
[clickhouse] 插入了3个 spans（app=chat, user=u_42）
[评估]       DeepEval faithfulness 0.82，toxicity 0.03
[漂移]       每周 PSI 0.08（低于 0.2 阈值）
[UI]         实时在 https://obs.example.com
```

## 产出

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，仪表盘接入其 trace，跑评估，漂移告警，在 Next.js 中呈现 cost/user 细分。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | Trace schema 覆盖率 | 产出规范 GenAI span 的 SDK 家族数（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数 vs 人工标注集 |
| 20 | 仪表盘 UX | 注入回归的 MTTR（目标5分钟以内） |
| 20 | 成本/规模 | 持续接入 1k spans/sec 无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链端到端运行 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证规范 span 带着忠诚的 `gen_ai.*` 属性落入 ClickHouse。

2. 在相同 trace 上将 DeepEval 换成 Phoenix 评估器。测量两个评估引擎之间的分数漂移。

3. 锐化漂移检测器：按 app-id 而非全局计算 PSI。展示每个 app 的漂移轨迹。

4. 添加"用户影响"页面：每用户成本和每用户失败率，带迷你图。

5. 构建尾采样策略，保留 toxicity > 0.5 的 100% trace 加其余的 10% 分层采样。测量引入的采样偏差。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM 属性" | 2025年 OpenTelemetry LLM span 属性规范（system、model、tokens） |
| Tail sampling | "trace 后采样" | Collector 在 trace 完成后决定保留还是丢弃（可以偷看错误） |
| PSI | "人口稳定性指数" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM-judge | "评估即模型" | 用一个 LLM 按评分标准评判另一个 LLM 的输出（faithfulness、toxicity、PII） |
| Tail-sampling policy | "保留规则" | 决定持久化 vs 丢弃哪些 trace 的规则；错误 + 采样率 |
| Eval span | "链接评估 trace" | 携带链接到原始 LLM 调用 span 的评估分数的子 span |
| Cost per user | "单位经济学" | 归因到 user_id 在一个窗口内的美元成本；关键产品指标 |

## 扩展阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 权威开源可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 带强大漂移支持的备选参考
- [OpenLLMetry（Traceloop）](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 家族
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 接入模式
- [Helicone](https://www.helicone.ai) — 备选托管可观测性
- [Braintrust](https://www.braintrust.dev) — 备选评估优先平台
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估器库