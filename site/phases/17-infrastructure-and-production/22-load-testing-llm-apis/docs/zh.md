# LLM API 负载测试 — 为什么 k6 和 Locust 说谎

> 传统负载测试工具不是为流式响应、可变输出长度、token 级指标或 GPU 饱和设计的。两个坑咬住大多数团队。GIL 陷阱：Locust 的 token 级测量在 Python GIL 下运行 token 化，这与重并发下请求生成竞争；token 化积压然后膨胀报告的 token 间延迟——你的客户端是瓶颈，不是服务器。提示词均匀性陷阱：循环中相同提示词只测试 token 分布的一个点；真实流量有可变长度和多样前缀匹配。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复。2026 年工具图谱：LLM 专用的（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）适合 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——流式感知、Kubernetes 原生、通过 TestRun/PrivateLoadZone CRD 分布式，CI/CD 关卡最佳；Vegeta 做 Go 常数率饱和；Locust 2.43.3 仅配合 LLM-Locust 扩展支持流式。负载模式：稳态、斜坡、尖峰（自动扩缩测试）、浸泡（内存泄漏）。

**类型：** 动手实现
**语言：** Python（标准库，玩具级真实感提示词生成器 + 延迟收集器）
**前置要求：** Phase 17 · 08（推理指标）、Phase 17 · 03（GPU 自动扩缩容）
**时长：** 约 75 分钟

## 学习目标

- 解释两个反模式（GIL 陷阱、提示词均匀性陷阱）如何让通用负载测试工具对 LLM API 说谎。
- 给定目的（基准运行、CI 关卡、大规模合成、NVIDIA 参考）选择工具。
- 设计四个负载模式（稳态、斜坡、尖峰、浸泡）及每个捕获的失败模式。
- 用均值 + 标准差而非固定长度构建真实感提示词分布。

## 背景问题

你用 k6 测试 LLM 端点，500 并发用户。它扛住了。你上线了。生产中 200 真实用户服务就挂了——P99 TTFT 爆炸，GPU  pinned。

发生了两件事。第一，k6 发了 500 个相同提示词——你的请求合并和前缀缓存让它看起来像处理 500 个并发 decode 而实际只处理一个。第二，k6 不像眼睛感受的那样追踪流式响应的 token 间延迟；它看一个 HTTP 连接，不看 500 个 token 以不同间隔到达。

LLM 的负载测试是自己的学科。

## 核心概念

### GIL 陷阱（Locust）

Locust 用 Python，在 GIL 下运行客户端 token 化。高并发下 tokenizer 排队在请求生成后面。报告的 token 间延迟包含客户端 token 化积压。你以为服务器慢；实际是测试工具。

修复：LLM-Locust 扩展将 token 化移到独立进程，或用编译语言工具（k6、LLMPerf 用 tokenizers.rs）。

### 提示词均匀性陷阱

所有已知负载测试工具让你配置一个提示词。10,000 次迭代的循环中每次发送完全相同的提示词。服务器每次看到相同前缀——前缀缓存命中接近 100%，吞吐看起来很好。

修复：从提示词分布采样。LLMPerf 用 `--mean-input-tokens 500 --stddev-input-tokens 150`——多样长度，多样内容。

### 四个负载模式

1. **稳态** — 常数 RPS 持续 30-60 分钟。捕获：基线性能回归。
2. **斜坡** — 15 分钟内从 0 线性升到目标 RPS。捕获：容量断点、预热异常。
3. **尖峰** — 突然 3-10 倍 RPS 持续 2 分钟然后回退。捕获：自动扩缩延迟、队列饱和、冷启动影响。
4. **浸泡** — 稳态 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具图谱

**LLMPerf**（Anyscale）— Python 但 Rust 后端 token 化。均值/标准差提示词。流式感知。性能运行默认工具。

**NVIDIA GenAI-Perf** — NVIDIA 参考。用 Triton client；全面指标覆盖。注意其 ITL 不含 TTFT；LLMPerf 的包含。同服务器两工具产生不同 TPOT。

**LLM-Locust**（TrueFoundry）— 修复 GIL 陷阱的 Locust 扩展。熟悉 Locust DSL + 流式指标。

**guidellm** — 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译，无 GIL）增加了流式感知指标。
- k6 Operator 用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 关卡和 SLA 测试。

**Vegeta** — Go，比 k6 简单。常数率 HTTP 饱和。不是 LLM 感知但适合网关/限流测试。

**Locust 2.43.3 原装** — 对 LLM 有 GIL 陷阱。仅配合 LLM-Locust 扩展。

### CI 中的 SLA 关卡

用 k6 在 PR 上跑：
- 每基线 RPS 跑 30-50 次迭代。
- 关卡：P50/P95 TTFT，5xx < 5%，TPOT 在阈值下。
- 突破则构建失败。

### 真实感提示词分布

从真实流量样本（如果有）或发布分布（如 ShareGPT 聊天的聊天提示词、HumanEval 的代码）构建。向 LLMPerf 喂均值 + 标准差。绝对避免循环单一提示词。

### 必须记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型 LLMPerf 运行：并发 X 下 100-1000 请求。
- 典型 CI 关卡：每个 PR 30-50 次迭代。
- 四个模式：稳态、斜坡、尖峰、浸泡。

## 用现成库

`code/main.py` 模拟带真实感提示词分布的负载测试，测量有效 TPOT，并演示均匀提示词陷阱。

## 产出

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选出工具并设计四个负载模式。

## 练习

1. 运行 `code/main.py`。对比均匀 vs 真实分布——差距在哪里？
2. 写 k6 CI 关卡脚本：100 并发 TTFT P95 < 800 ms，运行时 5 分钟。
3. 你的浸泡测试显示内存每小时增长 50 MB。说出三个原因及确认每个的插桩。
4. 从 10 RPS 尖峰到 100 RPS。如果 Karpenter + vLLM 生产栈在位（Phase 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 在同一服务器上报告 TPOT=11ms。解释。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| LLMPerf | "LLM 工具" | Anyscale 基准工具，流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考工具 |
| LLM-Locust | "LLM 版 Locust" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准" | 大规模合成工具 |
| k6 Operator | "K8s k6" | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | "Python 客户端开销" | Token 化积压膨胀报告延迟 |
| 提示词均匀性陷阱 | "单一提示词谎言" | 循环相同提示词命中缓存，膨胀吞吐 |
| 稳态 | "恒定负载" | N 分钟平 RPS |
| 斜坡 | "线性爬升" | 持续时间内 0 到目标 |
| 尖峰 | "突发测试" | 突然倍数后回退 |
| 浸泡 | "长测试" | 数小时检漏 |

## 扩展阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)