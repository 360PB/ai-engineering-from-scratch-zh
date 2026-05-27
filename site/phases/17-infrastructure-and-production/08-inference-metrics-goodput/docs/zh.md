# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否合格。TTFT 是 prefill 加队列加网络。TPOT（即 ITL）是每个 token 的内存受限 decode 开销。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐是跨集群聚合的每秒 token 数。但产品真正在乎的指标是 goodput——同时满足所有 SLO 的请求比例。高吞吐低 goodput 意味着你处理的 token 从未按时到达用户手中。参考数据：Llama-3.1-8B-Instruct 在 TRT-LLM 上，2026 年平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。永远报告 P50、P90、P99——不要只报均值。还要注意测量陷阱：GenAI-Perf 在计算 ITL 时不包含 TTFT，LLMPerf 包含；同一运行中两个工具对 TPOT 的数值不一致。

**类型：** 精读
**语言：** Python（标准库，玩具级百分位计算器和 goodput 报告器）
**前置要求：** Phase 17 · 04（vLLM 推理内部原理）
**时长：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐、goodput，并说明每个指标衡量的是哪部分。
- 解释为什么均值是 LLM 服务的错误统计量，以及如何读取 P50/P90/P99。
- 构建 SLO 多约束条件（如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），并针对它计算 goodput。
- 说出两个在同一次运行中对 TPOT 存在分歧的基准工具，并解释原因。

## 背景问题

"我们的吞吐是 15000 token/秒。"那又怎样？如果 40% 的请求端到端超过 2 秒，用户已经放弃会话了。单独看吞吐无法判断产品是否正常。

推理有多个延迟维度，每个维度的失败模式各不相同。Prefill 是计算密集型，随提示词长度增长。Decode 是内存密集型，随批大小变化。队列延迟是运营问题。网络是物理距离问题。你需要对每个维度设置独立指标，还需要百分位数，还需要一个综合指标来回答"用户是否得到了期望的结果"——这就是 goodput。

## 核心概念

### TTFT — 首个 token 的时间

`TTFT = queue_time + network_request + prefill_time`

Prefill 在提示词较长时占主导。在 Llama-3.3-70B FP8 H100 上，32k 提示词的纯 prefill 约 800 ms。队列时间是负载下的调度器行为。Network request 包含 TLS 的 wire time。TTFT 是用户看到流式返回之前的延迟。

### TPOT / ITL — 输出 token 间的延迟

多个名字指代同一个量。`TPOT`（每个输出 token 的时间）、`ITL`（token 间延迟）、逐 token decode 延迟——都是同一个东西。是首个 token 之后连续流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一 Llama-3.3-70B H100 配置加分块 prefill 下，平均 TPOT 约 7 ms。没有分块 prefill 时，在长 prefill 处理相邻序列期间，TPOT 可能飙升至 50 ms。看 P99，别看均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 token），E2E 由 TPOT 主导。对于短输出但长提示词，E2E 由 TTFT 主导。报告时按输出长度分段 E2E。

### 吞吐

`throughput = total_output_tokens / elapsed_time`

聚合指标。反映集群效率。不反映单个请求的健康状态。

### Goodput — 你真正在意的指标

`goodput = 满足 (TTFT <= a) 且 (TPOT <= b) 且 (E2E <= c) 的请求比例`

SLO 是多约束条件。一个请求只有在所有约束同时满足时才是"合格"的。Goodput 就是这个比例。高吞吐但 goodput 只有 60% 是失败。低吞吐但 99% goodput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交和 AI 平台内部 SLA 追踪使用的指标。

### 为什么均值是错误的统计量

LLM 延迟分布是右偏的。一个 decode 批中，一个长 prefill 邻居可以吐出 500 token 且 TPOT 约 7 ms，而另一个只有 20 token 但 TPOT 约 60 ms。平均 TPOT 是 9 ms。P99 TPOT 是 65 ms。用户经常遇到 P99——所以他们离开了。

永远报告三元组（P50、P90、P99）。对于用户体验，P99 是你优化的目标。

### 参考数据 — Llama-3.1-8B-Instruct 在 TRT-LLM 上，2026 年

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：取决于分块 prefill 配置，在 10-25 ms 之间。

这些是 NVIDIA 发布的参考数据。会随模型规模（70B 会显示 3-5 倍差异）、硬件（H100 vs B200 约 3 倍）和负载变化。

### 测量陷阱

2026 年两个最常用的基准工具在同一次运行的 TPOT 上存在分歧：

- **NVIDIA GenAI-Perf**：在 ITL 计算中不包含 TTFT。ITL 从 token 2 开始。
- **LLMPerf**：包含 TTFT。ITL 从 token 1 开始。

对于 TTFT 500 ms、100 个输出 token、总 decode 时间 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择改变数值。

永远说明用的是什么工具。永远发布定义。

### 构建 SLO

2026 年 70B 聊天模型面向消费者的合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- E2E P99 <= 3 秒（对于 <300 token 的输出）。
- Goodput 目标 >= 99%。

企业 SLO 收紧 TTFT（200-400 ms），放宽 E2E。重点是写下来，测量全部三个，用 goodput 作为单一综合指标追踪。

### 如何测量

- 跑真实流量或真实感强的合成流量（LLMPerf 用 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准测试时瞄准 2 倍峰并发。
- 跑 30-50 次迭代，取合并样本的百分位数。
- 发布时附上工具名、工具版本、模型、硬件、并发度、提示词分布。

## 用现成库

`code/main.py` 是一个玩具 goodput 计算器。生成合成延迟分布，应用 SLO，计算 goodput。同时展示 GenAI-Perf 与 LLMPerf 在同一 trace 上 TPOT 的差异。

## 产出

本课产出 `outputs/skill-slo-goodput-gate.md`。给定负载和 SLO，生成一个 CI/CD 就绪的基准测试配方，在 goodput（而非吞吐）上卡部署。

## 练习

1. 运行 `code/main.py`。生成有 1% 尾部尖峰的分布。将 P99 TPOT 从 30 ms 收紧到 15 ms，goodput 如何变化？
2. 某供应商报价"Llama 3.3 70B H100 上 15000 tok/s"。在信任之前，你需要问哪三个问题？
3. 为什么分块 prefill 保护 P99 TPOT 但不保护平均 TPOT？
4. 为语音助手设计消费者 SLO（首个 token 是听到的，不是读到的）。哪个指标用户感知最直接？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出三个两工具还存在分歧的其他指标。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| TTFT | "首个 token 的时间" | 队列 + 网络 + prefill；在长提示词下由 prefill 主导 |
| TPOT | "每输出 token 的时间" | 首个 token 之后每个 token 的内存受限 decode 开销 |
| ITL | "token 间延迟" | 与 TPOT 在大多数工具中相同（但不全是这样——见 GenAI-Perf） |
| E2E | "端到端" | TTFT + TPOT * output_len；再加响应侧网络 |
| 吞吐 | "tok/s" | 集群效率；没有延迟百分位数则毫无意义 |
| Goodput | "SLO 达标率" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "尾部" | 百分之一最差情况延迟；用户体验指标 |
| SLO 多约束 | "联合约束" | 三个延迟上界的 AND；任一违规则请求不合格 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具在 ITL 是否包含 TTFT 上存在分歧 |

## 扩展阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的规范定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量方法。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署中的测量实践。
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray 生态开源基准工具。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 基准工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 业界公认的基于 goodput 的基准测试。