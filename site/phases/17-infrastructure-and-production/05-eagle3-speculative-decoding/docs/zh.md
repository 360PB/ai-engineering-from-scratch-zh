# EAGLE-3 推测解码生产实践

> 推测解码将快速草稿模型与目标模型配对。草稿提出 K 个 token；目标在单次前向中验证；接受的 token 免费。2026 年，EAGLE-3 是生产级变体——它基于目标模型的隐藏状态而非原始 token 训练草稿头，使接受率 alpha 进入 0.6-0.8 区间（通用聊天）。正确的问题不是"草稿有多快"，而是"在我的流量上 alpha 是多少"。如果 alpha 降到约 0.55 以下，在高并发下推测解码是净负的，因为每个被拒绝的草稿都花费一次额外的目标前向传递。本课教你先测量 alpha，再开 flag。

**类型：** 精读
**语言：** Python（标准库，玩具级接受率模拟器）
**前置要求：** Phase 17 · 04（vLLM 推理内部原理）、Phase 10 · 18（多 Token 预测）
**时长：** 约 60 分钟

## 学习目标

- 说出三代推测解码，并解释 EAGLE-3 相比 EAGLE-2 和经典草稿模型改变了什么。
- 定义接受率 alpha，从 alpha 和 K（草稿长度）计算预期加速，并识别目标并发的盈亏平衡 alpha。
- 解释为什么推测解码在 vLLM 2026 是 opt-in（不是默认），以及为什么不开 alpha 测量就开启是生产反模式。
- 写出测量计划：哪个基准、哪种提示词分布、哪个并发点、哪个指标作为门控。

## 背景问题

Decode 是内存密集型。在一块运行 Llama 3.3 70B FP8 的 H100 上，每个解码 token 读取约 140 GB/s 的权重并发出一个 token。GPU 计算在 decode 期间几乎空闲——瓶颈是 HBM 带宽，不是矩阵乘法吞吐。

推测解码利用了这个差距。用便宜的草稿模型生成 K 个候选 token，然后让目标模型在单次前向中验证全部 K 个。每个被验证的 token 实际上是免费的（分摊到目标本来就需要做的一次 K 个 forward 中）。

经典草稿模型方法使用同一家族的小模型（Llama 3.2 1B 为 Llama 3.3 70B 起草）。有效但接受率一般——小模型分布与目标偏离。EAGLE，然后 EAGLE-2，然后 EAGLE-3 在目标模型内部状态上直接训练轻量草稿头，所以草稿分布更紧密地跟踪目标。这就是 alpha 从草稿模型的 0.4 到 EAGLE-3 的 0.6-0.8 的原因。

陷阱：EAGLE-3 在 vLLM 2026 是 opt-in。必须显式设置 `speculative_config`。没有 flag，没有加速。在真实流量上不测量 alpha 就开启的团队往往看到尾延迟变差，而非变好。

## 核心概念

### 推测解码实际买到了什么

没有 spec decode，每 token 成本是一次目标前向。在草稿长度 K 和接受率 alpha 下，预期每目标前向的 token 数是 `1 + K * alpha`。加速是 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证开销。对于 K=5，alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。真实世界数字集中在 2-3x，因为生产流量上 alpha 很少那么高，且 epsilon 在大 batch 时增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会消失——它们为第一个被拒绝的 token 强制执行第二次目标前向。在 alpha 降到 0.4 的工作负载上，你付草稿开销加验证加重跑。在高并发（比如说 256 并发）下，decode 批已经大到"仅目标"和"目标加验证"之间的内存带宽差距缩小。2026 年大多数硬件上 alpha 低于 0.55 时，spec decode 是净负的。

Alpha 因工作负载而异。在 ShareGPT 风格通用聊天上，在 ShareGPT 上训练的 EAGLE-3 达到 0.6-0.8。在领域特定流量（代码、医疗、法律）上，在通用数据上训练的草稿头降到 0.4-0.6。训练领域特定的 EAGLE-3 草稿头可以恢复 alpha——与目标微调相比，这是一个轻量、快速训练任务。

### EAGLE 代际一览

- **经典草稿模型**：同家族小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，草稿每目标前向运行 K 次前向。
- **EAGLE-1（2024）**：在目标隐藏状态（最后一层）上训练的单草稿头。Alpha 约 0.5-0.6。目标之上的小参数开销。
- **EAGLE-2（2025）**：自适应草稿长度和基于树的草稿（一次目标传递验证多个分支）。Alpha 约 0.6-0.7。草稿调度器更复杂。
- **EAGLE-3（2025-2026）**：在多个目标层（不仅是最后一层）上训练的草稿头，更好对齐。在通用聊天上 alpha 约 0.6-0.8。

### 2026 生产配方

1. 裸发目标模型。测量基准 TTFT、ITL、目标并发的吞吐。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重跑基准。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的草稿长度得到 alpha。
4. 如果生产流量分布上 alpha < 0.55，禁用 spec decode 或训练领域特定的 EAGLE-3 草稿。
5. 在生产并发下重跑。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

平均 ITL 随 spec decode 下降。但如果不调优，P99 可能变差。被拒绝的草稿触发两遍序列（草稿 + 验证失败 + 重跑）。在满批下，这两遍序列化。看 P99 ITL，不是 P50。

### 已部署 EAGLE-3 的地方

Google 于 2025 年在 AI Overviews 中部署了推测解码（相同质量，更快响应）。vLLM V1 以 `speculative_config` 作为记录接口发货；V1 中 N-gram GPU 推测解码是与分块 prefill 兼容的变体。SGLang 将 EAGLE-3 定位前缀繁重工作负载的推荐草稿路径。

### 一行盈亏平衡数学

预期加速：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。设 `S = 1` 解 alpha：`alpha_breakeven = verify_overhead / K`。对于典型 verify_overhead ~0.15，K=5：`alpha_breakeven = 0.03`。但这是原始 decode 数学。在高并发下验证开销上升，且 decode 批已经将内存读取分摊到多个序列，所以实践中有效 alpha_breakeven 攀升至约 0.45-0.55。

### 何时不用推测解码

- 批处理 1 离线生成，延迟不重要。用裸目标。
- 很短的输出（低于 50 tokens）。草稿开销和验证成本占主导。
- 没有领域训练草稿头的专业领域。Alpha 太低。
- vLLM v0.18.0 加草稿模型 spec decode 加 `--enable-chunked-prefill`。此组合无法编译。记录在案的例外是 V1 中的 N-gram GPU spec decode。

## 运用它

`code/main.py` 模拟有/无推测解码的 decode 循环，跨一系列 alpha 值和草稿长度 K。打印盈亏平衡 alpha、测得加速和尾部行为。在多种（alpha, K）组合上运行它，精确看到推测解码何时停止值得。

## 交付它

本课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，它产生分阶段 EAGLE-3 上线计划——基准测量、启用配置、测量 alpha、alpha >= 0.55 门控、观察 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，你需要多少 alpha 才能获得 2x 加速？3x 呢？它对 verify_overhead 有多敏感？
2. 假设生产流量是 70% 通用聊天、30% 代码。通用聊天在 ShareGPT 上训练的 EAGLE-3 上 alpha 达 0.7；代码上 alpha 达 0.4。混合 alpha 是多少，spec decode 是净正的吗？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram），以及哪个与分块 prefill 兼容。
4. 启用 EAGLE-3 后平均 ITL 下降 25%，但 P99 ITL 上升 15%。诊断并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头内存成本。与用 Llama 3.2 1B 作为经典草稿相比如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 推测解码 | "草稿加验证" | 用便宜模型提出 K 个 token，在一次目标前向中验证全部 K 个 |
| 接受率 alpha | "spec 接受率" | 目标接受的草稿 token 比例；唯一重要的指标 |
| 草稿长度 K | "spec k" | 草稿每目标前向提出多少 token；典型 4-8 |
| 验证开销 epsilon | "spec 开销" | 验证并重跑相比裸目标前向的额外成本；随 batch 增长 |
| EAGLE-3 | "最新 EAGLE" | 2025-2026 变体；在多个目标层上训练草稿头；通用聊天上 alpha 0.6-0.8 |
| `speculative_config` | "vLLM spec 配置" | vLLM V1 中的显式 opt-in；无默认意味着无加速 |
| N-gram spec decode | "N-gram 草稿" | 使用提示词中 N-gram 查找的 GPU 侧草稿；与分块 prefill 兼容 |
| 盈亏平衡 alpha | "无操作 alpha" | spec decode 零加速的 alpha；在生产并发下观察它 |
| 拒绝草稿两遍 | "重跑成本" | 草稿拒绝时的两次目标前向；驱动 P99 尾部 |

## 延伸阅读

- [vLLM — 推测解码文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — `speculative_config` 和 V1 中分块 prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 确切字段集。
- [EAGLE 论文（arXiv:2401.15077）](https://arxiv.org/abs/2401.15077) — 原创 EAGLE 草稿头形式化。
- [EAGLE-2 论文（arXiv:2406.16858）](https://arxiv.org/abs/2406.16858) — 自适应草稿和树。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 带推测解码的高效 LLM 系统。
- [BentoML — 推测解码](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产上线清单。