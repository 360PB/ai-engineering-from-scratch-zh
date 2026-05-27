# TensorRT-LLM on Blackwell：FP8 与 NVFP4

> TensorRT-LLM 是 NVIDIA 独家方案，但在 Blackwell 上表现最优。在 GB200 NVL72 + Dynamo 编排下，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型 $0.012/百万 token，对比 H100 + vLLM 的 $0.09/M——经济性差距达 7 倍。技术栈由三个浮点层级叠加：FP8 依然对 KV 缓存和注意力 kernel 不可或缺，因其动态范围恰好满足需求；NVFP4（4 位微缩放）处理权重和激活；多 token 预测（MTP）和分解式 prefill/decode 在此基础上再增 2-3 倍。Day-0 FP4 支持直接加载 FP4 权重，无需后训练转换。2026 年工程团队的代价：TRT-LLM 是封闭的 NVIDIA 技术栈，选用它意味着用可移植性换取吞吐。投入之前，先算清楚你的模型和硬件组合。

**类型：** 精读
**语言：** Python（标准库，玩具级 FP8/NVFP4 内存与成本计算器）
**前置要求：** Phase 17 · 04（vLLM 推理内部原理）、Phase 10 · 13（量化基础）
**时长：** 约 75 分钟

## 学习目标

- 解释为什么 FP8 即使在权重使用 NVFP4 的情况下依然对 KV 缓存和注意力不可或缺。
- 计算 frontier 模型在 BF16、FP8、NVFP4 下的 HBM 占用，并说清楚节省空间在哪里。
- 列举 TRT-LLM 利用的 Blackwell 特性（day-0 FP4、MTP、分解式服务、全互联原语）。
- 判断 TRT-LLM 的 NVIDIA 绑定代价（H100 + vLLM 相比 7 倍经济差距）是否值得。

## 背景问题

2026 年推理经济性的前沿课题是"每美元多少 token"。答案取决于四个叠加的选择：硬件代数（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、推理引擎（vLLM vs SGLang vs TRT-LLM）、编排方式（普通 vs 分解式 vs Dynamo）。

Hopper 上 vLLM 跑 120B MoE 约 $0.09/百万 token。Blackwell 上 TRT-LLM + Dynamo 跑同一模型约 $0.012——便宜 7 倍。其中部分差距来自硬件（Blackwell 单 GPU LLM 吞吐比 Hopper 高 11-15 倍），部分是技术栈带来的：FP4 权重、MTP draft、分解式 prefill/decode、NVLink 5 全互联加速 MoE 专家通信。

在 NVIDIA 技术栈外无法复现这一点。这就是核心取舍——用可移植性换经济性。理解每项选择贡献了差距的哪部分，是本课的意义。

## 核心概念

### 为什么 FP8 依然是 KV 缓存的底线

2026 年一个常见误解：以为 NVFP4 适用所有场景。并非如此。KV 缓存需要 FP8（8 位浮点），因为它存储的注意力 key 和 value 跨越宽动态范围。量化成 FP4 会造成灾难性的精度损失——分布尾部被截断，注意力分数塌陷。FP8 的指数位赋予 KV 缓存所需的动态范围。

NVFP4（2025-2026）应用于权重和激活。微缩放：每块权重有独立的缩放因子，小块之间可以有不同的动态范围，避免逐张量缩放带来的损失。对于激活，FP4 表现可以接受，因为激活在单层内动态范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV 缓存：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特有原语

- **Day-0 FP4 权重**：模型供应商直接发布 FP4 格式的权重；TRT-LLM 无需后训练转换即可加载。FP4 无需 AWQ/GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE 相同思路（Phase 17 · 05），但集成在 TRT-LLM 构建流程中。
- **分解式服务**：prefill 和 decode 分置于独立 GPU 池，KV 缓存通过 NVLink 或 InfiniBand 传输。与 Dynamo 思路相同（Phase 17 · 20）。
- **全互联通信原语**：NVLink 5 将 MoE 专家通信延迟削减至 Hopper 的三分之一。TRT-LLM 的 MoE kernel 针对此做了调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 硬件加速缩放因子处理。

### 必须记住的数字

- HGX B200 通过 TRT-LLM 跑 GPT-OSS-120B：$0.02/M token。
- GB200 NVL72 + Dynamo：$0.012/M token。
- H100 + vLLM 同等负载：约 $0.09/M token。
- TRT-LLM 三个月更新带来 2.8 倍吞吐提升（2026 年）。
- Blackwell 单 GPU LLM 吞吐比 Hopper 高 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在所有提交任务中占主导。

### FP4 的实际质量代价

NVFP4 激进度较高。在推理密集型负载（链式思维、数学、需长上下文的代码生成）上，FP4 权重质量下降明显。逐块校准能缓解但无法消除。发布推理模型的团队通常采用 FP8 权重 + FP4 激活作为折中方案，或继续使用 H200 + 全 FP8 配置。

原则：在将 NVFP4 权重模型上线前，一定在评估集上验证任务质量。

### 为什么这是 NVIDIA 锁定的决策

TRT-LLM 是 C++ + CUDA + 闭源 kernel。模型需要针对特定 GPU SKU 编译。不支持 AMD、Intel、ARM。如果你的基础设施策略是多供应商，TRT-LLM 对于 TRT-LLM 服务层是非 starter——你仍可以在异构硬件上用 vLLM 服务。如果你是 NVIDIA only，7 倍差距足以弥补绑定成本。

### 2026 年实践配方

对于年均推理账单超过 1 亿美元的团队，继续用 Hopper + vLLM 等于把 7-10 倍的优化空间留在桌上。将成本主导的负载迁移到 Blackwell + TRT-LLM + Dynamo。将实验层保留在 H100 + vLLM 以加速模型迭代。在每个 NVFP4 转换模型上线前验证质量。

### 分解的红利

TRT-LLM 的分解式服务（独立的 prefill 和 decode 池）在 Blackwell 上产生叠加效应：FP4 权重 × MTP 加速 × 分解式布局 × 缓存感知路由。7 倍数字基于这个完整技术栈。

## 用现成库

`code/main.py` 计算 HBM 占用、decode 吞吐（内存带宽受限区间）以及三个技术栈下的 $/M-token：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它看叠加效应和各层变化的贡献占比。

## 产出

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定负载、模型规模和年 token 量，判断 Blackwell + TRT-LLM 技术栈是否值得 NVIDIA 绑定。

## 练习

1. 运行 `code/main.py`。在 120B MoE（30% 激活参数）上，计算 H100 BF16、H100 FP8、B200 NVFP4/FP8 的内存带宽受限 decode 吞吐。最大跃升来自哪一步？
2. 某客户每年在 H100 + vLLM 上花 $2M。按 7 倍经济差距，12 个月内摊平迁移到 TRT-LLM 需要买多少台 Blackwell GPU？
3. NVFP4 权重转换后 MATH 精度下降 3 分。给出两条恢复路径：质量优先（保留 FP8 权重）、成本优先（用域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell 相对 Hopper 优势最小？为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV 缓存 + 128k 上下文下的 HBM 需求。能否装进单台 GB200 NVL72 节点？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| FP8 | "八位浮点" | 8 位浮点；用于 KV 缓存和注意力，因动态范围需求 |
| NVFP4 | "四位列微缩放" | NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上处理权重和激活 |
| MXFP8 | "MX 八位" | 微缩放 FP8 变体；Blackwell Tensor Core 硬件加速 |
| Day-0 FP4 | "直接发 FP4 权重" | 模型供应商直接发布 FP4 权重；无需后训练转换 |
| MTP | "多 token 预测" | TRT-LLM 集成的推测解码 draft（Phase 17 · 05） |
| 分解式服务 | "prefill/decode 分离" | Prefill 和 decode 分置于独立 GPU 池；KV 通过 NVLink/IB 传输 |
| 全互联 | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 削减 3 倍延迟 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年业界公认的每 token 成本基准 |

## 扩展阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 全互联与 MoE kernel。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 上层的分解式编排器。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数据的基准测试套件。