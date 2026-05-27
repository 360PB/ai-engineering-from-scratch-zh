# 生产量化 — AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化格式不是通用选择——它取决于硬件、推理引擎和负载。GGUF Q4_K_M 或 Q5_K_M 是 CPU 和边缘部署的首选，通过 llama.cpp 和 Ollama 提供。GPTQ 在 vLLM 内部提供多 LoRA 支持时胜出。AWQ 配合 Marlin-AWQ kernel 在 7B 级模型上达到约 741 tok/s，INT4 格式中 Pass@1 最优——2026 年数据中心生产默认。FP8 在 Hopper、Ada、Blackwell 上是可靠的中间地带——近无损，广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）激进，需要逐块验证。两个坑要避开：校准数据集必须匹配部署域；KV 缓存独立于权重量化——AWQ 的"模型现在是 4 GB"忽略了生产批次规模下 10-30 GB 的 KV 缓存。

**类型：** 精读
**语言：** Python（标准库，玩具级跨格式内存与吞吐对比）
**前置要求：** Phase 10 · 13（量化基础）、Phase 17 · 04（vLLM 推理内部原理）
**时长：** 约 75 分钟

## 学习目标

- 说出 2026 年的六种生产量化格式及其适用场景。
- 根据硬件（CPU vs GPU、Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和负载（日常聊天、推理、多 LoRA）选择格式。
- 计算选定格式下节省的权重内存和不受影响的 KV 缓存。
- 说出导致量化模型在领域流量上质量下降的校准数据集陷阱。

## 背景问题

量化减少内存和 HBM 带宽，而这正是 decode 所需要的。FP16 70B 模型有 140 GB 权重。量化为 INT4（AWQ 或 GPTQ）后是 35 GB——装进一块 H100 还有空间放 KV 缓存，而这很关键——128 并发、2k 上下文时，KV 缓存本身就要 20-30 GB。

但量化不是免费的。激进量化会降低质量，在推理密集型任务上尤其明显。不同格式配合不同引擎。不同硬件原生支持不同精度。2026 年的格式动物园是真实的，你不能照搬别人的选择——必须根据你的技术栈来决策。

## 核心概念

### 六种格式

| 格式 | 位宽 | 最佳场景 | 引擎 |
|------|------|----------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM (Marlin-AWQ)、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF — CPU/边缘首选

GGUF 是文件格式，不完全是量化方案——它把 K-quants 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器里。Q4_K_M 和 Q5_K_M 是生产默认——4-5 位接近 BF16 质量。在 CPU 或边缘推理时，llama.cpp 是最快的 CPU 推理引擎。

在 vLLM 中使用吞吐惩罚：7B 模型约 93 tok/s——该格式没有为 GPU kernel 做优化。CPU/边缘场景用 GGUF，其他场景不用。

### GPTQ — vLLM 多 LoRA

GPTQ 是带校准过程的训练后量化算法。Marlin kernel 在 GPU 上实现了 2.6 倍加速（对比非 Marlin GPTQ）。7B 模型约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA adapter。如果你服务一个基础模型加 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。NVFP4 在 2026 年初尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认

Activation-aware Weight Quantization，感知激活的权重量化。在量化时保护约 1% 最敏感的权重。Marlin-AWQ kernel：10.9 倍加速比朴素实现。7B 模型约 741 tok/s，INT4 格式中 Pass@1 最优。

新 GPU 服务除非需要多 LoRA（选 GPTQ）或激进的 Blackwell FP4（选 NVFP4），选 AWQ。

### FP8 — 可靠的中间地带

8 位浮点。近无损。广泛支持。Hopper Tensor Core 原生加速 FP8。Blackwell 继承之。FP8 是 2026 年质量不能打折时（推理、医疗、代码生成）的安全默认。内存节省是 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4 — Blackwell 激进方案

微缩放 FP4。每块权重有独立的缩放因子。在 Blackwell Tensor Core 上硬件加速。比 FP8 少一半 bytes/token——这就是 Phase 17 · 07 中的经济收益。

注意事项：
- 尚不支持 LoRA（2026 年初）。
- 在推理密集型负载上质量下降可见。
- 每个模型上线前在评估集上验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常用 C4 或 WikiText。对于领域模型（代码、医疗、法律），在通用网页文本上校准会让算法对哪些权重需要保护做出错误决策。HumanEval 上 Pass@1 可能下降好几分。

修复：用领域内数据校准。几百条领域样本通常就够。在上线前在评估集上测试。

### KV 缓存陷阱

AWQ 把权重压缩到 4 位。KV 缓存是独立的，保持 FP16/FP8。对于使用 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB INT4 压缩）。
- KV 缓存 128 并发 × 2k 上下文：约 20 GB。
- 激活：约 5 GB。
- 总计：约 60 GB——装进 H100 80GB。

简单地说"我的模型量化到 4 GB"忽略了另外 30-50 GB。HBM 预算要整体考虑。

另外，KV 缓存量化（FP8 KV 或 INT8 KV）是另一个独立选择，有自己的取舍——直接影响注意力精度，不是白捡的收益。

### AWQ INT4 对推理任务有风险

链式思维、数学、长上下文代码生成——这些在激进量化下质量下降明显。AWQ INT4 在 MATH 上损失约 3-5 分。对于推理密集型负载，上 FP8 或 BF16；接受内存成本。

### 2026 年选型指南

- CPU/边缘服务：GGUF Q4_K_M。定了。
- GPU 服务，日常聊天，无 LoRA：AWQ。
- GPU 服务，多 LoRA：带 Marlin 的 GPTQ。
- 推理负载：FP8。
- Blackwell 数据中心，经质量验证：NVFP4 + FP8 KV。
- 拿不准：每种候选格式跑 1000 样本评估。

## 用现成库

`code/main.py` 计算六种格式下跨模型规模的内存占用（权重 + KV + 激活）和相对吞吐。展示 KV 缓存何时占主导、权重压缩何时划算、FP8 何时是安全选择。

## 产出

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型规模、负载类型和质量容忍度，选出格式并给出校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 70B 模型在 128 并发、2k 上下文下，计算每种格式的总 HBM。哪种格式能装进一块 H100 80GB？
2. 你有一个 7B 编程模型。选一个格式并说明理由。如果对质量容忍度判断错了，恢复路径是什么？
3. 计算用领域内数据校准医疗模型所需的数据集大小。为什么更多数据不一定更好？
4. 阅读 Marlin-AWQ kernel 论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s，而原始 GPTQ 约 712。
5. 什么时候用 AWQ 权重 + FP8 KV 缓存比 KV 保持 BF16 更合理？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GGUF | "llama.cpp 格式" | 打包 K-quants 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4 位 K-quants 中等；GGUF 生产默认 |
| GPTQ | "gee pee tee q" | 训练后 INT4 带校准；vLLM 中支持 LoRA |
| AWQ | "a w q" | 感知激活的 INT4；Marlin kernel；INT4 中 Pass@1 最优 |
| Marlin kernel | "快速 INT4 kernel" | Hopper 上 INT4 的自定义 CUDA kernel；10 倍加速 |
| FP8 | "八位浮点" | Hopper/Ada/Blackwell 上的安全精度默认 |
| MXFP4 / NVFP4 | "微缩放四位" | Blackwell 4 位 FP，带逐块缩放因子 |
| 校准数据集 | "cal data" | 用于选取量化参数的输入文本；必须匹配领域 |
| KV 缓存量化 | "KV INT8" | 独立于权重的选择；直接影响注意力精度 |

## 扩展阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 各格式吞吐数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选型指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和参数。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ 原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ 原始论文。