# 自托管推理引擎选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年四个引擎主导自托管推理。按硬件、规模、生态选择。**llama.cpp** 是 CPU 上最快——最广泛模型支持，量化控制和线程控制全在手。**Ollama** 是开发笔记本 one-command 安装，比 llama.cpp 慢 15-30%（Go + CGo + HTTP 序列化），生产负载下吞吐差距 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式**——仅修复 bug，raw 吞吐比 vLLM 慢约 10%，但历史上观测性和 HF 生态集成一流。维护状态使它成为长期赌注的风险——SGLang 或 vLLM 是新项目更安全默认。**vLLM** 是通用生产默认——v0.15.1（2026 年 2 月）添加 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是 Agent 多轮/前缀密集型专家——生产中 40 万+ GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：CPU only → llama.cpp only。AMD / 非 NVIDIA → vLLM only（TRT-LLM 是 NVIDIA 锁定）。2026 年流水线模式：dev = Ollama，staging = llama.cpp，prod = vLLM 或 SGLang。全程用相同 GGUF/HF 权重。

**类型：** 精读
**语言：** Python（标准库，引擎决策树行走器）
**前置要求：** Phase 17 · 04、06、07、09、18（覆盖引擎的全部课程）
**时长：** 约 45 分钟

## 学习目标

- 给定硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和负载（日常聊天 / Agent / 长上下文）选引擎。
- 说出 2026 年 TGI 维护模式状态（2025 年 12 月 11 日）以及为什么它让新项目偏向 vLLM 或 SGLang。
- 描述用相同 GGUF 或 HF 权重贯穿 dev/staging/prod 的流水线。
- 解释为什么"仅 CPU"强制 llama.cpp 以及"AMD"排除 TRT-LLM。

## 背景问题

你的团队启动一个新的自托管 LLM 项目。一个工程师说 Ollama，另一个说 vLLM，第三个说"TGI 不就是开箱即用吗？"三个在各自上下文都对。没有一个适合所有。

2026 年选择树很重要：硬件优先，规模第二，负载第三。还有一个 2025 年特殊事件——TGI 2025 年 12 月 11 日进入维护模式——改变了新项目的默认。

## 核心概念

### 五个引擎

| 引擎 | 最佳场景 | 说明 |
|------|----------|------|
| **llama.cpp** | CPU / 边缘 / 最小依赖 / 最广泛模型支持 | CPU 上最快，全面控制 |
| **Ollama** | 开发笔记本、单用户、one-command 安装 | 比 llama.cpp 慢 15-30%；生产负载吞吐差距 3 倍 |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 广泛生产默认；v0.15.1 2026 年 2 月 |
| **SGLang** | Agent 多轮、前缀密集负载 | 生产中 40 万+ GPU |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也可以但更慢。其他引擎在 CPU 上无竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也可。TRT-LLM 是 NVIDIA 锁定，排除。

**NVIDIA Hopper（H100 / H200）** → vLLM 或 SGLang 或 TRT-LLM。三个都是顶级。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐领先（Phase 17 · 07）。vLLM 和 SGLang 紧随。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 包装这个。

### 规模次优决策

**1 用户 / 本地 dev** → Ollama。一条命令，秒出首个 token。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10k 用户 / 生产** → vLLM 生产栈（Phase 17 · 18）或 SGLang。

**10k+ 用户 / 企业** → vLLM 生产栈 + 分解式（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 负载第三决策

**日常聊天 / 问答** → vLLM 广泛默认胜出。

**Agent 多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）主导。

**重前缀复用的 RAG** → SGLang。

**代码生成** → vLLM 可以；SGLang 在缓存上略好。

**长上下文（128K+）** → vLLM + 分块 prefill；SGLang + 分层 KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——仅修复 bug。历史上：一流可观测性、最好-in-class HF 生态集成（model cards、安全工具）、raw 吞吐略落后 vLLM。

2026 年新项目：默认远离 TGI。现有 TGI 部署可以继续但应该迁移。SGLang 和 vLLM 是更安全默认。

### 流水线模式

Dev（Ollama）→ staging（llama.cpp）→ prod（vLLM）。全程相同 GGUF 或 HF 权重。工程师在笔记本上快速迭代；staging 镜像生产量化；prod 是服务目标。

### Ollama 注意

Ollama 适合 dev。不适合共享生产：Go HTTP 序列化增加开销，并发管理比 vLLM 简单，OpenTelemetry 支持落后。在它擅长的地方用——单用户，一条命令——然后为共享场景切换到 vLLM。

### 自托管 vs 托管是独立决策

Phase 17 · 01（托管超大规模）、· 02（推理平台）覆盖托管。本课假设你已经决定自托管。自托管原因：数据驻留、自定义微调、大规模总拥有成本、领域模型在托管上不可用。

### 必须记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产足迹：40 万+ GPU。
- Ollama 吞吐差距比 llama.cpp：慢 15-30%；生产负载下 3 倍。

## 用现成库

`code/main.py` 是决策树行走器：给定硬件 + 规模 + 负载，选引擎并解释原因。

## 产出

本课产出 `outputs/skill-engine-picker.md`。给定约束，选引擎并写迁移计划。

## 练习

1. 用你的硬件/规模/负载运行 `code/main.py`。输出与你的直觉一致吗？
2. 你的基础设施是 12 张 H100 和 8 张 MI300X AMD。选什么引擎？为什么 TRT-LLM 排除？
3. 一个团队 2026 年想用 TGI 因为"我们熟悉"。论证迁移案例。
4. Ollama dev 到 vLLM prod：量化、配置和可观测性上哪些变化？
5. RAG 产品 P99 前缀长度 8K，跨租户高复用。选引擎并用 Phase 17 · 11 + 18 叠加。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| llama.cpp | "CPU 的那个" | 最广泛模型支持，CPU 上最快 |
| Ollama | "笔记本的那个" | one-command 安装，开发级吞吐 |
| TGI | "HF 的服务" | 2025 年 12 月起维护模式 |
| vLLM | "默认那个" | 2026 年广泛生产基线 |
| SGLang | "Agent 那个" | 前缀密集，RadixAttention |
| TRT-LLM | "NVIDIA 锁定" | Blackwell 吞吐领先，NVIDIA only |
| GGUF | "llama.cpp 格式" | 打包 K-quants 变体 |
| 生产栈 | "vLLM K8s" | Phase 17 · 18 参考部署 |
| 流水线模式 | "dev→stage→prod" | Ollama → llama.cpp → vLLM 全程相同权重 |

## 扩展阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)