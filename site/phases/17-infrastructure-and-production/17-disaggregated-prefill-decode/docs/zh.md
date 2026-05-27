# 分解式 Prefill/Decode — NVIDIA Dynamo 和 llm-d

> Prefill 是计算密集型；decode 是内存密集型。两者跑在同一 GPU 上浪费一种资源。分解把它们分到独立池，通过 NIXL（RDMA/InfiniBand 或 TCP 回退）传输 KV 缓存。NVIDIA Dynamo（GTC 2025 宣布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 上层——Planner Profiler + SLA Planner 自动匹配 prefill:decode 比率以满足 SLO。NVIDIA 发布吞吐提升在这个量级——developer.nvidia.com（2025-06）显示 GB200 NVL72 + Dynamo 在中等延迟区间 DeepSeek-R1 MoE 约 6 倍提升；Dynamo 产品页（developer.nvidia.com，未标注日期）广告 GB300 NVL72 + Dynamo 相对 Hopper 高达 50 倍 MoE 吞吐。"30 倍"是整个 Blackwell + Dynamo + DeepSeek-R1 报告的社区汇总；我们没找到单一原始来源说确切的 30 倍，所以将其视为方向性主张。llm-d（Red Hat + AWS）是 Kubernetes 原生：prefill / decode / 路由器作为独立 Service，带 per-role HPA。llm-d 0.5 增加分层 KV offloading、缓存感知 LoRA 路由、UCCL 网络、缩容到零。经济性：多个客户披露的内部分汇总暗示从同地点服务迁移到 Dynamo 分解式，$2M 级推理支出可节省 30-40%（即 $60-80 万/年），SLA 不变；具体 $2M→$600-800K 数字是内部综合，不是单一已发布案例——用作数量级锚点，不是引用参考。短提示词（<512 token，短输出）不能抵消传输成本。

**类型：** 精读
**语言：** Python（标准库，玩具级分解式 vs 同地点模拟器）
**前置要求：** Phase 17 · 04（vLLM 推理内部原理）、Phase 17 · 08（推理指标）
**时长：** 约 75 分钟

## 学习目标

- 解释为什么 prefill 和 decode 有不同的最优 GPU 分配，并量化同地点下的浪费。
- 画出分解式架构：prefill 池、decode 池、通过 NIXL 的 KV 传输、路由器。
- 说出分解式不划算的条件（短提示词、短输出）。
- 区分 NVIDIA Dynamo（栈上层）和 llm-d（Kubernetes 原生），并匹配各自运维场景。

## 背景问题

你在 8 张 H100 上跑 Llama 3.3 70B。混合负载下（长提示词 + 短输出），GPU 在 decode 期间空闲因为大部分计算花在 prefill 上。在不同负载下（短提示词 + 长输出），情况相反。同地点 prefill + decode 意味着你为两者都过量配置了。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你买 H100 算力来跑内存受限的 decode，或者买 H100 HBM 带宽来跑计算受限的 prefill。两者都是昂贵的浪费。

分解把 prefill 和 decode 分到独立池，各按各自的瓶颈规格配置。KV 缓存通过高速互联从 prefill 池传输到 decode 池。

## 核心概念

### 为什么瓶颈不同

**Prefill** — 在一次前向中运行完整输入提示词的 transformer。矩阵乘法主导；计算密集型。H100 FP8 约 2000 TFLOPS 有效吞吐。批效率高——一次前向处理多个 token。

**Decode** — 逐 token 生成，每次迭代读取完整权重。内存带宽密集型。HBM3 约 3 TB/s。批效率仅在高并发时才高——权重读取在批上摊销。

同地点：买针对两者优化的 GPU。H100 两者都行但成本相同。在规模上，你想让 prefill 池在 H100/计算密集型；decode 池在 H200/内存密集型，或用激进量化。

### 架构

```
            ┌──────────────┐
  请求 →   │    路由器    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (仅提示词)                      │
            ┌──────────────┐    KV 缓存    ┌───────▼──────┐
            │ Prefill 池  │ ─── NIXL ────► │ Decode 池   │
            │  (计算型)    │                │  (内存型)    │
            └──────────────┘                └──────┬───────┘
                                                   │ token
                                                   ▼
                                                 客户端
```

NIXL 是 NVIDIA 的节点间传输协议。有 RDMA/InfiniBand（可用时）和 TCP 回退。传输延迟真实——70B FP8、4K-token 提示词 KV 缓存典型 20-80 ms。这就是为什么短提示词不能抵消分解式：传输税超过节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 宣布，1.0 GA）：
- 作为 vLLM、SGLang、TRT-LLM 上的协调器。
- Planner Profiler 测量负载，SLA Planner 自动配置 prefill:decode 比率。
- Rust 核心，Python 可扩展。
- 吞吐提升：NVIDIA 报告 GB200 NVL72 + Dynamo 在中等延迟区间 DeepSeek-R1 MoE 约 6 倍（developer.nvidia.com，2025-06）；社区"高达 30 倍"的报告在完整 Blackwell + Dynamo + DeepSeek-R1 栈上缺乏单一原始来源，应作为方向性参考。
- GB300 NVL72 + Dynamo：相对 Hopper 广告高达 50 倍 MoE 吞吐（developer.nvidia.com，未标注日期）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- Prefill / decode / 路由器作为独立 Kubernetes Service。
- 带队列深度（prefill）/KV 利用率（decode）信号的 per-role HPA。
- `topologyConstraint packDomain: rack` 将 prefill+decode Clique 打包在同一 rack 以实现高速 KV 传输。
- llm-d 0.5（2026）：分层 KV offloading、缓存感知 LoRA 路由、UCCL 网络、缩容到零。

如果想要托管式栈上层协调器，用 Dynamo。想要 Kubernetes 原语并深耕 CNCF 生态，用 llm-d。

### 经济性

内部综合（不是单一已发布案例——数量级锚点）：

- 同地点服务年均推理支出 $200 万。
- 迁移到 Dynamo 分解式。
- 请求量不变，P99 延迟 SLA 不变。
- 报告节省：$60-80 万/年（30-40% 削减）。
- 无新硬件。

我们从多个客户披露综合此数字而非单一可引用案例；最接近的已发布数据点是 Baseten 的 Dynamo KV 路由下 TTFT 快 2 倍 / 吞吐高 61%（baseten.co，2025-10），以及 VAST + CoreWeave 在 40-60% KV 命中率下投影每 token 成本节省 60-130%（vastdata.com，2025-12）。节省来自各池正确配置；RAG 8K+ 前缀等 prefill 密集型负载比均衡负载受益更多。

### 何时不分拆

- 提示词 <512 token 且输出 <200 token：传输税主导收益。
- 小集群（<4 GPU）：池多样性不足。
- 团队无法运维两个 GPU 池带 per-role 扩缩容：Dynamo 有帮助但不简单。
- 无 RDMA 架构：TCP 传输税更重。

### 路由器与 Phase 17 · 11 集成

分解式路由器是有 KV 感知的（Phase 17 · 11）。请求落在持有其前缀的 decode 池——如无匹配，它流向 prefill→decode。命中率和分解式是叠加的——缓存感知路由器决定是否需要新的 prefill。

### MoE on Blackwell 是真正数字所在

GB300 NVL72 + Dynamo 相对 Hopper 基线显示 50 倍 MoE 吞吐。MoE 专家路由在 prefill 是计算密集（在 decode 是内存密集——专家缓存），所以分解式是双倍收益。2026 年 frontier 模型服务以 MoE 为主（DeepSeek-V3、未来 GPT-5 变体）。

### 必须记住的数字

基准数字漂移——NVIDIA 和推理栈每季度发更新结果。引用前请重新核查。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo：相对基线中等延迟区间约 6 倍（developer.nvidia.com，2025-06）；完整 Blackwell + Dynamo 栈上社区"高达 30 倍"缺乏单一原始来源，作为方向性汇总。
- GB300 NVL72 + Dynamo：相对 Hopper 高达 50 倍 MoE 吞吐（developer.nvidia.com，未标注日期）。
- 节省锚点（内部综合，不是单一案例）：SLA 不变，年支出 $200 万减少 $60-80 万。
- 分解阈值：提示词 >512 token + 输出 >200 token。
- NIXL KV 传输：70B FP8 4K 提示词约 20-80 ms。

## 用现成库

`code/main.py` 模拟同地点 vs 分解式服务。报告吞吐、每请求成本和提示词长度交叉点。

## 产出

本课产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否分拆。

## 练习

1. 运行 `code/main.py`。在什么提示词长度下分解式超过同地点？
2. 为 P99 前缀长度 8K、输出 300 的 RAG 服务设计 prefill 池和 decode 池。
3. Dynamo vs llm-d：为纯 Kubernetes 团队选一个，不需要 Python 运行时偏好。
4. 计算 KV 传输成本：70B FP8 4K prefill = 约 500 MB KV。RDMA 100 GB/s = 5 ms。TCP 10 GB/s = 50 ms。哪个影响你的 SLA？
5. MoE 专家路由改变 KV 访问模式。分解式在激活不同专家的 MoE 上行为如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 分解式服务 | "prefill/decode 分离" | 各阶段独立 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "协调器" | vLLM/SGLang/TRT-LLM 上层的栈上层 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 分解式栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量负载，配置池比率 |
| SLA Planner | "Dynamo 策略" | 自动匹配 prefill:decode 以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将 prefill+decode 打包在同一 rack 以实现快速 KV |
| UCCL | "统一集合通信" | llm-d 0.5 网络层支持缩容到零 |
| MoE 专家路由 | "每 token 专家" | DeepSeek-V3 模式；分解式有帮助 |

## 扩展阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)