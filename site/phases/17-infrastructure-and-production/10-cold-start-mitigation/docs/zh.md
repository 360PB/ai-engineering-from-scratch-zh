# 无服务器 LLM 冷启动缓解

> 20 GB 模型镜像从冷到服务需要 7B 模型 5-10 分钟到 70B 的 20+ 分钟。在真正的无服务器世界里，这不是预热——这是故障。缓解措施分五层：预置节点镜像（Bottlerocket on AWS，双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU 内存快照（Modal，比重新初始化快 10 倍）、暖池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 流水线，延迟削减 10-200 倍）、以及实时迁移——传输输入 token（KB）而非 KV 缓存（GB）。Modal 发布冷启动 2-4 秒为下限；Baseten 默认 5-10 秒，预热后亚秒。本课教你测量、预算和叠加五层缓解措施。

**类型：** 精读
**语言：** Python（标准库，玩具级冷启动路径模拟器）
**前置要求：** Phase 17 · 02（推理平台经济）、Phase 17 · 03（GPU 自动扩缩容）
**时长：** 约 60 分钟

## 学习目标

- 枚举冷启动缓解的五层结构，说出每层的一个工具或模式。
- 计算 70B 模型的冷启动总时间 =（节点提供）+（权重下载）+（权重加载到 HBM）+（引擎初始化）。
- 解释为什么实时迁移传输输入 token（KB）而非 KV 缓存（GB），以及代价是什么（重计算）。
- 说清楚暖池的取舍（为闲置 GPU 付费 vs 接受冷启动尾部），以及 SLA 达到何种阈值时 `min_workers > 0` 成为必选项。

## 背景问题

你的无服务器 LLM 端点会在夜间缩容到零。早上 8 点流量高峰。第一条请求等待时：

1. Karpenter 提供 GPU 节点：45-60 秒。
2. 容器拉取 30 GB 含权重的镜像：120-300 秒。
3. 引擎将权重加载到 HBM：45-120 秒（取决于模型规模和存储速度）。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV 缓存池、分词器：10-30 秒。

总计：220-510 秒（约 3-8 分钟）才返回一个 token。你的 SLA 是 2 秒。你上了暖池（`min_workers=1`），问题似乎消失了——但现在 7×24 小时为一个闲置 GPU 付费。如果你的服务有 5 个产品各一个暖副本，就是 5 × 24 × 30 = 3,600 GPU 小时/月，不管有没有用户调用。

冷启动缓解是在保持无服务器经济性的同时，逼近常驻延迟的方法。

## 核心概念

### 第一层 — 预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将 OS 和数据分离。快照数据卷（已预拉取容器镜像），在 `EC2NodeClass` 中引用快照 ID。新节点启动时权重已在本地 NVMe 上——步骤 2 和部分步骤 3 消失。与 Karpenter 原生配合。大型模型每次冷启动节省 2-4 分钟。

GCP 上等价方案：含预烘焙容器层级的自定义 VM 镜像。Azure 上：含相同模式的管理磁盘快照。

### 第二层 — 模型流式加载（Run:ai Model Streamer）

不用等完整文件加载完再响应首个请求，而是按层流式将权重加载到 GPU 内存，处理在首个 transformer 块就绪后立即开始。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生提供。支持 S3、GCS 和本地 NVMe。通过将 I/O 和计算设置重叠，将大模型权重加载时间削减约一半。

### 第三层 — GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA graphs、KV 缓存区域）做检查点。后续重启时直接反序列化到 HBM——比重新初始化快 10 倍。这相当于"2 秒内启动一个热 GPU"的最近方案。取舍：快照按 GPU 拓扑生成，如果 Karpenter 把你迁移到不同 SKU，需要重新做检查点。

### 第四层 — 暖池（min_workers=1）

最简单的缓解：保持一个副本始终就绪。成本是 24×7 占用一个 GPU 的每小时费率。小模型算术很残酷（花 $0.85-1.50/小时避免 30 秒冷启动），大模型相对温和（花 $4/小时避免 5 分钟冷启动）。暖池成为必选的 SLA 阈值：70B+ 模型上通常 TTFT P99 < 60 秒。

### 第五层 — 分层加载（ServerlessLLM）

ServerlessLLM 将存储视为层级结构：NVMe（快但大）、DRAM（中等但分层）、HBM（小但即时）。权重预加载到 DRAM，按需加载到 HBM。论文报告冷加载相对朴素磁盘→HBM 延迟削减 10-200 倍。生产采用还在早期，但已与 vLLM 集成。

### 第六层 — 实时迁移（附加模式）

当节点不可用时（spot 驱逐、节点排空），传统做法是冷启动另一个副本并排空请求队列。实时迁移将输入 token（KB）迁移到已加载模型的目标节点，在目标节点重计算 KV 缓存。重计算比跨网络传输 GB 级 KV 缓存更便宜。适用于分解式部署。

### 暖池数学

对于 P99 TTFT SLA 为 2 秒的服务，问题是"要不要暖池"而是"几个暖副本，哪些路径用"。

- 高价值交互路径（在线聊天、语音 Agent）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：接受缩容到零，5-10 分钟冷启动可接受。
- 高级层：每个租户 `min_workers` 配专属容量。

### 先测量再优化

70B 模型在新节点上的冷启动解剖（示意）：

| 阶段 | 时间 | 缓解措施 |
|------|------|----------|
| 节点提供 | 50 秒 | Bottlerocket + 预置镜像，暖池 |
| 镜像拉取 | 180 秒 | 预置数据卷（消除） |
| 权重到 HBM | 75 秒 | 模型流式加载（减半）；GPU 快照（消除） |
| 引擎初始化 | 20 秒 | 持久化 CUDA graph 缓存 |
| 首次前向 | 3 秒 | 最小固有延迟 |
| **总冷启动** | **328 秒** | |
| **叠加缓解后** | **约 15 秒** | 22 倍削减 |

### 必须记住的数字

- Modal 冷启动：2-4 秒（GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预热后亚秒。
- 70B 原始冷启动：3-8 分钟。
- Run:ai Model Streamer：权重加载约 2 倍加速。
- ServerlessLLM 分层加载：论文数字 10-200 倍延迟削减。

## 用现成库

`code/main.py` 对有/无每种缓解措施的冷启动路径建模。报告总冷启动时间、暖池成本，以及暖池比冷启动税（通过 SLO 违约请求流失计算）更合算的平衡点请求率。

## 产出

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型规模和流量特征，选出要叠加的缓解措施组合。

## 练习

1. 运行 `code/main.py`。计算暖副本比通过额外请求流失支付冷启动税更合算的平衡点请求率。
2. 部署一个 TTFT P99 SLA 为 3 秒的 13B 模型。选出最少层数的缓解措施组合来实现它。
3. Bottlerocket 预置消除了镜像拉取，但权重仍从快照加载到 HBM。如果快照支撑的 NVMe 读取速度 7 GB/s，计算 70B 模型从 NVMe 到 HBM 的耗时。
4. 你的无服务器提供商提供 GPU 快照（Modal），团队拒绝因为"快照会泄露 PII"。论证正反两面——实际风险是什么？缓解措施（临时快照、加密、命名空间隔离）是什么？
5. 设计分层暖池策略：付费用户、试用用户、批处理工作负载各多少暖副本？给出数学计算。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 冷启动 | "大停顿" | 从新副本上请求到首个 token 的时间 |
| 暖池 | "常驻最小值" | `min_workers >= 1` 保持至少一个副本就绪 |
| 预置镜像 | "baked AMI" | 节点镜像中容器权重已预置 |
| Bottlerocket | "AWS 节点 OS" | AWS 容器优化 OS，支持双卷快照 |
| 模型流式加载 | "流式加载" | I/O 与计算设置重叠加载权重 |
| GPU 快照 | "检查点到 HBM" | 序列化加载后 GPU 状态；重启时反序列化 |
| 分层加载 | "NVMe + DRAM + HBM" | 存储层级体系；按需加载 |
| 实时迁移 | "移动 token" | 传输输入（KB），在目标节点重计算 KV |
| `min_workers` | "暖副本数" | 无服务器最小保活副本数 |
| 缩容到零 | "完全无服务器" | 闲置时零成本；接受完整冷启动税 |

## 扩展阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 发布的基准和检查点架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — I/O 与计算重叠加载权重。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — 预热 playbook。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 分解式部署的实时迁移。