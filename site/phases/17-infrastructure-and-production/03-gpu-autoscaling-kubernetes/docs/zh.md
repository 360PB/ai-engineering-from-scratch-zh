# GPU 在 Kubernetes 上的自动扩缩容 — Karpenter、KAI Scheduler、组调度

> 三层，不是一层。Karpenter 动态配置节点（1 分钟内，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理组调度、拓扑感知和层级队列——它防止"8 中取 7"的局部分配陷阱，7 个节点空等并烧钱等待那 1 个缺失的 GPU。应用层自动扩缩器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）根据推理特定信号（队列深度、KV 缓存利用率）扩缩，而非 CPU/DCGM 利用率。经典 HPA 陷阱是 `DCGM_FI_DEV_GPU_UTIL` 是占空比测量：100% 可能是 10 个请求，也可能是 100 个。vLLM 预分配 KV 缓存内存，所以内存永远不会触发缩容。本课教你组合这三层，并避免默认的 Karpenter `WhenEmptyOrUnderutilized` 策略——它会在推理过程中终止正在运行的 GPU 作业。

**类型：** 精读
**语言：** Python（标准库，玩具级队列深度自动扩缩模拟器）
**前置要求：** Phase 17 · 02（推理平台经济学）、Phase 17 · 04（vLLM 推理内部原理）
**时长：** 约 75 分钟

## 学习目标

- 绘制三层自动扩缩（节点配置、组调度、应用层）并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 的错误 HPA 信号，并说出两个替代方案（队列深度、KV 缓存利用率）。
- 描述组调度以及 KAI Scheduler 防止的局部分配失败模式（7/8 GPU 空闲）。
- 说出 Karpenter 的合并策略（`WhenEmptyOrUnderutilized`）会终止正在运行的 GPU 作业，并说出 2026 年的安全替代方案。

## 背景问题

你的团队在 Kubernetes 上运行 LLM 推理服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置了 HPA。服务在工作时间 pinned 在 100% 利用率。HPA 从不向上扩缩——它已经认为你满了。手动加了一个副本；TTFT 下降了。HPA 仍然不扩缩。信号在骗你。

另外，你用 Cluster Autoscaler 来管理节点。凌晨 2 点，一个 100 万 token 的提示词到达；集群花了 3 分钟配置一个节点，请求超时了。

再次，你的 70B 模型部署需要 8 块 GPU 跨 2 个节点。集群有 7 块 GPU 空闲，1 块分散在 3 个节点上。Cluster Autoscaler 为缺失的那 1 块 GPU 配置一个节点。七个节点等待 4 分钟烧钱，而 Kubernetes 让最后那块 GPU 上线。

三层，三个不同失败模式。2026 年的 GPU 感知自动扩缩不是"开启 HPA"，而是组合节点配置、组调度和应用信号扩缩。

## 核心概念

### 第一层——节点配置（Karpenter）

Karpenter 监控待调度的 pod，在约 45-60 秒内配置节点（Cluster Autoscaler 对 GPU 节点通常需要 90-120 秒）。它按 `NodePool` 约束动态选择实例类型——如果你的 pod 需要 8 块 H100 而集群没有匹配节点，Karpenter 直接配置一个，而不是扩容已有节点组。

**合并陷阱**：Karpenter 默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池很危险。它会终止正在运行的 GPU 节点，将 pod 迁移到更便宜的大小合适的实例。这意味着驱逐正在处理的请求，重新加载 70B 模型到新节点。损失是几分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让 Karpenter 在一小时后将真正为空的节点合并，但从不驱逐正在运行的作业。

### 第二层——组调度（KAI Scheduler）

KAI Scheduler（项目"Karp"后改名）处理默认 kube-scheduler 无法处理的：

**组调度** — 全有或全无。一份需要 8 块 GPU 的分布式推理 pod 要么 8 块一起启动，要么都不启动。没有这个，你会陷入局部分配陷阱：7/8 pod 启动，无限等待，烧钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一机架，哪些之间有 InfiniBand。按此放置 pod。一个 DeepSeek-V3 67B 张量并行工作负载必须留在一个 NVLink 域内；KAI Scheduler 遵守这一点。

**层级队列** — 多个团队共享同一 GPU 池，按优先级和配额竞争。A 团队的生产高峰只在优先级规则允许时才被 B 团队的训练作业抢占。

KAI 作为次级调度器部署在 kube-scheduler 旁边；你为工作负载加上注解以使用它。Ray 和 vLLM production-stack 都有集成。

### 第三层——应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比指标——它测量 GPU 在每个采样间隔是否在工作。100% 利用率可能意味着 10 个并发请求，也可能意味着 100；GPU 两种情况下都是忙的。根据占空比扩缩等于盲目扩缩。

更糟的是，vLLM 和类似引擎预分配 KV 缓存内存（高达 `--gpu-memory-utilization`）。内存使用即使在一个请求下也接近 90%。基于内存的 HPA 永远不会缩容。

**2026 年替换信号**：

- 队列深度（等待 prefill 的请求数量）。
- KV 缓存利用率（分配给活跃序列的块的比例）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩副本。它们完全取代了 LLM 推理的 HPA。

### 何时用哪个

| 扩缩决策 | 工具 |
|---------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA）|
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 分解式 prefill/decode 让一切复杂化

如果你运行分解式 prefill/decode（Phase 17 · 17），你有两种 pod 类，它们有不同的扩缩触发：prefill pod 按队列深度扩缩，decode pod 按 KV 缓存压力扩缩。llm-d 将这些作为独立 `Services` 提供，每个 role 有单独的 HPA。不要试图在两者前面放一个 HPA。

### 冷启动在这里也很重要

冷启动缓解（Phase 17 · 10）是节点配置时间对用户可见的地方。Karpenter 的 45-60 秒预热加 20GB 模型加载加引擎初始化意味着从零开始的请求需要 2-5 分钟。为 SLA 关键路径保持一个暖池（`min_workers=1`），或在应用层使用 Modal 风格的检查点。

### 记忆数字

- Karpenter 节点配置：约 45-60 秒 vs Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI Scheduler 防止局部分配浪费——7/8 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：无效；用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止正在运行的 GPU 作业。推理用 `WhenEmpty + consolidateAfter: 1h`。

## 运用它

`code/main.py` 模拟三层自动扩缩器在突发 GPU 工作负载上的表现。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI 组调度扩缩。报告未满足请求、空闲 GPU 分钟数和综合得分。

## 交付它

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形态和 SLA，设计三层自动扩缩方案。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比 HPA 丢弃了多少队列深度 HPA 能捕获的请求？差异来自哪里？
2. 为运行 Llama 3.3 70B FP8 on H100 SXM5 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个将非 GPU 工作负载隔离在这些节点外的 taint。
3. 你的团队报告部署卡在 Pending，因为"GPU 可用但 pod 无法调度"。诊断——这是 Karpenter、kube-scheduler 还是 KAI Scheduler？哪些指标可以确认？
4. 为分解式 prefill pod 选择一个信号，为 decode pod 选择一个不同的信号。两边都论证。
5. 计算 `WhenEmptyOrUnderutilized` 合并陷阱在 7×24 生产服务上的成本：平均每天 60 次请求丢弃事件，P99 TTFT > 10 秒。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Karpenter | "节点配置器" | Kubernetes 节点自动扩缩器；分钟级配置 |
| Cluster Autoscaler | "旧扩缩器" | Kubernetes 节点自动扩缩器前身；较慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于组调度 + 拓扑 + 队列的次级调度器 |
| 组调度 | "全有或全无" | 原子调度 N 个 pod，要么全调度要么全推迟 |
| 拓扑感知 | "机架感知" | 基于 NVLink/IB/机架放置放置 pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比指标；不是 LLM 的扩缩信号 |
| 队列深度 | "等待中的请求" | prefill 绑定扩缩的正确 HPA 信号 |
| KV 缓存利用率 | "内存压力" | decode 绑定扩缩的正确 HPA 信号 |
| 合并 | "Karpenter 合并" | 终止节点迁移到更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全合并" | 不驱逐正在运行的 GPU 作业的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。
- [Karpenter 干扰控制](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义和 GPU 安全默认值。
- [NVIDIA — Kubernetes 上的分解式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩信号。
- [Ray 文档 — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS 计算和自动扩缩最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 特定指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。