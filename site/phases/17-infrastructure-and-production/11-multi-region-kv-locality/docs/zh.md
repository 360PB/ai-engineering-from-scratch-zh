# 多区域 LLM 服务与 KV 缓存局部性

> 轮询负载均衡对缓存 LLM 推理有害无益。不在持有前缀的节点上的请求要付完整 prefill 代价——在长提示词上 P50 约 800 ms vs 缓存命中约 80 ms，相差 10 倍。2026 年生产模式是缓存感知路由器（vLLM Router in Rust，llm-d router）消费 KV-cache 事件并按前缀哈希匹配路由。最新研究（GORGO）将跨区域网络延迟作为路由目标的显式项。商业"跨区域推理"产品（Bedrock 跨区域推理、GKE 多集群网关）将推理视为黑盒——处理可用性，不处理 TTFT。摩根大通和妙佑诊所 2024 年 11 月在 us-east-1 做故障恢复耗时约 22 分钟。DR 现实：32% 的 LLM DR 失败是因为团队备份了权重但忘了分词器文件或量化配置。

**类型：** 精读
**语言：** Python（标准库，玩具级前缀缓存感知路由器模拟器）
**前置要求：** Phase 17 · 04（vLLM 服务）、Phase 17 · 06（SGLang RadixAttention）
**时长：** 约 60 分钟

## 学习目标

- 解释为什么轮询负载均衡破坏缓存推理，并量化 TTFT 惩罚。
- 画出缓存感知路由器：输入（KV-cache 事件）、算法（前缀哈希匹配）、平局决胜（GPU 利用率）。
- 说出 LLM DR 失败中 32% 的驱动因素（缺少分词器文件/量化配置），并给出三文件 DR 检查清单。
- 区分商业跨区域产品（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV 感知路由。

## 背景问题

你的服务部署在 us-east-1、us-west-2 和 eu-west-1，前面放一个 ALB 轮询。前缀缓存命中率在生产中跌到 8%。TTFT P50 翻了三倍。你的 vLLM 日志显示每条请求都在付完整 prefill 代价。

轮询对无状态服务是最优的。LLM 推理按设计是有状态的——KV 缓存编码了模型见过的所有内容。盲目路由就是路由到错误的缓存。

另外，你的团队有 DR 计划。你把模型权重备份到 S3 跨区域。区域故障；你尝试故障转移；副本拒绝启动。你忘了 tokenizer.json、量化配置和 RoPE 缩放配置在一个你没有同步的独立 bucket 里。

多区域 LLM 服务是缓存问题、路由问题和 DR 卫生问题——不是负载均衡器问题。

## 核心概念

### 缓存感知路由

请求到达时带提示词。路由器对前缀做哈希（比如前 512 个 token）；询问每个副本"你有这个前缀的缓存吗？"副本在分配和驱逐 KV 块时通过发布/订阅通道发布 KV-cache 事件。路由器选择匹配的副本；如果无人匹配，平局决胜到 GPU 利用率最低的节点。

**vLLM Router**（Rust，2026 年生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希→副本索引路由，用 O(1) 查找。平局决胜到最少队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是同副本内的对应方案。跨副本路由是其严格的上游。

### 数字

2K-token 提示词，Llama 3.3 70B FP8，H100 上 P50 TTFT：
- 缓存命中（同副本，前缀在内存）：约 80 ms。
- 缓存未命中（冷 prefill）：约 800 ms。

10 倍差距。如果你的路由器跨副本达到 60-80% 前缀缓存命中，你用 N 副本容量逼近单副本性能。如果只有 10%，你逼近朴素扩展。

### 跨区域新增了一个约束——网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由器将请求从 us-east-1 路由到 ap-southeast-1 的热前缀，节省的 prefill（800 → 80 ms）被 440 ms 往返抵消。GORGO（2026 年研究）将此作为显式目标——联合最小化 `prefill_time + network_latency`，而非只看 prefill。通常答案是：除超大型多 MB 前缀（prefill 主导）外，保持区域路由。

### 商业"跨区域推理"在这里不起作用

AWS Bedrock 跨区域推理在容量压力时自动将请求路由到其他区域。它优化可用性，不优化 TTFT，把推理当黑盒。GKE Multi-Cluster Gateway 相同——服务级故障转移，对 KV 缓存无感知。

用了这些你仍需要在应用层加缓存感知路由器。这些处理"us-east-1 着火"的情况。缓存感知路由处理 TTFT 情况。

### DR 卫生——32% 的缺文件问题

2026 年广泛引用的数据：32% 的 LLM DR 失败发生在团队备份了权重但忘了：
- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE 缩放、attention masks、聊天模板）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA adapter manifests）

修复：三文件最小 DR 清单：

1. HF 模型仓库下所有文件（权重 + 配置 + 分词器）。
2. 引擎特定服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定）。

另外：每季度做一次 DR 演练。摩根大通 us-east-1 演练 2024 年 11 月耗时 22 分钟恢复，只因 playbook 事先演练过。

### 数据驻留是独立的正交问题

欧盟客户 PHI 不能离开欧盟。如果你的缓存感知路由器把来自巴黎的请求发到 us-east-1 匹配前缀，不管 TTFT 收益多少，你已违反 GDPR。先按驻留边界划分路由器，再优化缓存。

### 必须记住的数字

- 缓存命中 vs 未命中 TTFT 差距：约 10 倍（2K 提示词上 80 ms vs 800 ms）。
- 区域间 RTT 欧美：约 75 ms。
- DR 失败原因 32%：分词器/量化配置缺失。
- 摩根大通 us-east-1 故障转移 2024 年 11 月：22 分钟（30 分钟 SLA）。

## 用现成库

`code/main.py` 模拟三种路由策略（轮询、缓存感知区域、缓存感知全局）在多区域负载下的表现。报告缓存命中率、TTFT P50/P99 和跨区域账单。

## 产出

本课产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计路由方案。

## 练习

1. 运行 `code/main.py`。给定 75 ms RTT，在什么提示词长度下跨区域路由优于仅本地路由？
2. 你的缓存命中率从 70% 跌到 12%。诊断三个可能原因，以及确认每个原因的观察指标。
3. 为 70B AWQ 量化模型在 vLLM 中服务加 5 个 LoRA adapter 设计 DR 清单。列出每个文件和配置。
4. 论证 Bedrock 跨区域推理对有严格 TTFT SLO 的金融科技公司是否"够用"。引用具体行为。
5. 一条来自巴黎的请求匹配 us-east-1 的前缀。你路由吗？写出策略。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 缓存感知路由 | "智能 LB" | 按前缀哈希匹配路由到持有 KV 缓存的副本 |
| KV-cache 事件 | "缓存发布/订阅" | 副本发布块加入/驱逐；路由器建立索引 |
| 前缀哈希 | "缓存键" | 前 N 个 token 的哈希作为路由查找键 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式路由项 |
| 跨区域推理 | "Bedrock CRI" | AWS 产品；可用性故障转移，无 TTFT 感知 |
| DR 清单 | "备份列表" | 恢复所需的所有文件——不只是权重 |
| 数据驻留 | "GDPR 边界" | 限制哪个区域能见到用户数据的法律约束 |
| RTT | "往返时间" | 网络延迟；欧美 75 ms，US-APAC 220 ms |
| LLM 感知 LB | "缓存命中 LB" | 缓存感知路由器作为一类产品 |

## 扩展阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 含网络延迟项的跨区域 KV 缓存复用。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — 缓存感知路由器源码。