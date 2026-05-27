# vLLM 生产栈与 LMCache KV Offloading

> vLLM 的生产栈是参考 Kubernetes 部署——路由器、引擎、可观测性全链路接好。LMCache 是 KV offloading 层，把 KV 缓存从 GPU 内存提取出来跨查询和引擎复用（CPU DRAM，然后 disk/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）使这成为异步可插拔的。Offload 延迟不面向用户。LMCache 即便没有共享前缀也有价值——GPU 的 KV 槽用完时，被抢占的请求可从 CPU 恢复而非重算 prefill。在 16x H100（80GB HBM）跨 4 个 a3-highgpu-4g 的已发布基准上：当 KV 缓存超 HBM，原生 CPU offload 和 LMCache 都显著改善吞吐；在低 KV 占用时，所有配置与基线相当，有小幅开销。

**类型：** 精读
**语言：** Python（标准库，玩具级 KV 溢出模拟器）
**前置要求：** Phase 17 · 04（vLLM 推理内部原理）、Phase 17 · 06（SGLang/RadixAttention）
**时长：** 约 60 分钟

## 学习目标

- 画出 vLLM 生产栈的层级：路由器、引擎、KV offload、可观测性。
- 解释 KV Offloading Connector API（v0.9.0+）以及 0.11.0 异步路径如何隐藏 offload 延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）何时只是额外开销（KV 小到能装进 HBM）。
- 在原生 vLLM CPU offload 和 LMCache connector 之间选择，给定部署约束。

## 背景问题

你的 vLLM 服务在并发上升时 GPU HBM 100%，且不断发生抢占事件。请求被驱逐，重新入队，同一 2K-token 提示词一分钟内被重新 prefill 四次。GPU 算力浪费在冗余 prefill 上；goodput 远低于原始吞吐。

加 GPU 线性增加成本。加 HBM 不可能。但 CPU DRAM 便宜——一个插槽 512 GB+ 以上，延迟比 HBM 差几个数量级，但对"暂时温热"的 KV 缓存来说足够好。

LMCache 把 KV 缓存提取到 CPU DRAM，这样被抢占的请求能快速恢复，跨引擎共享缓存的前缀无需各自重 prefill。

## 核心概念

### vLLM 生产栈

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **路由器** — 有缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎** — vLLM worker。每 GPU 一个或每个 TP/PP 组一个。
- **KV 缓存 offload** — LMCache 部署或原生 connector。
- **可观测性** — Prometheus 抓取、Grafana 仪表板、OTel traces。
- **控制平面** — 服务发现、配置、滚动更新。

以 Helm chart + operator 形式发布。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了可插拔 KV 缓存后端的 Connector API。你的引擎 offload 块到 connector；connector 存储它们（RAM、disk、对象存储、LMCache）。请求需要块时，connector 加载回来。

vLLM 0.11.0（2026 年 1 月）添加了异步 offload 路径——offload 可以在后台进行，所以引擎在常见情况下不会阻塞在上面。端到端延迟和吞吐仍取决于负载形状、KV 缓存命中率和系统压力；vLLM 自己的说明指出自定义 kernel offload 在低命中时可能降低吞吐，且异步调度与推测解码存在已知的交互问题。

### 原生 CPU offload vs LMCache

**原生 vLLM CPU offload**：引擎本地。存储 KV 块到宿主机 RAM。实现快，零网络跳。不跨引擎。

**LMCache connector**：集群规模。存储块到共享 LMCache 服务器（CPU DRAM + Ceph/S3 层）。块可被任何引擎访问。已发布 16x H100 基准。

当单个引擎有 HBM 压力时选原生。当多引擎共享前缀（RAG 加常见系统提示词、多租户加共享模板）时选 LMCache。

### 基准行为

16x H100（80 GB HBM）跨 4 个 a3-highgpu-4g 测试：

- 低 KV 占用（短提示词、低并发）：所有配置与基线相当，LMCache 加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用上帮助。
- KV 超出 HBM：原生 CPU offload 和 LMCache 都显著改善吞吐；LMCache 增益更大，因为跨引擎共享。

### LMCache 起决定性作用的场景

- 多租户服务，系统提示词跨租户共享。
- RAG，文档块在查询间重复。
- 同基模型上的微调变体（LoRA），基模型 KV 复用削减冗余工作。
- 抢占密集型负载：从 CPU 恢复比重 prefill 便宜。

### 何时不启用

- HBM 压力小——你付出开销没收益。
- 短上下文（<1K token）——传输时间 > 重 prefill。
- 单租户单提示词负载——无可用复用。

### 与分解式服务集成

Phase 17 · 17 分解式服务 + LMCache 叠加：KV 从 prefill 池传输到 decode 池后如未使用则落入 LMCache；后续查询从 LMCache 拉取。Phase 17 · 11 缓存感知路由器可路由到其本地或 LMCache 共享缓存匹配 Engine。

### 必须记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步 offload 路径；端到端延迟影响取决于负载、KV 命中率、系统压力（非绝对保证）。
- 16x H100 基准：KV 占用超出 HBM 时 LMCache 有帮助。
- 小 HBM 压力：无收益时 3-5% 开销。

## 用现成库

`code/main.py` 模拟有/无 LMCache 的抢占密集型负载。报告避免的重 prefill 次数、吞吐增益和盈亏平衡 HBM 利用率。

## 产出

本课产出 `outputs/skill-vllm-stack-decider.md`。给定工作负载形状和 vLLM 部署，决定原生 vs LMCache vs 两者都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始合算？
2. 某租户在 200 查询/小时中共享 6K-token 系统提示词。计算每租户预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计 HA 策略（副本、回退到原生）。
4. LMCache 存到机械盘 Ceph。4K-token KV 在 70B FP8（500 MB），读取时间 vs 重 prefill 是多少？
5. 论证 vLLM 0.11.0 异步路径是否是"免费"的——开销藏在哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 生产栈 | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| 原生 CPU offload | "引擎本地溢出" | 存储 KV 到同引擎宿主机 RAM |
| LMCache | "集群 KV 缓存" | 跨引擎 KV 缓存服务器，在 CPU DRAM + disk |
| 0.11.0 async | "非阻塞 offload" | Offload 隐藏在引擎流后面 |
| 抢占 | "驱逐腾空间" | HBM 满时 KV 缓存腾挪 |
| 前缀复用 | "相同系统提示词" | 多查询共享开头；缓存命中 |
| Ceph 层 | "disk 层" | 缓存在 DRAM 以下耐用存储层级 |

## 扩展阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector 实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径详情。