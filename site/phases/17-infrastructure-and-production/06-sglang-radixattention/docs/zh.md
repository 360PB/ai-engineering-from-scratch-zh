# SGLang 和 RadixAttention 处理前缀繁重工作负载

> SGLang 将 KV 缓存作为一等可复用资源存储在基数树中。vLLM 按 FCFS（先到先服务）调度，SGLang 的缓存感知调度器优先处理具有较长共享前缀的请求——实际上是深度优先的基数遍历，使热分支保留在 HBM 中。在 Llama 3.1 8B 上使用 ShareGPT 风格 1K 提示词，SGLang 达到约 16,200 tok/s，vLLM 约 12,500——约 29% 的优势。在前缀繁重的 RAG 工作负载上优势达到 6.4 倍。在语音克隆形态的工作负载上缓存命中率突破 86%。2026 年部署在 400,000+ GPU 上，涵盖 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS。陷阱：6.4x 数字在前缀排序不一致时完全消失——排序是工程师的杠杆。

**类型：** 精读
**语言：** Python（标准库，玩具级基数树缓存 + 缓存感知调度器）
**前置要求：** Phase 17 · 04（vLLM 推理内部原理）、Phase 14（Agentic RAG）
**时长：** 约 75 分钟

## 学习目标

- 绘制 RadixAttention：前缀如何存储在基数树中，以及 KV 块如何共享到同一分支根的序列。
- 解释缓存感知调度，以及为什么 FCFS 对前缀繁重流量是错误的。
- 给定前缀缓存命中率和提示词长度分布，计算预期加速。
- 说出使 6.4x 数字成真（而非失去的上行）的提示词排序规范。

## 背景问题

经典服务将每个请求的提示词视为不透明的。即使 5,000 个 RAG 请求都以相同的 2,000-token 系统提示词加相同检索前言开头，vLLM 也要将这 2,000-token 前缀 prefill 5,000 次。GPU 做同样的工作一遍又一遍。

观察：agentic 和 RAG 工作负载中的提示词几乎总是共享很长的前缀。系统提示词、工具 schema、少样本示例、检索头、对话历史——所有这些在请求间重复。如果你将那个前缀的 KV 缓存存储一次并复用，就不需要再 prefill 它了。

RadixAttention 正是这样做的。Token 在基数树中索引；每个节点拥有其路径上 token 序列的 KV 块。一个新请求遍历树：任何 token 匹配的节点复用那个节点的 KV 块。Prefill 成本与"新"后缀成正比，而非整个提示词。

挑战是调度。如果两个请求共享 2,000-token 前缀，第三个只共享同一前缀的 200 tokens，你希望一起服务两个长共享请求，这样长前缀保留在 HBM 中。FCFS 做相反的事——谁先到服务谁，可能会在下一个长前缀请求到达前驱逐热分支。

## 核心概念

### 基数树作为 KV 索引

基数树（压缩 trie）存储 token 序列。每个节点拥有一个 token 范围和为该范围计算的 KV 块。子节点将序列延长一个或多个 tokens。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个新请求到来，带着系统提示词 + "Context: <doc A>" + "Question: Carol"。调度器遍历：系统前缀匹配（124 块复用），doc-A 分支匹配（31 块复用），然后只为 "Question: Carol"（4 块）分配新块。Prefill 成本：4 块新 tokens。无树：160 块。Prefill 约 40 倍节省。

### 缓存感知调度

基数树支持的复用如果缓存抖动就毫无意义。两个关键策略：

1. **深度优先分派**。从队列中选择下一个请求时，优先选择在当前运行集同一分支上根植的请求。这使热分支 pinned。
2. **分支级 LRU，而非块级**。从最短使用的叶子开始逐分支驱逐，而非逐块驱逐，这样缓存形状与基数形状匹配。

FCFS 违反两者。共享 2,000 tokens 的请求排在共享 50 tokens 的请求后面，然后 2,000-token 分支被驱逐以容纳 50-token 请求。

### 记忆基准数字

- Llama 3.1 8B，H100，ShareGPT 1K 提示词：SGLang 约 16,200 tok/s vs vLLM 约 12,500（约 29% 优势）。
- 前缀繁重 RAG（相同系统 + 相同文档，不同问题）：SGLang 高达 6.4 倍。
- 语音克隆工作负载：86.4% 前缀缓存命中率。
- SGLang 客户生产命中率：50-99%，取决于提示词规范。
- 2026 年部署在 400,000+ GPU 上。

### 排序陷阱

6.4x 数字依赖于一致的提示词模板排序。如果你的客户端有时将提示词构造为 `[system, tools, context, history, question]`，有时为 `[system, context, tools, history, question]`，树找不到共享前缀。对人类看起来像共享前缀的，对基数树是两个不同序列。

工程师的杠杆：你的提示词模板是一个缓存键。固定顺序。将所有不可变内容（system、tools、schemas）放在前面。接下来放检索上下文。最后放用户问题。不要将动态内容混入前缀。

真实案例：研究之一，将动态内容移出可缓存前缀，一次部署从 7% 到 74% 缓存命中率。

### RadixAttention 胜出和失败的地方

胜出：
- RAG（相同检索前言，不同问题）。
- Agent（相同工具 schema，不同查询）。
- 带长系统提示词的聊天。
- 语音/视觉工作负载与重复前言。

失败（回到 vLLM 级别吞吐）：
- 带唯一提示词的单次生成（代码补全、无系统提示词的开放式聊天）。
- 每个请求将唯一内容混入前缀的动态提示词。

### 为什么这是调度器问题，不只是 kernel 问题

你可以将 KV 复用实现为 kernel 技巧。SGLang 的洞察是：只有当调度器保持热分支 resident 时，复用才值得。在混合负载下，朴素"可用则复用"策略会折腾缓存。基数树索引调度器是将 kernel 技巧转化为生产级 29% 优势的关键。

### 与 vLLM 的相互作用

两个系统不是严格竞争者。2026 年 vLLM 添加了前缀缓存（`--enable-prefix-caching`）和缓存感知路由器（Rust 中的 vLLM Router）。差距缩小但没有完全消失——SGLang 整个栈是基数优先；vLLM 是后来加的。对于前缀复用主导的工作负载，SGLang 仍是默认。对于没有强前缀模式的通用服务，vLLM 仍相当或更好。

## 运用它

`code/main.py` 实现了一个玩具基数树 KV 缓存加两个策略的调度器：FCFS 和缓存感知。对同一工作负载运行两者，报告前缀缓存命中率和吞吐增量。然后运行"打乱排序"工作负载来展示 6.4x 崩溃。

## 交付它

本课产出 `outputs/skill-radix-scheduler-advisor.md`。给定工作负载描述（提示词模板形状、检索模式、并发租户数），它产生提示词排序规范和 SGLang 采用 go/no-go。

## 练习

1. 运行 `code/main.py`。在同一工作负载上比较 FCFS 和缓存感知。差距来自哪里——prefill 节省、decode 节省还是队列延迟？
2. 修改工作负载，随机打乱 `[system, tools, context]` 的顺序。重跑。命中率发生了什么？为什么？
3. 计算在 Llama 3.1 8B 上将 2,000-token 系统提示词保留为一个基数分支的 HBM 成本。与无前缀复用的 16 序列批成本比较。
4. 阅读 SGLang RadixAttention 论文。用三句话解释为什么树形 LRU 驱逐在前缀繁重负载下优于块形 LRU。
5. 一位客户报告只有 8% 的缓存命中率。说出三个可能原因，以及针对每个的诊断方法。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| RadixAttention | "SGLang 特性" | KV 缓存索引为基数树，共享前缀复用块 |
| 基数树 | "压缩 trie" | 每个节点拥有一个 token 范围及其 KV 块的树 |
| 缓存感知调度器 | "热分支优先" | 优先处理与 resident 分支共享的请求的调度器 |
| 前缀缓存命中率 | "你的提示词有多少免费" | 从复用 KV 块服务的提示词 token 比例 |
| FCFS | "先到先服务" | 默认调度，破坏前缀局部性 |
| 分支级 LRU | "驱逐叶子" | 与基数形状匹配的驱逐策略 |
| 提示词模板排序 | "缓存键" | 提示词组件顺序决定了树能共享什么 |
| 系统提示词 pinning | "resident 前缀" | 保持不可变系统部分 pinned 以避免驱逐抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源码和文档。
- [SGLang 文档](https://sgl-project.github.io/) — RadixAttention 和调度详情。
- [SGLang 论文 — 高效编程大型语言模型（arXiv:2312.07104）](https://arxiv.org/abs/2312.07104) — 设计参考。
- [LMSYS 博客 — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — 基准数字和调度原理。
- [vLLM — 前缀缓存](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自己的类基数实现，用于比较。