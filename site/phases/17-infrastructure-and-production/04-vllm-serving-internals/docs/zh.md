# vLLM 推理内部原理：PagedAttention、连续批处理、分块 Prefill

> vLLM 在 2026 年的主导地位建立在三个复合默认配置上，而非单一诀窍。PagedAttention 始终开启。连续批处理在 decode 迭代之间注入新请求。分块 Prefill 将长提示词切片，使 decode token 永不饿死。三个全开，一块 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下达 2,200-2,400 tok/s——比 vLLM 自有默认值高约 25%，是朴素 PyTorch 循环的 3-4 倍。本课从可画图层面阅读调度器和注意力 kernel，最终用 `code/main.py` 中的玩具连续批处理器，按 vLLM 的方式调度 prefill 和 decode。

**类型：** 精读
**语言：** Python（标准库，玩具级连续批处理调度器）
**前置要求：** Phase 17 · 04（模型服务）、Phase 11（LLM 工程）
**时长：** 约 75 分钟

## 学习目标

- 将 PagedAttention 解释为 KV 缓存分配器：块、块表，以及为什么生产负载下碎片化保持在 4% 以下。
- 在迭代层面绘制连续批处理：完成的序列如何离开批、新序列如何在不排空的情况下加入。
- 用一句话描述分块 prefill，并说出它保护哪个延迟指标（提示：是 TTFT 尾部，不是平均吞吐）。
- 说出 2026 年 vLLM v0.18.0 中同时开启所有优化时会踩到的坑。

## 背景问题

朴素 PyTorch serve 循环一次运行一个请求：tokenize、prefill、decode 直到 EOS、返回。一个用户时正常运行。一百个用户时就是一条耐心等待的队列。明显修复——静态批处理——将批中每个请求 padding 到窗口中最长提示词的长度，将每个 decode padding 到最长预期输出，并让整个批在最慢序列上停滞。你为从未使用过的 padding 付钱，快请求等慢请求。

vLLM 一次性解决三个问题。PagedAttention 阻止 KV 缓存碎片化——经典连续分配会吞噬 60-80% 的 GPU 内存。连续批处理让请求在每个 decode 迭代之间加入和离开批，所以批中始终充满真实工作。分块 prefill 将 32k-token 提示词切片为约 512-token 片，与 decode 交错，所以一个长提示词不会冻结 GPU 上每个 decode token。

2026 年生产默认是三者全开。你需要理解每个的作用，因为失败模式全在调度器上，不在模型上。

## 核心概念

### PagedAttention 作为虚拟内存系统

KV 缓存是每个序列 `层数 × 2 × 头数 × 头维度 × 序列长度 × 每元素字节数`。对于 Llama 3.3 70B at 8192 tokens，每序列约 1.25 GB（BF16）。如果你为每个请求预分配 8192 个槽但平均只用 1500 个 tokens，你就浪费了约 82% 的 HBM。经典批处理为此付出代价。

PagedAttention 从 OS 虚拟内存借用思想。KV 缓存不是每个序列连续的。它以固定大小块分配（默认 16 tokens）。每个序列有一个块表，将逻辑 token 位置映射到物理块 ID。当序列增长超过分配块时，添加一个新块。完成时，它的块归还池。

碎片化从 60-80%（经典）降到 4% 以下（PagedAttention）。你不用 flag 开启 PagedAttention——它是 vLLM 唯一发货的分配器。旋钮是 `--gpu-memory-utilization`（默认 0.9），告诉 vLLM 在加载权重和激活后为 KV 块预留多少 HBM。

### 迭代级连续批处理

旧的"动态批处理"等待一个窗口（比如 10 ms）填满一个批，然后运行 prefill + decode + decode + decode 直到每个序列完成。快序列提前离开，空等 GPU 做完慢序列。

连续批处理在每个 decode 步骤之间运行。称运行中序列集为 `RUNNING` 列表。每次迭代：

1. `RUNNING` 中刚达到 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV 块，它允许新序列（prefill 或恢复）。
3. 前向传递在 `RUNNING` 中当前的任何内容上运行，每个序列发出一个新 token。

批大小从不 padding 到固定数量。处于输出不同位置的序列共享一次融合前向。2026 年 vLLM 这称为 `V1 调度器`。关键不变量：调度器每个 decode 迭代运行一次，不是每个请求一次。

### 分块 Prefill 保护 TTFT 尾部

Prefill 是计算密集型。Llama 3.3 70B 在一块 H100 上，32k-token 提示词的纯 prefill 约 800 ms。Prefill 运行时，批中每个其他序列的 decode token 等待。在服务循环中，一个长提示词的首次 token 延迟（TTFT）成为其他数十个用户的 token 间延迟（ITL）尖峰。

分块 prefill 将 prefill 分割为固定大小块（默认 512 tokens），将每个块作为一个单元调度。在块之间，调度器可以推进 decode 序列一个 token。你用每个块几毫秒的绝对 prefill 延迟损失，换取 decode 时低得多的抖动。发布基准测试中，混合负载下 P99 ITL 从约 50 ms 降到约 15 ms。

### 三个默认配置相互作用

三者互相假设。PagedAttention 给调度器细粒度 KV 资源来交易。连续批处理需要那种细粒度资源，所以允许新序列不需要全局重组。分块 prefill 是调度器对同一 `RUNNING` 列表做出的决策——它是又一个调度器策略，不是独立系统。

你不需要知道每个 flag。你需要知道调度器优化的是什么：在 KV 块预算下的 goodput，受分块 prefill 切片约束。

### 2026 年 v0.18.0 坑

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与草稿模型推测解码（`--speculative-model`）结合使用。记录在案的例外是 V1 调度器中的 N-gram GPU 推测解码。将所有 flag 同时开启而不阅读发布说明的团队在启动时遇到运行时错误，而非软回归。如果你的推测收益值得为分块 prefill 开启它，重新审视选择——2026 年正确答案通常是 EAGLE-3 而不分块 prefill，而非草稿模型加无法编译的分块 prefill。

### 记忆数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同模型，vLLM 默认（无分块 prefill）：约 1,800 tok/s。
- 同模型，朴素 PyTorch 前向循环：约 600 tok/s。
- PagedAttention 下生产负载 KV 碎片化浪费：<4%。
- 分块 prefill 下 P99 ITL：约 15 ms；无分块：约 50 ms。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个循环，用标准库 Python 加假 token 数和假前向延迟。运行它可以看到分块 prefill 如何在长 prefill 期间保持 decode 序列存活。

## 运用它

`code/main.py` 模拟一个 vLLM 风格调度器，带可切换功能。运行它来看：

- `NAIVE` 模式：一次一个请求，无批处理。
- `STATIC` 模式：pad 并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级准入和释放。
- `CONTINUOUS + CHUNKED` 模式：prefill 片与 decode 交错。

输出显示总吞吐（每虚拟秒 token 数）、TTFT 均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行在混合流量下应该占主导。

## 交付它

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定服务配置（批大小、KV 内存利用率、分块 prefill 大小、推测配置），它产生一个调度器诊断，指出三个默认配置中哪个是瓶颈以及如何调优。

## 练习

1. 运行 `code/main.py`。在混合短请求和长请求的工作负载上，比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里——prefill 效率、decode 效率还是尾延迟？
2. 修改玩具调度器，添加 `--max-num-batched-tokens`。对于一块运行 Llama 3.3 70B FP8 的 H100，正确的值是多少？（提示：它是 KV 块大小和空闲块数的函数，不是原始 HBM。）
3. 重读 vLLM v0.18.0 发布说明。哪些 flag 组合是互斥的？列出它们。
4. 计算在 8192 最大、连续按请求分配下，1000 个请求追踪（均值 1500 输出 tokens，std 600 tokens）的 KV 缓存碎片化浪费；(a) vs (b) PagedAttention，16-token 块。
5. 用一段话解释为什么分块 prefill 帮助 P99 ITL 但单独不帮助吞吐。实际上吞吐收益来自哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| PagedAttention | "KV 技巧" | KV 缓存的固定大小块分配器；碎片化 <4% |
| 块表 | "页表" | 每个序列从逻辑 token 位置到物理 KV 块的映射 |
| 连续批处理 | "正确的动态批处理" | 每个 decode 迭代做出准入/释放决策 |
| 分块 prefill | "prefill 分片" | 将长 prefill 分解为 512-token 片与 decode 交错 |
| TTFT | "首 token 时间" | prefill + 队列 + 网络；长提示词时 prefill 主导 |
| ITL | "token 间延迟" | 连续 decode token 间的时间；批大小主导 |
| Goodput | "满足 SLO 的吞吐" | 每请求仍达到 TTFT 和 ITL 目标的 tok/s |
| V1 调度器 | "新调度器" | vLLM 2026 调度器；N-gram spec decode 是与分块 prefill 兼容的路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 加载权重和激活后为 KV 块预留的 HBM 比例 |

## 延伸阅读

- [vLLM 文档 — 推测解码](https://docs.vllm.ai/en/latest/features/spec_decode/) — 分块 prefill 和推测解码兼容性的官方来源。
- [vLLM 发布说明（NVIDIA）](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 发布节奏和版本特定行为。
- [vLLM 博客 — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 原创文章，仍是理解分配器的基准。
- [PagedAttention 论文（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) — 碎片化分析和调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带火焰图的详细 V1 调度器讲解。