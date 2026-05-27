# KV 缓存、Flash Attention 与推理优化

> 训练是并行的，受 FLOPs 限制。推理是串行的，受内存限制。不同的瓶颈，不同的技巧。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 第 2 课（自注意力）、Phase 7 第 5 课（完整 Transformer）、Phase 7 第 7 课（GPT）
**时长：** ~75 分钟

## 问题

朴素的的自回归解码器生成 `N` 个 token 做 `O(N²)` 工作：在每步它重新计算对完整前缀的注意力。对于 4K token 响应，那是 16M 次注意力操作，大部分是冗余的。前缀 token 的每个隐藏状态一旦计算就是确定的——你只需要用新 token 的 query 针对之前所有内容的缓存 key 和 value 运行。

在此之上，注意力本身移动大量数据。标准注意力物化一个 N×N 分数矩阵、N×d softmax 输出、N×d 最终输出——太多对 HBM 的读写。对于 N≥2K，注意力在成为 FLOP 受限之前就变成内存受限。经典注意力内核在现代 GPU 上低效使用 4–10×。

Dao 等人的两个优化，将前沿推理从"慢"推到"快"：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力是针对缓存 key 的一次 query。推理从每步 `O(N²)` 减少到 `O(N)`。
2. **Flash Attention。** 平铺注意力计算，使完整 N×N 矩阵永不触碰 HBM。全部 softmax + 矩阵乘法在 SRAM 中发生。在 A100 上 2–4× 墙上时钟加速；在 H100 上 FP8 时 5–10×。

截至 2026 年两者都是通用的。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）假设它们。每个前沿模型以 Flash Attention 启用发货。

## 概念

![KV 缓存增长和 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存数学

每解码器层、每 token、每头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K 和 V
```

对于 7B 模型，32 层，32 头，d_head=128，fp16：

```
每 token 每层 = 2 * 128 * 2 = 512 bytes
每 token（32 层） = 16 KB
32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA，8 个 KV 头）：

```
每 token 每层 = 2 * 8 * 128 * 2 = 4096 bytes（4 KB）
32K 上下文 = 10.4 GB
```

那 10 GB 就是为什么 Llama 3 70B 在 128K 上下文、batch size 1 时需要大部分 40 GB A100 仅用于 KV 缓存。

**GQA 是 KV 缓存的胜利。** MHA 64 头将是 32 GB。MLA 进一步压缩。

### Flash Attention — 分块技巧

标准注意力：

```
S = Q @ K^T          (HBM 读，N×N，HBM 写)
P = softmax(S)       (HBM 读，HBM 写)
O = P @ V            (HBM 读，HBM 写)
```

三次 HBM 往返。在 H100 上，HBM 带宽是 3 TB/s；SRAM 是 30 TB/s。每次 HBM 往返是保持一切在芯片上的 10 倍减速。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 一次 HBM 往返。总内存占用从 `O(N²)` 减少到 `O(N)`。反向传递从正向传递重新计算一些值而非存储它们——又一个内存节省。

**数值技巧。** 运行 softmax 在 tile 之间维护 `(max, sum)`，因此最终归一化是精确的。不是近似——Flash Attention 计算与标准注意力位相同输出（modulo fp16 非结合性）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上加速 |
|------|------|----------|--------------|
| Flash 1 | 2022 | 平铺 SRAM 内核 | A100 上 2× |
| Flash 2 | 2023 | 更好并行性、因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步、FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级管道、软件 exp2 | 推理优先（初始仅前向） |

Flash 4 启动时仅前向传递。训练仍使用 Flash 3。Flash 4 的 GQA 和 varlen 支持待定（2026 年中）。

### 投机解码 — 另一个延迟优势

便宜模型提议 N 个 token。大模型并行验证所有 N。如果验证接受 k 个 token，你用一个前向传递支付 k 次生成。典型 k=3–5 在代码和散文上。

2026 年默认：
- **EAGLE 2 / Medusa。** 集成草稿头，共享验证器的隐藏状态。2–3× 加速，无质量损失。
- **带草稿模型的投机解码。** 消费硬件上 2–4× 加速。
- **Lookahead 解码。** 雅可比迭代；不需要草稿模型。利基但免费。

### 连续批处理

经典批处理推理：等待最慢序列完成，然后开始新批次。当短响应提前完成时浪费 GPU。

连续批处理（首次在 Orca 发货，现已在 vLLM、TensorRT-LLM、SGLang 中）：一旦旧序列完成就将新请求交换到批次。典型聊天工作负载 5–10× 吞吐量提升。

### PagedAttention — 作为虚拟内存的 KV 缓存

vLLM 的标志性特性。KV 缓存在 16-token 块中分配；页表将逻辑位置映射到物理块。让你在并行采样（束搜索、并行采样）之间共享 KV，热交换前缀用于提示缓存，以及碎片整理内存。相较于朴素连续分配 4× 吞吐量提升。

## 构建

见 `code/main.py`。我们实现：

1. 朴素的 `O(N²)` 增量解码器。
2. `O(N)` KV 缓存解码器。
3. 平铺 softmax，模拟 Flash Attention 的运行最大算法。

### 第一步：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每层、每头的列表中持续增长每 token K、V 向量。

### 第二步：平铺 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention 风格的 softmax(qK^T)V，带运行 max/sum。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

一次性 `softmax(qK) V` 的位相同输出，但在任何时候工作集是 `tile × d_head` 块，而非完整的 `N × d_head`。

### 第三步：在 100-token 生成上比较朴素 vs 缓存解码

计数注意力操作。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码打印两者。

## 使用

```python
# HuggingFace transformers 在 decoder-only generate() 上自动启用 KV 缓存。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 上用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的提示缓存是 2026 年的一个大胜利——相同的系统提示、few-shot 示例或长上下文文档在调用之间重用 KV。对于带重复工具提示的 agent 工作负载，提示缓存通常达到 5× 吞吐量提升。

## 交付

见 `outputs/skill-inference-optimizer.md`。该 skill 为新推理部署选择注意力实现、KV 缓存策略、量化和投机解码。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素和缓存解码器产生相同输出；注意操作数差异。
2. **中等。** 实现提示缓存：给定提示 P 和几个补全，运行一次前向传递过 P 以填充 KV 缓存，然后每个补全分支。测量 vs 每个重新编码 P 的加速。
3. **困难。** 实现一个玩具 PagedAttention：KV 缓存在固定 16-token 块中配空闲列表。当序列完成时，将其块返回到池。模拟 1,000 个不同长度聊天补全。比较内存碎片 vs 连续分配。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| KV 缓存 | "使解码变快的技巧" | 存储每个前缀 token 的 K 和 V；新 query 关注它们而非重新计算。 |
| HBM | "GPU 主内存" | 高带宽内存；H100 上 80 GB，B200 上 192 GB。~3 TB/s 带宽。 |
| SRAM | "芯片上内存" | 每个 SM 的快速内存，H100 上每 SM ~256 KB。~30 TB/s 带宽。 |
| Flash Attention | "平铺注意力内核" | 在 HBM 中不物化 N×N 的情况下计算注意力。 |
| 连续批处理 | "无等待批处理" | 序列完成时换出，新序列进入，不清空批次。 |
| PagedAttention | "vLLM 的标志性特性" | KV 缓存在固定块中分配，配页表；消除碎片。 |
| 提示缓存 | "重用长提示" | 在请求间缓存共享前缀的 KV；agent 成本大幅降低。 |
| 投机解码 | "草稿 + 验证" | 便宜草稿模型提议 token；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 发布说明（Dao-AILab，2026）](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 级管道和软件-exp2 技巧；阅读 repo README 获取本课提到的前向传递启动注意事项。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 本课引用的集成草稿方法的 EAGLE-1/2 论文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 本课在 EAGLE 旁边引用的 Medusa 方法。
- [vLLM 文档 — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16-token 块和页表设计的标准深度解析。