# 梯度检查点与激活重计算

> 反向传播会保存每一个中间激活。对于 70B 参数和 128K 上下文，每个 rank 需要 3 TB 的激活。检查点用 FLOPs 换内存：删掉激活，在需要时重新计算。问题是丢弃哪些片段，答案不是"全部丢弃"。

**类型：** Build（构建）
**语言：** Python（含 numpy，可选 torch）
**前置要求：** Phase 10 Lesson 04（从零训练 Mini-GPT）、Phase 10 Lesson 05（扩展与分布式）
**时长：** 约 70 分钟

## 问题背景

训练一个 transformer 需要为每一层存储每个可微分操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出和残差流。对于隐藏维度 `d`、序列长度 `L`、批次 `B` 的层，这大约是 `12 * B * L * d` 个浮点数每层。

对于 `d=8192, L=8192, B=1`，每层 800 MB（BF16）。一个 64 层模型需要 51 GB 的激活——这还没算上微批次大小、注意力-softmax 中间结果（每头 `L^2`）以及张量并行部分拷贝。

双重账本：BF16 权重加优化器状态可能勉强装进 80GB，但激活让总量超出。梯度检查点（又名激活重计算）是标准解决方案。丢弃大部分激活；在反向传播时重新运行前向传播来恢复它们。代价：额外 FLOPs。收益：按检查点片段数与总层数之比节省内存。

简单做，检查点每步大约多花 33% 的前向传播 FLOPs。做得好——按 Korthikanti 等人的"智能选择"进行选择性检查点——可以在 FLOPs 开销小于 5% 的情况下节省 5 倍内存。加上 FP8 矩阵乘法、FSDP 卸载和专家并行 MoE，这就非常重要了：你既负担不起内存，也负担不起浪费的计算。

## 核心概念

### 反向传播实际需要什么

`output = layer(input)`。反向传播想要 `grad_input` 和 `grad_params`。计算它们需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 一些激活值的导数中间结果（ReLU/GELU/softmax 的导数依赖于激活值）

前向传播自动在 autograd 图中存储这些。每一个 `tensor.retain_grad()` 和每个需要其输入的操作都保留一个引用。

### 朴素全量检查点

将网络分成 `N` 段。前向传播时，只存储每个段的*输入*。当反向传播需要中间结果时，重新运行该段的前向传播来实例化它们，然后求导。

示例：将 32 层 transformer 切分成每段 1 层的 32 个段。

- 内存：32 个层输入（小）vs 32 ×（每层激活体积）（大）。
- 额外计算：每段 1 次额外前向，即总前向 FLOPs 的约 33%（因为反向是前向的 2 倍，完整步骤变为 1 + 1 + 2 = 4 而非 1 + 2 = 3）。

这是原始 Chen 等人 2016 的配方：每 `sqrt(L)` 层设置一个检查点，以平衡内存和计算。对于 L=64，那是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

不是所有激活代价都一样。注意力 softmax 输出是 `B*L*L*heads`，随序列长度呈二次增长。FFN 隐藏激活是 `B*L*4d`，呈线性增长。对于长序列，softmax 占主导。

选择性检查点保留廉价存储的激活（线性投影、残差），只重计算昂贵的激活（注意力）。用最少 FLOPs 重计算，但节省 O(L^2) 的内存。

Megatron-Core 将此实现为"选择性"激活重计算。用于大多数 2024 年后的前沿训练运行。

### 卸载

重计算的替代方案：在前向和反向传播之间将激活发送到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重实例化成本时有益。混合策略很常见：对某些层检查点，对其他层卸载。

FSDP2 将卸载作为一等公民选项。当 GPU 受内存瓶颈但 CPU-GPU 传输有余裕时，卸载表现突出。

### 重计算成本模型

每步朴素检查点（每 `k` 层中有 `L` 层）的 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 段内每层一次额外前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性检查点只重计算注意力核，而非整层：

```
flops_recompute_selective = L * f_attention ≈ L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活体积：`A`。对于 `L` 层，总激活内存：`L * A`。

全量检查点（段大小 1）：只存储 `L * input_volume`（对于标准 transformer 约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层检查点：存储 `L/k * A` 加上活跃段内 `k-1` 层的量。

当 `k = sqrt(L)` 时，内存和重计算成本都随 `sqrt(L)` 缩放——对均一层代价的最优权衡。

### 何时不检查点

- 流水线阶段内部正在飞行的最内层。它们无论如何都要完成。
- 如果首层和末层主导阶段的计算，则不检查点（在 transformer 中很少见）。
- 已使用 FlashAttention 的注意力核——Flash 已经快速重计算了 softmax，所以额外的层间检查点在此之上添加很少。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装一段。PyTorch 只存储 `input`，在反向传播时重计算其他所有内容。

2. **装饰器方式：** 标记层为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重计算：** 自己编写反向传播，调用一个重复前向传播的 `recompute_forward`（使用存储的输入）。

三者给出相同的功能结果。包装器是标准用法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点输入必须在重计算时聚集或重新分散；处理通信成本。
- **流水线并行：** 典型模式是对每个流水线阶段的前向传播做检查点，以便反向顺序的微批次可以复用激活内存。
- **FP8 重计算：** 重计算期间更新的 amax 历史必须与原始前向传播的相匹配，否则 FP8 缩放会漂移。大多数框架会快照缩放值。

## 构建

### 步骤 1：带分段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤 2：朴素反向传播（需要所有激活）

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤 3：每 k 层检查点

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤 4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤 5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤 7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch 中的标准包装器。包装一个函数；只存储输入，在反向传播时重计算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式。2024 年后前沿训练的标准配置。
- **FSDP2 卸载**：`module.to_empty(device="cpu")` 配合 FSDP2 中的 `offload_policy`，将激活分片到 CPU 而非重计算。
- **DeepSpeed ZeRO-Offload**：优化器状态和激活的 CPU 卸载，与检查点互补。

## 发布

本课生成 `outputs/prompt-activation-recompute-policy.md`——一份提示词，接收你的模型配置（层数、隐藏维度、序列、批次）和可用 GPU 内存，输出逐层重计算策略（无 / 选择性 / 全量 / 卸载）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（全量激活）对比 `model_forward_checkpointed` + `model_backward_checkpointed`（分段落）。参数梯度必须在机器精度下完全相同。

2. 扫描段大小 `k` 从 1 到 `L`。绘制 FLOPs 开销和内存。找到曲线拐点。

3. 实现选择性检查点：存储注意力模块输入，但不存储其中间结果。对于 seq=8192 的 32 层模型，测量与全层层检查点相比的 FLOPs 开销。

4. 添加卸载。将段输入保存到模拟的"CPU 缓冲区"（一个单独的列表）。测量"PCIe 带宽"为 bytes/time，找到卸载和重计算之间的盈亏平衡点。

5. 对真实 PyTorch transformer 有无 `torch.utils.checkpoint` 进行基准测试。通过 `torch.cuda.max_memory_allocated` 测量内存和每步时间。

## 关键术语

| 术语 | 别人怎么说 | 实际含义 |
|------|-----------|----------|
| Gradient checkpointing | "通过重做前向传播来节省内存" | 只存储段输入；在反向传播时重计算中间结果以获得梯度支持张量 |
| Activation recomputation | "等同于检查点" | 同一技术的高性能计算风格名称 |
| Segment size (k) | "每检查点多少层" | 丢弃其中间结果并一起重实例化的层数 |
| Selective checkpointing | "Korthikanti 的技巧" | 只重计算存储代价昂贵的激活（注意力 softmax）；保留廉价的 |
| Full checkpointing | "朴素版本" | 重计算每段每层的中间结果 |
| Block checkpointing | "粗粒度" | 检查点整个 transformer 块；最粗粒度 |
| FLOP overhead | "计算税" | 每步额外 FLOPs = 重计算 FLOPs / (前向 + 反向 FLOPs)；朴素 33%，选择性 5% |
| Activation offload | "转移到 CPU" | 在前向→反向传播之间将激活移到 CPU RAM；重计算的替代方案 |
| sqrt-L rule | "经典最优" | 对于均一层代价，最优检查点间隔为 sqrt(L) 层 |
| Attention-softmax volume | "O(L^2) 问题" | L^2 × heads × batch 浮点数；在长上下文时主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 — "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) — 形式化梯度检查点的原始论文
- [Korthikanti et al., 2022 — "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) — 选择性激活重计算及正式成本分析
- [Pudipeddi et al., 2020 — "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) — 通过逆序重实例化实现恒定内存的替代方法
- [Ren et al., 2021 — "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) — 大规模激活卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) — 标准 API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) — 选择性、全量和块模式