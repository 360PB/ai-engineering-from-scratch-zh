# 注意力变体 — 滑动窗口、稀疏、微分

> 全注意力是一个圆。每个 token 看到每个 token，内存付出代价。四个变体弯曲圆的形状并恢复一半成本。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 第 2 课（自注意力）、Phase 7 第 3 课（多头）、Phase 7 第 12 课（KV 缓存 / Flash Attention）
**时长：** ~60 分钟

## 问题

全注意力的成本在序列长度上是 `O(N²)` 内存和 `O(N²)` 计算。对于 128K 上下文的 Llama 3 70B，那是每层 160 亿注意力条目，乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 激活内存但不改变算术成本——每个 token 仍然关注每个其他 token。

三类变体改变注意力矩阵本身的拓扑：

1. **滑动窗口注意力（SWA）。** 每个 token 仅关注固定窗口的邻居，而非完整前缀。内存和计算降至 `O(N · W)` 其中 W 是窗口。Gemma 2/3、Mistral 7B 前几层、Phi-3-Long。
2. **稀疏 / 块注意力。** 仅选择特定 pairs `(i, j)` 计分；其余强制为零权重。Longformer、BigBird、OpenAI 稀疏 transformer。
3. **微分注意力。** 用单独的 Q/K 投影计算两个注意力图，相减。杀死"注意力 sink"，将权重渗入前几个 token。微软的 DIFF Transformer（2024）。

它们共存。2026 年前沿模型经常混合它们：大多数层是 SWA-1024，每第五层是全局完整注意力，少数是清理检索的微分头。Gemma 3 的 5:1 SWA 到全局比率是当前教科书默认。

## 概念

### 滑动窗口注意力（SWA）

每个在位置 `i` 的 query 仅关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）中的位置。窗口外的 token 在分数矩阵中得到 `-inf`。

```
完整因果：           滑动窗口 (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵期望有 1024 × 8192 非零行——8× 减少。

**KV 缓存在 SWA 下缩小。** 每层只需保留最后 `W` 个 token 的 K 和 V。对于 Gemma-3ish 配置（1024 窗口，128K 上下文），KV 缓存减少 128×。

**质量成本。** 仅 SWA transformer 努力进行长程检索。修复：SWA 层与全注意力层交错。Gemma 3 使用 5:1 SWA:全局。Mistral 7B 使用因果-SWA 堆栈，其中信息通过重叠窗口"向前流动"——每层有效感受野扩展 `W`，L 层后模型可以关注 `L × W` token。

### 稀疏 / 块注意力

提前选择 `N × N` 稀疏模式。三种标准形状：

- **局部 + 步幅（OpenAI 稀疏 transformer）。** 关注最后 `W` token 加之前每隔 `stride` 的 token。捕获局部和长程，`O(N · √N)` 计算。
- **Longformer / BigBird。** 局部窗口 + 一小组全局 token（例如 `[CLS]`）关注所有人并被所有人关注 + 随机稀疏链接。经验 2× 上下文在匹配质量。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要；在内核级别跳过零块。FlashAttention 兼容。

稀疏注意力是一个内核工程故事。数学简单（掩码分数矩阵）；胜利来自从不将零条目加载到 SRAM。FlashAttention-3 和 2026 年 FlexAttention API 在 PyTorch 中使自定义稀疏模式成为一等公民。

### 微分注意力（DIFF Transformer，2024）

常规注意力有"注意力 sink"问题：softmax 强制每行和为 1，因此没有特别想关注的 token 将权重倾倒在第一个 token（或前几个）上。这窃取了应该进入真实内容的容量。

微分注意力通过计算**两个**注意力图并相减来修复：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是学习的标量（通常 0.5–0.8）。A1 捕获真实内容权重；A2 捕获 sink。相减取消 sink，将权重重新分配给相关 token。

报告结果（微软 2024）：困惑度低 5–10%，在相同训练长度下有效上下文长 1.5–2×，针尖检索更锐利。

### 变体比较

| 变体 | 计算 | KV 缓存 | 质量 vs 全 | 生产使用 |
|------|------|----------|------------|---------|
| 全注意力 | O(N²) | 每层 O(N) | 基线 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | -0.1 ppl，与全局层配合好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步幅稀疏 | O(N·√N) | 混合 | 类似于 SWA | OpenAI 稀疏 transformer、Longformer |
| BigBird（局部 + 全局 + 随机） | 近似 O(N) | 混合 | 在 2× 上下文匹配全 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃分数) | O(N) | 在 0.05 ppl 内 | DeepSeek-V3.2，2025 |
| 微分 | O(2·N²) | O(2N) | -5 到 -10% ppl | DIFF Transformer，2026 年初模型 |

## 构建

见 `code/main.py`。我们实现因果掩码比较器，在玩具序列上并排显示全、SWA、局部+步距和微分注意力。

### 第一步：全因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

第 7 课的基线。下三角；对角线上方权重为零。

### 第二步：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。对于 `window >= n`，你恢复全因果注意力。对于 `window = 1`，每个 token 仅关注自己。

### 第三步：局部 + 步幅稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集局部窗口加每隔 `stride` 的 token 回溯到序列开始。附加层感受野在对数步骤中增长。

### 第四步：微分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，用学习的混合系数相减。在代码中我们比较单 vs 微分的注意力-sink 热力图，看着 sink 崩溃。

### 第五步：KV 缓存大小

打印每个变体在 `N = 131072` 时每层缓存大小。SWA 和稀疏变体减少 10–100×。微分翻倍。明智地支付你的内存账单。

## 使用

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 以 5:1 混合 SWA（窗口=1024）和全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这编译为自定义 Triton 内核。对于常见模式在 FlashAttention-3 速度的 10% 以内，掩码函数是一个 Python 可调用对象。

**何时选每个：**

- **纯全注意力** — 每层高达 ~16K 上下文，或当检索质量至关重要时。
- **SWA + 全局混合** — 长上下文（>32K），训练和推理内存受限。2026 年 32K 以上的默认。
- **稀疏块注意力** — 自定义内核、自定义模式。保留用于专门工作负载（检索、音频）。
- **微分注意力** — 任何注意力-sink 污染有害的工作负载（长上下文 RAG、针尖在干草堆中）。

## 交付

见 `outputs/skill-attention-variant-picker.md`。该 skill 根据目标上下文长度、检索需求和训练/推理计算配置为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 在每行最后 4 个 token 之外归零。验证 `window=n` 位相同地重现全因果注意力。
2. **中等。** 在第 7 课顶点项目之上实现因果 SWA（`window=1024`）。在 tinyshakespeare 上训练 1,000 步。验证损失 vs 全注意力退化了多少？峰值内存下降了多少？
3. **困难。** 在顶点项目中实现 Gemma-3 风格的 5:1 层混合（5 SWA，1 全局）。在匹配参数下比较损失、内存和生成质量 vs 纯 SWA 和纯全基线。
4. **困难。** 用每头学习的 `λ` 实现微分注意力。在合成检索任务上训练（一个针，2,000 个干扰项）。在匹配参数下测量检索准确率 vs 单注意力基线。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| 滑动窗口注意力（SWA） | "局部注意力" | 每个 query 关注其最后 `W` 个 token；KV 缓存缩小到 `O(W)`。 |
| 有效感受野 | "模型回看多远" | 在 L 层 SWA 堆栈（窗口 W）中，高达 `L × W` token。 |
| Longformer / BigBird | "局部 + 全局 + 随机" | 带几个始终关注的全局 token 的稀疏模式；早期长上下文方法。 |
| 原生稀疏注意力 | "DeepSeek 的内核技巧" | 学习块级稀疏度；在内核级别跳过零块同时保持质量。 |
| 微分注意力 | "两张图，一个相减" | DIFF Transformer：从第一个减去学习的 `λ` 倍第二个注意力图以取消注意力 sink。 |
| 注意力 sink | "权重渗入 token 0" | Softmax 归一化强制行和为 1；无信息 query 将权重倾倒在位置 0。 |
| FlexAttention | "掩码即 Python" | PyTorch 2.5+ API，将任意掩码函数编译为 FlashAttention 形状内核。 |
| 层类型混合 | "5:1 SWA 到全局" | 在堆栈中交错稀疏和全注意力层，以更低内存保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 标准的滑动窗口 + 全局 token 论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的局部+步距模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:全局混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 混合，窗口=1024，现在是教科书默认。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的学习稀疏注意力。
- [PyTorch — FlexAttention 博客和文档](https://pytorch.org/blog/flexattention/) — 使用部分的掩码即可调用模式 API 参考。