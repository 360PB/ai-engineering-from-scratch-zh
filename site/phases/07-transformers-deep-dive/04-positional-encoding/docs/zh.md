# 位置编码 — 正弦、RoPE、ALiBi

> 注意力是置换不变的。"The cat sat on the mat" 和 "mat the on sat cat the" 没有位置信号会产生相同输出。三种算法修复了它——每种对"位置"的含义有不同的押注。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 第 2 课（自注意力）、Phase 7 第 3 课（多头注意力）
**时长：** ~45 分钟

## 问题

缩放点积注意力是顺序盲的。注意力矩阵 `softmax(Q K^T / √d) V` 由成对相似度计算。随机打乱 `X` 的行，输出的行也被同样打乱。注意力内部没有任何东西关心位置。

在词袋模型中这不是 bug。对于语言、代码、音频、视频——任何顺序承载意义的东西——这是致命的。

修复方案：以某种方式将位置注入嵌入。三代答案：

1. **绝对正弦**（Vaswani 2017）。将位置的 `sin/cos` 加到嵌入上。简单，无需学习，在训练长度之外外推差。
2. **RoPE — 旋转位置嵌入**（Su 2021）。将 Q 和 K 向量旋转一个与位置成比例的角度。在点积中直接编码**相对**位置。2026 年占主导地位。
3. **ALiBi — 带线性偏差的注意力**（Press 2022）。完全跳过嵌入；根据距离在注意力分数上添加每头线性惩罚。优秀的长度外推。

截至 2026 年，实际上每个前沿开放模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦是历史。

## 概念

![绝对正弦 vs RoPE 旋转 vs ALiBi 距离偏差](../assets/positional-encoding.svg)

### 绝对正弦

预计算一个固定矩阵 `PE`，形状 `(max_len, d_model)`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后 `X' = X + PE[:N]` 在注意力之前。每维是不同频率的正弦曲线。模型学习从相位模式中读取位置。在 `max_len` 之外失败：当模型只见过位置 0–2047 时，没有任何东西告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（不是嵌入）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 默认值
```

对 key 在位置 `pos_k` 处应用相同旋转。点积 `q'_m · k'_n` 成为 `(m - n)` 的函数。也就是说：**注意力分数只取决于相对距离**，尽管旋转是基于绝对位置键的。漂亮的技巧。

扩展 RoPE：`base` 可以缩放（NTK-aware、YaRN、LongRoPE）以在不重新训练的情况下外推到更长上下文。Llama 3 以此方式从 8K 扩展到 128K 上下文。

### ALiBi

跳过嵌入技巧。直接偏置注意力分数：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是每头特定的斜率（例如 `1 / 2^(8·h/H)`）。更近的 token 得到提升；远的 token 受到惩罚。无训练时间成本。论文显示长度外推优于正弦，并在其原始训练长度上匹配 RoPE。

### 2026 年选哪个

| 变体 | 外推 | 训练成本 | 使用者 |
|------|------|----------|--------|
| 绝对正弦 | 差 | 免费 | 原始 transformer、早期 BERT |
| 学习绝对 | 无 | 微小 | GPT-2、GPT-3 |
| RoPE | 通过缩放良好 | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 优秀 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 优秀 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出是因为它插入注意力而不改变架构，编码相对位置，其 `base` 超参数为长上下文微调提供了干净的旋钮。

## 构建

### 第一步：正弦编码

见 `code/main.py`。4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将此加到嵌入矩阵。

### 第二步：RoPE 应用于 Q、K

RoPE 在原地操作 Q 和 K。对每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置 `m` 处的 Q 和位置 `n` 处的 K 应用相同函数。它们的点积在每个坐标对上获得 `cos((m-n)·θ_i)` 因子。注意力免费学习相对位置。

### 第三步：ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 在 softmax 之前加到注意力分数
```

将 `bias[h]` 加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵，然后 softmax。

### 第四步：验证 RoPE 的相对距离属性

取两个随机向量 `a, b`。用 `(pos_a, pos_b)` 旋转。然后用 `(pos_a + k, pos_b + k)` 旋转。两个点积必须在浮点误差范围内匹配。该属性是 RoPE 的全部意义——它对绝对偏移不变，只关心相对差距。

## 使用

PyTorch 2.5+ 在 `torch.nn.functional` 中提供 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力内核内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年长上下文技巧：**

- **NTK-aware 插值。** 将 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`，当从 4K 扩展到 16K+ 时。
- **YaRN。** 更智能的插值，在长上下文上保留注意力熵。Llama 3.1 128K 使用它。
- **LongRoPE。** 微软 2024 年方法，使用进化搜索来选取每维缩放因子。Phi-3-Long 使用它，并在使用部分引用。
- **位置插值 + 微调。** 只需将位置按扩展因子缩小，并微调 1–5B token。出人意料地有效。

## 交付

见 `outputs/skill-positional-encoding-picker.md`。该 skill 根据目标上下文长度、外推需求和训练预算为新模型选择编码策略。

## 练习

1. **简单。** 将正弦 `PE` 矩阵绘制为 `max_len=512, d=128` 的热力图。确认"条纹随维度索引增长而变宽"的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度 256 的序列上训练一个 tiny LM，然后在长度 1024 上测试，有和无缩放。测量困惑度。
3. **困难。** 在同一注意力模块中实现 ALiBi 和 RoPE。在长度 512 的复制任务上训练 4 层 transformer。在测试时外推到 2048。比较退化。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| 位置编码 | "告诉注意力顺序" | 添加到嵌入或注意力的任何编码位置的信号。 |
| 正弦 | "原始的一个" | 以几何频率添加到嵌入的 `sin/cos`；不能外推。 |
| RoPE | "旋转嵌入" | 通过与位置相关的角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "线性偏差技巧" | 添加 `-m·|i-j|` 到注意力分数；无需嵌入，外推优秀。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放器；增加以在推理时扩展上下文。 |
| NTK-aware | "RoPE 缩放技巧" | 重新缩放 `base` 以便在上下文扩展时不被挤压高频维度。 |
| YaRN | "花哨的那个" | 保留注意力熵的每维插值+外推。 |
| 外推 | "在训练长度之外工作" | 位置方案在训练中看到的 `max_len` 之后能否提供正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 微软方法，被 Phi-3-Long 使用，并在使用部分引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每个 RoPE 缩放方案（默认、线性、动态、YaRN、LongRoPE、Llama-3）的生产级实现。