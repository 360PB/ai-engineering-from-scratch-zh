# 从零实现自注意力

> 注意力是一张查找表，每个词问："谁与我相关？"——然后学习答案。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3（深度学习核心），Phase 5 第 10 课（序列到序列）
**时长：** ~90 分钟

## 学习目标

- 仅使用 NumPy 从头实现缩放点积自注意力，包括 query/key/value 投影和 softmax 加权和
- 构建多头注意力层，分割头、并行计算注意力、拼接结果
- 追踪注意力矩阵如何捕获 token 之间的关系，并解释为什么除以 sqrt(d_k) 可防止 softmax 饱和
- 应用因果掩码，将双向注意力转换为自回归（解码器风格）注意力

## 问题

RNN 逐个 token 处理序列。当到达 token 50 时，token 1 的信息已经被压缩通过了 50 次压缩步骤。长距离依赖被压成固定大小的隐藏状态——这是一个没有任何 LSTM 门控能完全解决的瓶颈。

2014 年的 Bahdanau 注意力论文展示了修复方案：让解码器回看每个编码器位置，并决定当前步骤哪些是重要的。但它仍然嫁接在 RNN 上。2017 年的"Attention Is All You Need"论文提出了一个更尖锐的问题：如果注意力是**唯一的机制**呢？没有循环。没有卷积。只有注意力。

自注意力让序列中的每个位置在一次并行步骤中关注所有其他位置。这就是使 Transformer 快速、可扩展且占主导地位的原因。

## 概念

### 数据库查找类比

把注意力想象成一个软数据库查找：

```
传统数据库：
  Query: "法国的首都"  -->  精确匹配  -->  "巴黎"

注意力：
  Query: "法国的首都"  -->  与所有 key 相似度  -->  所有 value 的加权混合
```

每个 token 生成三个向量：
- **Query（Q）**："我在找什么？"
- **Key（K）**："我包含什么？"
- **Value（V）**："如果被选中，我提供什么信息？"

query 和所有 key 之间的点积产生注意力分数。高分意味着"这个 key 与我的 query 匹配"。这些分数加权 value。输出是 value 的加权求和。

### Q、K、V 计算

每个 token 嵌入通过三个学习权重矩阵投影：

```
输入嵌入（n 个 token 的序列，每个 d 维）：

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

三个权重矩阵：

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

投影：

  Q = X @ Wq    shape: (n, dk)      每个 token 的 query
  K = X @ Wk    shape: (n, dk)      每个 token 的 key
  V = X @ Wv    shape: (n, dv)      每个 token 的 value
```

直观上，对于一个 token：

```
             Wq
  x_i ------[*]------> q_i    "我在找什么？"
       |
       |     Wk
       +----[*]------> k_i    "我包含什么？"
       |
       |     Wv
       +----[*]------> v_i    "如果被选中，我提供什么？"
```

### 注意力矩阵

一旦你有了所有 token 的 Q、K、V，注意力分数形成一个矩阵：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- q1 对每个 key 的关注程度
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

每行：一个 token 对整个序列的注意力
```

### 为什么缩放？

点积随维度 dk 增长。如果 dk = 64，点积可能在几十的范围内，将 softmax 推入梯度消失的区域。修复方法：除以 sqrt(dk)。

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

这使值保持在 softmax 产生有用梯度的范围内。

### Softmax 将分数转化为权重

Softmax 将原始分数转换为每行上的概率分布：

```
q1 的原始分数：   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
注意力权重：   [0.52, 0.09, 0.07, 0.14, 0.08]   （和约为 ~1.0）
```

现在每个 token 有一套权重，说明对每个其他 token 关注多少。

### Value 的加权求和

每个 token 的最终输出是所有 value 向量的加权和：

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

对于 token 1：
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流程

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

单行公式：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 构建

### 第一步：从零实现 Softmax

Softmax 将原始 logits 转换为概率。减去最大值以保证数值稳定性。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 第二步：缩放点积注意力

核心函数。接受 Q、K、V 矩阵，返回注意力输出和权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 第三步：带学习投影的自注意力类

一个完整的自注意力模块，具有 Wq、Wk、Wv 权重矩阵，用类似 Xavier 的缩放初始化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 第四步：在句子上运行

为一个句子创建假嵌入，观察注意力权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("注意力权重（每行：该 token 看哪里）：\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 第五步：用 ASCII 热力图可视化注意力

将注意力权重映射为字符，快速可视化。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 使用

PyTorch 的 `nn.MultiheadAttention` 正是我们构建的功能，加上多头分割和输出投影：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"输入形状：            {X_torch.shape}")
print(f"输出形状：            {output.shape}")
print(f"注意力权重形状：    {attn_weights.shape}")
print(f"\n注意力权重（平均到头）：")
print(attn_weights[0].detach().numpy().round(3))
```

关键区别：多头注意力并行运行多个注意力函数，每个都有自己大小为 dk = d_model / n_heads 的 Q、K、V 投影，然后拼接结果。这让模型同时关注不同类型的关系。

## 交付

本课产出：
- `outputs/prompt-attention-explainer.md` — 通过数据库查找类比解释注意力的提示词

## 练习

1. 修改 `scaled_dot_product_attention` 以接受一个可选掩码矩阵，在 softmax 之前将某些位置设置为负无穷（这就是因果/解码器掩码的工作方式）
2. 从零实现多头注意力：将 Q、K、V 分割成 `n_heads` 块，在每个块上运行注意力，拼接，并通过最终权重矩阵 Wo 投影
3. 取两个不同长度的相同句子，用相同的 SelfAttention 实例运行，比较它们的注意力模式。什么变了？什么没变？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| Query（Q） | "问题向量" | 输入的学习投影，表示该 token 正在寻找什么信息 |
| Key（K） | "标签向量" | 学习投影，表示该 token 包含什么信息，与 query 匹配 |
| Value（V） | "内容向量" | 携带实际信息的学习投影，根据注意力分数聚合 |
| 缩放点积注意力 | "注意力公式" | softmax(QK^T / sqrt(dk)) @ V — 缩放防止高维时 softmax 饱和 |
| 自注意力 | "token 看到自己和他人" | Q、K、V 都来自同一序列的注意力，让每个位置关注每个其他位置 |
| 注意力权重 | "关注多少" | 根据缩放点积的 softmax 产生的位置概率分布 |
| 多头注意力 | "并行注意力" | 用不同投影运行多个注意力函数，然后拼接结果以获得更丰富的表示 |

## 延伸阅读

- [Attention Is All You Need（Vaswani et al., 2017）](https://arxiv.org/abs/1706.03762) — 原始 transformer 论文
- [The Illustrated Transformer（Jay Alammar）](https://jalammar.github.io/illustrated-transformer/) — 最佳视觉走查全架构
- [The Annotated Transformer（Harvard NLP）](https://nlp.seas.harvard.edu/annotated-transformer/) — 带解释的行对行 PyTorch 实现