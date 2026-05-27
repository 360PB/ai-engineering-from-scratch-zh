# 从零构建 Transformer — 顶点项目

> 十三课。一模型。无捷径。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 第 1–13 课。不要跳过。
**时长：** ~120 分钟

## 问题

你已经读了每篇论文。你已经实现了注意力、多头分割、位置编码、编码器和解码器块、BERT 和 GPT 损失、MoE、KV 缓存。现在让它们在一个真实任务上一起工作。

顶点项目：在字符级语言建模任务上端到端训练一个小型的纯解码器 transformer。它读莎士比亚。它生成新莎士比亚。它小到可以在笔记本电脑上在 10 分钟内训练。它正确到换入更大的数据集和更长时间训练能得到一个真正的 LM。

这是课程的"nanoGPT"。它不是原创的——Karpathy 的 2023 nanoGPT 教程是每个学生至少写一次的标准实现。我们提升形状并根据我们涵盖的内容重新调整它。

## 概念

![从零开始的 Transformer 块图](../assets/capstone.svg)

架构，注解：

```
输入 token (B, N)
   │
   ▼
token 嵌入 + 位置嵌入  ◀── 第 4 课（RoPE 选项）
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── 第 5 课
│  MultiHeadAttention (causal)      │  ◀── 第 3 课 + 07（因果掩码）
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 第 5 课
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head（绑定到 token 嵌入）
   │
   ▼
logits (B, N, V)
   │
   ▼
移位一交叉熵            ◀── 第 7 课
```

### 我们交付的内容

- `GPTConfig` — 一处配置所有超参数。
- `MultiHeadAttention` — 因果、批处理，带可选 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — pre-norm、残差包裹注意力 + FFN。
- `GPT` — 嵌入、堆叠块、LM head、generate()。
- 带 AdamW、余弦 LR、梯度裁剪的训练循环。
- 莎士比亚文本上的字符级分词器。

### 我们不交付的内容

- RoPE — 在第 4 课概念性实现。这里为简单起见使用学习位置嵌入。练习要求你换入 RoPE。
- 生成期间的 KV 缓存 — 每次生成步骤重新计算对完整前缀的注意力。更慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention — PyTorch 2.0+ 自动分派如果输入匹配；我们使用 `F.scaled_dot_product_attention`。
- MoE — 每块单个 FFN。你在第 11 课看到了 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，在 `tinyshakespeare.txt` 上训练 2,000 步的 4 层、4 头、d_model=128 GPT：

- 训练损失从约 4.2（随机）收敛到约 1.5，约 6 分钟。
- 采样输出看起来像莎士比亚：古语词、换行符、"ROMEO:" 这样的专有名词出现。
- 验证损失（保留文本最后 10%）紧密跟踪训练损失；在此规模/预算下无过拟合。

## 构建

本课使用 PyTorch。安装 `torch`（CPU 构建即可）。见 `code/main.py`。脚本处理：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 90/10 的训练/验证拆分。
- 支持硬件上 bf16 autocast 的训练循环。
- 训练完成后采样。

### 第一步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。微小词汇。适合 4 字节词汇表。无 BPE，无分词器戏剧。

### 第二步：模型

见 `code/main.py`。块是第 5 课的教科书——pre-norm、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数计数：~800K。

### 第三步：训练循环

获取长度-256 token 窗口的随机批次。前向。移位一交叉熵。反向。AdamW 步。日志。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第四步：采样

给定提示，反复前向，从 top-p logit 采样，附加，继续。500 token 后停止。

### 第五步：读输出

2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但是莎士比亚形状的。对于约 800K 参数和笔记本电脑上 6 分钟，明显的胜利。

## 使用

此顶点项目是一个参考架构。三个扩展将其交付到真实的东西：

1. **换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词汇表从 65 跳到约 50,000。模型容量需要相应扩展。
2. **在更大语料上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单个 A100 上 10B token 的 125M 参数 GPT 训练需要约 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习引导你完成每一步。

最终结果是 125M 参数 GPT 生成流利英语。不是前沿模型。但相同代码路径——只是更大——是 Karpathy、EleutherAI 和艾伦研究所在 2026 年训练研究检查点使用的。

## 交付

见 `outputs/skill-transformer-review.md`。该 skill 审查从零开始的 transformer 实现在所有 13 个前置课程上的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证训练模型的最终步验证损失低于 2.0。将 `max_steps` 从 2,000 更改为 5,000——验证损失继续改善吗？
2. **中等。** 将学习位置嵌入替换为 RoPE。在 `MultiHeadAttention` 内部将旋转应用于 Q 和 K。训练并验证验证损失至少一样低。
3. **中等。** 在采样循环中实现 KV 缓存。有和无缓存生成 500 token。墙钟应在笔记本电脑上改善 5–20×。
4. **困难。** 添加第二个 head 预测下一个加一 token（MTP——来自 DeepSeek-V3 的多 token 预测）。联合训练。有帮助吗？
5. **困难。** 将每块单个 FFN 替换为 4 专家 MoE。路由器 + top-2 路由。看在匹配激活参数下验证损失如何变化。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| nanoGPT | "Karpathy 的教程仓库" | 最小纯解码器 transformer 训练代码，约 300 行；标准参考。 |
| tinyshakespeare | "标准玩具语料" | ~1.1 MB 文本；2015 年以来每个字符-LM 教程使用它。 |
| 绑定嵌入 | "共享输入/输出矩阵" | LM head 权重 = token 嵌入矩阵的转置；节省参数，改善质量。 |
| bf16 autocast | "训练精度技巧" | 前向/反向用 bf16 运行，保持优化器状态在 fp32；2021 年以来的标准。 |
| 梯度裁剪 | "阻止尖峰" | 将全局梯度范数上限为 1.0；防止训练爆炸。 |
| 余弦 LR 日程 | "2020+ 默认" | LR 线性上升（预热）然后余弦衰减到峰值的 10%。 |
| MFU | "模型 FLOP 利用率" | 达到 FLOPs / 理论峰值；2026 年 40% 密集、30% MoE 是强值。 |
| 验证损失 | "保留损失" | 模型从未见过的数据的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer（Harvard NLP）](https://nlp.seas.harvard.edu/annotated-transformer/) — 带注解的实现经典。