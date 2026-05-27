# 用于文本的 CNN 和 RNN

> 卷积学习 n-gram。循环记住。两者都被注意力超越。两者在受限硬件上仍然重要。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 3 · 11（PyTorch 入门）、Phase 5 · 03（词嵌入）、Phase 4 · 02（卷积从零开始）
**耗时：** 约 75 分钟

## 问题

TF-IDF 和 Word2Vec 产生忽略词序的平坦向量。建立在它们之上的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时携带信号。

Transformer 到来之前，两类架构填补了这个空白。

**用于文本的卷积网络（TextCNN）。** 在词嵌入序列上应用 1D 卷积。宽度为 3 的滤波器是学习的 trigram 检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）以检测多尺度模式。最大池化到固定大小表示。平坦、并行、快。

**循环网络（RNN、LSTM、GRU）。** 一次处理一个 token，维护一个向前传递信息的隐藏状态。顺序、记忆、灵活输入长度。从 2014 年到 2017 年主导序列建模，然后注意力发生了。

本课构建两者，然后指出激励注意力的那个失败模式。

## 概念

**TextCNN**（Kim, 2014）。Token 被嵌入。宽度-`k` 的 1D 卷积在嵌入的连续 `k`-gram 上滑动滤波器，产生特征图。在该图上进行全局最大池化选取最强激活。对几个滤波器宽度进行最大池化输出拼接。送入分类头。

为什么有效。滤波器是可学习的 n-gram。最大池化是位置不变的，所以 "not good" 在评论开头或中间触发相同的特征。三个滤波器宽度各 100 个滤波器给你 300 个学习的 n-gram 检测器。训练是并行的；没有顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。在时间上共享 `W`、`U`、`b`。时间 `T` 的隐藏状态是整个前缀的摘要。对于分类，在 `h_1 ... h_T` 上池化（最大、平均或最后）。

普通 RNN 遭受梯度消失。**LSTM** 添加门来决定忘掉什么、存储什么、输出什么，通过长序列稳定梯度。**GRU** 将 LSTM 简化为两个门；用更少参数达到类似性能。

**双向 RNN** 运行一个前向 RNN 和一个后向 RNN，拼接隐藏状态。每个 token 的表示看到左右上下文。对标注任务至关重要。

## 构建

### 步骤 1：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴作为通道。池化输出是固定大小的，与输入长度无关。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

在序列上最大池化，而不是最后状态池化。对于分类，最大池化通常优于取最后隐藏状态，因为长序列末尾的信息往往主导最后状态。

### 步骤 3：梯度消失演示（直觉）

没有门控的普通 RNN 无法学习长期依赖。考虑一个玩具任务：预测 token `A` 是否出现在序列中任意位置。如果 `A` 在位置 1 而序列有 100 个 token 长，损失的梯度必须通过 99 次循环权重的乘法反向传播。如果权重小于 1，梯度消失。如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 用一个细胞状态修复这个问题，它只在网络中流动，通过加性交互（遗忘门用它来缩放，但梯度仍然沿着"高速公路"流动）。GRU 用更少参数做类似的事情。两者都能在 100+ 步序列中稳定训练。

### 步骤 4：为什么这仍然不够

即使有 LSTM，三个问题仍然存在。

1. **顺序瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 个串行前向/后向步骤。无法在时间上并行化。
2. **编码器-解码器设置中的固定大小上下文向量。** 解码器只看到编码器的最后隐藏状态，压缩在整段输入上。长输入丢失细节。第 09 课直接涵盖这一点。
3. **远距离依赖准确率上限。** LSTM 优于普通 RNN，但仍难以在 200+ 步上传播特定信息。

注意力解决了所有三个。Transformer 完全丢弃了循环。第 10 课是转折点。

## 使用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已可投入生产。训练代码是标准的。

Hugging Face 提供预训练嵌入，你可以作为输入层插入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

当符合约束时使用的检查清单。

- **边缘/设备推理。** TextCNN 加 GloVe 嵌入比 Transformer 小 10-100 倍。如果你的部署目标是手机，这就是方案。
- **流式/在线分类。** RNN 一次处理一个 token；Transformer 需要完整序列。对于实时输入文本，LSTM 仍然胜出。
- **小型模型的基线。** 在新任务上快速迭代。在 CPU 上 5 分钟训练 TextCNN。
- **有限数据的序列标注。** BiLSTM-CRF（第 06 课）仍然是 1k-10k 标注句子级生产 NER 架构。

其他一切用 Transformer。

## 交付

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: 为给定约束集选择文本编码器架构。
phase: 5
lesson: 08
---

给定约束（任务、数据量、延迟预算、部署目标、算力预算），输出：

1. 编码器架构：TextCNN、BiLSTM、BiLSTM-CRF、Transformer 微调，或"使用预训练 Transformer 作为冻结编码器 + 小头"。
2. 嵌入输入：随机初始化、冻结 GloVe / fastText，或上下文 Transformer 嵌入。
3. 训练配方（5 行）：优化器、学习率、批量大小、轮次、正则化。
4. 一个监控信号。对于 RNN/CNN 模型：缺少注意力意味着错过远距离依赖；检查按长度准确率。对于 Transformer：学习率太高导致微调崩溃；检查训练损失。

当用户有少于约 500 条标注样本时，拒绝推荐微调 Transformer，除非他们展示了 TextCNN / BiLSTM 基线已 plateau。将边缘部署标记为需要架构优先于一切。
```

## 练习

1. **简单。** 在 3 类玩具数据集（你创造数据）上训练 TextCNN。验证滤波器宽度（2、3、4）平均优于单一宽度（3）的平均 F1。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和最后状态池化。在小数据集上比较；记录哪个池化胜出并假设原因。
3. **困难。** 构建 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课的 CRF 单独基线和 BERT 微调比较。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| TextCNN | 用于文本的 CNN | 在词嵌入上堆叠 1D 卷积，加全局最大池化。Kim (2014)。 |
| RNN | 循环网络 | 隐藏状态在每个时间步更新：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 添加输入/遗忘/输出门加细胞状态。在长序列上稳定训练。 |
| GRU | 更简单的 LSTM | 两个门而非三个。相似准确率，更少参数。 |
| 双向 | 双向 | 前向 + 后向 RNN 拼接。每个 token 看到其上下文两侧。 |
| 梯度消失 | 训练信号死亡 | 普通 RNN 中小于 1 的权重重复乘法使早期步梯度实际上为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) —— TextCNN 论文。八页。可读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) —— LSTM 论文。出人意料地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) —— 让 LSTM 对所有人可及的图解。