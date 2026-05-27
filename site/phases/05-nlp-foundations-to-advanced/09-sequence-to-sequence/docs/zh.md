# 序列到序列模型

> 两个 RNN 假装是翻译器。它们遇到的瓶颈正是注意力存在的原因。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 08（CNN + RNN 用于文本）、Phase 3 · 11（PyTorch 入门）
**耗时：** 约 75 分钟

## 问题

分类将可变长度序列映射到单一标签。翻译将可变长度序列映射到另一个可变长度序列。输入和输出使用不同词汇表，可能是不同语言，长度不保证相等。

seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的方案破解了它。两个 RNN。一个读取源句子并产生固定大小的上下文向量。另一个读取该向量并逐 token 生成目标句子。和第 08 课写的代码一样，只是粘合方式不同。

这值得研究，有两个原因。首先，上下文向量瓶颈是 NLP 中最具教学意义的失败。它激发了注意力机制和 Transformer 的所有优点。其次，训练方案（教师强制、计划采样、推理时的束搜索）仍然适用于包括 LLM 在内的每个现代生成系统。

## 概念

**编码器（Encoder）。** 一个读取源句子的 RNN。它的最终隐藏状态是**上下文向量**——整个输入的固定大小摘要。理论上只丢失源。

**解码器（Decoder）。** 另一个从上下文向量初始化的 RNN。每一步它接受之前生成的 token 作为输入，生成目标词汇表上的分布。采样或 argmax 选择下一个 token。反馈回去。重复直到产生 `<EOS>` token 或达到最大长度。

**训练：** 每步解码器的交叉熵损失，在序列上求和。通过两个网络进行标准的时间反向传播。

**教师强制（Teacher forcing）。** 训练时，解码器在步 `t` 的输入是位置 `t-1` 的**真实** token，而不是解码器自己之前的预测。这稳定了训练；没有它，早期错误会级联，模型永远学不会。推理时，你必须用模型的预测，所以总有一个训练/推理分布差距。这个差距叫做**曝光偏差（exposure bias）**。

**瓶颈。** 编码器学到的关于源的一切必须压缩进那个单一的上下文向量。长句子丢失细节。罕见词模糊。重新排序（chat noir vs. black cat）必须记住，而不是计算出来。

注意力（第 10 课）通过让解码器看到*每个*编码器隐藏状态而非只有最后一个来修复这个问题。这就是全部卖点。

## 构建

### 步骤 1：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 形状为 `[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden` 形状为 `[1, batch, hidden_dim]`——最后一步。第 08 课说"池化 outputs 用于分类"。这里我们保留最后隐藏状态作为上下文向量，忽略逐步 outputs。

### 步骤 2：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器一次调用一步。输入：一批单个 token 和当前隐藏状态。输出：下一个 token 的词汇表 logits 和更新后的隐藏状态。

### 步骤 3：带教师强制训练的循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得指出的旋钮。`ignore_index=0` 跳过填充 token 的损失。`teacher_forcing_ratio` 是每步使用真实 token 与模型预测的概率。从 1.0（全教师强制）开始，在训练过程中退火到约 0.5 以缩小曝光偏差差距。

### 步骤 4：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码每步选择最高概率 token。它可能会走偏：一旦你提交了一个 token，你就无法收回。**束搜索（Beam search）** 保持 top-`k` 个部分序列存活，并在最后选择得分最高的完整序列。束宽 3-5 是标准。

### 步骤 5：瓶颈演示

在玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损地记忆 40 token 的输入。每个编码器步骤中都存在信息，但解码器只看到最后状态。注意力直接修复了这个问题。

## 使用

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供在数十亿 token 上训练的完整编码器-解码器模型（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器用 Transformer 替代了 RNN。每个块内部机制不同，但高层形态（编码器、解码器、逐 token 生成）与 2014 年 seq2seq 论文相同。

### 何时仍然使用基于 RNN 的 seq2seq

对于新项目几乎从不。特定例外：

- 流式翻译，你以有界内存逐 token 消费输入。
- 设备上文本生成，Transformer 内存成本过高。
- 教学。理解编码器-解码器瓶颈是理解 Transformer 为何胜出的最快路径。

### 曝光偏差及其缓解方法

- **计划采样（Scheduled sampling）。** 在训练期间退火教师强制比例，让模型学会从自己的错误中恢复。
- **最小风险训练。** 在句子级 BLEU 分数而非 token 级交叉熵上训练。更接近你实际想要的。
- **强化学习微调。** 用指标奖励序列生成器。用于现代 LLM RLHF。

三者仍然适用于基于 Transformer 的生成。

## 交付

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计序列到序列流水线。
phase: 5
lesson: 09
---

给定任务（翻译、摘要、改写、问题重写），输出：

1. 架构。预训练 Transformer 编码器-解码器（BART、T5、mBART、NLLB）是默认。只有在特定约束下才用基于 RNN 的 seq2seq。
2. 起始检查点。命名它（`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`）。将检查点与任务和语言覆盖匹配。
3. 解码策略。贪心用于确定性输出，束搜索（宽度 4-5）用于质量，带温度的采样用于多样性。一句话说明理由。
4. 上线前要验证的一个失败模式。曝光偏差表现为较长输出上的生成漂移；取第 90 百分位长度处的 20 个输出肉眼检查。

当用户有少于 100 万平行样本时，拒绝推荐从零训练 seq2seq。标记任何为用户面向内容使用贪心解码的流水线为脆弱（贪心会重复和循环）。
```

## 练习

1. **简单。** 实现玩具复制任务。在输入等于输出的输入-输出对上训练 GRU seq2seq。测量长度 5、10、20 处的准确率。复现瓶颈。
2. **中等。** 添加束宽为 3 的束搜索解码。在小型平行语料库上用 BLEU 与贪心比较。记录束搜索在哪里胜出（通常是最后 token）在哪里没有区别。
3. **困难。** 在 10k 对改写数据集上微调 `facebook/bart-base`。将微调模型的束-4 输出与基模型在留出输入上比较。报告 BLEU 并挑选 10 个定性示例。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 编码器 | 输入 RNN | 读取源。产生逐步隐藏状态和最终上下文向量。 |
| 解码器 | 输出 RNN | 从上下文向量初始化。逐个生成目标 token。 |
| 上下文向量 | 摘要 | 最终编码器隐藏状态。固定大小。注意力解决的就是这个瓶颈。 |
| 教师强制 | 使用真实 token | 训练时喂入真实的前一个 token。稳定学习。 |
| 曝光偏差 | 训练/测试差距 | 在真实 token 上训练的模型从未练习过从自己的错误中恢复。 |
| 束搜索 | 更好的解码 | 在每步保持 top-k 个部分序列存活，而不是贪心提交。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) —— 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) —— 引入了 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力论文。本课之后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) —— 可构建的 seq2seq + 注意力代码。