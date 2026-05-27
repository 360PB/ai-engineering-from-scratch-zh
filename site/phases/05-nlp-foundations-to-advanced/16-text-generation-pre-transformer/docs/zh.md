# Transformer 前的文本生成——N-gram 语言模型

> 如果一个词令人惊讶，模型就不好。困惑度将惊讶变成数字。平滑使其有限。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 01（文本处理）、Phase 2 · 14（朴素贝叶斯）
**耗时：** 约 45 分钟

## 问题

在 Transformer 之前，在 RNN 之前，在词嵌入之前，语言模型通过计数来预测下一个词：它跟随前 `n-1` 个词有多频繁。"the cat" → "sat" 计数 47 次，"the cat" → "jumped" 12 次，"the cat" → "refrigerator" 0 次。归一化为概率分布。

这就是 N-gram 语言模型。它驱动了从 1980 年到 2015 年的每个语音识别器、每个拼写检查器和每个基于短语的机器翻译系统。当你需要便宜的设备语言建模时，它仍然在运行。

有趣的问题是如何处理未见的 N-gram。基于原始计数的模型对任何未见过的内容分配零概率，这是灾难性的，因为句子很长，几乎每个长句子都包含至少一个未见的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑是结果，现代深度学习继承了其经验传统。

## 概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常三元组用 3，四元组用 4）。从计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中未见过的任何 N-gram 得到零概率。2007 年对 Brown 语料库的研究发现，即使四元组模型也有 30% 的留出四元组在训练中未见。在不平滑的情况下，你无法在任何真实文本上评估。

**平滑方法，按复杂度排序：**

1. **拉普拉斯（加一）。** 每个计数加 1。简单，对稀有事件糟糕。
2. **Good-Turing。** 根据频率的频率将高频事件的概率质量重新分配给未见事件。
3. **插值。** 结合 N-gram、(n-1)-gram 等估计，带可调权重。
4. **回退。** 如果 N-gram 计数为零，回退到 (n-1)-gram。Katz 回退将其规范化。
5. **绝对折扣。** 从所有计数中减去固定折扣 `D`，重新分配给未见。
6. **Kneser-Ney。** 绝对折扣加上下阶模型的一个聪明选择：使用*连续概率*（词出现在多少个上下文中）而非原始频率。

Kneser-Ney 的洞察是深刻的。"San Francisco" 是一个常见二元组。Unigram "Francisco" 大多出现在 "San" 之后。朴素绝对折扣给 "Francisco" 高 unigram 概率（因为计数高）。Kneser-Ney 注意到 "Francisco" 只出现在一个上下文中，并相应降低其连续概率。结果：以 "Francisco" 结尾的新二元组得到适当的低概率。

**评估：困惑度（Perplexity）。** 在留出测试集上每个词的平均负对数似然的指数。越低越好。困惑度 100 意味着模型就像在 100 个词中均匀选择一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 构建

### 步骤 1：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词句子列表。输出是 N-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界。

### 步骤 2：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每个计数加 1。平滑但对未见事件过度分配质量，也伤害已知稀有事件。

### 步骤 3：Kneser-Ney（二元组，插值）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个运动部分。`continuation_prob` 捕捉"这个词出现在多少个不同上下文中？"（Kneser-Ney 创新）。`lambda_prev` 是折扣释放的质量，用于加权回退。最终概率是折扣主项加加权连续项。

### 步骤 4：用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率采样。每个种子总是给出不同输出。对于类似束搜索的输出，在每步取 argmax（贪心）并添加小随机性旋钮（温度）。

### 步骤 5：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，调优良好的 4 元组 KN 模型达到困惑度约 140。Transformer LM 在同一测试集上达到 15-30。差距约 10 倍。这就是该领域继续前进的原因。

## 使用

- **经典 NLP 教学。** 你能得到的最清晰的平滑、MLE 和困惑度暴露。
- **KenLM。** 生产 N-gram 库。在语音和 MT 系统中用作重评分器，低延迟很重要。
- **设备自动补全。** 键盘中的三元组模型。仍然如此。
- **基线。** 在声明你的神经 LM 好之前始终计算 N-gram LM 困惑度。如果你的 Transformer 没有大幅超越 KN，某处有问题。

## 交付

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: 在训练神经 LM 之前构建可复现的 N-gram 语言模型基线。
phase: 5
lesson: 16
---

给定语料库和目标用途（下一个词预测、重评分、困惑度基线），输出：

1. N-gram 阶数。通用英语用三元组，语料库大用四元组，语音重评分用五元组。
2. 平滑。修正 Kneser-Ney 是默认；拉普拉斯仅用于教学。
3. 库。生产用 `kenlm`，教学用 `nltk.lm`，只有为了学习才自己写。
4. 评估。留出困惑度，训练和测试集之间的分词一致。

拒绝报告在不同分词之间计算 perplexity 时比较的系统——perplexity 数字仅在相同分词下可比。标记测试集中的 OOV 率；KN 处理 OOV 不好，除非你在训练期间预留一个特殊 <UNK> token。
```

## 练习

1. **简单。** 在 1000 句莎士比亚语料库上训练三元组 LM。生成 20 句。它们局部可信但全局不连贯。这是经典演示。
2. **中等。** 在留出的莎士比亚分割上为你的 KN 模型实现困惑度。与拉普拉斯比较。你应该看到 KN 困惑度低 30-50%。
3. **困难。** 构建一个三元组拼写纠正器：给定拼写错误的词及其上下文，在 LM 下按上下文概率生成更正并排名。在 Birkbeck 拼写字典（公开）上评估。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| N-gram | 词序列 | `n` 个连续 token 的序列。 |
| 平滑 | 避免零 | 重新分配概率质量，使未见事件得到非零概率。 |
| 困惑度 | LM 质量指标 | 留出数据上的 `exp(-average log-prob)`。越低越好。 |
| 回退 | 回退到更短上下文 | 如果三元组计数为零，使用二元组。Katz 回退将其形式化。 |
| Kneser-Ney | N-gram 最佳平滑 | 绝对折扣 + 下阶模型的连续概率。 |
| 连续概率 | KN 特有 | 按词出现的上下文数量加权的 `P(w)`，而非原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) —— N-gram LM 和平滑的规范论述。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) —— 确定 Kneser-Ney 作为最佳 N-gram 平滑器的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) —— 原始 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) —— 快速生产 N-gram LM，2026 年仍用于延迟敏感应用。