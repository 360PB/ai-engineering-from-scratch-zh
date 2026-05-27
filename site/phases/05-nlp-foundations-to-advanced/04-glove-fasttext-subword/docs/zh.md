# GloVe、FastText 与子词嵌入

> Word2Vec 每个词训练一个嵌入。GloVe 分解了共现矩阵。FastText 嵌入了部件。BPE 桥接到 Transformer。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 03（Word2Vec 从零开始）
**耗时：** 约 45 分钟

## 问题

Word2Vec 留下了两个悬而未决的问题。

首先，有一条平行研究路线直接分解共现矩阵（LSA、HAL），而不是做在线 skip-gram 更新。Word2Vec 的迭代方法本质上更好，还是两种方法处理计数的差异造成的？**GloVe** 回答了这个问题：带有经过深思熟虑设计的损失函数的矩阵分解，与 Word2Vec 持平或更好，且训练成本更低。

其次，两种方法都没有处理未见词的故事。`Zoomer-approved`、`dogecoin`、上周才创造的专有名词、罕见词根的每个屈折形式。**FastText** 用嵌入字符 n-gram 修复了这个问题：一个词是其各部分之和，包括词素，所以未登录词也能得到合理的向量。

第三，Transformer 出现后，问题又变了。词级词汇表在百万条目左右达到上限；真实语言比这更开放。**字节对编码（BPE）** 及其衍生方法通过学习覆盖所有内容的常用子词单元词汇表解决了这个问题。每个现代 LLM 的每个现代分词器都是子词分词器。

本课走遍三者，然后解释何时用哪个。

## 概念

**GloVe（Global Vectors）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 是词 `j` 在词 `i` 的上下文中出现的频率。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失加权，使高频词对不主导。搞定。

**FastText。** 一个词是其字符 n-gram 加词本身之和。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些分量向量之和。以 Word2Vec 方式训练。好处：未登录词（`whereupon`）由已知 n-gram 组合而成。

**BPE（字节对编码）。** 从单个字节（或字符）的词汇表开始。统计语料库中每个相邻对。合并最频繁的一对为新 token。重复 `k` 次。得到 `k + 256` 个 token 的词汇表，其中常用序列（`ing`、`tion`、`the`）是单个 token，罕见词被分解为熟悉的片段。每个句子都能分词成某个东西。

## 构建

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个值得指出的部分。权重函数 `f(x) = (x/x_max)^alpha` 降低高频词对（如 `(the, and)`）的权重，使它们不主导损失。最终嵌入是 `W`（中心）和 `W_tilde`（上下文）表之和。求和两者是一个已发表的技巧，通常比只用其中一个好。

### FastText：子词感知嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合（通常 3 到 6 个字符）表示。词嵌入是其 n-gram 嵌入之和。对于 skip-gram 训练，将这里替换 Word2Vec 用的单个向量即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未登录词，只要它的某些 n-gram 是已知的，你仍然能得到向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，所以两者在空间中靠近。

### BPE：学习子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻对。经过足够次迭代，常用子串（`low`、`est`、`tion`）变成单个 token，罕见词被干净地拆分。

真正的 GPT / BERT / T5 分词器学习 30k-100k 次合并。结果：任何文本都能分词成已知 ID 的有界长度序列，永无未登录词。

## 使用

实际上，你很少自己训练这些。你加载预训练检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Transformer 时代的 BPE 风格子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2 约定）。每个现代分词器都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 何时选哪个

| 场景 | 选择 |
|------|------|
| 预训练通用词向量，不需要处理未登录词 | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误/新词/形态丰富语言 | FastText |
| 进入 Transformer 的任何东西（训练或推理） | 用模型自带的分词器。别换。 |
| 从零训练自己的语言模型 | 先在语料库上训练 BPE 或 SentencePiece 分词器 |
| 用线性模型做生产文本分类 | 仍然是 TF-IDF。第 02 课。 |

## 交付

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: 为新的语言模型或文本流水线选择分词方案。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定任务和数据集描述，输出：

1. 分词策略（词级、BPE、WordPiece、SentencePiece、字节级）。一句话理由。
2. 词汇表大小目标（例如，英语 LM 用 32k，多语言用 64k-100k）。
3. 库调用及具体训练命令。命名库。引用参数。
4. 一个可复现性陷阱。分词器-模型不匹配是生产环境中最常见的静默 bug；指出哪一对必须一起使用。

当用户微调预训练 LLM 时，拒绝推荐训练自定义分词器。拒绝为任何针对生产推理的模型推荐词级分词。标记非英语/多脚本语料库需要 SentencePiece 加字节回退。
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠。你应该看到大量共享片段（`pla`、`lay`、`play`），这就是为什么 FastText 在形态变体间迁移性好。
2. **中等。** 扩展 `learn_bpe` 跟踪词汇表增长。绘制每个语料库字符的 token 数作为合并次数的函数。你应该看到起初压缩很快，之后趋近于约 2-3 字符每 token。
3. **困难。** 在莎士比亚全集中训练 1k 次合并的 BPE。比较常用词与罕见专有名词的分词。测量前后每词平均 token 数。写出让你惊讶的地方。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 共现矩阵 | 词-词频率表 | `X[i][j]` = 词 `j` 在词 `i` 周围窗口中出现的频率。 |
| 子词 | 词的片段 | 字符 n-gram（FastText）或学习到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并最频繁的相邻对，直到词汇表达到目标大小。 |
| OOV | 未登录词 | 模型从未见过的词。Word2Vec/GloVe 失败。FastText 和 BPE 处理它。 |
| 字节级 BPE | 在原始字节上做 BPE | GPT-2 的方案。词汇表从 256 字节开始，所以永无未登录词。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) —— GloVe 论文，七页，仍然是最好的损失推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) —— FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— 将 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) —— BPE、WordPiece 和 SentencePiece 在实践中的实际差异。