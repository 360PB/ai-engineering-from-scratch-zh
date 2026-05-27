# 词嵌入——Word2Vec 从零开始

> 观其伴，知其义。训练一个浅层网络在这个思想上训练，几何性质随之涌现。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 02（BoW + TF-IDF）、Phase 3 · 03（反向传播从零开始）
**耗时：** 约 75 分钟

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词。它不知道它们几乎表示同一个事物。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列出同义词来掩盖这一点，但这在罕见词、专业术语以及每种你没想到的语言上都会失败。

你需要一个表示方法，让 `dog` 和 `puppy` 在空间中彼此靠近。`king - man + woman` 落在 `queen` 附近。在 `dog` 上训练的模型能够免费将一些信号迁移到 `puppy`。

Word2Vec 给了我们这个空间。两层神经网络，万亿 token 的训练运行，2013 年发表。架构简单得近乎尴尬。结果重塑了 NLP 十年。

## 概念

**分布假说（Distributional hypothesis）**（Firth, 1957）："观其伴，知其义。"如果两个词出现在相似的上下文中，它们很可能意思相近。

Word2Vec 有两种变体，都利用了这个思想。

- **Skip-gram。** 给定中心词，预测周围词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（连续词袋）。** 给定周围词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练较慢但更好地处理罕见词。它成为默认选项。

网络有一个隐藏层，没有非线性。输入是对词汇表的 one-hot 向量。输出是对词汇表的 softmax。训练完成后，丢掉输出层。隐藏层权重就是嵌入向量。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          这就是嵌入向量
```

关键技巧：对 10 万词做 softmax 开销大得无法承受。Word2Vec 使用**负采样（negative sampling）**将其转化为二元分类任务。预测"这个上下文词是否出现在这个中心词附近，是还是否"。每个训练对采样几个负（非共现）词，而不是对整个词汇表计算 softmax。

## 构建

### 步骤 1：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内每个（中心词，上下文词）对都是一个正训练样本。

### 步骤 2：嵌入表

两个矩阵。`W` 是中心词嵌入表（你保留的那个）。`W'` 是上下文词表（通常丢弃，有时与 `W` 平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词汇表 10k 和维度 100 是现实的；教学用词汇 50 × 维度 16 足以看到几何性质。

### 步骤 3：负采样目标函数

对于每个正对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负样本。训练模型使得正样本的点积 `W[center] · W'[context]` 高，负样本的点积低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式：正对的逻辑损失（希望 sigmoid 接近 1）加上负对的逻辑损失（希望 sigmoid 接近 0）。梯度流向两个表。原始论文有完整推导；如果想牢记，用铅笔和纸走一遍。

### 步骤 4：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上训练足够的轮次后，共享上下文的词具有相似的中心嵌入。在玩具语料库上，你只能微弱地看到这个效果。在数十亿 token 上，效果显著。

### 步骤 5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300d Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是皇室。是因为向量 `(king - man)` 捕捉到了类似"皇室"的东西，把它加到 `woman` 上就落在了皇室-女性区域。

## 使用

从零写 Word2Vec 是教学。生产 NLP 用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

对于真实工作，你几乎从不自己训练 Word2Vec。你下载预训练向量。

- **GloVe** —— 斯坦福的共现矩阵分解方法。50d、100d、200d、300d 检查点。泛化覆盖好。第 04 课专门讲 GloVe。
- **fastText** —— Facebook 的 Word2Vec 扩展，用字符 n-gram 做嵌入。处理未登录词——通过组合子词。第 04 课。
- **Google News 预训练 Word2Vec** —— 300d，300 万词词汇表，2013 年发布。至今每日仍在下载。

### 2026 年 Word2Vec 仍然胜出的场景

- 轻量级领域专用检索。在笔记本上花一小时在医学摘要上训练，获得任何通用模型都捕捉不到的专用向量。
- 类比风格特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减去它以获得性别中立轴。公平性研究中仍在使用。
- 可解释性。100d 小到可以用 PCA 或 t-SNE 绘制并实际看到聚类形成。
- 任何需要无 GPU 在设备上运行推理的地方。Word2Vec 查找就是单行查找。

### Word2Vec 失败的地方

多义词之墙。`bank` 有一个向量。`river bank` 和 `financial bank` 共享它。`table`（电子表格 vs. 家具）共享它。下游分类器无法从向量中区分这些语义。

上下文嵌入（ELMo、BERT、之后的每个 Transformer）通过为每个词根据周围上下文产生不同向量解决了这个问题。这就是从 Word2Vec 到 BERT 的跳跃：从静态到上下文。Phase 7 涵盖 Transformer 部分。

另一个失败是未登录词问题。Word2Vec 从未见过 `Zoomer-approved` 如果它不在训练数据中。没有回退。fastText 用子词组合修复了这个问题（第 04 课）。

## 交付

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: 检查 word2vec 模型。运行类比、找近邻、诊断质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你探测训练好的词嵌入以验证它们正常工作。给定一个 `gensim.models.KeyedVectors` 对象和词汇表，你运行：

1. 三个规范类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦相似度。
2. 五个近邻测试，用户提供领域专用词。打印 top-5 近邻及其余弦相似度。
3. 一个对称性检查。`similarity(a, b) == similarity(b, a)` 在浮点精度范围内。
4. 一个退化检查。如果任何嵌入的范数低于 0.01 或高于 100，模型有训练 bug。标记它。

不要仅凭类比准确率就宣称模型好。类比基准可以作弊且不能迁移到下游任务。建议同时进行内在评估和下游评估。
```

## 练习

1. **简单。** 在小型语料库（20 句关于猫和狗的句子）上运行训练循环。200 轮后，验证 `nearest(vocab, W, W[vocab["cat"]])` 在 top-3 中返回 `dog`。如果没有，增加轮次或词汇量。
2. **中等。** 添加高频词下采样。频率高于 `10^-5` 的词按与其频率成正比的概率从训练对中丢弃。测量对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练模型。计算两个偏差轴：`he - she` 和 `doctor - nurse`。将职业词投影到两个轴上。报告哪些职业偏差最大。这是公平性研究员使用的探测类型。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 词嵌入 | 词即向量 | 从上下文学习的密集低维（通常 100-300）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，对罕见词更好。 |
| 负采样 | 训练捷径 | 用对 `k` 个随机词的二元分类替代对完整词汇表的 softmax。 |
| 静态嵌入 | 每词一向量 | 不论上下文，向量相同。在多义词上失败。 |
| 上下文嵌入 | 上下文敏感向量 | 根据周围词，每个出现产生不同向量。Transformer 产生的就是这种。 |
| OOV | 未登录词 | 模型训练时未见过的词。Word2Vec 无法为这些词生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) —— 负采样论文。简短且可读性强。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) —— 最清晰的梯度推导，如果原始论文的数学让你觉得吃力。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) —— 生产训练设置，实际能用。