# 词袋、TF-IDF 与文本表示

> 先计数，后思考。2026 年 TF-IDF 在定义明确的任务上仍然胜过嵌入向量。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 01（文本处理）、Phase 2 · 02（线性回归从零开始）
**耗时：** 约 75 分钟

## 问题

模型需要数字。你只有字符串。

每个 NLP 流水线都必须回答同一个问题：如何将可变长度的 token 流转换为分类器可以消费的固定大小向量。该领域最初给出的答案是最简单粗暴但能工作的方案——数单词，建向量。

这个向量承载了比任何嵌入模型更多的生产级 NLP。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP 基准测试的最初十年。2026 年的从业者仍然在窄范围分类任务上首先想到它。它快速、可解释，在只关心词出现与否的任务上往往与 4 亿参数嵌入模型没有区别。

本课从零构建词袋，然后是 TF-IDF。然后展示 scikit-learn 三行代码完成同样的工作。然后指出它的失败模式——正是这个失败模式让你转向嵌入向量。

## 概念

**词袋（Bag of Words, BoW）** 丢弃顺序。对每个文档，统计每个词汇表中单词出现的次数。向量长度是词汇表大小。位置 `i` 是单词 `i` 的计数。

**TF-IDF** 对 BoW 重新加权。出现在每个文档中的词没有信息量，所以降低权重。罕见跨文档但在单个文档中频繁出现的词是信号，所以提升权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档内词频，`df` 是文档频率（多少个文档包含该词），`N` 是总文档数。`log` 保持 ubiquitous 词的权重有界。

关键特性：两者都产生稀疏向量，具有可解释的轴。你可以直接查看训练好的分类器的权重，读出哪些词将文档推向某个类别。你无法对 768 维的 BERT 嵌入做到这一点。

## 构建

### 步骤 1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本课的 `code/main.py` 使用简化的小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词汇表中第一个词对应索引 0。惯例各不相同；scikit-learn 按字母顺序排序。

### 步骤 2：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词汇表索引。条目 `[i][j]` 是"文档 `i` 中单词 `j` 出现了多少次"。文档 1 的 `cat` 出现两次，因为它确实出现了两次。文档 0 的 `ran` 是零次，因为它确实没出现。

### 步骤 3：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得指出的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。末尾的 `+1` 确保出现在每个文档中的词仍有 IDF 1（不是 0），与 scikit-learn 默认值匹配。其他实现使用原始的 `log(N/df)`。两者都有效；平滑版本更友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在全部三个文档中，所以它的 IDF 低。`dog` 只出现在一个文档中，所以它的 IDF 高。向量是稀疏的（大多数条目很小），有区分力的词会凸显出来。

### 步骤 5：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

没有归一化的话，较长的文档会得到更大的向量，并在相似度得分中占主导地位。L2 归一化将每个文档放在单位超球面上。现在行之间的余弦相似度就是简单的点积。

## 使用

scikit-learn 提供生产级版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一次调用完成分词、词汇表和 BoW。`TfidfVectorizer` 添加 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万个文档，密集版本放不进内存；保持稀疏直到分类器要求密集。

改变一切的旋钮：

| 参数 | 效果 |
|------|------|
| `ngram_range=(1, 2)` | 包含二元组。通常能提升分类效果。 |
| `min_df=2` | 去掉出现在少于 2 个文档中的词。在脏数据上修剪词汇表。 |
| `max_df=0.95` | 去掉出现在超过 95% 文档中的词。在不使用硬编码停用词表的情况下近似停用词去除。 |
| `stop_words="english"` | scikit-learn 内置停用词表。任务相关——情感分析**不应该**去掉否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 而不是原始 `tf`。当一个词在一个文档中重复多次时有帮助。 |

### TF-IDF 仍然胜出的场景（2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词出现与否就是关键；语义细微差别不是。
- 低数据场景（数百个标注样本）。TF-IDF 加逻辑回归没有预训练成本。
- 任何需要低延迟的地方。TF-IDF 加线性模型以微秒级响应。嵌入一个文档通过 transformer 需要 10-100ms。
- 必须解释其预测的系统。检查分类器的系数。正向权重最高的词就是原因。

### TF-IDF 失败的场景

语义盲点失败。考虑以下两个文档：

- "The movie was not good at all."
- "The movie was excellent."

一个是差评。一个是好评。它们的 TF-IDF 重叠正好是 `{the, movie, was}`。词袋分类器必须死记硬背单词 `not` 在 `good` 附近会翻转标签。它可以在足够的数据上学会这一点，但永远不会像理解句法的模型那样优雅。

另一个失败：推理时的未登录词（out-of-vocabulary）。在 IMDb 评论上训练的 BoW 模型对 `Zoomer-approved` 毫无概念，如果该 token 从未出现在训练中。子词嵌入（第 04 课）处理这种情况。TF-IDF 不能。

### 混合方法：TF-IDF 加权嵌入

2026 年中等数据分类的务实默认：用 TF-IDF 权重作为词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入向量获得语义容量，从 TF-IDF 获得罕见词强调。分类器在聚合向量上训练。在约 5 万标注样本以下的情感、主题和意图分类任务上，这比单独的任何一个都好。

## 交付

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定文本分类任务，推荐 BoW、TF-IDF、嵌入或混合方法。
phase: 5
lesson: 02
---

你推荐文本向量化策略。给定任务描述，输出：

1. 表示方法（BoW、TF-IDF、Transformer 嵌入或混合）。用一句话解释原因。
2. 具体向量化器配置。命名库。引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 上线前要测试的一个失败模式。

当用户有少于 500 个标注样本时，拒绝推荐嵌入，除非他们展示了 TF-IDF 基线存在语义失败的证据。拒绝为情感分析移除停用词（否定词携带信号）。标记类别不平衡需要的不只是向量化器的改动。

示例输入："将 30k 个客户支持工单分类到 12 个类别。大多数工单 2-3 句话。英语。只有。需要可解释性用于审计日志。"

示例输出：

- 表示方法：TF-IDF。30k 样本不小；可解释性要求排除密集嵌入。
- 配置：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词因为类别关键词有时就是停用词（"not working" vs "working"）。
- 待测试的失败：验证 `min_df=3` 不会丢掉罕见类别关键词。运行 `get_feature_names_out` 按类别过滤后肉眼检查。
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分 1.0，词汇完全不同的文档得分 0.0。
2. **中等。** 给 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 产生 `n-gram` 的计数。测试 `n=2` 对 `["the", "cat", "sat"]` 产生 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 使用 GloVe 100d 向量构建上面的 TF-IDF 加权嵌入混合（下载一次，缓存）。在 20 Newsgroups 数据集上与纯 TF-IDF 和纯平均池化嵌入比较分类准确率。报告各自在哪些情况下胜出。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BoW | 词频向量 | 一个文档中词汇表单词的计数。丢弃顺序。 |
| TF | 词频 | 一个词在文档中的计数，可选地按文档长度归一化。 |
| DF | 文档频率 | 至少包含一次该词的文档数量。 |
| IDF | 逆文档频率 | `log(N / df)` 加平滑。降低出现在所有地方的词的权重。 |
| 稀疏向量 | 大多数为零 | 词汇表通常有 1 万到 10 万个词；任何给定文档中大多数都不存在。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— 标准 API 参考，以及每个旋钮的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) —— 使 TF-IDF 成为十年默认标准的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) —— 2026 年视角：旧方法何时胜出及原因。