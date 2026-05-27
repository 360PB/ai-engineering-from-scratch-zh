# 信息检索与搜索

> BM25 精确但脆弱。密集检索撒网广但漏关键词。混合检索是 2026 年的默认。其他都是调优。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 02（BoW + TF-IDF）、Phase 5 · 04（GloVe、FastText、子词）
**耗时：** 约 75 分钟

## 问题

用户输入"what happens if someone lies to get money"，期望找到实际涵盖该行为的法规："Section 420 IPC"。关键词搜索完全漏掉它（无共享词汇）。语义搜索漏掉它（如果嵌入未在法律文本上训练）。真实搜索必须处理两者。

IR 是每个 RAG 系统、每个搜索栏、每个文档站模糊查找的底层流水线。2026 年在生产中有效的架构不是一个方法。它是一组互补方法的链条，每个捕捉前一个的失败。

本课构建每个部分并命名每个捕捉的失败。

## 概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。选择你需要的。

1. **稀疏检索（BM25）。** 快、精确匹配好、语义差。在倒排索引上运行。百万文档上每查询亚 10ms。正确获取法规引用、产品代码、错误信息、命名实体。
2. **密集检索。** 将查询和文档编码为向量。近邻搜索。捕捉改写和语义相似度。漏掉差一个字符的精确关键词匹配。每查询 50-200ms（FAISS 或向量数据库）。
3. **融合。** 合并稀疏和密集的排序列表。倒数排名融合（RRF）是简单默认，因为它忽略原始分数（在不同尺度上）而只使用排名位置。当你知道一个信号在你的领域占主导时，加权融合是一个选项。
4. **交叉编码器重排。** 从融合中取 top-30。运行交叉编码器（查询 + 文档一起，对每对评分）。保留 top-5。交叉编码器每对比双编码器慢但准确得多。通过只在 top-30 上运行来摊销。

三路检索（BM25 + 密集 + 如 SPLADE 的学习稀疏）在 2026 年基准测试中优于两路，但需要学习稀疏索引基础设施。对大多数团队，两路加交叉编码器重排是最佳选择。

## 构建

### 步骤 1：从零构建 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得知道的参数。`k1=1.5` 控制词频饱和；越高越重视词重复。`b=0.75` 控制长度归一化；0 忽略文档长度，1 完全归一化。默认值是 Robertson 原始论文的建议，很少需要调优。

### 步骤 2：双编码器密集检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2 归一化嵌入使点积等于余弦。`all-MiniLM-L6-v2` 是 384 维、快速、对大多数英语检索足够强。对于多语言工作，使用 `paraphrase-multilingual-MiniLM-L12-v2`。最高准确率用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。较高的 `k` 平抑排名差异的贡献；较低的 `k` 使排名靠前的占主导。60 是发布默认值，很少需要调优。

### 步骤 4：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 找到词汇匹配。密集找到语义匹配。RRF 在不需要分数校准的情况下合并两个排名。交叉编码器使用查询-文档对一起重评分 top-30，捕捉双编码器遗漏的细粒度相关性。保留 top-5。

### 步骤 5：评估

| 指标 | 含义 |
|------|------|
| Recall@k | 在正确答案存在的查询中，有多少次在 top-k 中？ |
| MRR（平均倒数排名） | 第一个相关文档的 1/排名平均值。 |
| nDCG@k | 考虑相关性程度，不只是二元相关/不相关。 |

对于 RAG specifically，**Retriever 的 Recall@k** 是最重要的数字。如果正确答案段落不在检索集中，你的阅读器无法回答。

调试提示：对于失败的查询，diff 稀疏和密集排名。如果一个找到了正确答案而另一个没有，你有词汇不匹配（修复：添加缺失的一半）或语义歧义（修复：更好的嵌入或重排器）。

## 使用

2026 年技术栈：

| 规模 | 技术栈 |
|------|-------|
| 1k-100k 文档 | 内存 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无独立数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 用于密集 + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 文档 | Qdrant / Weaviate / Vespa / Milvus 带混合支持。Top-30 上交叉编码器重排。 |
| 最高质量前沿 | 三路（BM25 + 密集 + SPLADE）+ ColBERT 后期交互重排 |

无论选什么，预算评估。在基准测试端到端 RAG 准确率之前先基准测试检索召回率。阅读器无法修复检索器遗漏的内容。

### 2026 年生产 RAG 的来之不易的经验

- **80% 的 RAG 失败追溯到摄入和分块，而不是模型。** 团队花数周交换 LLM 和调优提示，而检索悄悄地在每三个查询中返回错误上下文。先修复分块。
- **分块策略比块大小更重要。** 固定大小拆分破坏表格、代码和嵌套标题。句子感知是默认；语义或基于 LLM 的分块对技术文档和产品手册值得付出。
- **父文档模式。** 检索小的"子"块以保证精确。当同一父部分的多个子出现时，换入父块以保留上下文。这持续提升答案质量，无需重训练。
- **k_rerank=3 通常最优。** 超过此的每个额外块增加 token 成本和生成延迟，而不提升答案质量。如果 k=8 对你仍然比 k=3 更好，重排器表现不佳。
- **HyDE / 查询扩展。** 从查询生成假设答案，嵌入那个，检索。弥合短问题与长文档之间的措辞差距。免费精确率提升，无需训练。
- **上下文预算低于 8K tokens。** 在该限制处持续命中意味着重排器阈值太松。
- **版本一切。** 提示、分块规则、嵌入模型、重排器。任何漂移都会悄悄破坏答案质量。在忠实度、上下文精确率和未回答问题率上用 CI gate 在用户看到之前阻止回归。
- **三路检索（BM25 + 密集 + 如 SPLADE 的学习稀疏）在 2026 年基准测试中优于两路**，尤其是对于混合专有名词和语义的查询。在基础设施支持 SPLADE 索引时上线。

根据 2026 年行业测量，正确的检索设计将幻觉减少 70-90%。大多数 RAG 性能提升来自更好的检索，而不是模型微调。

## 交付

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: 为给定语料库和查询模式选择检索技术栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

给定需求（语料库大小、查询模式、延迟预算、质量标准、基础设施约束），输出：

1. 技术栈。仅 BM25、仅密集、混合（BM25 + 密集 + RRF）、混合 + 交叉编码器重排，或三路（BM25 + 密集 + 学习稀疏）。
2. 密集编码器。命名具体模型。与语言、领域和上下文长度匹配。
3. 重排器。如果使用，命名具体交叉编码器模型。标记重排为 top-30 上增加 30-100ms 延迟。
4. 评估计划。Recall@10 是主要检索器指标。多答案用 MRR。先基线，用它衡量增量改进。

当用户有证据表明密集检索处理精确匹配时，拒绝推荐仅密集检索用于有命名实体、错误代码或产品 SKU 的语料库。拒绝在高风险检索（法律、医疗）中跳过重排，因为最终 top-5 决定用户的答案。
```

## 练习

1. **简单。** 在 500 文档语料库上实现上面的 `hybrid_search`。测试 20 个查询。在 BM25-only、密集-only 和混合中比较 Recall@5。
2. **中等。** 添加 MRR 计算。对于每个有已知正确答案的测试查询，在 BM25、密集和混合排名中找到正确答案的排名。报告每个的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）微调你领域的密集编码器。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BM25 | 关键词搜索 | Okapi BM25。按词频、IDF 和长度对文档评分。 |
| 密集检索 | 向量搜索 | 将查询 + 文档编码为向量，找近邻。 |
| 双编码器 | 嵌入模型 | 独立编码查询和文档。查询时快。 |
| 交叉编码器 | 重排模型 | 一起编码查询 + 文档。慢但准确。 |
| RRF | 排名融合 | 通过对 `1/(k + rank)` 求和组合两个排名。 |
| Recall@k | 检索指标 | 相关文档在 top-k 中的查询比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) —— 最权威的 BM25 论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) —— DPR，规范双编码器。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) —— 缩小与密集差距的学习稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) —— RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) —— 后期交互检索。