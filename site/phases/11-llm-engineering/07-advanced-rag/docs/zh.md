# Advanced RAG (Chunking, Reranking, Hybrid Search)

> 基础 RAG 检索 top-k 最相似的 chunks。这对简单问题有效。对于多跳推理、模糊查询和大型语料库就崩溃了。Advanced RAG 是 demo 在 10 个文档上工作与系统在 1000 万文档上工作的区别。

**类型:** 构建
**语言:** Python
**前置知识:** Phase 11 · 06（RAG）
**时间:** 约 90 分钟
**相关:** Phase 5 · 23（RAG 的 Chunking 策略）涵盖所有六种 chunking 算法——递归、语义、句子、父子文档、晚期 chunking、上下文检索——及 Vectara/Anthropic 基准。本课在此基础上构建：混合搜索、重排序、查询转换。

## 学习目标

- 实现保留文档结构和上下文的高级 chunking 策略（语义、递归、父子）
- 构建结合 BM25 关键词匹配与语义向量搜索和 cross-encoder 重排序器的混合搜索 pipeline
- 应用查询转换技术（HyDE、多查询、step-back）改善模糊或复杂问题的检索
- 诊断和修复常见 RAG 失败：检索了错误的 chunk、答案不在上下文中、多跳推理崩溃

## 问题

你在第 06 课构建了一个基础 RAG pipeline。它在小型语料库的简单问题上有效。现在试试这些：

**模糊查询**："上季度收入是多少？"语义搜索返回关于收入策略、收入预测和 CFO 对收入增长的想法的 chunks。都语义上相似于"收入"这个词。都不包含实际数字。正确的 chunk 说"$47.2M in Q3 2025"但使用"earnings"而不是"revenue"。Embedding 模型认为"收入策略"比"Q3 盈利为 $47.2M"更接近查询。

**多跳问题**："哪个团队的客户满意度分数提升最高？"这需要找到每个团队的满意度分数，比较它们，并识别最大值。没有单个 chunk 包含答案。信息分散在团队报告中。

**大型语料库问题**：你有 200 万个 chunks。正确答案在 chunk #1,847,293。你的 top-5 检索拉取 chunks #14、#89,201、#1,200,000、#44 和 #901,333。在嵌入空间中接近，但都不包含答案。在这个规模上，近似最近邻搜索引入足够的误差，使相关结果被推出 top-k。

基础 RAG 失败是因为向量相似度不等于相关性。一个 chunk 可能语义上相似于查询但对回答它没有用处。Advanced RAG 用四种技术解决这个：混合搜索（添加关键词匹配）、重排序（更仔细地对候选项评分）、查询转换（搜索前修复查询）和更好的 chunking（在正确的粒度上检索）。

## 概念

### 混合搜索：语义 + 关键词

语义搜索（向量相似度）擅长理解含义。"How do I cancel my subscription?" 匹配"Steps to terminate your plan"，尽管它们没有共享词。但它错过精确匹配。"Error code E-4021"可能不匹配包含"E-4021"的 chunk，如果 embedding 模型将其视为噪声。

关键词搜索（BM25）相反。它擅长精确匹配。"E-4021"完美匹配。但"cancel my subscription"如果文档说"terminate your plan"则返回零结果。

混合搜索同时运行两者，然后合并结果。

**BM25**（Best Matching 25）是标准关键词搜索算法。自 1990 年代以来一直是搜索引擎的支柱。公式：

```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是词 t 在文档 d 中的词频，IDF(t) 是逆文档频，|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和（默认 1.2），b 控制长度归一化（默认 0.75）。

简而言之：BM25 当文档包含查询词时（尤其是罕见词）给文档更高分数，但对重复词有递减收益。一个包含"revenue"50 次的文档并不比包含一次的相关 50 倍。

### 倒数排名融合（RRF）

你有两个排名列表：一个来自向量搜索，一个来自 BM25。如何组合它们？倒数排名融合是标准方法。

```
RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))
```

其中 k 是一个常数（通常 60），防止排名最高的结果主导。

在向量搜索中排名第 1、在 BM25 中排名第 5 的文档得：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

在向量搜索中排名第 3、在 BM25 中排名第 2 的文档得：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 自然平衡两个信号。在两个列表中都排名高的文档获得最佳分数。在一个列表中排名第 1 但在另一个中缺席的文档获得中等分数。这是稳健的，因为它使用排名而非原始分数，因此两个系统之间的分数分布差异无关紧要。

### 重排序

检索（无论是向量、关键词还是混合）快但不精确。它使用 bi-encoders：查询和每个文档独立嵌入，然后比较。Embedding 计算一次并缓存。这可以扩展到数百万文档。

重排序使用 cross-encoders：查询和候选文档一起输入到一个模型中，输出相关性分数。模型同时看到两个文本，可以捕获它们之间的细粒度交互。Cross-encoder 可以理解"What were Q3 earnings?" 与包含"$47.2M in Q3"的 chunk 高度相关，即使 bi-encoder 错过了这个连接。

权衡：cross-encoders 比 bi-encoders 慢 100-1000 倍，因为它们联合处理查询-文档对。你无法为百万文档预计算 cross-encoder 分数。解决方案：从混合搜索中检索更大的候选集（top-50），然后用 cross-encoder 重排序得到最终 top-5。

```mermaid
graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 候选项"]
    C50 --> RR["Cross-Encoder 重排序器"]
    RR --> C5["Top 5 最终结果"]
    C5 --> P["构建 prompt"]
    P --> LLM["Generate answer"]
```

常见重排序模型（2026 阵容）：
- Cohere Rerank 3.5：托管 API，多语言，混合语料库上最佳召回增益
- Voyage rerank-2.5：托管 API，托管选项中最低延迟
- Jina-Reranker-v2 Multilingual：开源权重，100+ 语言
- bge-reranker-v2-m3：开源权重，强基线
- cross-encoder/ms-marco-MiniLM-L-6-v2：开源权重，原型在 CPU 上运行
- ColBERTv2 / Jina-ColBERT-v2：晚期交互多向量重排序器——O(tokens) 而非 O(docs) 在评分时

### 查询转换

有时问题不在检索而在查询本身。"关于新政策变化的那件事是什么？"是一个糟糕的搜索查询。它不包含具体术语。Embedding 模糊。没有检索系统能从中找到正确的文档。

**查询重写**：将用户的查询重新表述为更好的搜索查询。LLM 可以做到：

```
User: "关于新政策变化的那件事是什么？"
Rewritten: "最近的策略变更和更新"
```

**HyDE（假设文档 Embedding）**：不搜索查询，而是生成一个假设答案，嵌入那个，搜索相似的真实文档。

```
Query: "企业计划的退款政策是什么？"
假设答案："企业客户有资格在购买后 60 天内获得全额退款。
退款根据剩余订阅期按比例计算，并在 5-7 个工作日内处理。"
```

嵌入假设答案并搜索与之相似的真实文档。直觉：假设答案在嵌入空间中比原始问题更接近真实答案。问题和答案有不同的语言结构。通过生成假设答案，你桥接了嵌入空间中"问题空间"和"答案空间"之间的差距。

HyDE 在检索前增加一个 LLM 调用。这将延迟增加 500-2000ms。当原始查询的检索质量差时值得。

### 父子 Chunking

标准 chunking 强制权衡：小 chunks 用于精确检索，大 chunks 用于足够上下文。父子 chunking 消除这个权衡。

索引小 chunks（128 token）用于检索。当一个小 chunk 被检索时，返回其父 chunk（512 token）用于 prompt。小 chunk 精确匹配查询。父 chunk 为 LLM 提供足够的上下文以生成好答案。

```mermaid
graph TD
    P["Parent chunk (512 tokens)<br/>关于退款政策的完整章节"]
    C1["Child chunk (128 tokens)<br/>标准计划：30 天退款"]
    C2["Child chunk (128 tokens)<br/>企业：60 天按比例"]
    C3["Child chunk (128 tokens)<br/>处理时间：5-7 天"]
    C4["Child chunk (128 tokens)<br/>如何提交请求"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"匹配 child"| C2
    C2 -.->|"返回 parent"| P
```

查询"enterprise refund?"精确匹配 child chunk C2。但 prompt 接收完整父 chunk P，其中包括关于处理时间和提交过程的周围上下文。

### 元数据过滤

在运行向量搜索之前，按元数据过滤语料库：日期、来源、类别、作者、语言。这减少搜索空间并防止不相关结果。

"上个月安全策略有什么变化？"应该只搜索最近 30 天安全类别中的文档。如果没有元数据过滤，你搜索整个语料库，可能检索到一个 2 年前的安全文档，恰好语义相似。

生产 RAG 系统将元数据与每个 chunk 一起存储：源文档、创建日期、类别、作者、版本。向量数据库支持在相似度搜索前按元数据预过滤，这对大规模性能至关重要。

### 评估

你构建了一个 RAG 系统。如何知道它是否有效？三个指标：

**检索相关性（Recall@k）**：对于一组带有已知相关文档的测试问题，top-k 结果中出现相关文档的百分比是多少？如果问题的答案在 chunk #47，chunk #47 是否出现在 top-5 中？

**Faithfulness**：生成的答案是否扎根于检索的文档？如果检索的 chunks 说"60 天退款窗口"而模型说"90 天退款窗口"，那是 faithfulness 失败。尽管有正确的上下文，模型还是产生了幻觉。

**答案正确性**：生成的答案是否与预期答案匹配？这是端到端指标。它结合检索质量和生成质量。

一个简单的 faithfulness 检查：取生成答案中的每个声明并验证它（实质上）出现在检索的 chunks 中。如果答案包含任何检索 chunk 中没有的事实，可能是幻觉。

```mermaid
graph TD
    subgraph "评估框架"
        Q["测试问题\n+ 预期答案\n+ 相关文档 ID"]
        Q --> Ret["检索评估\nRecall@k：正确\n文档被检索了吗？"]
        Q --> Faith["Faithfulness 评估\n答案扎根于\n检索的文档了吗？"]
        Q --> Correct["正确性评估\n答案匹配\n预期答案了吗？"]
    end
```

## 构建

### 第 1 步：BM25 实现

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 第 2 步：倒数排名融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 第 3 步：混合搜索 Pipeline

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 第 4 步：简单重排序器

在生产中，你将使用 cross-encoder 模型。这里我们构建一个使用词重叠、词重要性和短语匹配对查询-文档相关性评分的重排序器。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in",
                  "to", "and", "or", "on", "at", "by", "it", "its", "this",
                  "that", "with", "from", "be", "has", "have", "had", "not",
                  "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 第 5 步：HyDE（假设文档 Embedding）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 第 6 步：父子 Chunking

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 第 7 步：Faithfulness 评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this",
                      "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 使用

使用真实的 cross-encoder 进行重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 的托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

使用真实 LLM 的 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

在 Weaviate 中使用生产混合搜索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制平衡：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权重。大多数生产系统使用 0.3 到 0.7 之间的 alpha。

## 交付

本课产生：
- `outputs/prompt-advanced-rag-debugger.md` —— 用于诊断和修复 RAG 质量问题的 prompt
- `outputs/skill-advanced-rag.md` —— 用于构建具有混合搜索和重排序的生产级 RAG 的 skill

## 练习

1. 在示例文档上比较 BM25 vs 向量搜索 vs 混合搜索。对于每个 5 个测试查询，记录哪种方法在第 1 位返回最相关的 chunk。混合搜索应该在至少 3/5 上赢。

2. 实现元数据过滤器。向每个文档添加"category"字段（security、billing、api、product）。在运行向量搜索之前，将 chunks 过滤到仅相关类别。用"What's encryption is used?"测试并验证它只搜索安全类别 chunks。

3. 使用第 06 课的 simple generate 函数构建完整 HyDE pipeline。在所有 5 个测试查询上比较直接查询搜索和 HyDE 搜索的检索质量（top-3 相关性）。对于模糊查询，HyDE 应该改善结果。

4. 在示例文档上实现父子 chunking 策略。使用 child_size=30 和 parent_size=100。用 child chunks 搜索但返回 prompt 中的父 chunks。将生成的答案与 chunk_size=50 的标准 chunking 进行比较。

5. 创建评估数据集：10 个带有已知答案 chunks 的问题。测量 (a) 仅向量搜索、(b) 仅 BM25、(c) 混合搜索、(d) 混合 + 重排序的 Recall@3、Recall@5 和 Recall@10。绘制结果并识别重排序在哪里帮助最多。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| BM25 | "关键词搜索" | 概率排序算法，通过词频、逆文档频和文档长度归一化对文档评分 |
| 混合搜索 | "兼两家之长" | 并行运行语义（向量）和关键词（BM25）搜索，然后用排名融合合并结果 |
| 倒数排名融合 | "合并排名列表" | 通过对所有列表中每个文档求和 1/(k + rank) 来组合多个排名列表 |
| 重排序 | "第二遍评分" | 使用更昂贵的 cross-encoder 模型对初始检索的候选集重新评分 |
| Cross-encoder | "联合查询-文档模型" | 将查询和文档作为单个输入的模型，产生相关性分数；比 bi-encoders 更准确，但对整个语料库搜索太慢 |
| Bi-encoder | "独立 embedding 模型" | 独立嵌入查询和文档的模型；快速因为 embedding 预计算，但不如 cross-encoders 准确 |
| HyDE | "用假答案搜索" | 生成对查询的假设答案，嵌入它，搜索与之相似的真实文档 |
| 父子 chunking | "小搜索、大上下文" | 索引小 chunks 用于精确检索但返回更大的父 chunk 以提供足够的上下文 |
| 元数据过滤 | "搜索前缩小范围" | 在运行向量搜索前按属性（日期、来源、类别）过滤文档以减少搜索空间 |
| Faithfulness | "它保持扎根了吗" | 生成的答案是否被检索的文档支持，而不是从模型的训练数据中幻觉出来 |

## 扩展阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) —— BM25 的确定性参考，解释公式背后的概率基础
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009) —— 原始 RRF 论文，显示它优于更复杂的融合方法
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) —— HyDE 论文，证明假设文档 embedding 在没有任何训练数据的情况下改善检索
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) —— 显示 BM25 之上 cross-encoder 重排序显著改善检索质量
- [Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714) —— 将 prompt 构建和权重选择视为检索 pipeline 上的优化问题；读这个"编程 LLMs"而不是"prompt LLMs"。
- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130) —— GraphRAG 论文：实体-关系提取 + Leiden 社区检测用于查询聚焦摘要；全局与本地检索的区别。
- [Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511) —— 带有反思 token 的自评估 RAG；静态 retrieve-then-generate 之后的 agent 前沿。
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/) —— 如何将自然语言查询翻译为结构化数据库查询（Text-to-SQL、Cypher）作为预检索步骤。