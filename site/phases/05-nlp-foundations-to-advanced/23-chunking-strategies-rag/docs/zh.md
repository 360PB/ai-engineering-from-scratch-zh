# RAG 分块策略

> 分块配置对检索质量的影响与嵌入模型选择一样大（Vectara NAACL 2025）。分块错了，多少重排都救不了你。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 14（信息检索）、Phase 5 · 22（嵌入模型）
**耗时：** 约 60 分钟

## 问题

你将一份 50 页合同放入 RAG 系统。用户问："终止条款是什么？"检索器返回封面。为什么？因为模型在 512 token 块上训练，而终止条款在第 20 页，被分页符分开，没有本地关键词将其与查询联系起来。

修复不是"买一个更好的嵌入模型"。修复是分块。多大？重叠？在哪里拆分？周围上下文？

2026 年 2 月基准测试显示令人惊讶的结果：

- Vectara 2026 研究：递归 512 token 分块以 69% → 54% 准确率击败语义分块。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠提供零可衡量收益。
- 上下文悬崖：响应质量在约 2,500 token 上下文附近急剧下降。

"显而易见"的答案（语义分块、20% 重叠、1000 tokens）通常是错的。本课为六种策略构建直觉并告诉你何时用哪个。

## 概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**固定分块。** 每 N 个字符或 token 拆分一次。最简单的基线。中间句拆分。好的压缩，差的连贯性。

**递归。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试在 `\n\n` 上拆分，然后 `\n`，然后 `.`，然后空格。干净回退。2026 年默认。

**语义。** 嵌入每个句子。计算相邻句子之间的余弦相似度。在相似度低于阈值的地方拆分。保持主题连贯。更慢；有时产生伤害检索的微小 40 token 片段。

**句子。** 在句子边界上拆分。每句一块或 N 句窗口。在约 ~5k tokens 以下与语义分块匹配，成本的一小部分。

**父文档。** 存储用于检索的小子块*以及*用于上下文的大父块。按子块检索；返回父块。优雅退化：坏的子块仍然返回合理的父块。

**后期分块（2024）。** 首先在 token 级嵌入整个文档，然后将 token 嵌入池化为块嵌入。保留跨块上下文。与长上下文嵌入器（BGE-M3、Jina v3）配合使用。更高计算。

**上下文检索（Anthropic，2024）。** 在每个块前面加上 LLM 生成的文档中其位置摘要（"This chunk is section 3.2 of the termination clauses..."）。在 Anthropic 自己的基准上 35-50% 检索改进。索引成本高。

### 击败每个默认的规则

将块大小与查询类型匹配：

| 查询类型 | 块大小 |
|---------|--------|
| 事实（"CEO 的名字是什么？"） | 256-512 tokens |
| 分析性 / 多跳 | 512-1024 tokens |
| 全节理解 | 1024-2048 tokens |

NVIDIA 2026 年基准。块应该足够大以包含答案加本地上下文，小到检索器的 top-K 返回聚焦于答案而非上下文噪音。

## 构建

### 步骤 1：固定和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调整 `threshold`。太高 → 碎片。太低 → 一个巨大块。

### 步骤 3：父文档

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：去重父块。多个子块可以映射到同一父块；返回所有会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文分块。在查询时，检索受益于额外的周围信号。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终基准测试。"最佳"策略可能与任何博客文章不匹配。

## 陷阱

- **仅在事实查询上评估分块。** 多跳查询揭示非常不同的赢家。使用按查询类型分层的评估集。
- **语义分块没有最小大小。** 产生伤害检索的 40-token 碎片。始终强制执行 `min_tokens`。
- **重叠作为 Cargo Cult。** 2026 年研究发现重叠通常提供零收益并使索引成本翻倍。测量，不要假设。
- **没有最小/最大强制。** 5 tokens 或 5000 tokens 的块都破坏检索。夹紧。
- **跨文档分块。** 永远不要让块跨越两个文档。始终按文档分块，然后合并。

## 使用

2026 年技术栈：

| 场景 | 策略 |
|------|------|
| 首次构建，未知语料库 | 递归，512 tokens，无重叠 |
| 事实 QA | 递归，256-512 tokens |
| 分析性 / 多跳 | 递归，512-1024 tokens + 父文档 |
| 重交叉引用（合同、论文） | 后期分块或上下文检索 |
| 对话 / 对话语料库 | 轮次级块 + 说话者元数据 |
| 短话语（推文、评论） | 一文档 = 一块 |

从递归 512 开始。在 50 查询评估集上测量 recall@5。从那里调整。

## 交付

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: 为给定语料库和查询分布选择分块策略、大小和重叠。
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

给定语料库（文档类型、平均长度、领域）和查询分布（事实 / 分析性 / 多跳），输出：

1. 策略。递归 / 句子 / 语义 / 父文档 / 后期 / 上下文。理由。
2. 块大小。Token 计数。与查询类型相关的理由。
3. 重叠。默认 0；如果 >0 则说明理由。
4. 最小/最大强制。`min_tokens`、`max_tokens` guard。
5. 评估计划。在 50 查询分层评估集（事实、分析性、多跳）上的 Recall@5。

拒绝任何没有最小/最大块大小强制的分块策略。拒绝超过 20% 的重叠而没有消融显示其有帮助。标记没有 min-token 底线的语义分块推荐。
```

## 练习

1. **简单。** 用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 块化一份 20 页文档。比较块数量和边界质量。
2. **中等。** 在 5 份文档上构建 30 查询评估集。测量递归、语义和父文档的 recall@5。哪个胜出？与博客文章匹配吗？
3. **困难。** 实现上下文检索。测量 vs 基线递归的 MRR 改进。报告索引成本（LLM 调用）vs 准确率收益。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 块 | 文档的一片 | 子文档单元，被嵌入、索引和检索。 |
| 重叠 | 安全边际 | 相邻块之间共享 N 个 token；在 2026 年基准中通常无用。 |
| 语义分块 | 智能分块 | 在相邻句子嵌入相似度下降处拆分。 |
| 父文档 | 两级检索 | 检索小子块，返回大父块。 |
| 后期分块 | 分块后嵌入 | 在 token 级嵌入完整文档，池化为块向量。 |
| 上下文检索 | Anthropic 的技巧 | 在索引前在每个块前面加上 LLM 生成摘要。 |
| 上下文悬崖 | 2500-token 墙 | 在 RAG 中约 2.5k 上下文 token 处观察到质量下降（2026 年 1 月）。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) —— 生产默认。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) —— 分块与嵌入选择一样重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) —— 后期分块论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) —— 带 LLM 生成上下文前缀的 35-50% 检索改进。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) —— 按查询类型的块大小。