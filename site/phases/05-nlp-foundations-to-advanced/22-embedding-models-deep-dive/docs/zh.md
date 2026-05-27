# 嵌入模型——2026 年深度探讨

> Word2Vec 给你每词一个向量。现代嵌入模型给你每段一个向量，跨语言，有稀疏、密集和多向量视图，大小适合你的索引。选错了，你的 RAG 检索到错误的东西。

**类型：** 学习
**语言：** Python
**先修课程：** Phase 5 · 03（Word2Vec）、Phase 5 · 14（信息检索）
**耗时：** 约 60 分钟

## 问题

你的 RAG 系统 40% 的时间检索到错误的段落。罪魁祸首很少是向量数据库或提示。是嵌入模型。

2026 年选择嵌入意味着在五个轴上挑选：

1. **密集 vs 稀疏 vs 多向量。** 每段一个向量，还是每 token 一个向量，还是稀疏加权的词袋。
2. **语言覆盖。** 仅英语模型在仅英语任务上仍然胜出。混合语料库时多语言模型胜出。
3. **上下文长度。** 512 tokens vs 8,192 vs 32,768——真实有效容量通常是广告最大值的 60-70%。
4. **维度预算。** 3,072 个浮点数在完整精度 = 每向量 12 KB。100M 向量时，存储 $1,300/月。Matryoshka 截断削减 4 倍。
5. **开放 vs 托管。** 开放权重意味着你控制栈和数据。托管意味着你用控制权换取始终最新。

本课命名权衡，以便你能根据证据选择，而不是根据上季度流行什么。

## 概念

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**密集嵌入。** 每段一个向量（通常 384-3,072 维）。余弦相似度按语义接近度对段落排名。OpenAI `text-embedding-3-large`、BGE-M3 dense 模式、Voyage-3。默认选择。

**稀疏嵌入。** SPLADE 风格。Transformer 为每个词汇表 token 预测一个权重，然后将大多数置零。结果是大小为 |vocab| 的稀疏向量。捕捉词汇匹配（像 BM25）但带有学习的词权重。在关键词重查询上强。

**多向量（后期交互）。** ColBERTv2、Jina-ColBERT。每 token 一个向量。用 MaxSim 评分：对每个查询 token，找到最相似的文档 token，汇总分数。存储和评分更贵，但在长查询和领域特定语料库上胜出。

**BGE-M3：一次搞定三种。** 单一模型同时输出密集、稀疏和多向量表示。每个可以独立查询；分数通过加权和融合。当你想从一个检查点获得灵活性时的 2026 年默认。

**Matryoshka 表示学习。** 训练使得向量的前 N 维形成独立的可用嵌入。将 1,536 维向量截断到 256 维，支付约 1% 准确率获得 6 倍存储节省。被 OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 支持。

### MTEB 排行榜讲了一个不完整的故事

大规模文本嵌入基准——在发布时（2022）56 个任务跨 8 种任务类型，在 MTEB v2 扩展到 100+ 任务。2026 年初，Gemini Embedding 2 在检索上领先（67.71 MTEB-R）。Cohere embed-v4 在通用上领先（65.2 MTEB）。BGE-M3 在开放权重多语言上领先（63.0）。排行榜是必要的但不是充分的——总是在你的领域上基准测试。

### 三层模式

| 用途 | 模式 |
|------|------|
| 快速第一遍 | 密集双编码器（BGE-M3、text-3-small） |
| 召回提升 | 稀疏（SPLADE、BGE-M3 sparse）+ RRF 融合 |
| Top-50 精确率 | 多向量（ColBERTv2）或交叉编码器重排器 |

大多数生产技术栈使用全部三种。

## 构建

### 步骤 1：基线——用 Sentence-BERT 的密集嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等于余弦相似度。始终设置它。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 训练有素，使前几个级别无损。非 Matryoshka 模型（原始 Sentence-BERT）在截断时急剧退化。

### 步骤 3：BGE-M3 多功能

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理调用。分数融合：

```python
dense_score = ... # dense_vecs 上的余弦
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调整权重。

### 步骤 4：在自定义任务上 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*代表性*子集上运行候选模型。不要只信任排行榜排名——你的领域很重要。

### 步骤 5：从零手工余弦

见 `code/main.py`。平均哈希技巧嵌入（仅 stdlib）。与 Transformer 嵌入没有竞争力，但展示了形态：分词 → 向量 → 归一化 → 点积。

## 陷阱

- **查询和文档用同一模型。** 某些模型（Voyage、Jina-ColBERT）使用非对称编码——查询和文档经过不同路径。始终检查模型卡。
- **缺少前缀。** `bge-*` 模型需要在查询前加上 `"Represent this sentence for searching relevant passages: "`。忘记的话 3-5 分召回差距。
- **过度修剪 Matryoshka。** 1,536 → 256 通常安全。1,536 → 64 不是。在你的评估集上验证。
- **上下文截断。** 大多数模型静默截断超过最大长度的输入。长文档需要分块（见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏 p99 延迟。600M 模型可能比 335M 模型高 2 分但每查询成本高 3 倍。

## 使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 仅英语，快速，API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开放权重，英语 | `BAAI/bge-large-en-v1.5` |
| 开放权重，多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词重查询 | 添加 SPLADE 稀疏，与密集 RRF 融合 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估，如果领域特定模型高超过 3 分则交换。

## 交付

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: 为给定语料库和部署选择嵌入模型、维度和检索模式。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定语料库（大小、语言、领域、平均长度）、部署目标（云 / 边缘 / 本地）、延迟预算和存储预算，输出：

1. 模型。命名检查点或 API。一句话理由。
2. 维度。完整 / Matryoshka 截断 / int8 量化。与存储预算相关的理由。
3. 模式。密集 / 稀疏 / 多向量 / 混合。理由。
4. 如果模型卡需要，查询前缀 / 模板。
5. 评估计划。与领域相关的 MTEB 任务 + 带 nDCG@10 的留出领域评估。

拒绝推荐将 Matryoshka 截断到 <64 维而不在领域上验证。拒绝为小于 10k 段落的语料库使用 ColBERTv2（开销不合理）。标记被路由到 512-token 窗口模型的长文档语料库（>8k tokens）。
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 在完整维（384）和 Matryoshka 128 上编码 100 句。在 10 个查询上测量 MRR 下降。
2. **中等。** 在你的领域 500 段上比较 BGE-M3 dense、sparse 和 colbert。在 recall@10 上哪个胜出？RRF 融合比单个最佳模式好吗？
3. **困难。** 在你的前两个领域任务上跨三个候选模型运行 MTEB。报告 MTEB 分数、100 查询批上的 p99 延迟和每百万查询成本。选择帕累托最优的。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 密集嵌入 | 向量 | 每文本一个固定大小向量。用余弦相似度排名。 |
| 稀疏嵌入 | 学习的 BM25 | 每个词汇表 token 一个权重；大多数为零；端到端训练。 |
| 多向量 | ColBERT 风格 | 每 token 一个向量；MaxSim 评分；索引更大，召回更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身是有效的较小嵌入。 |
| MTEB | 基准 | 大规模文本嵌入基准——发布时 56 个任务，v2 超过 100 个。 |
| BEIR | 检索基准 | 18 个零样本检索任务；经常被引用用于跨域鲁棒性。 |
| 非对称编码 | 查询 ≠ 文档路径 | 模型对查询和文档使用不同投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) —— 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) —— 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) —— 统一三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) —— 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) —— 生产中的后期交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) —— 实时排名。