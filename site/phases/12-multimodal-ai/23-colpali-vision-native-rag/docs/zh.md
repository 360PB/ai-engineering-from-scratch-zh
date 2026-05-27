# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本，分块，embedding，存储向量。每步都丢失信号：OCR 丢弃图表数据，分块破坏表格行，文本 embedding 忽略图形。ColPali（Faysse 等，2024 年 7 月）问了更简单的问题：为什么要提取文本？直接通过 PaliGemma embedding 页面图像，使用 ColBERT 风格晚期交互进行检索，保留文档携带的所有布局、图形、字体和格式信号。已发布基准：在视觉丰富文档上端到端准确率比文本 RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这一模式。本节解读视觉原生 RAG 论文并构建一个小型类 ColPali 索引器。

**类型：** Build
**语言：** Python（标准库，多向量索引器 + MaxSim 评分器）
**前置知识：** Phase 11（LLM Engineering — RAG 基础），Phase 12 · 05（LLaVA）
**时间：** 约 180 分钟

## 学习目标

- 解释双编码器检索（每个文档一个向量）和晚期交互检索（每个文档多个向量）之间的区别。
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 泛化到图像 patches。
- 构建一个小型类 ColPali 索引器：页面 → patch embedding → 在查询 term embedding 上 MaxSim → top-k 页面。
- 在发票/财务报告用例上比较 ColPali + Qwen2.5-VL 生成器 vs 文本 RAG + GPT-4。

## 问题背景

PDF 上的文本 RAG 丢弃了大多数文档。财务报告的 Q3 营收增长通常在图表中；医疗报告的发现在带注释的图像中；法律合同的签名块是布局事实，不是文本事实。

文本 RAG 流水线：

1. PDF → 文本通过 OCR / pdftotext。
2. 文本 → 300-500 token 分块。
3. 块 → 双编码器 embedding（一个向量）。
4. 用户查询 → embedding → 余弦相似度 → top-k 块。
5. 块 + 查询 → LLM。

五个有损步骤。图表未捕获。表格跨块断裂。多栏布局扁平化。图形注释消失。

ColPali 的修复：跳过 OCR，直接 embedding 页面图像。使用 ColBERT 风格晚期交互进行检索，模型可以在查询时关注细粒度 patches。

## 核心概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。不是每个文档一个向量，而是每个 token 一个向量。查询时：

- 查询 token 获得自己的 embedding（N_q 个向量）。
- 文档 token 获得 embedding（N_d 个向量，通常缓存）。
- 分数 = 查询 token 上的和 × 文档 token 上的最大余弦相似度：Σ_i max_j cos(q_i, d_j)。

这是 MaxSim 操作。每个查询 token"选择"其最佳匹配文档 token。最终分数是总和。

优点：召回强，处理 token 级语义。缺点：每个文档 N_d 个向量，存储昂贵。

### ColPali

ColPali（Faysse 等，arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每页由 PaliGemma（ViT + 语言）编码为 patch embedding：每页 N_p 个向量。
- 每个用户查询（文本）编码为查询 token embedding：N_q 个向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，即查询文本 token 和页面图像 patches 之间的 MaxSim。
- 按总分检索 top-k 页面。

文档 ingestion 时：用 PaliGemma embedding 每页，存储所有 patch embedding。查询时：embedding 查询 token，对所有存储的页面 embedding 计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富文档上端到端比文本 RAG 高 20-40%。每个 patch 向量捕获局部布局和内容。

缺点：N_p patches × 4 字节浮点数 × D 维向量/页 = 存储快速增长。通过 PQ / OPQ 量化缓解。

### ColQwen2 和 ColSmol

ColQwen2（illuin-tech，2024-2025）将 PaliGemma 换成 Qwen2-VL。更的基础编码器，更好的检索。

ColSmol 是用于本地/边缘的更小规模变体。约 10 亿参数的 ColSmol retriever 可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等，arXiv:2410.10594）是不同变体：不是 patches 上的 MaxSim，而是用 VLM 将每页池化为单个向量然后双编码器检索。索引更快 + 存储更小，召回更弱。

质量-成本权衡：质量选 ColPali，规模选 VisRAG。

### M3DocRAG

M3DocRAG（Cho 等，arXiv:2411.04952）将多模态检索扩展到多页多文档推理。跨文档检索页面，组合多页上下文供 VLM 使用。

### ViDoRe——基准

ColPali 的配套基准。视觉文档检索评估。任务包括财务报告、科学论文、行政文档、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上得分约 80% nDCG@5；相同文档上文本 RAG 得分约 50-60%。

### 端到端 RAG 流水线

视觉原生 RAG：

1. Ingest：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch embedding。
2. 查询：用户文本 → 查询 token embedding → 在所有索引页面上 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

无处使用 OCR。图形、图表、字体、布局全部流入答案。

### 存储数学

50 页财务报告每页 729 patches 和 128 维 embedding：

- ColPali：50 * 729 * 128 * 4 字节 ≈ 18 MB 原始，约 4 MB PQ 后。
- 文本 RAG：50 块 * 768 维 * 4 字节 ≈ 150 kB。

ColPali 每文档存储约 30 倍。通过 OPQ / PQ 降至约 5-10 倍，通常可接受。

### 文本 RAG 何时仍然胜出

- 无布局信号的纯文本文档（维基文章、聊天日志）。文本 RAG 更简单且存储便宜。
- 存储主导成本的几百万页档案。
- 要求可提取 OCR 文本以及检索的严格监管要求。

2026 年其他一切——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 胜出。

## 使用方法

`code/main.py`：

- 玩具 patch 编码器：将"页面"（小特征向量网格）映射到 patch embedding 数组。
- MaxSim 评分器：计算查询 token embedding 集和页面 patch 集之间的 ColBERT 风格分数。
- 索引 5 个玩具页面，运行 3 个查询，返回带分数的 top-k。

## 输出作品

本节生成 `outputs/skill-vision-rag-designer.md`。给定文档 RAG 项目，选择 ColPali / ColQwen2 / VisRAG / 文本 RAG 并确定存储规模。

## 练习

1. 200 页年报每页 729 patches，128 维 emb，4 字节浮点。计算原始存储和 PQ 压缩（8x）存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个和捕获了简单均值相似度没有捕获的什么？

3. ColPali 将页面索引为 patch 集。如果我们改为在词级索引（如 ColBERT）会怎样？权衡？

4. 为 100 万页语料库设计端到端流水线，延迟预算每查询 500ms。选择 ColQwen2 / VisRAG 并提供理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页注意模式以及它如何与单页 ColPali 检索不同。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Late interaction | "ColBERT 风格" | 使用逐 token 或逐 patch embedding + MaxSim 检索，而非单一文档向量 |
| MaxSim | "跨 patches 最大" | 对每个查询 token，选择最高相似度文档 token；跨查询求和 |
| Bi-encoder | "单一向量" | 每个文档一个向量；更快但丢失细粒度 |
| Multi-vector | "每文档多向量" | 每个文档/页面存储 N_p 个向量；存储成本增加但召回改善 |
| Patch embedding | "页面特征" | 来自 VLM 编码器的每个图像 patch 的一个向量，按页面缓存 |
| ViDoRe | "视觉文档基准" | ColPali 的视觉文档检索评估套件 |
| PQ quantization | "乘积量化" | 在缩小存储约 8 倍的同时保持向量相似度的压缩 |

## 延伸阅读

- [Faysse 等 — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu 等 — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho 等 — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)