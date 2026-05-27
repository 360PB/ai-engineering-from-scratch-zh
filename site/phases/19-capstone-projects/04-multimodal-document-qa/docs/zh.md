# Capstone 04 — 多模态文档问答（视觉优先的 PDF、表格、图表）

> 2026年的文档问答前沿已从"OCR优先再转文本"转向"视觉优先的延迟交互"。ColPali、ColQwen2.5 和 ColQwen3-omni 将每一页 PDF 视为一张图像，用多向量延迟交互对其进行嵌入，让查询直接attend到patch上。在财务10-K报告、科学论文和手写笔记上，这种模式以很大优势超越了OCR优先方案。从零构建端到端流水线，处理10k页文档，并与OCR优先方案进行对比。

**类型：** 毕业项目
**语言：** Python（流水线），TypeScript（查看器 UI）
**前置知识：** Phase 4（计算机视觉）、Phase 5（NLP）、Phase 7（Transformers）、Phase 11（LLM工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时长：** 30小时

## 问题

企业积累了大量被OCR流水线处理得支离破碎的PDF：带旋转表格的扫描10-K报告、满是公式的科学论文、只能作为图像理解才有意义的图表、手写批注。以文本优先的方式处理这些文档意味着丢失一半的信号。2026年的解决方案是对原始页面图像进行延迟交互多向量检索。ColPali（Illuin Tech）首创了这种方法；ColQwen2.5-v0.2 和 ColQwen3-omni 将精度进一步推高。在 ViDoRe v3 上，视觉优先检索以显著优势超越了OCR优先方案——在图表、表格和手写内容上差距更大。

代价是存储和延迟。一个 ColQwen 嵌入每页约2048个patch向量，而不是单一的1024维向量。原始存储量暴涨。DocPruner（2026）实现了50%剪枝且几乎不影响精度。你将对10k页建立索引，测量 ViDoRe v3 的 nDCG@5，控制在2秒内返回答案，并直接与OCR优先方案对比。

## 核心概念

延迟交互意味着每个查询token对每个patch token独立打分，每个查询token取最大分数后求和。你无需单一的池化向量就能获得细粒度匹配。多向量索引（Vespa、Qdrant multi-vector 或 AstraDB）存储每个patch的嵌入，在检索时运行 MaxSim。

回答器是一个视觉语言模型，输入查询和top-k检索页面图像，输出带证据区域（边界框或页码引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是2026年的前沿选择。对于公式和科学符号，OCR后备（Nougat、dots.ocr）作为可选文本通道接入。

评估是二维矩阵。一维：内容类型（普通文本段落、密集表格、柱状/折线图、手写笔记、公式）。另一维：检索方法（视觉优先延迟交互 vs OCR优先 vs 混合）。每个单元格给出 nDCG@5 和答案准确率。报告就是交付物。

## 架构

```
PDFs -> 页面渲染器（PyMuPDF，180 DPI）
           |
           v
    ColQwen2.5-v0.2 嵌入（每页多向量，约2048个patch）
           |
           +------> DocPruner 50% 压缩
           |
           v
    多向量索引（Vespa 或 Qdrant multi-vector）
           |
查询 ----+----> 检索 top-k 页面（MaxSim）
           |
           v
   VLM 回答器：Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
     输入：查询 + top-k 页面图像 + 可选 OCR 文本
           |
           v
   带页码引用和证据区域的答案
           |
           v
   Streamlit / Next.js 查看器：在源页面上高亮边界框
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，竖屏归一化
- 延迟交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face vidore team）
- 索引：Vespa multi-vector 字段，或 Qdrant multi-vector，或 AstraDB + MaxSim
- 剪枝：DocPruner 2026 策略（保留高方差patch，50%压缩，精度损失<0.5%）
- OCR后备（公式/密集表格）：dots.ocr 或 Nougat
- VLM回答器：Qwen3-VL-30B 自托管 或 Gemini 2.5 Pro 托管；InternVL3 作为后备
- 评估：ViDoRe v3 基准测试，M3DocVQA 用于多页推理
- 查看器UI：Next.js 15 + canvas 叠加显示证据区域

## 动手实现

1. **摄取。** 遍历10k页PDF语料库，涵盖10-K报告、科学论文和扫描文档。将每页渲染为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 用 ColQwen2.5-v0.2 处理每页图像。输出形状约为2048个patch嵌入，dim=128。用 DocPruner 保留最高信号的一半。写入 Vespa multi-vector 字段或 Qdrant multi-vector。

3. **查询。** 对每个输入查询，用查询塔嵌入（token级嵌入）。对索引运行 MaxSim：对每个查询token，取页面patch嵌入的最大点积，求和。返回 top-k 页面。

4. **合成。** 用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词："仅使用提供的页面作答。每个论点需注明 (doc_id, page)，并指出区域类型（figure、table、paragraph）。"

5. **证据区域。** 后处理答案，提取被引用的区域。如果 VLM 输出边界框（Qwen3-VL 会），在查看器中将其渲染为叠加层。

6. **OCR后备。** 对于被识别为公式密集的页面（基于图像方差的启发式判断），运行 Nougat 或 dots.ocr，将 OCR 文本作为附加通道与图像一起传入。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。同时在相同语料库上运行 OCR 优先流水线，使用相同的合成器。生成内容类型 × 方法矩阵。

8. **UI。** 先做 Streamlit 原型；Next.js 15 生产级查看器，支持逐页证据区域叠加。

## 用现成库

```bash
$ doc-qa ask "2024财年 EMEA 部门的营业利润率变化是多少？"
[检索]     top-5 页面，320ms（ColQwen2.5，MaxSim，Vespa）
[合成]     qwen3-vl-30b，1.4s，已引用 (form-10k-2024, p.88) + (..., p.92)
答案:
  EMEA 营业利润率从 18.2% 降至 16.8%，下降了 140 个基点。
  引用: 10-K-2024.pdf p.88（表4，部门营业利润率）
         10-K-2024.pdf p.92（MD&A，运营表现）
[查看器]   打开 p.88 表4 带高亮边界框叠加
```

## 产出

`outputs/skill-doc-qa.md` 描述交付物：一个针对特定语料库调优的视觉优先多模态文档问答系统，在 ViDoRe v3 上与 OCR 优先方案对比评估。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 基准分数 vs OCR-文本基线和公开排行榜 |
| 20 | 证据区域定位 | 被引用区域中真正包含答案跨度的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩比，索引 p95，答案 p95 |
| 20 | 多页推理 | 手标100题多页测试集准确率 |
| 15 | 源文档检查UX | 查看器清晰度，叠加保真度，并排对比工具 |
| **100** | | |

## 练习

1. 在相同语料库上比较 ColQwen2.5-v0.2 和 ColQwen3-omni。哪些页面一个做对、另一个做错？给索引添加"内容类别"标签按类型路由。

2. 激进剪枝嵌入（75%、90%）。找到压缩悬崖：ViDoRe nDCG@5 跌破 OCR 基线的那个点。

3. 构建混合方案：OCR优先和 ColQwen 并行运行，用 RRF 融合，用 cross-encoder 重排。混合方案能否超越单独使用任一个？在哪类场景帮助最大？

4. 把 Qwen3-VL-30B 换成更小的 VLM（Qwen2.5-VL-7B）。测量准确率-成本曲线。

5. 添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，测量检索效果。与手写 OCR 流水线对比。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Late interaction | "ColPali式检索" | 查询token独立对页面patch打分；MaxSim聚合 |
| Multi-vector | "Per-patch 嵌入" | 每个文档有多个向量，而非一个池化向量 |
| MaxSim | "延迟交互打分" | 对每个查询token，取文档向量的最大相似度后求和 |
| DocPruner | "Patch 压缩" | 2026年剪枝方案，保留50%的patch，精度损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026年视觉文档检索测量标准 |
| Evidence region | "引用边界框" | 源页面上定位答案跨度的边界框 |
| OCR fallback | "公式通道" | 在视觉方案之外，对公式或表格密集页面使用的文本流水线 |

## 扩展阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 延迟交互文档检索参考实现
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [ColQwen 系列（Hugging Face）](https://huggingface.co/vidore) — 生产级检查点
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa multi-vector 教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant multi-vector 支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 后备