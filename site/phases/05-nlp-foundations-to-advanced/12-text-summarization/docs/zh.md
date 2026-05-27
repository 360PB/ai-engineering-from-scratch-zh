# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者的意思。不同任务，不同陷阱。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 02（BoW + TF-IDF）、Phase 5 · 11（机器翻译）
**耗时：** 约 75 分钟

## 问题

一篇 2000 字的新闻文章进入你的信息流。你需要 120 字来概括它。你可以从文章中选取三个最重要的句子（抽取式），或者用自己的话重写内容（生成式）。两者都叫摘要。它们是完全不同的问题。

抽取式摘要是排序问题。给每个句子打分，返回 top-`k`。输出总是语法的，因为是逐字提升的。风险是遗漏分布在整个文章中的内容。

生成式摘要是生成问题。Transformer 在输入条件下产生新文本。输出流畅且压缩，但可能产生源中没有的事实幻觉。风险是自信的捏造。

本课构建两者，每个都有其拥有的失败模式。

## 概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**抽取式。** 将文章视为图，其中节点是句子，边是相似度。在图上运行 PageRank（或类似算法）给句子评分，评分依据是它与所有其他内容的连接程度。得分最高的句子是摘要。规范实现是 **TextRank**（Mihalcea and Tarau, 2004）。

**生成式。** 在文档-摘要对上微调 Transformer 编码器-解码器（BART、T5、Pegasus）。推理时，模型读取文档并通过交叉注意力逐 token 生成摘要。Pegasus 特别使用间隙句预训练目标，使其无需太多微调就在摘要上表现出色。

用 **ROUGE**（面向召回率的自动摘要评估研究）评估。ROUGE-1 和 ROUGE-2 评分 unigram 和 bigram 重叠。ROUGE-L 评分最长公共子序列。越高越好，但 40 ROUGE-L 是"好"，50 是"出色"。每篇论文都报告所有三个。使用 `rouge-score` 包。

## 构建

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两个值得指出的东西。相似度函数使用对数归一化词重叠，这是原始 TextRank 变体。TF-IDF 向量的余弦也可以工作。阻尼因子 0.85 和迭代次数是 PageRank 默认值。

### 步骤 2：用 BART 做生成式

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上微调。开箱即用产生新闻风格摘要。对于其他领域（科学论文、对话、法律），使用相应的 Pegasus 检查点或在目标数据上微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取。没有它，"running" 和 "run" 算作不同的词，ROUGE 低估。

### ROUGE 之外（2026 摘要评估）

ROUGE 作为主导摘要指标已二十年，2026 年单凭它是不够的。NLG 论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）在 2023 年获得关注，现在与 ROUGE 一起在大多数摘要论文中报告。
- **BARTScore** 将评估视为生成：用预训练 BART 给定源对摘要分配的可能性评分。
- **MoverScore**（上下文嵌入上的 Earth Mover's 距离）在 2025 年摘要基准测试中达到最高位置，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **基于 QA 的忠实度** 在 2021-2023 年常见，现在经常被 **G-Eval**（GPT-4 提示链，用思维链推理在连贯性、一致性、流畅性、相关性上评分）取代。
- **G-Eval** 和类似 LLM 评判方法在评分标准设计良好时与人类判断一致性约 80%。

生产建议：报告 ROUGE-L 用于传统比较，BERTScore 用于语义重叠，G-Eval 用于连贯性和事实性。用 50-100 个人类标注摘要校准。

### 步骤 4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要携带低得多的幻觉风险，因为输出是从源逐字提升的，尽管如果源句子被去语境化、过时或乱序引用仍可能误导。这是生产系统仍然在合规相关内容上首选抽取式方法的首要原因。

需要命名的幻觉类型：

- **实体替换。** 源说"John Smith"。摘要说"John Brown"。
- **数字漂移。** 源说"25,000"。摘要说"25 million"。
- **极性翻转。** 源说"rejected the offer"。摘要说"accepted the offer"。
- **事实捏造。** 源未提及 CEO。摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 在源句子和摘要句子之间的 entailment 上训练的二元分类器。预测事实/非事实。
- **基于 QA 的事实性。** 向 QA 模型提出答案在源中的问题。如果摘要支持不同答案，标记。
- **实体级 F1。** 比较源与摘要中的命名实体。仅在摘要中的实体值得怀疑。

对于任何事实性重要的用户面向内容（新闻、医疗、法律、金融），抽取是更安全的默认。生成式需要在循环中有事实性检查。

## 使用

2026 年技术栈：

| 用途 | 推荐 |
|------|------|
| 新闻，3-5 句摘要，英语 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优的 T5 |
| 多文档、长篇 | 任何 32k+ 上下文的 LLM，带提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，结构上低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

2026 年当算力不是约束时，长上下文 LLM 通常优于专业模型。权衡是成本和可复现性；专业模型给出更一致的输出。

## 交付

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择抽取式或生成式，命名库，事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定任务（文档类型、合规要求、长度、算力预算），输出：

1. 方法。抽取式或生成式。用一句话解释原因。
2. 起始模型/库。命名它。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed` 或 LLM 提示。
3. 评估计划。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干提取的 rouge-score）。如果是生成式加上事实性检查。
4. 一个要探查的失败模式。实体替换是生成式新闻摘要中最常见的；在源实体未出现在摘要中的样本上标记。

未经事实性门控，拒绝医疗、法律、金融或受监管内容的生成式摘要。标记超过模型上下文窗口的输入为需要分块 map-reduce 摘要（不仅仅是截断）。
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。将 top-3 句子与参考摘要比较。测量 ROUGE-L。在 CNN/DailyMail 风格文章上你应该看到 30-45 ROUGE-L。
2. **中等。** 实现实体级事实性：从源和摘要（spaCy）中提取命名实体，计算源实体在摘要中的召回率和摘要实体对源的精确率。高精确率和低召回率意味着安全但简洁；低精确率意味着幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（按实体 F1）和每个摘要的成本。记录每个胜出的地方。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 抽取式 | 选句子 | 从源逐字返回句子。永不合幻觉。 |
| 生成式 | 重写 | 在源条件下生成新文本。可能幻觉。 |
| ROUGE | 摘要指标 | 系统输出和参考之间的 N-gram / LCS 重叠。 |
| TextRank | 基于图的抽取 | 在句子相似度图上的 PageRank。 |
| 事实性 | 是否正确 | 摘要声称是否被源支持。 |
| 幻觉 | 编造的内容 | 摘要中源不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) —— 抽取式经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) —— BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) —— Pegasus 和间隙句目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) —— ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) —— 事实性格局论文。