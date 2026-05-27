# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接将其锚定。关系抽取找出它们之间的边。知识图谱是节点、边及其来源的总和。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 06（NER）、Phase 5 · 25（实体链接）
**时长：** 约 60 分钟

## 问题

分析师阅读："Tim Cook 于 2011 年成为苹果公司 CEO。"四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（Relation Extraction，RE）将自由文本转化为结构化三元组 `(subject, relation, object)`。在语料库上聚合得到知识图谱。再聚合与查询，就有了用于 RAG、分析或合规审计的推理底层。

2026 年的问题：LLM 抽取关系非常积极——过于积极了。它们会产生源文本并不支持的幻觉三元组。没有来源追踪，你就无法区分真实三元组和看似合理的虚构。2026 年的答案是 AEVS 风格的锚定-抽取-验证-补充管道。

## 概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体论（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，一切皆可）。

**三种抽取方法。**

1. **规则 / 模式匹配。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工编写的正则表达式。脆弱但精确、可解释。
2. **监督分类器。** 给定句子中两个实体mention，预测固定集合中的关系。在 TACRED、ACE、KBP 上训练。2015–2022 年标准方法。
3. **生成式 LLM。** 提示模型输出三元组。开箱即用。需要来源追踪，否则会幻觉出看起来像样的垃圾。

**AEVS（Anchor-Extraction-Verification-Supplement，2026）。** 当前抑制幻觉的框架：

- **锚定（Anchor）。** 用精确位置标识每个实体跨度和关系短语跨度。
- **抽取（Extract）。** 生成与锚定跨度关联的三元组。
- **验证（Verify）。** 将每个三元组元素与源文本匹配；拒绝任何不支持的内容。
- **补充（Supplement）。** 覆盖率通过确保没有锚定跨度被遗漏。

幻觉率大幅下降。需要更多算力，但可审计。

**开放与封闭的权衡。**

- **封闭本体论。** 固定属性列表（如 Wikidata 的 11000+ 属性）。可预测、可查询、不易杜撰。
- **开放 IE。** 任何语言短语都可以作为关系。高召回、低精确、难以查询。

生产知识图谱通常混合使用：开放 IE 做发现，然后在合并到主图之前将关系规范化为封闭本体论。

## 构建

### 步骤 1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

参见 `code/main.py` 中的完整演示抽取器。Hearst 模式仍在领域特定管道中使用，因为它们可调试。

### 步骤 2：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个序列到序列的关系抽取器：文本输入，三元组输出，已经是 Wikidata 属性 ID 格式。在远程监督数据上微调。标准开源基座模型。

### 步骤 3：LLM 提示抽取 + 锚定

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证每个返回的跨度与源文本的一致性。凡 `text[start:end] != triple_entity` 的一律拒绝。这是 AEVS "验证"步骤的最小化形式。

### 步骤 4：规范化为封闭本体论

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (subject/object 反转)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 丢弃未映射的开放关系，或转人工审核
```

规范化通常是工程量的 60–80%。留足时间。

### 步骤 5：构建小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是所有 RAG-over-KG 系统的原子单元。可以用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储扩展。

## 陷阱

- **RE 前先做共指消解。** "He founded Apple"——RE 需要知道 "he" 是谁。先运行共指（课程 24）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须解析到同一节点。先做实体链接（课程 25）。
- **幻觉三元组。** LLM 会发出文本不支持的三元组。强制执行跨度验证。
- **关系规范化漂移。** 开放 IE 的关系不一致（"was born in"、"came from"、"is a native of"）。折叠为规范 ID，否则图无法查询。
- **时间错误。** "Tim Cook is CEO of Apple"——现在是 true，2005 年是 false。许多关系是有时间限制的。使用限定符（Wikidata 的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要领域微调的 RE 模型。

## 使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 快速上线，通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域特定（生物医学、法律） | SciREX 风格领域微调 + 自定义本体论 |
| LLM 提示，可审计输出 | AEVS 管道：锚定 → 抽取 → 验证 → 补充 |
| 高容量新闻 IE | 基于模式 + 监督混合 |
| 从零构建 KG | 开放 IE + 人工规范化环节 |
| 时序知识图谱 | 带限定符抽取（开始/结束时间、时点） |

集成模式：NER → 共指 → 实体链接 → 关系抽取 → 本体映射 → 图加载。每个环节都是潜在的质量门。

## 上线

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: 设计带来源追踪和规范化的关系抽取管道。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定语料库（领域、语言、体量）和下游用途（KG-RAG、分析、合规），输出：

1. 抽取器。基于模式 / 监督 / LLM / AEVS 混合。理由与精确率/召回率目标挂钩。
2. 本体论。封闭属性列表（Wikidata / 领域）或开放 IE + 规范化环节。
3. 来源追踪。每个三元组携带源 char-span + 文档 ID。审计需求不可妥协。
4. 合并策略。规范实体 ID + 关系 ID + 时间限定符；去重策略。
5. 评估。200 个人工标注三元组的精确率/召回率 + LLM 抽取样本的幻觉率。

拒绝任何无跨度验证（来源追踪）的 LLM RE 管道。拒绝未规范化的开放 IE 输出流入生产图谱。标记有时间限制关系（如雇主、配偶、职位）但无时间限定符的管道。
```

## 练习

1. **简单。** 在 5 篇新闻文章句子上运行 `code/main.py` 中的模式抽取器。人工检查精确率。
2. **中等。** 使用 REBEL（或小型 LLM）在同一组句子上抽取。比较三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 管道：LLM 抽取 + 验证源文本跨度。在 50 个 Wikipedia 风格句子上测量加入验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|---------|
| Triple | 主语-谓语-宾语 | 构建 KG 的基本单元 `(s, r, o)` 三元组。 |
| Open IE | 开放抽取 | 开放词汇关系短语；高召回、低精确。 |
| Closed ontology | 封闭本体论 | 有限关系类型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | 规范化 | 将表面名称 / 关系映射到规范 ID。 |
| AEVS | 锚定抽取 | Anchor-Extraction-Verification-Supplement 管道（2026）。 |
| Provenance | 来源追踪 | 每个三元组携带文档 ID + char-span 指向其源。 |
| Distant supervision | 远程监督 | 将文本与现有 KG 对齐以创建训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE主力模型。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合 IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年幻觉抑制设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图查询。