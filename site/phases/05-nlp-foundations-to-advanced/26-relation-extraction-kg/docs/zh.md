# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接将它们锚定。关系抽取找到它们之间的边。知识图谱是节点、边及其溯源的总和。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 5 · 06（NER），Phase 5 · 25（实体链接）
**时长：** 约60分钟

## 问题

分析师读到："Tim Cook 于2011年成为苹果 CEO。"四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）将自由文本转化为结构化三元组 `(subject, relation, object)`。跨语料库聚合，你就有了知识图谱。聚合并查询，你就有了用于 RAG、分析或合规审计的推理底层。

2026年的难题：LLM 热情地抽取关系。太热情了。它们幻觉出源文本不支持的三元组。没有溯源，你无法区分真实三元组和看似合理的虚构。2026年的答案是 AEVS 风格的锚定-验证流水线。

## 核心概念

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体（Wikidata 属性、FIBO、UMLS）或开放集（OpenIE 风格，什么都行）。

**三种抽取方法。**

1. **规则 / 基于模式。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加手工正则。脆弱，精确，可解释。
2. **监督分类器。** 给定句子中两个实体提及，从固定集合预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年标准。
3. **生成式 LLM。** 提示模型输出三元组。开箱即用。需要溯源，否则幻觉看似合理但无用的东西。

**AEVS（锚定-抽取-验证-补充，2026）。** 当前的幻觉缓解框架：

- **锚定。** 用精确位置识别每个实体跨和关系短语跨。
- **抽取。** 生成链接到锚定跨的三元组。
- **验证。** 将每个三元组元素匹配回源文本；拒绝任何不支持的内容。
- **补充。** 覆盖率确保没有锚定跨被丢弃。

幻觉急剧下降。需要更多计算但可审计。

**开放 vs 封闭的权衡。**

- **封闭本体。** 固定属性列表（如 Wikidata 的 11,000+ 属性）。可预测。可查询。难以发明。
- **开放 IE。** 任何口头短语都成为关系。高召回。低精确。查询麻烦。

生产 KG 通常混合：开放 IE 用于发现，然后在合并到主图之前将关系规范化为封闭本体。

## 动手实现

### 步骤 1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

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

### 步骤 3：带锚定的 LLM 提示抽取

```python
prompt = f"""从文本中抽取 (subject, relation, object) 三元组。
每个三元组包含源文本中的精确字符跨。

文本：{text}

输出 JSON：
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}, ...]

只包含完全由文本支持的三元组。不做超出所陈述内容的推理。
```

### 步骤 4：规范化为封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by"（反转主语/宾语）
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 丢弃未映射的开放关系或路由给人工审核
```

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

## 陷阱

- **RE 前先做共指。** "He founded Apple"——RE 需要知道"他"是谁。先跑共指（第24课）。
- **实体规范。** "Apple Inc"和"Apple"必须解析到同一节点。先做实体链接（第25课）。
- **幻觉三元组。** LLM 发出文本不支持的三元组。强制执行跨验证。
- **关系规范漂移。** 开放 IE 关系不一致（"was born in," "came from," "is a native of"）。折叠到规范 id，否则图谱不可查询。
- **时间错误。** "Tim Cook is CEO of Apple"——现在为真，2005年为假。许多关系是有时间限制的。使用限定符（Wikidata 的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在维基百科上训练。法律、医疗和科学文本通常需要领域微调 RE 模型。

## 用现成库

2026年技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域特定（生物医学、法律） | SciREX 风格领域微调 + 自定义本体 |
| LLM 提示，可审计输出 | AEVS 流水线：锚定 → 抽取 → 验证 → 补充 |
| 大容量新闻 IE | 基于模式 + 监督混合 |
| 从零构建 KG | 开放 IE + 人工规范化通过 |
| 时间 KG | 带限定符抽取（开始/结束时间、时点） |

## 产出

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: 设计带溯源和规范化的关系抽取流水线。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定语料库（领域、语言、规模）和下游用途（KG-RAG、分析、合规），输出：

1. 抽取器。基于模式 / 监督 / LLM / AEVS 混合。与精确召回目标关联的原因。
2. 本体。封闭属性列表（Wikidata / 领域）或带规范化通过的开放 IE。
3. 溯源。每个三元组带源字符跨 + doc id。审计不可妥协。
4. 合并策略。规范实体 id + 关系 id + 时间限定符；去重策略。
5. 评估。200 个人工标注三元组精确率/召回率 + LLM 抽取样本的幻觉率。

拒绝任何没有跨验证（源溯源）的基于 LLM 的 RE 流水线。拒绝没有规范化的开放 IE 输出流入生产图谱。标记任何在有时间限制关系上没有时间限定符的流水线（雇主、配偶、职位）。
```

## 练习

1. **简单。** 在5个新闻句子文章上运行 `code/main.py` 中的模式抽取器。人工检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或小 LLM）。对比三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 流水线：LLM 抽取 + 根据源验证跨。在50个维基百科风格句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|-----------------------|
| Triple | 主语-关系-宾语 | `(s, r, o)` 元组，是 KG 的原子单位。 |
| Open IE | 抽取任何东西 | 开放词汇关系短语；高召回，低精确。 |
| Closed ontology | 固定模式 | 关系类型的有界集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | 规范化一切 | 将表面名称 / 关系映射到规范 id。 |
| AEVS | 接地抽取 | 锚定-抽取-验证-补充流水线（2026）。 |
| Provenance | 溯源链接 | 每个三元组带 doc id + 字符跨到其源。 |
| Distant supervision | 廉价标签 | 将文本与现有 KG 对齐以创建训练数据。 |

## 扩展阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 工作马。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合 IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026年幻觉缓解设计。
- [Wikidata SPARQL 教程](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图查询。