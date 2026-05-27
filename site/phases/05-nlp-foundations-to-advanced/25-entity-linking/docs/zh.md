# 实体链接与消歧

> NER 找到"Paris。"实体链接决定：法国巴黎？Paris Hilton？德克萨斯州巴黎？Paris（特洛伊王子）？不链接，你的知识图谱保持歧义。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 06（NER）、Phase 5 · 24（共指消解）
**耗时：** 约 60 分钟

## 问题

一句话读："Jordan beat the press." 你的 NER 标注"Jordan"为 PERSON。好。但*哪个* Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（UC Berkeley ML 教授——是的，在 ML 论文中这种混淆是真实的）？
- Jordan（那个国家）？
- Jordan（希伯来语名字）？

实体链接（EL）将每个提及解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域 KB。两个子任务：

1. **候选生成。** 给定"Jordan"，哪些 KB 条目是可信的？
2. **消歧。** 给定上下文，哪个候选是正确的？

两个步骤都可学习。两者都有基准。组合流水线已经稳定了十年——改变的是消歧器的质量。

## 概念

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**候选生成。** 给定提及表面形式（"Jordan"），在别名索引中查找候选。Wikipedia 别名词典覆盖大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型索引每提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好，快，不需要训练。
2. **基于嵌入（ESS / REL / BLINK）。** 编码提及 + 上下文。编码每个候选的描述。取最大余弦。2020-2024 年默认。
3. **生成式（GENRE, 2021；基于 LLM, 2023+）。** 逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）确保只输出有效标题作为 KB id。现代后裔是 REL-GEN 和带结构化输出的 LLM 提示 EL。

**端到端 vs 流水线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）一次运行 NER + 候选生成 + 消歧。流水线系统在生产中仍占主导，因为你可交换组件。

### 两个测量

- **提及召回率（候选生成）。** 黄金提及中正确 KB 条目出现在候选列表中的比例。整个流水线的地板。
- **消歧准确率 / F1。** 给定正确候选，top-1 有多少次是对的。

始终报告两者。80% 候选召回上 99% 消歧的系统是 80% 流水线。

## 构建

### 步骤 1：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 1800 万（别名，实体）对。从 Wikidata 转储下载。存储为倒排索引。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠是玩具。用嵌入上的余弦相似度替换（见 `code/main.py` 步骤 2 的 Transformer 版本）。

### 步骤 3：基于嵌入（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引时，一次嵌入每个 KB 实体。查询时，一次嵌入提及 + 上下文，对候选池做点积，取最大。

### 步骤 4：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）确保只输出有效标题。与 KB 支持的 trie 紧密集成。现代后裔是 REL-GEN 和带结构化输出的 LLM 提示 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年发货的最简单 EL 流水线。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准：1,393 篇路透社文章，34k 提及，Wikipedia 实体。报告在-KB 准确率（`P@1`）和 KB 外 NIL 检测率。

## 陷阱

- **NIL 处理。** 某些提及不在 KB 中（新兴实体、不知名人物）。系统必须预测 NIL 而非猜错实体。单独测量。
- **提及边界错误。** 上游 NER 漏掉部分跨度（"Bank of America" 只标注为"Bank"）。EL 召回下降。
- **流行度偏差。** 训练系统过度预测频繁实体。ML 论文中"Michael I. Jordan"的提及经常链接到篮球 Jordan。
- **跨语言 EL。** 将中文文本中的提及映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **KB 陈旧。** 新公司、事件、人物不在去年的 Wikipedia 转储中。生产流水线需要刷新循环。

## 使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 通用英语 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好，每日提及少 | 提示 Claude/GPT-4 带候选列表 + 约束 JSON |
| 领域特定 KB（医疗、法律） | 带 KB 感知检索的定制 BERT + 在领域 AIDA 风格集上微调 |
| 极低延迟 | 仅精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年发货的生产模式：NER → 共指 → 在每个提及上 EL → 将簇折叠为每簇一个规范实体。输出：文档中每个实体一个 KB id，而非每提及一个。

## 交付

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: 设计实体链接流水线——KB、候选生成器、消歧器、评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定用例（领域 KB、语言、体积、延迟预算），输出：

1. 知识库。Wikidata / Wikipedia / 自定义 KB。版本日期。刷新节奏。
2. 候选生成器。别名索引、嵌入或混合。目标提及召回 @ K。
3. 消歧器。先验 + 上下文、基于嵌入、生成式或 LLM 提示。
4. NIL 策略。顶部分数阈值、分类器或显式 NIL 候选。
5. 评估。在留出集上的提及召回 @ 30、top-1 准确率、NIL 检测 F1。

拒绝任何没有提及召回基线的 EL 流水线（你无法评估消歧器而不知道候选生成浮现了正确实体）。拒绝任何使用 LLM 提示 EL 而不约束输出到有效 KB id 的流水线。标记系统中的流行度偏差影响少数实体（例如名称冲突）而无领域微调。
```

## 练习

1. **简单。** 在 10 个歧义提及（Paris、Jordan、Apple）上实现 `code/main.py` 中的先验+上下文消歧器。人工标注正确实体。测量准确率。
2. **中等。** 用句子 Transformer 编码 50 个歧义提及。嵌入每个候选的描述。比较基于嵌入的消歧与 Jaccard 上下文重叠。
3. **困难。** 构建 1k-实体领域 KB（例如你公司的员工 + 产品）。端到端实现 NER + EL。在 100 条留出句子上测量精确率和召回率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 实体链接（EL） | 链接到 Wikipedia | 将提及映射到唯一 KB 条目。 |
| 候选生成 | 可能是谁？ | 为提及返回可信 KB 条目的短列表。 |
| 消歧 | 选正确的 | 使用上下文对候选评分，取胜者。 |
| 别名索引 | 查找表 | 从表面形式 → 候选实体映射。 |
| NIL | 不在 KB 中 | 显式预测无 KB 条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | 基准 | 带有黄金实体链接的 1,393 篇路透社文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) —— 基础先验+上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) —— 基于嵌入的主力。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) —— 带约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) —— 基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) —— 开放生产技术栈。