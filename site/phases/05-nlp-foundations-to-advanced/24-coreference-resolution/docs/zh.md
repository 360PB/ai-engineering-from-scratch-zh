# 共指消解

> "她打了电话。他没接。医生在吃午饭。"三个人称，两个被提到但没人有名字。共指消解弄清楚谁是谁。

**类型：** 学习
**语言：** Python
**先修课程：** Phase 5 · 06（NER）、Phase 5 · 07（POS 和句法分析）
**耗时：** 约 60 分钟

## 问题

从一篇 300 词的文章中提取对 Apple Inc. 的每次提及。当文章说"Apple"时容易。当它说"the company"、"they"、"Cupertino's technology giant"或"Jobs's firm"时困难。不将这些提及解析到同一实体，你的 NER 流水线会漏掉 60-80% 的提及。

共指消解将每个指代同一真实世界实体的表达式链接成一个簇。它是表层 NLP（NER、句法分析）和下游语义（IE、QA、摘要、KG）之间的粘合剂。

2026 年为什么重要：

- 摘要："The CEO announced..." vs "Tim Cook announced..."——摘要应该点名 CEO。
- 问答："谁她打电话？"需要解析"她"。
- 信息抽取：知识图谱中"PER1 创立了 Apple"和"Jobs 创立了 Apple"作为单独条目是错误的。
- 多文档 IE：跨关于同一事件的文章合并提及是跨文档共指。

## 概念

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任务。** 输入：文档。输出：提及（跨度）的聚类，其中每个簇指代一个实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **名词性提及。** "the CEO", "the company"
- **代词性提及。** "he", "she", "they", "it"
- **同位语。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Hobbs, 1978）。** 使用语法规则的句法树形代词解析。好的基线。在代词上出人意料地难打败。
2. **提及对分类器。** 对于每对提及 (m_i, m_j)，预测它们是否共指。通过传递闭包聚类。2016 年前的标准。
3. **提及排序。** 对于每个提及，对候选先行词（包括"无先行词"）排序。取最高。
4. **基于跨度的端到端（Lee et al., 2017）。** Transformer 编码器。列举长度上限以下的所有候选跨度。预测提及分数。为每个跨度预测先行词概率。贪婪聚类。现代默认。
5. **生成式（2024+）。** 提示 LLM："列出本文中每个代词及其先行词。"在简单案例上效果良好，在长文档和罕见指代上挣扎。

**评估指标。** 五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标捕捉聚类质量。报告前三个的平均作为 CoNLL F1。2026 年在 CoNLL-2012 上的最先进：约 83 F1。

**已知困难案例。**

- 指向前几页引入的实体的确定性描述。
- 桥接回指（"the wheels" → 前面提到的汽车）。
- 中文和日语等语言中的零回指。
- 逆 Antecedent（代词在指代之前）："当**她**走进来时，Mary 笑了。"

## 构建

### 步骤 1：预训练神经共指（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你得到类似：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 步骤 2：基于规则的代词解析器（教学）

见 `code/main.py` 获取仅 stdlib 实现：

1. 提取提及：命名实体（大写跨度）、代词（dict 查找）、确定性描述（"the X"）。
2. 对于每个代词，查看前 K 个提及并评分：
   - 性别/数一致（启发式）
   - 近因（近者胜）
   - 句法角色（优先主语）
3. 链接得分最高的先行词。

与神经模型没有竞争力。但它展示了搜索空间和端到端模型必须做出的决策。

### 步骤 3：使用 LLM 做共指

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

两个失败模式要注意。首先，LLM 过度合并（"him" 和 "her" 指两个不同的人）。其次，LLM 在长文档中静默丢弃提及。始终用跨度偏移检查验证。

### 步骤 4：评估

标准 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，从标注测试集上的跨度级精确率和召回率开始，然后添加提及链接 F1。

## 陷阱

- **单例爆炸。** 某些系统报告每个提及为自己的簇。B³ 宽容。MUC 惩罚这个。始终检查所有三个指标。
- **长上下文中的代词。** 性能在超过 2,000 token 的文档上下降约 15 F1。小心分块。
- **性别假设。** 硬编码性别规则在非二元指代、组织、动物上失效。使用学习模型或中性评分。
- **长文档上的 LLM 漂移。** 单个 API 调用无法可靠地在 50+ 段落间聚类提及。使用滑动窗口 + 合并。

## 使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 英语，单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经共指 |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专业端到端模型（2025-26 SOTA） |
| 快速 LLM 基线 | 带结构化输出共指提示的 GPT-4o / Claude |
| 生产对话系统 | 基于规则的回退 + 神经主要 + 关键槽人工审查 |

2026 年上线的集成模式：先运行 NER，运行共指，将共指簇合并到 NER 实体。下游任务每个簇看到一个实体，而非每个提及一个实体。

## 交付

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: 选择共指方法、评估计划和集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定用例（单文档 / 多文档、领域、语言），输出：

1. 方法。基于规则 / 神经基于跨度 / LLM 提示 / 混合。一句话理由。
2. 模型。如果神经，命名检查点。
3. 集成。操作顺序：分词 → NER → 共指 → 下游任务。
4. 评估。在留出集上的 CoNLL F1（MUC + B³ + CEAF-φ4 平均）+ 20 文档上手工簇审查。

拒绝超过 2,000 token 的文档使用仅 LLM 共指而无滑动窗口合并。拒绝任何没有提及级精确率-召回率报告的共指流水线。标记在人口多样性文本上部署的性别启发式系统。
```

## 练习

1. **简单。** 在 5 个人工制作的段落上运行 `code/main.py` 中的基于规则解析器。测量 vs 地面真值的提及链接准确率。
2. **中等。** 在新闻文章上使用预训练神经共指模型。将簇与你自己的人工标注比较。哪里失败了？
3. **困难。** 构建共指增强 NER 流水线：先 NER，然后通过共指簇合并。在 100 篇文章上测量 vs 仅 NER 的实体覆盖改进。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Mention | 指代 | 指代实体的文本跨度（名称、代词、名词短语）。 |
| Antecedent | "it" 指的是什么 | 较晚提及与较早提及共指的先前提及。 |
| Cluster | 实体的提及 | 都指代同一真实世界实体的提及集合。 |
| Anaphora | 向后引用 | 较晚提及指较早提及（"he" → "John"）。 |
| Cataphora | 向前引用 | 较早提及指较晚提及（"When he arrived, John..."）。 |
| Bridging | 隐含引用 | "I bought a car. The wheels were bad."（那辆车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) —— 规范教科书章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) —— 基于跨度的端到端。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) —— 改进共指的预训练。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) —— 基准。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) —— 基于规则的经典。