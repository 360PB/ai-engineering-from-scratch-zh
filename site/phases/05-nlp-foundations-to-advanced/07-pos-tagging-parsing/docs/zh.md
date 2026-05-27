# 词性标注与句法解析

> 语法有一段时间不流行了。然后每个 LLM 流水线都需要验证结构化抽取，它又回来了。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 01（文本处理）、Phase 2 · 14（朴素贝叶斯）
**耗时：** 约 45 分钟

## 问题

第 01 课承诺词形还原需要词性标注。不知道 `running` 是动词，词形还原器无法将其还原为 `run`。不知道 `better` 是形容词，无法还原为 `good`。

这个承诺背后藏着一个完整的子领域。词性标注赋予每个 token 语法类别。句法解析恢复句子的树结构：哪个词修饰哪个，哪个动词支配哪些论元。经典 NLP 花二十年改进两者。然后深度学习将它们压缩成预训练 Transformer 之上的 token 分类任务，研究社区转向了。

应用社区没有。 每个结构化抽取流水线仍在底层使用 POS 和依存树。LLM 生成的 JSON 根据语法约束验证。问答系统使用依存解析分解查询。机器翻译质量评估器检查解析树的对齐。

值得了解。本课介绍标注集、基线和何时停止从零实现并调用 spaCy。

## 概念

**词性标注（POS tagging）** 给每个 token 标注语法类别。**Penn Treebank（PTB）** 标注集是英语默认。36 个标签，有随意读者觉得过于细致的区分：`NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在时，等等。**通用依存（Universal Dependencies, UD）** 标注集更粗（17 个标签），且语言无关；它成为跨语言工作的默认。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法解析（Syntactic parsing）** 生成树结构。两种主要风格：

- **成分句法解析（Constituency parsing）。** 名词短语、动词短语、介词短语相互嵌套。输出是一个非终结符类别（NP、VP、PP）的树，单词作为叶子。
- **依存解析（Dependency parsing）。** 每个词有一个它依赖的中心词，用语法关系标注。输出是一个树，每条边是一个（中心词、从属词、关系）三元组。

依存解析在 2010 年代胜出，因为它能干净地跨语言泛化，尤其是自由词序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 构建

### 步骤 1：最常见标签基线

最简单但能工作的 POS 标注器。对每个词，预测它在训练中最常出现的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线达到约 85% 准确率。不够好，但严肃模型不应低于此。

### 步骤 2：二元组 HMM 标注器

建模序列的联合概率：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（给定前一标签的标签）、发射概率（给定标签的词）。用拉普拉斯平滑从计数中估计两者。用 Viterbi 解码（标签格上的动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

二元组 HMM 在 Brown 上达到约 93% 准确率。从 85% 到 93% 的跳跃主要来自转移概率——模型学到 `DET NOUN` 常见而 `NOUN DET` 罕见。

### 步骤 3：为什么现代标注器胜出

转移 + 发射概率是局部的。它们无法捕捉 `saw` 在 "I bought a saw" 中是名词但在 "I saw the movie" 中是动词。具有任意特征（后缀、词形、前面和后面的词、词本身）的 CRF 达到约 97%。BiLSTM-CRF 或 Transformer 达到约 98%+。

这个任务的上限由标注者不一致决定。Penn Treebank 上人类标注者约 97% 的时间一致。超过 98% 的模型可能过拟合测试集。

### 步骤 4：依存解析概述

从零实现完整依存解析超出范围；标准教科书处理在 Jurafsky and Martin 中。两种经典家族需要了解：

- **基于转移的解析器**（arc-eager、arc-standard）像移位-归约解析器一样工作：读取 token，将它们移到栈上，应用创建弧的归约动作。贪心解码快。经典实现是 MaltParser。现代神经版本：Chen and Manning 的基于转移的解析器。
- **基于图的解析器**（Eisner 算法、Dozat-Manning 双仿射）为每个可能的中心-从属边打分，选择最大生成树。更慢但更准确。

对于大多数应用工作，调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就浮现出来了。

## 使用

每个生产 NLP 库将 POS 和依存解析器作为标准流水线的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快、准确，与分词 + NER + 词形还原集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP (stanza)**。Stanford CoreNLP 的继任者。60+ 语言上达到最优。
- **trankit**。基于 Transformer，UD 准确率高。
- **NLTK**。`pos_tag`。可用，慢，旧。适合教学。

### 2026 年这在哪里仍然重要

- **词形还原。** 第 01 课需要 POS 才能正确词形还原。始终需要。
- **从 LLM 输出进行结构化抽取。** 验证生成的句子符合语法约束（如主谓一致、必需修饰语）。
- **方面级情感。** 依存解析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过解析分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系是语言无关的，支持新语言的零样本结构化分析。
- **低计算流水线。** 如果你无法部署 Transformer，POS + 依存解析 + 词典能让你走得很远。

## 交付

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: 为下游 NLP 任务设计经典 POS + 依存流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定下游任务（信息抽取、重写验证、查询分解、词形还原），输出：

1. 要使用的标注集。英语-only 旧流水线用 Penn Treebank，多语言或跨语言用通用依存。
2. 库。多数生产用 spaCy，学术级多语言用 stanza，最高 UD 准确率用 trankit。命名具体模型 ID。
3. 集成模式。展示 3-5 行调用库的代码，消费所需属性（`.pos_`、`.dep_`、`.head`）。
4. 要测试的失败模式。名-动歧义（`saw`、`book`、`can`）和 PP 附着歧义是经典陷阱。取 20 个输出肉眼检查。

拒绝推荐自己写解析器。从零构建解析器是研究项目，不是应用任务。标记任何消费 POS 标签但不处理大小写变体的流水线为脆弱。
```

## 练习

1. **简单。** 在小型标注语料库（例如 NLTK 的 Brown 子集）上使用最常见标签基线，测量在留出句子上的准确率。验证约 85% 的结果。
2. **中等。** 训练上面的二元组 HMM 并报告每标签精确率/召回率。HMM 最混淆哪些标签？
3. **困难。** 使用 spaCy 的依存解析从 1000 句子样本中抽取主谓宾三元组。在 50 个手动标注的三元组上评估。记录抽取失败的地方（通常是是被动句、并列句和省略主语）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| POS tag | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标注集 | 针对英语。细粒度动词时态和名词数。 |
| 通用依存 | 多语言标注集 | 比 PTB 粗；语言中立；跨语言工作默认。 |
| 依存解析 | 句子树 | 每个词有一个中心词，每条边有语法关系。 |
| Viterbi | 动态规划 | 给定发射和转移，找到最高概率标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) —— POS 和解析的经典教科书处理。
- [Universal Dependencies project](https://universaldependencies.org/) —— 每个多语言解析器使用的跨语言标注集和树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) —— `Token` 上每个暴露属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) —— 将神经解析器带入主流的论文。