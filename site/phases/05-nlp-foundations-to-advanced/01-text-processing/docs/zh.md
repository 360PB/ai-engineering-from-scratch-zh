# 文本处理——分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是二者之间的桥梁。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 2 · 14（朴素贝叶斯）
**耗时：** 约 45 分钟

## 问题

模型无法读取 "The cats were running."。它只能读取整数。

每个 NLP 系统的起点都面临同样的三个问题。单词从哪里开始。单词的词根是什么。如何将 "run"、"running"、"ran" 在有帮助时视为同一事物，在不需要时视为不同事物。

分词做错了，模型就学不到正确的东西。如果分词器把 `don't` 当成一个 token，却把 `do n't` 当成两个，训练分布就会分裂。如果词干提取器把 `organization` 和 `organ` 压缩成同一个词干，主题建模就会失效。如果词形还原器需要词性上下文而你没有传入，动词就会被当成名词处理。

本课从零开始构建三个预处理原语，然后展示 NLTK 和 spaCy 如何完成同样的工作，让你看清其中的权衡。

## 概念

三个操作。每个都有其职责和失败模式。

**分词（Tokenization）** 将字符串拆分为 token。"token" 一词故意含糊，因为正确的粒度取决于任务。经典 NLP 用词级。Transformer 用子词级。无空白字符的语言用字符级。

**词干提取（Stemming）** 用规则切掉词缀。快、激进、笨。`running -> run`。`organization -> organ`。第二个就是它的失败模式。

**词形还原（Lemmatization）** 利用语法知识将单词还原为其词典形式。更慢、精确、需要查找表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则。速度优先且可以容忍噪音时用词干提取（搜索索引、粗略分类）。语义优先时用词形还原（问答、语义搜索、用户会阅读的任何内容）。

## 构建

### 步骤 1：正则表达式词级分词器

最简单的实用分词器在非字母数字字符处切分，同时将标点符号保留为独立 token。不完美，不是最终版本，但一行代码就能跑。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

按优先级顺序排列的三个模式。带可选内撇号的单词（`don't`、`it's`）。纯数字。任何单个非空白非字母数字字符作为独立 token（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要留意的失败模式。`3pm` 被拆分成 `['3', 'pm']`，因为我们在字母串和数字串之间交替切换。对大多数任务来说足够好了。URL、电子邮件、话题标签都会出问题。生产环境中，在通用模式之前添加特殊模式。

### 步骤 2：Porter 词干提取器（仅第 1a 步）

完整的 Porter 算法有五个阶段的规则。单独第一步 1a 就覆盖了最常见的英语词缀，也能教会你基本模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

自上而下读规则。`ies -> i` 规则是为什么 `ponies -> poni` 而不是 `pony` 的原因。真正的 Porter 有第 1b 步可以修复它。规则之间存在竞争。先出现的规则优先。顺序比任何单一规则都重要。

### 步骤 3：基于查表的词形还原器

真正的词形还原需要形态学知识。一个可讲授的简化版本使用小型词元表加回退机制。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一种情况是关键的教学时刻。`watched` 不在表中，我们的回退只处理 `ing`。真正的词形还原需要覆盖 `ed`、不规则动词、比较级形容词、有音变的复数（`children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整的形态分析器。

### 步骤 4：串联使用

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是词性标注器。Phase 5 · 07（词性标注）会构建一个。目前默认所有词为 `NOUN`，并承认这个局限。

## 使用

NLTK 和 spaCy 提供生产级版本。各用几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩略形式、Unicode 和你正则表达式漏掉的边界情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将 POS 标签从 NLTK 的 Penn Treebank 标注集翻译成 WordNet 的缩写集。上面的翻译连接代码是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个流程隐藏在 `nlp(text)` 之后。分词、词性标注和词形还原全部运行。比 NLTK 快且开箱即用更准确。代价是你无法轻易替换单个组件。

### 何时选哪个

| 场景 | 选择 |
|------|------|
| 教学、研究、需要替换组件 | NLTK |
| 生产、多语言、速度优先 | spaCy |
| Transformer 流水线（你会用模型的 tokenizer 做分词） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人警告你的两个失败模式

大多数教程只教算法就停了。两个问题会在真实预处理流水线中咬你一口，而且几乎从不被覆盖。

**可复现性漂移（Reproducibility drift）。** NLTK 和 spaCy 在不同版本之间改变分词和词形还原的行为。spaCy 2.x 中产生 `['do', "n't"]` 的文本在 3.x 中可能产生 `["don't"]`。你的模型训练于一种分布。推理现在运行在不同的分布上。准确率悄悄下降，没人知道为什么。在 `requirements.txt` 中锁定库版本。写一个预处理回归测试，用 20 个示例句子冻结预期的分词结果。每次升级时运行它。

**训练 / 推理不匹配（Training / inference mismatch）。** 用激进的预处理训练（转小写、去停用词、词干提取），在原始用户输入上部署，看着性能崩溃。这是生产 NLP 中最常见的单点故障。如果你在训练时做了预处理，推理时必须运行完全相同的函数。将预处理作为函数放在模型包内，而不是作为笔记本单元格让部署团队重写。

## 交付

一个可重用的提示词，帮助工程师在不读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-sentiment-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐分词、词干提取和词形还原配置。
phase: 5
lesson: 01
---

你提供经典 NLP 预处理建议。给定任务描述，输出：

1. 分词选择（正则表达式、NLTK word_tokenize、spaCy 或 Transformer tokenizer）。解释原因。
2. 是否需要词干提取、词形还原、两者都要或都不要。解释原因。
3. 具体库调用。命名函数。如果涉及 NLTK，引用 POS 标签翻译代码。
4. 用户应测试的一个失败模式。

拒绝为用户可见的文本推荐词干提取。拒绝在无 POS 标签时推荐词形还原。标记非英语输入需要不同的流水线。
```

## 练习

1. **简单。** 扩展 `tokenize` 函数将 URL 作为单个 token 保留。测试：`tokenize("Visit https://example.com today.")` 应产生一个 URL token。
2. **中等。** 实现 Porter 第 1b 步。如果一个词包含元音且以 `ed` 或 `ing` 结尾，则去掉它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个使用 WordNet 作为查表但当 WordNet 无条目时回退到 Porter 词干提取器的词形还原器。在带标注的语料库上测量准确率，与纯 WordNet 和纯 Porter 比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Token | 词 | 模型消耗的任何单位。可以是词、子词、字符或字节。 |
| Stem | 词干 | 基于规则的词缀剥离结果。不一定是真实单词。 |
| Lemma | 词元 | 词典形式。即你在词典中查找时使用的形式。需要语法上下文才能正确计算。 |
| POS tag | 词性标注 | 类别，如 NOUN、VERB、ADJ。准确词形还原需要它。 |
| Morphology | 形态学 | 词形规则。单词如何根据时态、数、格变化。词形还原依赖于此。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) —— 原始论文，五页，仍然是 最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) —— 真实流水线如何连接。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) —— 你还没想到的分词边界情况。