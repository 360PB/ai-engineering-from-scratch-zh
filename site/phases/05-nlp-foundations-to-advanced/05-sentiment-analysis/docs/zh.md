# 情感分析

> 经典的 NLP 任务。关于经典文本分类，你需要知道的几乎都在这里了。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 02（BoW + TF-IDF）、Phase 2 · 14（朴素贝叶斯）
**耗时：** 约 75 分钟

## 问题

"The food was not great." 正向还是负向？

情感听起来简单。评论者说他们喜欢或不喜欢某样东西。给句子打标签。它成为经典 NLP 任务的原因是：每个看似简单的案例背后都藏着一个困难的。否定翻转含义。反讽反转含义。"Not bad at all" 是正向的，尽管有两个负向编码的词。表情符号比周围文本携带更多信号。领域词汇很重要（音乐评论中的 `tight` 与时尚评论中的 `tight` 意思不同）。

情感是经典 NLP 的实验场。如果你理解为什么每个朴素基线都有一个特定的失败模式，你就理解为什么每个更丰富的模型被发明出来。本课从零构建朴素贝叶斯基线，加上逻辑回归，并指出那些使生产情感成为合规级别问题的陷阱。

## 概念

经典情感分析是一个两步流程。

1. **表示。** 将文本转化为特征向量。BoW、TF-IDF 或 n-gram。
2. **分类。** 在标注样本上拟合线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是最简单但能工作的模型。假设每个特征在给定标签时相互独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时乘以概率。"朴素"的独立性假设错得可笑，但结果却惊人地强。原因：对于稀疏文本特征和中度数据，分类器更关心每个词倾向于哪一边，而不是强度。

逻辑回归修正了独立性假设。它学习每个特征的权重，包括负权重。`not good` 作为二元组特征得到负权重。朴素贝叶斯无法为从未标注过的二元组做到这一点。

## 构建

### 步骤 1：一个真实的迷你数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

有意做小。真实工作使用数万样本（IMDb、SST-2、Yelp 极性）。数学相同。

### 步骤 2：从零构建多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）是拉普拉斯平滑。没有它，未在类别中出现的词的概率为零，对数就会爆炸。`alpha=0.01` 在实践中常见。`alpha=1.0` 是教学默认值。

### 步骤 3：从零构建逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里很重要。文本特征是稀疏的；没有 L2 模型会记住训练样本。从 `0.01` 开始调参。

### 步骤 4：处理否定（失败模式）

考虑 "not good" 和 "not bad"。BoW 分类器看到 `{not, good}` 和 `{not, bad}`，从训练中更常出现的那个学习。二元组分类器看到 `not_good` 和 `not_bad`，将它们作为不同特征学习。通常这就够了。

一个更粗糙但在没二元组时的有效修复：**否定作用域（negation scoping）**。在否定词后给 token 加上 `NOT_` 前缀，直到下一个标点符号。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同特征。分类器可以给它们相反的权重。三行预处理，在情感基准上有可测量的准确率提升。

### 步骤 5：重要的评估指标

类别不平衡时单独看准确率会误导人。真实情感语料库通常 70-80% 正向或 70-80% 负向；常数多数分类器得 80% 准确率但毫无价值。报告以下全部：

- **每类别精确率和召回率。** 每个类别一对。宏观平均得到一个尊重类别平衡的单一数字。
- **宏-F1（不平衡数据的主要指标）。** 每类别 F1 的均值，等权重。当类别不平衡时用它代替准确率。
- **加权-F1（备选）。** 与宏-F1 相同，但按类别频率加权。当不平衡本身有业务含义时与宏-F1 一起报告。
- **混淆矩阵。** 原始计数。在信任任何标量指标之前总要检查；它揭示模型在哪对类别上混淆。
- **每类别错误样本。** 每个类别取 5 个错误预测。读它们。读实际错误无可替代。

对于严重不平衡数据（>95-5 比例），报告 **AUROC** 和 **AUPRC** 而非准确率。AUPRC 对少数类更敏感，而这通常才是你关心的（垃圾邮件、欺诈、罕见情感）。

**需要避免的常见 bug。** 在不平衡数据上报告微-F1 而非宏-F1 会给出一个看起来很高的数字，因为它被多数类主导。宏-F1 迫使你看到少数类表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用

scikit-learn 六行代码，正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

三件事要注意。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加二元组使 `not_good` 成为特征。`sublinear_tf=True` 抑制重复词。这三个标志是 SST-2 上 75% 准确率基线和 85% 准确率之间的差异。

### 何时使用 Transformer

- 反讽检测。经典模型在这里失败。不用想了。
- 情感在文档中途转变的长评论。
- 方面级情感。"Camera was great but battery was terrible." 你需要将情感归因到方面。只有 Transformer 或结构化输出模型。
- 非英语、低资源语言。多语言 BERT 免费给你一个零样本基线。

如果你需要以上任何一种，跳到 phase 7（Transformer 深潜）。否则，TF-IDF 加二元组加否定处理的朴素贝叶斯或逻辑回归是你 2026 年的生产基线。

### 可复现性陷阱（再次）

重新训练情感模型是常规操作。重新评估不是。论文中报告的准确率数字使用特定的数据划分、特定预处理、特定分词器。如果不使用相同流水线就拿你的新模型和基线比较，你会得到误导性的差异。始终在你自己流水线上重新生成基线，而不是论文的数字。

## 交付

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: 为新数据集设计情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、大小、标签粒度、延迟预算），输出：

1. 特征提取配方。指定分词器、n-gram 范围、停用词策略（通常保留）、否定处理（作用域前缀或二元组）。
2. 分类器。基线用朴素贝叶斯，生产用逻辑回归，只有当领域需要反讽/方面/跨语言时才用 Transformer。
3. 评估计划。报告精确率、召回率、F1、混淆矩阵和每类别错误样本（不只是标量）。
4. 上线后要监控的一个失败模式。领域漂移和反讽是前两名。

拒绝为情感任务推荐去掉停用词。拒绝在类别不平衡时（如 90% 正向）将准确率作为唯一指标推荐。标记子词丰富语言需要 FastText 或 Transformer 嵌入而非词级 TF-IDF。
```

## 练习

1. **简单。** 在 scikit-learn 流水线中添加 `apply_negation` 作为预处理步骤，在小型情感数据集上测量 F1 差值。
2. **中等。** 实现类别加权逻辑回归（向 scikit-learn 传入 `class_weight="balanced"`，或自己推导梯度）。在合成的 90-10 类别不平衡上测量效果。
3. **困难。** 在情感模型残差上训练第二个分类器构建反讽检测器。记录你的实验设置。当你的准确率低于随机（2 类反讽的随机基线约 50%，大多数初次尝试落在这里）时警告读者。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 极性 | 正向或负向 | 二元标签；有时扩展到中性或细粒度（五星）。 |
| 方面级情感 | 每方面极性 | 将情感归因到文本中提到的特定实体或属性。 |
| 否定作用域 | 反转附近 token | 在 "not" 后的 token 加上 `NOT_` 前缀直到标点符号。 |
| 拉普拉斯平滑 | 计数加 1 | 防止朴素贝叶斯中出现零概率特征。 |
| L2 正则化 | 收缩权重 | 在损失中添加 `lambda * sum(w^2)`。对稀疏文本特征必不可少。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) —— 奠基性综述。长，但前四节覆盖了所有经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) —— 这篇论文证明二元组 + 朴素贝叶斯在短文本上难有对手。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— `CountVectorizer`、`TfidfVectorizer` 和你将调的所有旋钮的参考文档。