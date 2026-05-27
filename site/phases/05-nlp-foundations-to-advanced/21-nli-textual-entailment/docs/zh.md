# 自然语言推理——文本蕴含

> "t 蕴含 h" 意味着人类读 t 会推断 h 为真。NLI 是预测蕴含 / 矛盾 / 中性的任务。表面无聊，在生产中举足轻重。

**类型：** 学习
**语言：** Python
**先修课程：** Phase 5 · 05（情感分析）、Phase 5 · 13（问答）
**耗时：** 约 60 分钟

## 问题

你构建了一个摘要器。它产生了摘要。你如何知道摘要不包含幻觉？

你构建了一个聊天机器人。它回答"是"。你如何知道答案被检索到的段落支持？

你需要按主题分类 10,000 篇新闻文章。你没有训练标签。你能复用模型吗？

所有三个问题都归约为自然语言推理。NLI 问：给定前提 `t` 和假设 `h`，`h` 是被 `t` 蕴含、矛盾还是中性（无关）？

- **幻觉检查：** `t` = 源文档，`h` = 摘要声明。非蕴含 = 幻觉。
- **接地 QA：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含 = 捏造。
- **零样本分类：** `t` = 文档，`h` = 言语化标签（"This is about sports"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架都在底层使用 NLI 模型。

## 概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签。**

- **蕴含。** `t` → `h`。"The cat is on the mat" 蕴含 "There is a cat."
- **矛盾。** `t` → ¬`h`。"The cat is on the mat" 矛盾 "There is no cat."
- **中性。** 双向都无推理。"The cat is on the mat" 对 "The cat is hungry." 是中性的。

**不是逻辑蕴含。** NLI 是*自然*语言推理——典型人类读者会推断的，而非严格逻辑。"John walked his dog" 在 NLI 中蕴含 "John has a dog"，但严格一阶逻辑只有在你将所有权公理化时才会承认它。

**数据集。**

- **SNLI**（2015）。57 万对人工标注对，图像描述作为前提。窄领域。
- **MultiNLI**（2017）。10 种体裁的 43.3 万对。2026 年的标准训练语料库。
- **ANLI**（2019）。对抗性 NLI。人类特意编写设计来破坏现有模型的例子。更难。
- **DocNLI, ConTRoL**（2020-21）。文档长度前提。测试多跳和长距离推理。

**架构。** Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示送入 3 路 softmax。在 MNLI 上训练，在留出基准上评估，在分布内对上获得 90%+ 准确率。

**通过 NLI 实现零样本。** 给定文档和候选标签，将每个标签转化为假设（"This text is about sports"）。计算每个的蕴含概率。取最大。这就是 Hugging Face `zero-shot-classification` 流水线背后的机制。

## 构建

### 步骤 1：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产 NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开放的默认。DeBERTa-v3 在排行榜上领先。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 02, 0.01]}
```

模板默认是 "This example is about {label}." 用 `hypothesis_template` 自定义。不需要训练数据。不需要微调。开箱即用。

### 步骤 3：RAG 的忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度的核心。将生成的答案分解为原子声明。检查每个声明对检索上下文的支持情况。报告蕴含的部分。

### 步骤 4：手工 NLI 分类器（概念）

见 `code/main.py` 获取仅 stdlib 的玩具：通过词重叠 + 否定检测比较前提和假设。与 Transformer 模型没有竞争力——但它展示了任务的形态：两个文本输入，3 路标签输出，损失 = `{entail, contradict, neutral}` 上的交叉熵。

## 陷阱

- **仅假设捷径。** 模型可以在 SNLI 上仅从假设预测标签约 60%，因为"not"、"nobody"、"never"与矛盾相关。标签泄露检测的强基线。
- **词重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）通过 SNLI 但在 HANS/ANLI 上失败。使用对抗基准。
- **文档长度退化。** 单句 NLI 模型在文档长度前提上下降 20+ F1。对于长上下文使用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}" vs "{label}" vs "The topic is {label}" 可以使准确率摆动 10+ 点。调优模板。
- **领域不匹配。** MNLI 在一般英语上训练。法律、医疗和科学文本需要领域特定的 NLI 模型（例如 SciNLI、MedNLI）。

## 使用

2026 年技术栈：

| 用途 | 模型 |
|------|------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速/边缘 | `cross-encoder/nli-deberta-v3-base` |
| 轻量级零样本分类 | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 幻觉检测 | RAGAS / DeepEval 中的 NLI 层 |

2026 年元模式：NLI 是文本理解的管道胶带。每当你需要"A 支持 B 吗？"或"A 矛盾 B 吗？"——在调用另一个 LLM 之前使用 NLI。

## 交付

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: 为分类 / 忠实度 / 零样本任务选择 NLI 模型、标签模板和评估设置。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

给定用例（忠实度检查、零样本分类、文档级推理），输出：

1. 模型。命名 NLI 检查点。与领域、长度、语言相关的理由。
2. 模板（如果零样本）。言语化模式。示例。
3. 阈值。决策规则的蕴含截止。理由基于校准。
4. 评估。留出标注集上的准确率、仅假设基线、对抗子集。

拒绝在没有 100 样本人工标注健全检查的情况下发货零样本分类。拒绝在文档长度前提上使用句级 NLI 模型。标记任何声称 NLI 解决幻觉的说法——它减少了幻觉；不能消除它。
```

## 练习

1. **简单。** 在 20 个人工制作的（前提、假设、标签）三元组（覆盖所有三个类别）上运行 `facebook/bart-large-mnli`。测量准确率。添加对抗"子序列启发式"陷阱（"I did not eat the cake" vs "I ate the cake"）看它是否打破。
2. **中等。** 在 100 条 AG News 标题上比较零样本模板 `"This text is about {label}"` vs `"The topic is {label}"` vs `"{label}"`。报告准确率摆动。
3. **困难。** 构建 RAG 忠实度检查器：原子声明分解 + 每个声明的 NLI。在 50 个带黄金上下文的 RAG 生成答案上评估。测量 vs 人工标签的假阳性和假阴性率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| NLI | 自然语言推理 | 前提-假设关系的三向分类。 |
| RTE | 识别文本蕴含 | NLI 的旧名称；相同任务。 |
| 蕴含 | "t 蕴含 h" | 典型读者在给定 t 的情况下会推断 h 为真。 |
| 矛盾 | "t 排除 h" | 典型读者在给定 t 的情况下会推断 h 为假。 |
| 中性 | "未决定" | 从 t 到 h 双向无推理。 |
| 零样本分类 | NLI 即分类器 | 将标签言语化为假设，取最大蕴含。 |
| 忠实度 | 答案被支持吗？ | 在（检索上下文、生成答案）上的 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) —— SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) —— MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) —— ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) —— NLI 即分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) —— 2026 年 NLI 主力。