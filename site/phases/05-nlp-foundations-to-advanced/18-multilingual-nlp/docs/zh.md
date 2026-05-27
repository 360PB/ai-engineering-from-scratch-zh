# 多语言 NLP

> 一个模型，100+ 种语言，对大多数零训练数据。跨语言迁移是 2020 年代的实践奇迹。

**类型：** 学习
**语言：** Python
**先修课程：** Phase 5 · 04（GloVe、FastText、子词）、Phase 5 · 11（机器翻译）
**耗时：** 约 45 分钟

## 问题

英语有数十亿标注样本。乌尔都语有数千个。Mithila 语几乎为零。任何服务全球观众的实用 NLP 系统都必须处理长尾语言，那里不存在特定任务的训练数据。

多语言模型通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型将高资源语言中学到的技能迁移到低资源语言。在英语情感分析上微调模型，它开箱就能对乌尔都语产生令人惊讶的良好情感预测。那是零样本跨语言迁移，它重塑了 NLP 向世界发货的方式。

本课命名权衡、规范模型和团队新接触多语言工作时绊倒的一个决定：选择迁移的源语言。

## 概念

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词汇表是共享的：相同的子词单元在相关语言中表示相同的词素。英语和意大利语中的 `anti-` 得到相同的 token。

**共享表示。** 在多种语言上通过掩码语言建模预训练的 Transformer 学习到不同语言中语义相似的句子产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出这一点。英语中"cat"的嵌入与法语"chat"和西班牙语"gato"相近，整句嵌入也是如此。

**零样本迁移。** 在一种语言（通常是英语）的标注数据上微调。推理时，在模型支持的任何其他语言上运行。不需要目标语言标签。在类型学相关语言上结果强，在遥远语言上较弱。

**少样本微调。** 在目标语言中添加 100-500 个标注样本。准确率跃升至英语基线的 95-98%。这是多语言 NLP 中最具成本效益的杠杆。

## 模型

| 模型 | 年份 | 覆盖 | 说明 |
|------|------|------|------|
| mBERT | 2018 | 104 种语言 | 在维基百科上训练。第一个实用多语言 LM。低资源上弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl（比维基百科大得多）上训练。设定跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 带有 100 万 token 词汇表（vs 250k）。低资源上更好。 |
| mT5 | 2020 | 101 种语言 | T5 架构用于多语言生成。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包括 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程 | 开放 176B LLM 多语言训练。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。阿拉伯语、印地语、斯瓦希里语强。 |

按用例选择。分类用 XLM-R-base 作为理智默认。生成任务取决于翻译 vs 开放生成，调用 mT5 或 NLLB。LLM 风格工作与 Aya-23 或使用显式多语言提示的 Claude 配对。

## 源语言决策（2026 年研究）

大多数团队默认以英语作为微调源。最新研究（2026 年）表明这往往是错误的。

语言相似度比原始语料库大小更能预测迁移质量。对于斯拉夫语目标，德语或俄语通常胜英语。对于印地语目标，印地语通常胜英语。**qWALS** 相似度指标（2026 年，基于世界语言结构地图集特征）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是一种单独的早期方法，从语言相似度、语料库大小和遗传相关性组合对候选源语言排名。

实用规则：如果你的目标语言有一个类型学上接近的高资源亲属，先尝试在该语言上微调，然后与英语微调比较。

## 构建

### 步骤 1：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同 API。在 NLI 数据上训练的 XLM-R 通过蕴含技巧很好地迁移到分类。

### 步骤 2：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译在嵌入空间中落在相近位置。不同的英语句子落在更远的地方。这就是跨语言检索、聚类和相似性工作的原因。

### 步骤 3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 个目标语言示例，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认值。更高的学习率导致多语言对齐崩溃，你得到一个仅英语模型。

## 真正有效的评估

- **每语言留出集准确率。** 不聚合。聚合隐藏了长尾。
- **与单语言基线比较。** 对于有足够数据的语言，从头训练的单语言模型有时胜过多语言模型。测试。
- **实体级测试。** 目标语言中的命名实体。多语言模型通常对远离拉丁语的脚本有弱的 tokenization。
- **跨语言一致性。** 两种语言中相同含义应产生相同预测。测量差距。

## 使用

2026 年技术栈：

| 任务 | 推荐 |
|------|------|
| 分类，100 种语言 | XLM-R-base（约 270M）微调 |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相关高资源语言上领域微调 |

如果性能重要，总是为目标语言预算微调。零样本是起点，不是最终答案。

### 分词税（低资源语言出问题的地方）

多语言模型在所有语言间共享一个分词器。该词汇表在英语、法语、西班牙语、汉语、德语主导的语料库上训练。对于主导集之外的任何语言，三种税悄悄地复合：

- **生育税。** 低资源语言文本每词 tokenize 成比英语多得多 token。印地语句子可能需要等效英语句子的 3-5x token。那 3-5x 吃掉你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每个拼写错误、音标变体、Unicode 归一化不匹配或大小写变体在嵌入空间中变成冷启动无关序列。模型无法学习母语者认为显而易见的正字法对应。
- **容量溢出税。** 税 1 和 2 消耗上下文位置、层深度和嵌入维度。剩余用于实际推理的系统地小于高资源语言从同一模型获得的。

实际症状：你的模型在印地语上正常训练，损失曲线看起来正确，eval 困惑度看起来合理，但生产输出微妙地错误。形态学在句中崩溃。罕见屈折形式保持不可恢复。**你无法通过数据规模摆脱破碎的分词器。**

缓解：为目标语言选择覆盖率好的分词器（XLM-V 的 100 万 token 词汇表是直接修复）；在留出的目标文本上验证分词生育率；对真正长尾脚本使用字节级回退（SentencePiece `byte_fallback=True`，GPT-2 风格字节级 BPE），使永无未登录词。

## 交付

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: 为多语言 NLP 任务选择源语言、目标模型和评估计划。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

给定需求（目标语言、任务类型、每语言可用标注数据），输出：

1. 微调的源语言。默认英语；如果目标语言有类型学上接近的高资源亲属，检查 LANGRANK 或 qWALS。
2. 基模型。XLM-R（分类）、mT5（生成）、NLLB（翻译）、Aya-23（生成式 LLM）。
3. 少样本预算。如果可用，从 100-500 个目标语言示例开始。仅在标注不可行时用零样本。
4. 评估计划。每语言准确率（非聚合）、跨语言一致性、非拉丁语脚本上的实体级 F1。

拒绝在没有每语言评估的情况下上线多语言模型——聚合指标隐藏长尾失败。标记分词覆盖率低的脚本（阿姆哈拉语、提格雷尼亚语、许多非洲语言）为需要带字节回退的模型（SentencePiece 带 byte_fallback=True，或字节级分词器如 GPT-2）。
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语各 10 句上运行零样本分类流水线。报告每语言准确率。你应该看到法语强，印地语体面，阿拉伯语多变。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在小型混合语言语料库上构建跨语言检索器。用英语查询，检索任何语言的文档。测量 Recall@5。
3. **困难。** 比较印地语分类任务的英语源和印地语源微调。在两种机制下用 500 个目标语言示例进行少样本微调。报告哪个源产生更好的印地语准确率及差距多少。这是 LANGRANK 论点的缩影。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享词汇表和参数。 |
| 跨语言迁移 | 在一种语言上训练，在另一种上运行 | 在源语言上微调，在目标语言上评估，不需要目标语言标签。 |
| 零样本 | 无目标语言标签 | 在目标语言上不用微调的迁移。 |
| 少样本 | 小目标标签 | 100-500 个目标语言示例用于微调。 |
| mBERT | 第一个多语言 LM | 104 种语言 BERT 在维基百科上预训练。 |
| XLM-R | 标准跨语言基线 | 100 种语言 RoBERTa 在 CommonCrawl 上预训练。 |
| NLLB | Meta 的 200 种语言 MT | 不落下任何语言。包括 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) —— XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) —— 开始跨语言迁移研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) —— NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) —— Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) —— qWALS / LANGRANK 源语言论文。