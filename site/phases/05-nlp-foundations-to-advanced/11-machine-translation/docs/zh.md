# 机器翻译

> 这项任务为 NLP 研究付费了三十年，现在还在继续。

**类型：** 构建
**语言：** Python
**先修课程：** Phase 5 · 10（注意力机制）、Phase 5 · 04（GloVe、FastText、子词）
**耗时：** 约 75 分钟

## 问题

一个模型读取一种语言的句子，产生另一种语言的句子。长度变化。词序变化。一些源词映射到多个目标词，反之亦然。习语拒绝一对一映射。"I miss you" 在法语中是 "tu me manques"——字面意思是"你对我欠缺"。没有词级对齐能承受这个。

机器翻译是迫使 NLP 发明编码器-解码器、注意力、Transformer，最终是整个 LLM 范式的任务。每一步前进都到来，因为翻译质量可衡量，人类和机器之间的差距顽固存在。

本课跳过历史课，教授 2026 年的工作流水线：预训练多语言编码器-解码器（NLLB-200 或 mBART）、子词分词、束搜索、BLEU 和 chrF 评估，以及仍然悄悄上线的少数失败模式。

## 概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代 MT 是在平行文本上训练的 Transformer 编码器-解码器。编码器用其语言的 tokenization 读取源。解码器通过交叉注意力（第 10 课）使用编码器的输出逐子词生成目标。解码使用束搜索避免贪心解码陷阱。输出被去 tokenization、去大小写，并对照参考评分。

三个操作选择驱动真实世界 MT 质量。

- **分词器。** 在混合语言语料库上训练的 SentencePiece BPE。跨语言共享词汇表是 NLLB 中零样本对的支持方式。
- **模型大小。** NLLB-200 蒸馏 600M 可在笔记本上运行。NLLB-200 3.3B 是发布的生产默认。54.5B 是研究天花板。
- **解码。** 一般内容用束宽 4-5。长度惩罚以避免输出过短。需要术语一致性时用约束解码。

## 构建

### 步骤 1：预训练 MT 调用

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三件事很重要。`src_lang` 告诉分词器应用哪种脚本和切分。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 使用各自的约定，不可互换。

### 步骤 2：BLEU 和 chrF

BLEU 衡量输出和参考之间的 n-gram 重叠。四个参考 n-gram 大小（1-4）、精确率的几何平均、过短输出的简短惩罚。分数在 [0, 100] 范围内。常用。解释起来令人沮丧：30 BLEU 是"可用"；40 是"好"；50 是"出色"；1 BLEU 以下的差异是噪音。

chrF 衡量字符级 F 分数。对形态丰富语言更敏感，BLEU 低估了匹配。经常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它规范化 tokenization，使分数跨论文可比。自己写 BLEU 计算是误导性基准的来源。

### 三层评估体系（2026）

现代 MT 评估使用三个互补的指标家族。至少使用两个上线。

- **启发式**（BLEU、chrF）。快、基于参考、可解释、对改写不敏感。用于传统比较和回归检测。
- **学习型**（COMET、BLEURT、BERTScore）。在人类判断上训练的神经模型；比较翻译与源和参考的语义相似度。2023 年以来 COMET 与 MT 研究的相关性最高，是 2026 年质量重要时的生产默认。
- **LLM 即评判**（无参考）。提示大模型在流畅性、充分性、语气、文化适当性上评分翻译。当评分标准设计良好时，GPT-4 即评判与人类一致性约 80%。用于无参考存在的开放式内容。

实用的 2026 技术栈：`sacrebleu` 做 BLEU 和 chrF，`unbabel-comet` 做 COMET，以及提示 LLM 做最终人类面向信号。在信任生产数据之前，用 50-100 个人类标注样本校准每个指标。

无参考指标（COMET-QE、BLEURT-QE、LLM 即评判）让你在没有参考的情况下评估翻译，这对不存在参考翻译的长尾语言对很重要。

### 步骤 3：生产中会出什么错

上面的工作流水线 80% 的时间流利翻译，其余 20% 默默失败。已命名的失败模式：

- **幻觉。** 模型虚构源中没有的内容。在不熟悉领域词汇中常见。症状：输出流畅但声称源未陈述的事实。缓解：对领域术语进行约束解码，受监管内容需人工审查，监控输出比输入长很多的情况。
- **目标语言错误。** 模型翻译成错误的语言。NLLB 在罕见语言对上出奇地容易出现这个问题。缓解：验证 `forced_bos_token_id` 并始终在输出上进行语言 ID 模型检查。
- **术语漂移。** "Sign up" 在文档 1 中变成 "s'inscrire"，在文档 2 中变成 "créer un compte"。对于 UI 文本和用户面向字符串，一致性比原始质量更重要。缓解：词典约束解码或后编辑词典。
- **正式程度不匹配。** 法语 "tu" vs "vous"、日语礼貌级别。模型选择训练中更常见的哪种形式。对于面向客户的内容，这通常是错误的。缓解：如果模型支持，用正式标记提示前缀，或在仅正式语料上微调小模型。
- **短输入长度爆炸。** 很短的输入句子经常产生过长的翻译，因为长度惩罚在约 5 个源 token 以下急剧下降。缓解：硬上限最大长度与源长度成正比。

### 步骤 4：领域微调

预训练模型是通才。法律、医疗或游戏对话翻译在领域平行数据上微调可衡量地受益。配方不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

数千个高质量平行样本胜过上百万个嘈杂的网络爬取样本。训练数据质量是最大的生产杠杆。

## 使用

2026 年 MT 生产技术栈：

| 用途 | 推荐起点 |
|------|---------|
| 任意到任意，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 英语中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短运行，便宜推理，英法/德/西班牙语 | Helsinki-NLP / Marian 模型 |
| 延迟敏感浏览器端 | ONNX 量化的 Marian（~50 MB） |
| 最高质量，愿意付费 | 带翻译提示的 GPT-4 / Claude / Gemini |

截至 2026 年，LLM 在多个语言对上已经优于专业 MT 模型，特别是在习语内容和长上下文上。权衡是每 token 成本和延迟。当上下文长度、文体一致性或通过提示的领域适配比吞吐量更重要时，选择 LLM。

## 交付

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: 评估机器翻译输出是否可上线。
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

给定源文本和候选翻译，输出：

1. 自动分数估计。你期望的 BLEU 和 chrF 范围。说明是否有参考可用。
2. 五点人工可验证检查清单：(a) 内容保持（无幻觉），(b) 语言正确，(c) 语域/正式程度匹配，(d) 术语与提供的词汇表一致性，(e) 无截断或长度爆炸。
3. 一个领域特定问题探查。例如，法律：命名实体和法规引用。医疗：药品名称和剂量。UI：`{name}` 占位符变量。
4. 置信度标记。"上线"/"附审查上线"/"不上线"。与第 2 步中发现问题的严重程度挂钩。

未经输出上的语言 ID 检查，拒绝上线翻译。未经参考明确选择无参考评分（COMET-QE、BLEURT-QE），拒绝评估。标记超过 1000 token 的内容为可能需要分块翻译。
```

## 练习

1. **简单。** 用 `nllb-200-distilled-600M` 将 5 句英文段落翻译成法语再翻译回英语。测量往返与原始的接近程度。你应该看到语义保持但词汇选择漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 实现翻译输出上的语言 ID 检查。集成到 MT 调用中，以便在返回之前捕获目标语言错误生成。
3. **困难。** 在你选择的 5000 对领域语料库上微调 `nllb-200-distilled-600M`。在留出集上测量微调前后的 BLEU。报告哪些类型的句子改善了，哪些退步了。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BLEU | 翻译分数 | 带简短惩罚的 N-gram 精确率。[0, 100]。 |
| chrF | 字符 F 分数 | 字符级 F-分数。对形态丰富语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的 Transformer 编码器-解码器。2017 年后的默认。 |
| NLLB | 不落下任何语言 | Meta 的 200 种语言 MT 模型系列。 |
| 约束解码 | 受控输出 | 强制特定 token 或 n-gram 出现或不出现。 |
| 幻觉 | 虚构内容 | 模型输出不被源支持。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) —— NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) —— 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) —— chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) —— 实用微调演练。