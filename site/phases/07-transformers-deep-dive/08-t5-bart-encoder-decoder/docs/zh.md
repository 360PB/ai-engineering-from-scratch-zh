# T5、BART — 编码器-解码器模型

> 编码器理解。解码器生成。把它们放回一起，你得到一个为输入→输出任务构建的模型：翻译、摘要、改写、转录。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 第 5 课（完整 Transformer）、Phase 7 第 6 课（BERT）、Phase 7 第 7 课（GPT）
**时长：** ~45 分钟

## 问题

纯解码器 GPT 和纯编码器 BERT 各自为不同目标剥离了 2017 架构。但许多任务天生是输入-输出的：

- 翻译：英语 → 法语。
- 摘要：5,000-token 文章 → 200-token 摘要。
- 语音识别：audio token → 文本 token。
- 结构化提取：散文 → JSON。

对这些任务，编码器-解码器是最干净的适配。编码器产生源的密集表示。解码器在每步交叉关注该表示，生成输出。训练在输出侧是移位一。与 GPT 相同的损失，只是以编码器输出为条件。

两篇论文定义了现代 playbook：

1. **T5**（Raffel et al. 2019）。"Text-to-Text Transfer Transformer。"每个 NLP 任务重新框定为文本输入、文本输出。单一架构、单一词汇、单一损失。预训练于掩码跨度预测（损坏输入中的跨度，在输出中解码它们）。
2. **BART**（Lewis et al. 2019）。"Bidirectional and Auto-Regressive Transformer。"去噪自编码器：以多种方式（打乱、掩码、删除、旋转）损坏输入，要求解码器重建原文。

2026 年编码器-解码器形式存活于输入结构重要的地方：

- Whisper（语音 → 文本）。
- 谷歌翻译栈。
- 一些具有不同上下文和编辑结构的代码补全/修复模型。
- Flan-T5 及变体用于结构化推理任务。

纯解码器赢得了聚光灯，但编码器-解码器从未消失。

## 概念

![带交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
源 token ─▶ 编码器 ─▶ (N_src, d_model)  ──┐
                                               │
目标 token ─▶ 解码器块                        │
                 ├─▶ 带掩码自注意力            │
                 ├─▶ 交叉注意力 ◀───────────┘
                 └─▶ FFN
                ↓
              下一个 token logit
```

关键：编码器对每个输入运行一次。解码器自回归运行但在每步交叉关注*相同*的编码器输出。缓存编码器输出是长输入的免费加速。

### T5 预训练——跨度腐败

随机选择输入的跨度（平均长度 3 token，15% 总计）。将每个跨度替换为唯一哨兵：`<extra_id_0>`、`<extra_id_1>` 等。解码器仅输出损坏的跨度及其哨兵前缀：

```
源：The quick <extra_id_0> fox jumps <extra_id_1> dog
目标：<extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更便宜的信号。在 T5 论文消融中与 MLM（BERT）和 prefix-LM（UniLM）持平。

### BART 预训练——多噪声去噪

BART 尝试五种噪声函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（掩码一个跨度，解码器插入正确长度）。
4. 句子排列。
5. 文档旋转。

结合文本填充 + 句子排列产生最佳下游数字。解码器始终重建原文。BART 的输出是完整序列，而不仅仅是损坏的跨度——所以预训练计算高于 T5。

### 推理

与 GPT 相同的自回归生成。贪心/束/top-p 采样适用。束搜索（宽度 4–5）是翻译和摘要的标准，因为输出分布比聊天更窄。

### 2026 年何时选各变体

| 任务 | 编码器-解码器？ | 为什么 |
|------|----------------|---------|
| 翻译 | 通常是 | 清晰的源序列；固定输出分布；束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器塑造音频特征 |
| 聊天/推理 | 否，纯解码器 | 无持久"输入"——对话就是序列 |
| 代码补全 | 通常否 | 纯解码器加长上下文胜出；代码模型如 Qwen 2.5 Coder 是纯解码器 |
| 摘要 | 两者皆可 | BART、PEGASUS 胜过早期纯解码器基线；现代纯解码器 LLM 匹配它们 |
| 结构化提取 | 两者皆可 | T5 是干净的，因为"text → text" 吸收任何输出格式 |

自约 2022 年以来的趋势：纯解码器接管编码器-解码器曾经拥有的任务，因为（a）指令微调的纯解码器 LLM 通过提示泛化到任何事情，（b）一种架构比两种更容易扩展，（c）RLHF 假设解码器。编码器-解码器在输入模态不同（语音、图像）或束搜索质量重要的地方保持优势。

## 构建

见 `code/main.py`。我们为玩具语料实现 T5 风格的跨度腐败——这是自那以来每个编码器-解码器预训练配方中最有用的单个部分。

### 第一步：跨度腐败

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选择加起来约 mask_rate 的跨度。返回 (损坏输入, 目标)。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是 T5 惯例：`<sent0> span0 <sent1> span1 ...`。损坏的输入将未改变的 token 与跨度位置的哨兵 token 交错。

### 第二步：验证往返

给定损坏的输入和目标，重建原始句子。如果你的腐败是可逆的，前向传播是良好定义的。这是一个健全性检查——真实训练从不这样做，但测试便宜，能捕获跨度簿记中的 off-by-one bug。

### 第三步：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 使用

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 技巧：任务名称进入输入文本。同一模型处理数十个任务，因为每个任务是文本输入、文本输出。2026 年这个模式已被指令微调纯解码器模型泛化，但 T5 最早编纂了它。

## 交付

见 `outputs/skill-seq2seq-picker.md`。该 skill 根据输入-输出结构、延迟和质量目标为新任务选择编码器-解码器和纯解码器。

## 练习

1. **简单。** 运行 `code/main.py`，对 30-token 句子应用跨度腐败，验证连接非哨兵源 token 与解码目标跨度再现原文。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机跨度，解码器必须推断正确的跨度长度加内容。展示一个例子。
3. **困难。** 在微型英语 → 猪拉丁语语料（200 对）上微调 `flan-t5-small`。在保留的 50 对上测量 BLEU。与在相同数据、相同计算上微调 `Llama-3.2-1B` 比较。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| 编码器-解码器 | "Seq2seq transformer" | 两堆：用于输入的双向编码器，用于输出的带交叉注意力的因果解码器。 |
| 交叉注意力 | "源与目标对话" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一地方。 |
| 跨度腐败 | "T5 的预训练技巧" | 用哨兵 token 替换随机跨度；解码器输出跨度。 |
| 去噪目标 | "BART 的游戏" | 对输入应用噪声函数，训练解码器重建干净序列。 |
| 哨兵 token | "`<extra_id_N>` 占位符" | 在源中标记损坏跨度并在目标中重新标记的特殊 token。 |
| Flan | "指令微调 T5" | 在 >1,800 个任务上微调的 T5；使编码器-解码器在指令遵循上具有竞争力。 |
| 束搜索 | "解码策略" | 在每步保持 top-k 部分序列；翻译/摘要的标准。 |
| 教师强制 | "训练时输入" | 在训练时将真实前一个输出 token 送入解码器，而非采样的那个。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年标准的编码器-解码器。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。