# 子词分词——BPE、WordPiece、Unigram、SentencePiece

> 词分词器被未见词噎住。字符分词器序列长度爆炸。子词分词器取中间。每个现代 LLM 都发货在一个上面。

**类型：** 学习
**语言：** Python
**先修课程：** Phase 5 · 01（文本处理）、Phase 5 · 04（GloVe / FastText / 子词）
**耗时：** 约 60 分钟

## 问题

你的词汇表有 50,000 个词。用户输入"untokenizable"。你的分词器返回 `[UNK]`。模型现在对这个词没有信号。更糟：语料库中第 90 百分位文档有 40 个稀有词，这意味着每文档 40 位的丢弃信息。

子词分词解决了这个问题。常用词保持为单个 token。稀有词分解为有意义的片段：`untokenizable` → `un`, `token`, `izable`。训练数据覆盖一切，因为任何字符串最终都是字节序列。

2026 年每个前沿 LLM 都发货在三种算法之一（BPE、Unigram、WordPiece）上，包装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中。你不能发货语言模型而不选择其中之一。

## 概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE（字节对编码）。** 从字符级词汇表开始。统计每个相邻对。合并最频繁的一对为新 token。重复直到达到目标词汇表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE。** 相同算法但基于原始字节（256 个基础 token）而非 Unicode 字符。保证零 `[UNK]` token——任何字节序列都编码。GPT-2 使用 50,257 个 token（256 字节 + 50,000 合并 + 1 个特殊）。

**Unigram。** 从巨大词汇表开始。为每个 token 分配 unigram 概率。迭代剪除移除后使语料库对数似然增加最少的 token。推理时是概率性的：可以对分词结果采样（用于通过子词正则化的数据增强）。被 T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并使训练语料库似然最大化而非原始频率的对。BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken。** SentencePiece 是*训练*词汇表的库（BPE 或 Unigram）直接在原始 Unicode 文本上，将空格编码为 `▁`。tiktoken 是 OpenAI 针对预构建词汇表的快速*编码器*；它不训练。

经验法则：

- **训练新词汇表：** SentencePiece（多语言，无预处理分词）或 HF Tokenizers。
- **针对 GPT 词汇表快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者都要：** HF Tokenizers——一个库，训练 + 服务。

## 构建

### 步骤 1：从零构建 BPE

见 `code/main.py`。循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码的三个事实。`</w>` 标记词尾，使"low"（后缀）和"lower"（前缀）保持不同。频率加权使高频词对早期胜出。合并列表是有序的——推理按训练顺序应用合并。

### 步骤 2：用学习的合并编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用合并等级查找加优先队列，近线性时间运行。

### 步骤 3：实践中的 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：不需要预处理分词，空格编码为 `▁`，`character_coverage` 控制稀有字符保留 vs 映射到 `<unk>` 的激进程度。

### 步骤 4：OpenAI 兼容 vocabs 的 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust 后端）。与 GPT-4/5 分词完全匹配，用于字节计数、成本估算、上下文窗口预算。

## 2026 年仍然上线的陷阱

- **分词器漂移。** 在词汇表 A 上训练，部署在词汇表 B 上。Token ID 不同；模型输出垃圾。在 CI 中检查 `tokenizer.json` 哈希。
- **空白歧义。** BPE "hello" vs " hello" 产生不同 token。始终明确指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足。** 英语主导语料库产生的词汇表将非拉丁语脚本分成 5-10x 更多 token。在 GPT-3.5 上日语/阿拉伯语的相同提示花费 5-10x 更多。o200k_base 部分修复了这一点。
- **Emoji 拆分。** 单个 emoji 可能占用 5 个 token。在预算上下文时检查 emoji 处理。

## 使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 从零训练单语言模型 | HF Tokenizers (BPE) |
| 训练多语言模型 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| 服务 OpenAI 兼容 API | tiktoken (`o200k_base` for GPT-4+) |
| 领域特定词汇（代码、数学、蛋白质） | 在领域语料库上训练自定义 BPE，与基词汇表合并 |
| 边缘推理，小模型 | Unigram（更小词汇表效果更好） |

词汇表大小是扩展决策，不是常数。粗略启发式：<1B 参数用 32k，1-10B 用 50-100k，多语言/前沿用 200k+。

## 交付

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: 为给定语料库和部署目标选择分词算法、词汇表大小、库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定语料库（大小、语言、领域）和部署目标（从零训练 / 微调 / API 兼容推理），输出：

1. 算法。BPE、Unigram 或 WordPiece。一句话理由。
2. 库。SentencePiece、HF Tokenizers 或 tiktoken。理由。
3. 词汇表大小。四舍五入到最近的 1k。与模型大小和语言覆盖的理由。
4. 覆盖率设置。`character_coverage`、`byte_fallback`、特殊 token 列表。
5. 验证计划。留出集上的平均 tokens-per-word、OOV 率、压缩比、往返解码相等性。

当语料库有稀有脚本内容时，拒绝训练 character-coverage <0.995 的分词器。拒绝在没有 CI 中冻结 `tokenizer.json` 哈希检查的情况下发货词汇表。标记任何低于 16k 词汇表的单语言分词器为可能规格不足。
```

## 练习

1. **简单。** 在 `code/main.py` 的微型语料库上训练 500 次合并 BPE。编码三个留出词。有多少正好产生 1 个 token vs >1 token？
2. **中等。** 在 100 个英语维基百科句子上比较 `cl100k_base`、`o200k_base` 和你用 vocab=32k 训练的 SentencePiece BPE 的 token 数。报告每个的压缩比。
3. **困难。** 用 BPE、Unigram 和 WordPiece 训练相同语料库。在小型情感分类器上使用每个时测量下游准确率。选择移动了超过 1 点 F1 吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BPE | 字节对编码 | 贪婪合并最频繁的字符对直到达到目标词汇表大小。 |
| 字节级 BPE | 永无未知 token | 在原始 256 字节上的 BPE；GPT-2 / Llama 使用。 |
| Unigram | 概率分词器 | 使用对数似然从大候选集剪枝；被 T5、Gemma 使用。 |
| SentencePiece | 空格那个 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 快的那个 | OpenAI 的 Rust 支持 BPE 编码器，用于预构建词汇表。不训练。 |
| 合并列表 | 神奇数字 | 有序的 `(a, b) → ab` 合并列表；推理按顺序应用。 |
| 字符覆盖率 | 多稀有算太稀有？ | 分词器必须覆盖的训练语料库中字符的分数；~0.9995 典型。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) —— Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) —— 库。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) —— 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) ——  cookbook + 编码列表。