# 从零构建分词器

> 第一节课给了你一个玩具。这节课给你一件武器。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 10，第一课（分词器：BPE、WordPiece、SentencePiece）
**时间：** 约 90 分钟

## 学习目标

- 构建一个生产级 BPE 分词器，处理 Unicode、空格规范化，和特殊 token
- 实现字节级回退，使分词器能编码任何输入（包括 emoji、CJK 和代码）而不产生未知 token
- 添加预分词正则表达式模式，在应用 BPE 合并前按词边界分割文本
- 在多语言文本上训练自定义分词器并用压缩比评估其与 tiktoken 的对比

## 问题所在

第一节课的分词器在英文文本上工作正常。现在扔给它日语。或者 emoji。或者混合了 Tab 和空格的 Python 代码。

它崩溃了。

不是因为 BPE 错了——是因为实现不完整。生产级分词器处理任何编码的原始字节，在分割前规范化 Unicode，管理从不参与 BPE 合并的特殊 token，链式预分词与子词分割，并且足够快以至于不会成为处理 15 万亿 token 的训练流水线瓶颈。

GPT-2 的分词器有 50,257 个 token。Llama 3 有 128,256。GPT-4 约有 100,000。这些不是玩具数字。那些词表背后的合并表在数百 GB 文本上训练，周围的机制——规范化、预分词、特殊 token 注入、聊天模板格式——是区分能处理 "hello world" 和能处理整个互联网的分词器之间的差距。

你将要构建那个机制。

## 概念

### 完整流水线

生产级分词器不是一个算法。它是一个流水线，五个阶段，每个解决不同问题。

```mermaid
graph LR
    A[原始文本] --> B[规范化]
    B --> C[预分词]
    C --> D[BPE 合并]
    D --> E[特殊 Token]
    E --> F[Token ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段有具体职责：

| 阶段 | 做什么 | 为什么重要 |
|-------|-------------|----------------|
| 规范化 | NFKC Unicode，可选小写，可选去除重音 | "fi" 连字（U+FB01）变为 "fi"（两个字符）。没有这个，同一个词得到不同 token。 |
| 预分词 | 在 BPE 前将文本分割成块 | 防止 BPE 跨词边界合并。"the cat" 永远不应产生 "e c" 这样的 token。 |
| BPE 合并 | 将学习到的合并规则应用于字节序列 | 核心压缩。将原始字节变成子词 token。 |
| 特殊 Token | 注入 [BOS]、[EOS]、[PAD]、聊天模板标记 | 这些 token 有固定 ID。它们从不参与 BPE 合并。模型需要它们来构建结构。 |
| ID 映射 | 将 token 字符串转换为整数 ID | 模型看到整数，不是字符串。 |

### 字节级 BPE

第一节课的分词器在 UTF-8 字节上操作。这是正确的调用。但我们跳过了一个重要的东西：当这些字节不是有效 UTF-8 时会发生什么？

字节级 BPE 的解决方案是将每个可能的字节值（0-255）视为有效 token。你的基础词表恰好是 256 项。任何文件——文本、二进制、损坏的——都可以分词而不会产生未知 token。

GPT-2 加了一个技巧：将每个字节映射到一个可打印的 Unicode 字符，使词表保持人类可读。字节 0x20（空格）在他们的映射中变成字符 "G"。这只是外观上的。算法不在乎。

真正的力量：字节级 BPE 处理地球上的每一种语言。汉字每个是 3 个 UTF-8 字节。日语可能是 3-4 字节。阿拉伯语、天城文、emoji——都只是字节序列。BPE 算法在这些字节序列中找模式的方式与在英文 ASCII 字节中完全相同。

### 预分词

在 BPE 接触你的文本之前，你需要将其分割成块。这防止合并算法创建跨词边界的 token。

GPT-2 使用正则表达式模式分割文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

此模式按缩约词分割（"don't" 变为 "don" + "'t"）、带可选前导空格的词、数字、标点符号和空格。前导空格与词保持连接——所以 "the cat" 变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，完全跳过正则表达式。它将原始字节流视为一个长序列，让 BPE 算法自己找出边界。这更简单但给 BPE 更多自由度来创建跨词 token。

选择很重要。GPT-2 的正则表达式防止分词器学到结尾的 "the" 和开头的 "the" 应该合并。SentencePiece 允许它，这有时产生更高效的压缩但更不易解释的 token。

### 特殊 Token

每个生产级分词器都为结构性标记保留 token ID：

| Token | 用途 | 使用者 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3, GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批量对齐的填充 | BERT, T5 |
| `[UNK]` | 未知 token（字节级 BPE 消除此需求） | BERT, WordPiece |
| `<|im_start|>` | 聊天消息边界开始 | ChatGPT, Qwen |
| `<|im_end|>` | 聊天消息边界结束 | ChatGPT, Qwen |
| `<|user|>` | 用户回合标记 | Llama 3 |
| `<|assistant|>` | 助手回合标记 | Llama 3 |

特殊 token 从不被 BPE 分割。它们在合并算法运行前精确匹配，替换为固定 ID，周围的文本正常分词。

### 聊天模板

这是大多数人困惑、大多数实现出错的地方。

当你向聊天模型发送消息时，API 接受一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON。它看到一个平面 token 序列。聊天模板用特殊 token 将消息转换为平面序列。每个模型做法不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板错误会导致模型输出垃圾。它按一个精确格式训练。任何偏差——缺少换行、token 交换、多余空格——都会使输入超出训练分布。

### 速度

Python 对生产级分词太慢。

tiktoken（OpenAI）用 Rust 编写，有 Python 绑定。HuggingFace 分词器也是 Rust。SentencePiece 是 C++。这些实现比纯 Python 快 10-100 倍。

从规模上看：在 100 万 tokens/秒（快速 Python）下，Llama 3 预训练分词 15 万亿 token 需要 174 天。在 1 亿 tokens/秒（Rust）下，需要 1.7 天。

你在 Python 中构建是为了理解算法。在生产中，你会使用编译实现，只碰 Python 包装器。

## 构建

### 步骤 1：字节级编码

基础。将任何字符串转换为字节序列，将每个字节映射到可打印字符用于显示，并反向操作。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试以查看字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 个字节。"你好" 是 6 个字节（每个字符 3 个）。fire emoji 是 4 个字节。字节级分词器不在乎它是什么语言。字节就是字节。

### 步骤 2：正则表达式预分词器

使用 GPT-2 正则表达式模式将文本分割成块。每个块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，因此回退到 ASCII 字符类。对于生产多语言分词器，安装 `regex`。

试试看：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格与词保持连接。缩约词在撇号处分割。标点成为独立块。BPE 永远不会跨这些边界合并 token。

### 步骤 3：字节序列上的 BPE

第一节课的核心算法，但现在在预分词块上独立操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 步骤 4：特殊 Token 处理

特殊 token 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 步骤 5：完整分词器类

将所有内容链接在一起：规范化、按特殊 token 分割、预分词、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 步骤 6：多语言测试

真正的测试。扔给它英文、中文、emoji 和代码。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个产生 3 个字节。emoji 产生 4 个字节。没有崩溃。没有产生未知 token。这就是字节级 BPE 的力量。

## 使用

### 比较真实分词器

加载 Llama 3、GPT-4 和 Mistral 的真实分词器。看每个如何处理同样的多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到相同文本产生不同的 token 数。128K 词表的 Llama 3 更激进地合并常见模式。100K 的 GPT-4 居中。32K 的 Mistral 产生更多 token 但嵌入层更小。

权衡总是一样的：更大的词表意味着更短的序列，但参数更多。

## 交付

本课产生一个用于构建和调试生产分词器的 prompt。见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加 `get_token_bytes(id)` 方法，显示任何 token ID 的原始字节。用它检查你最常见合并 token 实际代表什么。

2. **中等：** 实现 Llama 风格的预分词器，按空格和数字分割但保留前导空格。在相同语料上与 GPT-2 正则方法比较词表。

3. **困难：** 添加聊天模板方法，接受 `{"role": ..., "content": ...}` 消息列表并产生 Llama 3 聊天格式的正确 token 序列。用 HuggingFace 实现测试它。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 字节级 BPE | "在字节上操作的分词器" | 基础词表为 256 字节值的 BPE——处理任何输入而没有未知 token |
| 预分词 | "BPE 前分割" | 正则或规则基础的分割，防止 BPE 跨词边界合并 |
| NFKC 规范化 | "Unicode 清理" | 规范分解后兼容性组合——"fi" 连字变为 "fi"，全角 "A" 变为 "A" |
| 聊天模板 | "消息如何变成 token" | 将角色/内容消息列表转换为平面 token 序列的精确格式——模型特定，必须匹配训练格式 |
| 特殊 token | "控制 token" | 绕过 BPE 的保留 token ID——[BOS]、[EOS]、[PAD]、聊天标记——在合并前精确匹配 |
| Fertility | "每词 token 数" | 输出 token 与输入词之比——GPT-4 英语为 1.3，韩语为 2-3，更高意味着浪费上下文 |
| tiktoken | "OpenAI 分词器" | Rust BPE 实现，有 Python 绑定——比纯 Python 快 10-100 倍 |
| 合并表 | "词表" | 训练期间学习的字节对合并有序列表——这就是分词器的学习知识 |

## 拓展阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken) -- GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace 分词器](https://github.com/huggingface/tokenizers) -- 支持 BPE、WordPiece、Unigram 的 Rust 分词器库
- [Llama 3 论文（Meta, 2024）](https://arxiv.org/abs/2407.21783) -- 128K 词表和分词器训练细节
- [SentencePiece（Kudo & Richardson, 2018）](https://arxiv.org/abs/1808.06226) -- 语言无关的分词
- [GPT-2 分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 原始字节到 Unicode 映射