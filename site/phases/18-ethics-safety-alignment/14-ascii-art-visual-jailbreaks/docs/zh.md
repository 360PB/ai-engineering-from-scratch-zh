# ASCII Art 和视觉越狱

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，"ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs"（ACL 2024，arXiv:2402.11753）。屏蔽有害请求中的安全相关 token，用相同字母的 ASCII 艺术渲染替换它们，然后发送隐蔽的 prompt。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都未能鲁棒识别 ASCII-art token。攻击绕过 PPL（困惑度过滤器）、Paraphrase 防御和 Retokenization。相关：ViTC 基准测量对非语义视觉 prompt 的识别；StructuralSleight 推广到 Uncommon Text-Encoded Structures（树、图、嵌套 JSON）作为编码攻击家族。

**类型：** 构建
**语言：** Python（标准库，ArtPrompt token 屏蔽工具）
**前置知识：** Phase 18 · 12（PAIR），Phase 18 · 13（MSJ）
**时长：** 约 60 分钟

## 学习目标

- 描述 ArtPrompt 攻击：词语识别步骤、ASCII-art 替换、最终隐蔽 prompt。
- 解释为什么标准防御（PPL、Paraphrase、Retokenization）在 ArtPrompt 上失败。
- 定义 ViTC 并描述它衡量的内容。
- 描述 StructuralSleight 作为推广到任意 Uncommon Text-Encoded Structures。

## 问题

通过释义和角色扮演（第 12 课）和长上下文（第 13 课）的攻击在文本级模式上操作。ArtPrompt 在识别级操作：模型不解析被禁止的 token。它解析字符渲染的图像。安全过滤器看到无害的标点符号。模型看到一个词。

## 概念

### ArtPrompt，两步

第 1 步：词语识别。给定有害请求，攻击者使用 LLM 识别安全相关词语（例如，"how to make a bomb"中的"bomb"）。

第 2 步：隐蔽 Prompt 生成。将每个识别的词替换为其 ASCII 艺术渲染（形成字母形状的 7x5 或 7x7 字符块）。模型收到一个标点和空格的网格，足以识别的模型将其读为一个词；安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失败。在其基准子集上攻击成功率高于 75%。

### 为什么标准防御失败

- **PPL（困惑度过滤器）。** ASCII 艺术有高困惑度——但所有新输入也是如此。阻止 ArtPrompt 的阈值选择也阻止合法结构化输入。
- **Paraphrase。** 释义 prompt 破坏 ASCII 艺术。实际上，释义 LLM 通常保留或重建艺术。
- **Retokenization。** 以不同方式分割 token 不会改变模型正在识别字母形状的事实。

底层问题：安全过滤器是 token 级或语义级；ArtPrompt 在视觉识别级操作。

### ViTC 基准

对非语义视觉 prompt 的识别。测量模型读取 ASCII 艺术、wingdings 和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型读取视觉文本越好，ArtPrompt 对其效果越好。这是一个能力-安全权衡。

### StructuralSleight

推广 ArtPrompt：Uncommon Text-Encoded Structures（UTES）。树、图、嵌套 JSON、CSV-in-JSON、diff 风格代码块。如果一种结构在训练安全数据中罕见但模型可解析，它可隐藏有害内容。

防御含义：安全必须泛化到模型可解析的结构化表示。集合是大的且在增长。

### 图像模态类似物

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展攻击面。带实际图像的 ArtPrompt 风格攻击比 ASCII 艺术类似物更强，因为图像编码器产生更丰富的信号。

### 为什么这在 Phase 18 中重要

第 12-14 课描述三个正交攻击向量：迭代细化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从模型中心攻击转向系统边界攻击（间接 prompt 注入）。第 16 课描述防御工具响应。

## 使用它

`code/main.py` 构建玩具 ArtPrompt。你可以屏蔽有害查询中的特定词为 ASCII 艺术字形，验证隐蔽字符串通过简单关键词过滤器，并（可选）使用简单识别器将隐蔽字符串解码回。

## 交付它

本课生成 `outputs/skill-encoding-audit.md`。给定越狱防御报告，枚举覆盖的编码攻击家族（ASCII art、base64、leet-speak、UTF-8 同形字、UTES）以及捕获每个的防御层。

## 练习

1. 运行 `code/main.py`。验证隐蔽字符串通过简单关键词过滤器。报告所需的字符级更改。

2. 实现第二编码：对相同目标词用 base64。比较过滤器绕过率与 ArtPrompt 和恢复难度。

3. 读 Jiang 等 2024 第 4.3 节（五模型结果）。提出为什么在同一基准上 Claude 的 ArtPrompt 抵抗力高于 Gemini 的原因。

4. 设计检测 prompt 中 ASCII 艺术形状区域的预生成防御。在合法代码、表和数学符号上测量误报率。

5. StructuralSleight 列出 10 种编码结构。勾勒处理全部 10 种的泛化防御并估计每个受保护 prompt 的计算成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| ArtPrompt | ASCII 艺术攻击 | 两步越狱：用 ASCII 艺术渲染屏蔽安全词 |
| 隐蔽 | 隐藏词 | 用模型读取但过滤器不读的视觉表示替换被禁止的 token |
| UTES | 不常见结构 | Uncommon Text-Encoded Structure——树、图、嵌套 JSON 等用于走私内容 |
| ViTC | 视觉-文本能力 | 模型读取非语义视觉编码的能力基准 |
| 困惑度过滤器 | PPL 防御 | 拒绝高困惑度 prompt；失败因为合法结构化输入也得高分 |
| Retokenization | tokenizer 移位防御 | 用不同 tokenizer 预处理 prompt；失败因为识别是视觉的 |
| 同形字 | 看起来像的字符 | 看起来与拉丁字母相同的 Unicode 字符；绕过子字符串检查 |

## 延伸阅读

- [Jiang 等 — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li 等 — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 推广
- [Chao 等 — PAIR（第 12 课，arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补迭代攻击
- [Anil 等 — Many-shot Jailbreaking（第 13 课）](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补长度攻击