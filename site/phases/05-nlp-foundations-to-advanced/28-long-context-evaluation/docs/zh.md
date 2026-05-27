# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 广告称 10M token 上下文。在 1M token 时，8针 MRCR 降至 26.3%。广告 ≠ 可用。长上下文评估告诉你实际在用哪个模型时可以工作的容量。

**类型：** Learn
**语言：** Python
**前置知识：** Phase 5 · 13（问答），Phase 5 · 23（分块策略）
**时长：** 约60分钟

## 问题

你有一份200页合同。模型声称有 1M-token 上下文。你粘贴合同并问："终止条款是什么？"模型回答了——但从封面页回答，因为终止条款在120k token 深处，超过了模型实际attend的范围。

这是2026年的上下文容量差距。规格表说 1M 或 10M。现实说其中 60-70% 可用，而且"可用"取决于任务。

- **检索（单针在草垛）：** 在前沿模型上接近完美，直到广告最大值。
- **多跳 / 聚合：** 在大多数模型上超过约 128k 后急剧退化。
- **分散事实推理：** 第一个失败的任务。

长上下文评估测量这些轴。本课命名基准测试，每个实际测什么，以及如何为你的领域构建自定义针测试。

## 核心概念

**大海捞针（NIAH，2023）。** 将一个事实（"神奇词是菠萝"）放在长上下文中的受控深度。问模型检索它。扫深度 × 长度。原始长上下文基准。前沿模型现在已饱和它；这是一个必要但不充分的基线。

**RULER（Nvidia，2024）。** 跨4类别的13种任务类型：检索（单键 / 多键 / 多值）、多跳追踪（变量跟踪）、聚合（常见词频率）、QA。上下文长度可配置（4k 到 128k+）。揭露饱和 NIAH 但在多跳上失败的模型。在2024年发布中，声称 32k+ 上下的17个模型中只有一半在 32k 时保持质量。

**LongBench v2（2024）。** 503个多项选择题，8k-2M 词上下文，六种任务类别：单文档 QA、多文档 QA、长上下文学习、长对话、代码仓库、长结构化数据。真实世界长上下文行为的生产基准。

**MRCR（多轮共指消解）。** 大规模多轮共指。8针、24针、100针变体。暴露模型在注意力退化前能 juggling 多少事实。

**NoLiMa。** "非词汇针。"针和查询没有字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 连接许多文档，从任何一个问问题。测试选择性注意力。

**BABILong。** 将 bAbI 推理链嵌入无关草垛中。测试草垛中的推理，而非仅检索。

### 实际报告什么

- **广告上下文窗口。** 规格表数字。
- **有效检索长度。** 某阈值（如 90%）的 NIAH 通过率。
- **有效推理长度。** 该阈值的多跳或聚合通过率。
- **退化曲线。** 准确性 vs 上下文长度，按任务类型绘制。

两个数字给你的规格表：检索有效性和推理有效性。通常推理有效是广告窗口的 25-50%。

## 动手实现

### 步骤 1：你的领域自定义 NIAH

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。那是你目标模型的 NIAH 卡。

### 步骤 2：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

### 步骤 4：LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

## 陷阱

- **仅 NIAH 评估。** 在 1M token 通过 NIAH 什么也说不说明多跳。总是运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。测试 depth=0, 0.25, 0.5, 0.75, 1.0——"迷失在中间"效应是真实的。
- **与填充物的词汇重叠。** 如果针与填充物共享关键词，检索变得 trivial。使用 NoLiMa 风格无重叠针。
- **忽略延迟。** 1M-token 提示预填充需 30-120 秒。同时测量首个 token 时间和准确性。
- **供应商自报数字。** OpenAI、Google、Anthropic 都发布自己的分数。总是在你的用例上独立重跑。

## 用现成库

2026年技术栈：

| 场景 | 基准 |
|-----------|-----------|
| 快速理智检查 | 自定义 NIAH 3深度 × 3长度 |
| 生产模型选择 | 目标长度的 RULER（13任务） |
| 真实世界 QA 质量 | LongBench v2 单文档 QA 子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话 / 对话 | 目标长度的 MRCR 8针 |
| 模型升级回归 | 固定 in-house NIAH + RULER 测试，每次新模型运行 |

生产经验法则：直到在目标长度上有 NIAH + 1个推理任务，不要信任上下文窗口。

## 产出

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: 为给定模型和用例设计长上下文评估电池。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

给定目标模型、目标上下文长度和用例，输出：

1. 测试。NIAH 深度 × 长度网格；RULER 多跳；自定义领域任务。
2. 采样。深度 0, 0.25, 0.5, 0.75, 1.0 在每个长度。
3. 指标。检索通过率；推理通过率；首个 token 时间；每查询成本。
4. 截止。有效检索长度（90% 通过）和有效推理长度（70% 通过）。报告两者。
5. 回归。固定测试工具，每次模型升级重跑，表面差值。

拒绝仅信任模型卡的上下文窗口。拒绝任何多跳工作负载的纯 NIAH 评估。拒绝供应商自报长上下文分数作为独立证据。
```

## 练习

1. **简单。** 构建 3深度（0.25, 0.5, 0.75）× 3长度（1k, 4k, 16k）的 NIAH。在任何模型上运行。将通过率绘制为 3×3 热力图。
2. **中等。** 添加3针变体。在每个长度测量所有3个的检索。与同长度单针通过率比较。
3. **困难。** 在 64k 填充中构建变量追踪任务（X1 → X2 → X3，3跳）。跨3个前沿模型测量准确性。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 在填充物中种一个事实，问模型检索它。 |
| RULER | 类固醇上的 NIAH | 跨检索 / 多跳 / 聚合 / QA 的13种任务类型。 |
| Effective context | 真实容量 | 准确性仍在阈值以上的长度。 |
| Lost in the middle | 深度偏差 | 模型对长输入中间的内容attend不足。 |
| Multi-needle | 一次多个事实 | 多个种植；测试注意力 juggling，而非仅检索。 |
| MRCR | 多轮共指 | 8、24或100针共指；暴露注意力饱和。 |
| NoLiMa | 非词汇针 | 针和查询共享字面 token；需要推理。 |

## 扩展阅读

- [kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov et al. (2024). BABILONG](https://arxiv.org/abs/2406.10149) — 草垛中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文。