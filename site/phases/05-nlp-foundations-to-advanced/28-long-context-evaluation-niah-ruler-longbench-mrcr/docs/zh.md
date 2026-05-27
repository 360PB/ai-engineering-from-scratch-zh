# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 标称 10M token 上下文。在 1M token 时，8-针 MRCR 降至 26.3%。标称 ≠ 可用。长上下文评估告诉你实际在用的模型容量是多少。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 5 · 13（问答）、Phase 5 · 23（分块策略）
**时长：** 约 60 分钟

## 问题

你有一份 200 页的合同。模型声称有 1M token 上下文。你把合同粘贴进去问："终止条款是什么？"模型回答了——但答案来自封面，因为终止条款在 120k token 深处，超过了模型实际参与运算的位置。

这就是 2026 年的上下文容量差距。规格表写着 1M 或 10M。实际情况是可用容量是宣传值的 60-70%，而且"可用"取决于任务类型。

- **检索（大海捞针）：** 在前沿模型上接近完美，可达宣传最大值。
- **多跳 / 聚合：** 在大多数模型上超过 ~128k 后急剧下降。
- **分散事实推理：** 第一个失效的任务。

长上下文评估测量这些维度。本课程命名各基准测试、每个实际测量什么，以及如何为你的领域构建自定义捞针测试。

## 概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**大海捞针（Needle-in-a-Haystack，NIAH，2023）。** 将一个事实（"魔法词是 pineapple"）放在长上下文中的受控深度位置。让模型检索它。扫描深度 × 长度。原始长上下文基准测试。前沿模型现在已在这一项上饱和；它是必要的但非充分基线。

**RULER（Nvidia，2024）。** 4 大类 13 种任务类型：检索（单键 / 多键 / 多值）、多跳追踪（变量追踪）、聚合（高频词）、问答。可配置上下文长度（4k 到 128k+）。揭示在 NIAH 上饱和但在多跳上失败的模型。在 2024 年版本中，声称 32k+ 上下的 17 个模型中，只有半数在 32k 时保持了质量。

**LongBench v2（2024）。** 503 道选择题，上下文 8k–2M 词，6 个任务类别：单文档 QA、多文档 QA、长上下文学习、长对话、代码库、长结构化数据。现实世界长上下文行为的生产基准。

**MRCR（Multi-Round Coreference Resolution，多轮共指消解）。** 规模化多轮共指。8 针、24 针、100 针变体。暴露模型在注意力衰减前能处理多少个事实。

**NoLiMa。** "非词汇针"。针与查询没有字面重叠；需要一步语义推理才能检索。比 NIAH 更难。

**HELMET。** 将许多文档拼接在一起，从任意一个提问。测试选择性注意力。

**BABILong。** 在无关 haystack 中嵌入 bAbI 推理链。测试 haystack 中的推理，而不仅仅是检索。

### 实际应报告的内容

- **宣传上下文窗口。** 规格表上的数字。
- **有效检索长度。** NIAH 在某个阈值（如 90%）下通过。
- **有效推理长度。** 多跳或聚合在该阈值下通过。
- **退化曲线。** 准确率 vs 上下文长度，按任务类型绘制。

规格表上写两个数字：有效检索长度和有效推理长度。通常推理有效长度是宣传窗口的 25–50%。

## 构建

### 步骤 1：为你的领域定制 NIAH

参见 `code/main.py`。骨架如下：

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

    # 重复填充文本直到足够长以填满 haystack 主体。
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

扫描 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热图。这就是你的目标模型的 NIAH 卡。

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

类似"三个魔法词是什么？"的问题需要检索全部三个。单针成功不能预测多针成功。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链接三次赋值。前沿模型在 128k 时往往降至 50–70% 准确率。

### 步骤 4：在你的技术栈上运行 LongBench v2

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

报告每个类别的准确率。聚合分数掩盖了大的任务级别差异。

## 陷阱

- **仅 NIAH 评估。** 在 1M token 通过 NIAH 不能说明任何多跳能力。始终运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。要测试 depth=0, 0.25, 0.5, 0.75, 1.0——"迷失在中间"效应是真实存在的。
- **与填充文本的词汇重叠。** 如果针与填充文本共享关键词，检索变得 trivial。使用 NoLiMa 风格的无重叠针。
- **忽视延迟。** 1M token 提示需要 30–120 秒预填充。除了准确率，还要测量首个 token 出来的时间。
- **供应商自报数字。** OpenAI、Google、Anthropic 都发布自己的分数。始终在你的用例上独立重新运行。

## 使用

2026 年技术栈：

| 场景 | 基准测试 |
|------|---------|
| 快速健全性检查 | 自定义 NIAH，3 深度 × 3 长度 |
| 生产选型 | RULER（13 种任务）在你的目标长度 |
| 真实世界 QA 质量 | LongBench v2 单文档 QA 子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话 / 语音 | MRCR 8 针在目标长度 |
| 模型升级回归 | 固定内部 NIAH + RULER 测试套件，在每个新模型上运行 |

生产经验法则：在有 NIAH + 1 个推理任务在目标长度通过之前，不要信任任何上下文窗口。

## 上线

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
2. 采样。0, 0.25, 0.5, 0.75, 1.0 各深度，每个长度。
3. 指标。检索通过率；推理通过率；首个 token 时间；每次查询成本。
4. 截断点。有效检索长度（90% 通过）和有效推理长度（70% 通过）。两个都报告。
5. 回归。固定测试套件，每次模型升级重新运行，输出差异。

拒绝仅凭模型卡信任上下文窗口。拒绝任何多跳工作负载的纯 NIAH 评估。拒绝供应商自报的长上下文分数作为独立证据。
```

## 练习

1. **简单。** 构建 NIAH，3 深度（0.25, 0.5, 0.75）× 3 长度（1k, 4k, 16k）。在任何模型上运行。将通过率绘制为 3×3 热图。
2. **中等。** 添加 3 针变体。在每个长度测量全部 3 个的检索。与同一长度的单针通过率比较。
3. **困难。** 构建变量追踪任务（X1 → X2 → X3，3 跳），嵌入 64k 填充文本。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|---------|
| NIAH | 大海捞针 | 在填充文本中埋入一个事实，让模型检索它。 |
| RULER | NIAH 强化版 | 13 种任务类型，涵盖检索 / 多跳 / 聚合 / 问答。 |
| Effective context | 真实容量 | 准确率仍高于阈值的最大长度。 |
| Lost in the middle | 深度偏差 | 模型对长输入中间内容注意力不足。 |
| Multi-needle | 多事实同时 | 多个埋点；测试注意力分配，不仅仅是检索。 |
| MRCR | 多轮共指 | 8、24 或 100 针共指；暴露注意力饱和。 |
| NoLiMa | 非词汇针 | 针与查询没有共同 token；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准测试。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 现实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — haystack 中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文。