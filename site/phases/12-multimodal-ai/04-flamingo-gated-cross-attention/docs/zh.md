# Flamingo 与 Few-Shot VLM 的门控交叉注意力

> DeepMind 的 Flamingo（2022）做了两件业内率先之事。它展示了一个模型可以处理任意交错的图像、视频和文本序列。还展示了 VLM 可以进行上下文学习——给一个 few-shot prompt，包含三对示例（图像，标题），模型就能为新图像生成标题，无需任何梯度步。机制：门控交叉注意力层，插入到冻结 LLM 现有层之间，带一个学习到的 tanh 门，初始化为零以保留 LLM 的文本能力。本节走读 Flamingo 的 Perceiver resampler 和门控交叉注意力架构——Gemini 交错输入和 Idefics2 视觉 token 的祖先。

**类型：** Learn
**语言：** Python（标准库，门控交叉注意力 + Perceiver resampler 演示）
**前置知识：** Phase 12 · 03（BLIP-2 Q-Former）
**时间：** 约 120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 走读 Perceiver resampler：N 个图像 patch → 通过交叉注意力产生 K 个固定"潜变量" queries。
- 描述 Flamingo 如何处理交错的图像-文本序列，并保持尊重图像放置的因果掩码。
- 复现一个 few-shot 多模态 prompt 结构（三对图文示例后接查询图像）。

## 问题背景

BLIP-2 将 32 个视觉 token 送入冻结 LLM 的输入层。对每 prompt 一张图像有效。但如果想馈入*多张*交错文本的图像，如"这是图像 A，描述它；这是图像 B，描述它；这是图像 C，描述它"呢？LLM 的自注意力需要在一个流中同时处理图像 token 和文本 token，哪些位置可以关注哪些图像，这个问题变得很棘手。

Flamingo 的答案：完全不改变 LLM 的输入流。在现有 LLM 块之间插入额外的交叉注意力层。文本 token 照常通过 LLM 的因果自注意力。在每隔几个 LLM 块之间，文本 token 也通过新的门控层对图像特征做交叉注意力。门（初始化为零）意味着在第零步，新的层是空操作——模型行为完全像预训练的 LLM。随着训练进行，门打开，视觉信息开始流动。

Flamingo 回答的第二个问题：如何处理 prompt 中可变数量的图像（0、1 或多张）？用 Perceiver resampler——一个小型交叉注意力模块，接受任意数量的 patch，输出固定数量的视觉潜变量 token。无论 prompt 中有多少图像，LLM 交叉注意力层看到的是相同形状。

## 核心概念

### 冻结的 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 开始。所有 700 亿权重不触碰。现有的文本自注意力和 FFN 正常运作。

### Perceiver resampler

对于 prompt 中的每张图像，ViT 产生 N 个 patch token。Perceiver resampler 有 K 个固定可学习潜变量（Flamingo 用 K=64）。每个 resampler 块是两步：

1. 交叉注意力：K 个潜变量关注 N 个 patch token（Q 来自潜变量，K/V 来自 patches）。
2. 潜变量内部的自注意力 + FFN。

经过 6 个 resampler 块后，输出是 K=64 个 dim 1024 的视觉 token，无论 ViT 产生了多少 patches。224x224 图像（196 patches）和 480x480 图像（900 patches）都输出为 64 个 resampler token。

对于视频，resampler 在时间维度应用：每帧的 patches 产生 64 个潜变量，时间位置编码让模型区分 t=0 和 t=N。完整视频变为 T * 64 个视觉 token。

### 门控交叉注意力

每隔 M 层冻结 LLM（Flamingo 用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个初始化为零的可学习标量。
- `tanh(0) = 0`，所以初始化时门控分支贡献为零。
- 当 `alpha` 偏离零，交叉注意力的贡献平滑增长。
- 残差连接意味着即使门完全打开，也不会覆盖 LLM 的文本表示；只是在上面添加视觉信息。

这是 Flamingo 最重要的设计选择：视觉条件化是加性的、有门的、初始化为零的。Flamingo 在第 0 步就是一个完美的 Chinchilla 70B 处理纯文本输入。

### 交错输入的掩码交叉注意力

在 prompt 如 `<image A> caption A <image B> caption B <image C> ?` 中，每个文本 token 只能看到序列中出现在它之前的图像。交叉注意力掩码强制：位置 `t` 处的文本 token 只关注图像索引 `i < i_t` 的图像 resampler token，其中 `i_t` 是位置 `t` 之前最近的图像。"只看前一张图像"和"看所有前图像"都是合理选择；Flamingo 选择了前者。

### 上下文 Few-Shot 学习

Flamingo prompt 形如：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到完成模式，输出"bird"（或 image3 实际显示的内容）。无需梯度步。冻结 LLM 的上下文学习能力穿过门控交叉注意力——这是论文的亮点，也是它重要的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb (M3W)：4300 万个包含交错图像和文本的网页，重建阅读顺序。
2. 图文对（ALIGN + LTIP）：44 亿对。
3. 视频-文本对（VTP）：2700 万个短视频片段。

OBELICS（2023）是对交错网页语料的开源复现，Idefics、Idefics2 和大多数开源"Flamingo-like"模型在上面训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开源复现。架构相同（Perceiver resampler + 冻结 LLaMA 或 MPT 上的门控交叉注意力）。3B、4B、9B checkpoint。由于基础 LLM 更小、数据更少，质量落后于 Flamingo。

Otter（2023）在 OpenFlamingo 基础上用 MIMIC-IT（多模态指令数据集）做指令微调，表明门控交叉注意力也适用于指令遵循。

### 后裔们

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力传承，逐渐简化（Idefics2 为直接 patch token 加自适应池化，放弃了 resampler）。
- Flamingo 到 Chameleon 的过渡：到 2024 年，许多团队转向早期融合（第 12.11 节）；Flamingo 风格的门控交叉注意力在必须冻结骨干网的场景中仍在生产使用。
- Gemini 的交错输入：概念上继承 Flamingo 的交错格式灵活性，确切机制是专有的。

### 与 BLIP-2 的比较

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥 | 在输入层一次 Q-Former | 在每 M 层门控交叉注意力 |
| 视觉 token | 每张图像 32 个 | 每交叉注意力层每张图像 64 个 |
| 冻结 LLM | 是 | 是 |
| Few-shot 上下文 | 弱 | 强——论文的核心 |
| 交错输入 | 无原生支持 | 有，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错网页 |
| 参数量 | 训练 1.88 亿 | 训练约 100 亿（交叉注意力层） |
| 计算量 | 8 块 A100 几天 | 数千块 TPUv4 几周 |

单图像 VQA 预算有限选 BLIP-2。交错、few-shot 或多图像推理选 Flamingo/Idefics2。

## 使用方法

`code/main.py` 演示：

1. 在 36 个假 patch token 上运行 Perceiver resampler，有 8 个可学习潜变量（纯 Python 交叉注意力）。
2. 一个门控交叉注意力步骤：`alpha = 0` → 输出等于输入（LLM 不变），然后 `alpha = 2.0` → 视觉贡献混入。
3. 一个交错掩码构建器，为"(image 1) (text 1) (image 2) (text 2)"序列产生 2D 注意力掩码。

## 输出作品

本节生成 `outputs/skill-gated-bridge-diagnostic.md`。给定开源 VLM 的配置（resampler 有/无，交叉注意力频率，门方案），识别 Flamingo 血统元素并解释冻结策略。用于调试为什么微调后文本性能下降（答案：门开得太快太大）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：90 亿 LLM + 14 亿门控交叉注意力层 + 6400 万 resampler。总参数中训练的比例是多少？

2. 在 PyTorch 中实现门控残差 `y = tanh(alpha) * cross + x`。实验证明 `alpha=0` 时，初始化时 `y==x` 精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390）关于 batch 中每 prompt 图像数量不同时的处理方式。描述填充策略。

4. 为什么 Flamingo 的交叉注意力掩码让文本 token 只关注*最近*的前一张图像而不是所有前图像？阅读 Flamingo 论文第 2.4 节并解释权衡。

5. 上下文 few-shot：为一个新 Flamingo 变体构造 prompt，包含 4 个"图像 → 主物体颜色"的例子。描述当你将示例数从 0 变化到 8 时的预期精度模式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Perceiver resampler | "固定潜变量交叉注意力" | 从可变数量输入 patch 产生 K 个固定 token 的模块 |
| Gated cross-attention | "Tanh 门控桥" | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，初始化为 0 |
| Interleaved input | "混合序列" | 图像和文本在阅读顺序中自由混合的 prompt 格式 |
| Frozen LLM | "无 LLM 梯度" | 文本 LLM 权重不更新；只有 resampler + 交叉注意力层训练 |
| Few-shot | "上下文示例" | 在 prompt 中给几对（图像，答案）；模型无需微调即可泛化 |
| OBELICS | "交错网页语料" | 1.41 亿个带图像和文本的网页开源数据集，按阅读顺序排列 |
| Chinchilla | "700 亿冻结基础" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | "alpha 如何变化" | 训练过程中交叉注意力门打开的速率 |
| Cross-attn frequency | "每隔 M 层" | 门控交叉注意力块插入的频率；Flamingo 用 M=4 |
| OpenFlamingo | "开源复现" | MosaicML/LAION 3-9B 开源 checkpoint；架构与 Flamingo 完全相同 |

## 延伸阅读

- [Alayrac 等 — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla 等 — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon 等 — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料。
- [Jaegle 等 — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li 等 — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后裔。
- [Laurençon 等 — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方案的现代简化。