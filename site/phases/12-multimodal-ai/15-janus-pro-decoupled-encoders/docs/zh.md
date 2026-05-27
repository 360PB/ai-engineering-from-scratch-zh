# Janus-Pro：统一多模态模型的解耦编码器

> 统一多模态模型有一个不可避免的张力。理解需要语义特征——SigLIP 或 DINOv2 输出的富含概念级信息的向量。生成需要重建友好的编码——VQ token 可以组合回清晰像素。一个编码器不能同时满足两个目标。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）认为解法是停止尝试：解耦两个编码器。在任务间共享 transformer 主体，但理解路由经 SigLIP，生成路由经 VQ 分词器。在 70 亿参数下，Janus-Pro 在 GenEval 上击败 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本节解读为什么两个编码器在一个失败的地方有效。

**类型：** Build
**语言：** Python（标准库，双编码器路由 + 共享主体信号）
**前置知识：** Phase 12 · 13（Transfusion），Phase 12 · 14（Show-o）
**时间：** 约 120 分钟

## 学习目标

- 解释为什么单一共享编码器在理解或生成质量上做出妥协。
- 描述 Janus-Pro 的路由：输入侧理解用 SigLIP 特征，生成在输入和输出侧都用 VQ token。
- 追踪使 Janus-Pro 在 Janus 不行的地方成功的数据混合规模化。
- 比较解耦（Janus-Pro）、耦合连续（Transfusion）和耦合离散（Show-o）架构。

## 问题背景

统一模型在理解和生成间共享 transformer 主体。此前的尝试（Chameleon、Show-o、Transfusion）都用一个视觉分词器处理两个方向。分词器是一种妥协：

- 针对重建优化（生成）：VQ-VAE 捕获细粒度像素细节，但产生语义一致性弱的 token。
- 针对语义优化（理解）：SigLIP embedding 将"cat"图像聚集在"cat"token 附近，但不允许良好重建。

Show-o 和 Transfusion 在一个方向上为此付出明显的质量税。Janus-Pro 问：当任务需求不同时，为什么要一个分词器？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构分离两个编码器：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → transformer 主体。
- 生成路径。输入图像（如果以现有图像为条件）→ VQ 分词器 → token ID → transformer 主体。
- 输出生成。Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 主体是共享的。主体上游和下游的一切都是任务特定的。

输入通过 prompt 格式区分：`<understand>` 标签路由经 SigLIP；`<generate>` 路由经 VQ。或者路由从任务隐式推断。

### 为什么有效

理解损失获得 SigLIP 特征，这是 CLIP 风格预训练调整用于语义相似性的。模型的感知基准在理解上优于 Show-o / Transfusion，因为输入特征对该任务更好。

生成损失获得 VQ token，这是分词器调整用于重建的。图像质量在理解上优于 Show-o，因为 VQ 编码可以干净地组合回像素。

共享 transformer 主体看到两种输入分布（SigLIP 和 VQ），并学习处理两者。主张：足够数据 + 足够参数，主体吸收切换。

### 数据规模化——Janus vs Janus-Pro

Janus（原始，arXiv 2410.13848）引入了解耦但在小型（13 亿参数，有限数据）。Janus-Pro（arXiv 2501.17811）规模化：

- 70 亿参数（vs 13 亿）。
- 第一阶段（对齐）从 7200 万增加到 9000 万图文对。
- 第二阶段（统一）从 2600 万增加到 7200 万。
- 第三阶段增加了 20 万图像生成指令样本。

要旨：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），在 GenEval 上击败 DALL-E 3（0.80 vs 0.67）。一个开源模型，在统一频谱两边都有竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径换成整流流生成路径（连续）。分割变为 SigLIP 用于理解 + 整流流用于生成。质量上限进一步提升。架构保持解耦编码器-共享主体。

### 共享主体的职责

Transformer 主体处理统一序列但有两种输入分布。它的职责是：

- 对于理解：消费 SigLIP 特征 + 文本 token → 自回归发出文本。
- 对于生成：消费文本 token +（可选图像 VQ token）→ 自回归发出图像 VQ token。

每个 block 没有模态特定的权重。它是你期望在 Qwen 或 Llama 内部找到的文本风格 transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主体可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这一选择重要：LLM 贡献的推理能力是从头开始的统一模型难以达到的。

### 与 InternVL-U 对比

InternVL-U（第 12.10 节）是 2026 年后续。它结合了：

- 原生多模态预训练（InternVL3 骨干）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散 head 输出）。
- 统一理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择吸收到更大的框架中。解耦编码器想法现在是规模化统一模型的默认选择。

### 局限性

解耦编码器增加架构复杂性。需要训练两个分词器、维护两个输入路径、两套失败模式。对于不需要生成的产品，Janus-Pro 过度工程化——选 LLaVA 家族的理解模型。

对于不需要理解的产品，Janus-Pro 资历过高——选 Stable Diffusion 3 / Flux 模型。

对于两者都需要的产品，Janus-Pro 现在是参考开源架构。

## 使用方法

`code/main.py` 模拟 Janus-Pro 路由：

- 两个模拟编码器：SigLIP 类（产生 256 维语义向量）和 VQ 类（产生整数编码）。
- 基于任务标签选编码器的 prompt 路由器。
- 共享主体（替代），处理无论由哪个编码器产生的 token 序列。
- 从第一阶段（对齐）到第三阶段（指令微调）加权采样调度的切换。

为 3 个示例打印路由路径：图像 QA、T2I、图像编辑。

## 输出作品

本节生成 `outputs/skill-decoupled-encoder-picker.md`。给定一个产品需要前沿质量级别的统一生成 + 理解，在 Janus-Pro、JanusFlow 或 InternVL-U 中选择，附具体数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败 DALL-E 3。解释为什么 70 亿开源模型能在生成上匹配前沿专有模型但在理解上不行。

2. 实现一个路由器函数：给定 prompt 文本，分类为 `understand` 或 `generate`。如何处理"先描述再画草图"这样的歧义 prompt？

3. JanusFlow 用整流流替换 VQ 路径。Transformer 主体现在输出什么，损失中有什么变化？

4. 提出 Janus-Pro 架构可以用再多一个解耦编码器处理的第四个任务。例子：图像分割（DINO 风格）、深度（MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据规模化。哪个数据阶段对 T2I 质量收益贡献最大 vs Janus？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Decoupled encoding | "两个视觉编码器" | 每个方向分离的分词器或编码器：一个语义用于理解，一个重建用于生成 |
| Shared body | "一个 transformer" | 单个 transformer 处理任一编码器的输出；无模态特定权重 |
| SigLIP for understanding | "语义特征" | CLIP 家族视觉塔，提供丰富的概念特征但重建能力差 |
| VQ for generation | "重建编码" | 向量量化 token，可以干净地解码回像素 |
| JanusFlow | "整流流变体" | Janus-Pro 用连续流匹配生成 head 替换 VQ |
| Routing tag | "任务标签" | 选择输入编码器的 prompt 标记（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu 等 — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen 等 — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma 等 — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong 等 — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)