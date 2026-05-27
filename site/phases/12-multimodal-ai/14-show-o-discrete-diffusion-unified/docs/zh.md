# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie 等，2024 年 8 月）走另一条路：文本 token 用因果下一 token 预测，图像 token 用 MaskGIT 风格的掩码离散扩散。两者都在一个带混合注意力掩码的 transformer 内。结果在一个骨干、一个每模态分词器、一个损失公式（扩展为掩码预测的下一 token）上统一了 VQA、文生图、修复和混合模态生成。本节走读 Show-o 设计——为什么掩码离散扩散是并行的、少步的图像生成器——并与 Transfusion 和 Emu3 对比。

**类型：** Learn
**语言：** Python（标准库，掩码离散扩散采样器）
**前置知识：** Phase 12 · 13（Transfusion）
**时间：** 约 120 分钟

## 学习目标

- 解释掩码离散扩散：统一掩码 token 然后要求 transformer 恢复的计划。
- 在速度和质量上比较并行图像解码（Show-o、MaskGIT）和自回归图像解码（Chameleon、Emu3）。
- 说出 Show-o 在一个 checkpoint 中处理的三个任务：T2I、VQA、图像修复。
- 选择掩码调度（余弦、线性、截断）并推理其对样本质量的影响。

## 问题背景

Transfusion 的双损失训练有效，但动态更棘手——连续扩散损失与离散 NTP 损失生活在不同的数值尺度上。平衡损失权重是超参数搜索。架构有效但复杂。

Show-o 的答案：保持两者都离散（如 Chameleon），但通过掩码离散扩散并行生成图像而非顺序生成。训练目标变成单一的掩码 token 预测，自然地泛化下一 token 预测。

## 核心概念

### 掩码离散扩散（MaskGIT）

原始 Chang 等人（2022）的 MaskGIT 技巧很优雅。从完全掩码的图像开始（每个 token 都是特殊的 `<MASK>` id）。每步，并行预测所有掩码 token，然后保留 top-K 最自信的预测，重新掩码其余。在约 8-16 次迭代后，所有 token 都填充完毕。每步掩码多少 token 的调度是调优的——余弦调度效果很好。

训练简单：从 [0, 1] 均匀采样掩码比例，应用到图像的 VQ token，训练 transformer 恢复掩码的那些。正好是 BERT 对文本做的，扩展到图像生成。

### Show-o：一个 transformer，混合掩码

Show-o 将 MaskGIT 放入因果语言模型 transformer。注意力掩码是：

- 文本 token：因果（标准 LLM）。
- 图像 token：在图像块内完全双向（因此掩码 token 在预测时可以查看所有其他图像 token）。
- 文本到图像：文本注意前面的图像，图像注意前面的文本。

训练交替于：
1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 图像，带掩码图像 token，掩码 token 预测损失。
3. VQA 样本：图像 → 文本，带掩码文本 token（实际上是 NTP）。

统一损失是对 `<MASK>` token 的交叉熵，覆盖文本 NTP（只有最后一个 token 是"掩码"）和图像掩码扩散（随机子集被掩码）。

### 并行采样

Show-o 生成图像约 16 步，而不是约 1000（逐 token 自回归）或约 20（扩散）。每步，并行预测所有掩码 token；提交 top-K 自信；重复。

对比：
- Chameleon / Emu3（逐 token 自回归）：N_tokens 次前向传播，每图像典型 1024-4096。
- Transfusion（连续扩散）：约 20 步，每步一次完整 transformer 传播。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整 transformer 传播。

Show-o 在相似规模模型上比 Chameleon 快，大致匹配 Transfusion 步数但每步成本更低（离散词汇表 logit vs 连续 MSE 损失）。

### 一个 checkpoint 中的任务

Show-o 在推理时支持四种任务，由 prompt 格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过掩码离散扩散输出图像。
- 修复：带部分 token 被掩码的图像，填入。

修复能力从掩码预测训练免费获得。掩码 VQ-token 网格的一个区域，喂入其余加上文本 prompt，预测掩码 token。

### 掩码调度

每步解掩多少 token 的调度影响质量。Show-o 推荐余弦：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

第 0 步，所有 token 掩码（比例 1.0）。第 T 步，无掩码。余弦将质量集中在预测最有信息的中间比例。线性调度也可以但 plateau 更快。

### Show-o2

Show-o2（2025 年续作，arXiv 2506.15564）扩展 Show-o：更大的 LLM 基座、更好的分词器、改进的掩码调度。相同架构模式。

### Show-o 所在位置

在 2026 年分类学中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍有分词器损失。
- 连续 + 扩散：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- VLM 中连续 + 流匹配：JanusFlow、InternVL-U。最新的。

按任务选择：当你想要在一个开源模型中同时需要 T2I + 修复 + VQA 且速度合理时选 Show-o；当你质量至上且能承受双损失管道时选 Transfusion。

## 使用方法

`code/main.py` 模拟 Show-o 采样：

- 一个 16 个 VQ token 的玩具网格。
- 一个模拟"transformer"，基于 prompt 和当前未掩码 token 预测 logit。
- 8 步余弦调度的并行掩码采样。
- 打印中间状态（掩码模式演化）和最终 token。

运行它，观察掩码逐步消解。

## 输出作品

本节生成 `outputs/skill-unified-gen-model-picker.md`。给定一个需要同时理解（VQA、标题生成）和生成（T2I、修复）且有开源权重约束的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3/Chameleon 家族之间选择，附具体权衡。

## 练习

1. 掩码离散扩散采样约 16 步。为什么不是 1？如果在第 0 步解掩所有 token，什么会坏？

2. 修复是掩码扩散的免费功能。提出一个产品用例（真实或假设），其中 Show-o 的修复胜过专家模型。

3. 余弦调度 vs 线性调度：追踪 T=8 时每步解掩的 token 数。哪个更平衡？

4. 512x512 Show-o 图像是 1024 token。在词汇表 K=16384 下，模型发出 1024 * log2(16384) = 14,336 比特（约 1.75 KiB）数据。Stable Diffusion 输出 512*512*24 比特 = 6,291,456 比特（约 768 KiB）原始像素。压缩比是多少，买到了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类别条件自回归图像模型与 Show-o 的掩码方法有什么不同？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Masked discrete diffusion | "MaskGIT 风格" | 训练预测掩码 token；推理时迭代解掩最自信的预测 |
| Cosine schedule | "解掩调度" | 推理步骤上掩码比例的衰减；在中间比例集中置信度增长 |
| Parallel decoding | "一次全部 token" | 每步在一次前向传播中预测完整序列的掩码 token，然后提交 top-K |
| Hybrid attention | "因果 + 双向" | 掩码在文本 token 上因果，在图像块内双向 |
| Inpainting | "填充生成" | 以带部分 token 被掩码的图像为条件，预测缺失的；来自训练目标免费获得 |
| Commitment rate | "每步 top-K" | 每迭代宣布多少 token"完成"；控制推理与质量的权衡 |

## 延伸阅读

- [Xie 等 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang 等 — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun 等 — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang 等 — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)