# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步）。VAR 在尺度上迭代采样——它预测 1×1 token，然后 2×2，然后 4×4，直到最终分辨率，每个尺度以前一个为条件。2024 年的论文表明 VAR 匹配 GPT 风格图像生成的缩放定律，并在相同计算预算下击败 DiT。本课构建核心机制。

**类型：** Build
**语言：** Python（带 PyTorch）
**前置知识：** Phase 7 · 03（多头注意力），Phase 8 · 06（DDPM）
**时间：** 约 90 分钟

## 问题

自回归生成在语言建模中占主导地位，因为它可预测地扩展：更多计算、更多参数、更低困惑度、更好输出。图像生成在 2024 年之前有两种主要 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（逐 VQ-VAE code 的 token）。

两者都受困于生成顺序问题。像素和 token 排列在 2D 网格中，但 AR 模型必须以 1D 光栅顺序访问它们。早期的角像素对图像最终变成什么一无所知。生成质量比 GPT 在文本上扩展得更差，从未在匹配计算下达到扩散模型的质量。

VAR 通过改变生成内容来解决生成顺序问题。不是在空间中逐个预测图像 token，而是 VAR 预测整个图像在递增分辨率。步骤 1：预测 1×1 token（整体图像"摘要"）。步骤 2：预测 2×2 token 网格（较粗糙的特征）。步骤 3：预测 4×4 网格。步骤 K：预测最终 (H/8)×(W/8) 网格。

每个尺度关注所有之前的尺度（在"尺度顺序"上是因果的）并在自身尺度内并行。顺序问题消失：整个图像在尺度 k 一次 transformer 前向传播产生。

## 核心概念

### VQ-VAE 多尺度 Tokenizer

VAR 需要一个**多尺度离散 tokenizer**。对于图像 x，它产生一个渐进更高分辨率 token 网格的序列：

```
x -> encoder -> latent f
f -> 在 1×1 tokenize：token 网格 z_1，形状 (1, 1)
f -> 在 2×2 tokenize：token 网格 z_2，形状 (2, 2)
...
f -> 在 (H/p)×(W/p) tokenize：token 网格 z_K，形状 (H/p, W/p)
```

每个 z_k 使用相同的 codebook（典型大小 4096-16384）。每个尺度的 tokenize 不是独立的——它被训练成这样：在每个尺度上残差求和重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是**残差 VQ** 变体。尺度 k 捕获尺度 1..k-1 遗漏的内容。解码器取所有尺度 embedding 的和并产生图像。

多尺度 VQ tokenizer 训练一次（如 VQGAN）然后冻结。所有生成工作由其上的自回归模型完成。

### 下一尺度预测

生成模型是一个 transformer，它看到所有先前尺度的 token 并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入同时编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k、位置 (i, j) 的 token 可以关注尺度 1..k 的所有 token 以及尺度 k 本身中按其内部顺序更早出现的 token（VAR 使用固定位置注意力，尺度内无因果性——尺度内的所有位置并行预测）。

训练损失：在每个尺度 k，给定所有先前尺度 token，预测 token z_k。关于离散 VQ codes 的交叉熵损失。与 GPT 相同的结构，只是"序列"现在是尺度结构的。

### 生成

在推理时：
```
生成 z_1 = 从 p(z_1) 采样                    # 1 token
生成 z_2 = 从 p(z_2 | z_1) 采样              # 4 token 并行
生成 z_3 = 从 p(z_3 | z_1, z_2) 采样         # 16 token 并行
...
解码：f = 所有尺度 1..K 的 embed-and-upsample 之和
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成是 10 次 transformer 前向传播。每次传播在一步中产生其整个尺度——尺度内无逐 token 自回归。对于 256×256 图像，这约是 10 次 vs DiT 的 28-50 次。

### 为什么下一尺度优于下一 token

三个结构优势：
1. **从粗到细与自然图像统计一致。** 人类视觉感知和图像数据集都表现出尺度依赖规律性：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用这一点。
2. **尺度内并行生成。** 与 GPT 风格 token AR 不同，VAR 在一步中产生一个尺度的所有 token。有效生成长度是对数级而不是线性的。
3. **无生成顺序偏差。** 尺度 k 的 token 看到尺度 k-1 的全部；没有"左边"或"上边"偏差，迫使早期 token 在后期上下文可用前就做出承诺。

### 缩放定律

Tian 等人证明 VAR 遵循 FID 在 ImageNet 上关于计算的幂律——就像 GPT 对困惑度一样。翻倍参数或计算可靠地减半错误。这是第一个以语言模型那种清晰方式表现出这种缩放行为的图像生成模型。结果是，VAR 规模的预测可以从计算中预测，而不是每次架构变体的经验猜测。

### 与扩散的关系

VAR 和扩散共享相同的数据压缩故事：都将生成问题分解为一序列更简单的子问题。

- 扩散：逐步添加噪声，学习撤销每一步。
- VAR：逐步添加分辨率，学习预测下一尺度。

它们是问题不同维度的切入。两者都产生可处理的条件分布。Empirically VAR 推理更快（更少传递，尺度内全并行），在匹配参数预算下匹配或超越 DiT 类条件 ImageNet。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 构建

在 `code/main.py` 中你将：
1. 在合成"图像"数据（2D 高斯环）上构建一个微型**多尺度 VQ tokenizer**。
2. 训练一个**VAR 风格 transformer** 来下一尺度预测 token。
3. 通过调用 transformer 4 次（4 个尺度）并解码来采样。
4. 验证尺度顺序训练使尺度内生成并行。

这是一个玩具实现。要点是看到尺度结构注意力掩码和尺度内并行生成实际工作。

## 发布

本课产生 `outputs/skill-var-tokenizer-designer.md`——一个用于设计多尺度 tokenizer 的 skill：尺度数量、尺度比例、codebook 大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量 vs 自回归传递数量。更多尺度 = 更细残差 = 更好质量但更多传递。
2. **Codebook 大小。** 用 codebook 大小 512、4096、16384 训练 tokenizer。更大的 codebook 给出更好重建但预测更难。找到拐点。
3. **尺度内并行检查。** 对于训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置但不关注尺度内？验证掩码实现。
4. **VAR vs DiT 缩放。** 对于相同的 ImageNet 类别条件任务，在匹配参数预算（例如 33M、130M、458M）下训练 VAR 和 DiT。绘制 FID vs 计算。VAR 应在每个规模上领先 DiT——在较小规模上复现论文结果。
5. **文本条件化。** 通过 adaLN 将文本 embedding（CLIP pooled）作为额外条件输入扩展 VAR。这是 HART 配方。在文本对齐采样上 FID 改善了多少？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| VAR | "视觉自回归" | 通过 VQ token 网格金字塔上的下一尺度预测进行图像生成 |
| 下一尺度预测 | "从粗到细预测" | 模型在递增分辨率尺度上预测 token，以前一个尺度为条件 |
| 多尺度 VQ tokenizer | "残差 VQ" | VQ-VAE 产生 K 个递增分辨率的 token 网格，解码器对所有尺度求和 |
| 尺度 k | "金字塔级别 k" | K 个分辨率级别之一，从 k=1 的 1×1 到 k=K 的 (H/p)×(W/p) |
| 尺度内并行 | "每尺度一次前向" | 尺度 k 的所有 token 在一次 transformer 传递中预测，不是自回归的 |
| 跨尺度因果 | "尺度顺序注意力" | 尺度 k 的 token 可以关注尺度 1..k 的全部，但不能关注 k+1..K |
| 残差 VQ | "加性 tokenize" | 每个尺度的 token 对较低尺度的残差进行编码；解码器对所有尺度 embedding 求和 |
| VAR 缩放定律 | "图像 GPT 缩放" | FID 在计算中遵循可预测的幂律，像语言模型的困惑度一样 |
| HART | "混合 VAR + 文本" | 结合 MaskGIT 风格迭代解码与 VAR 尺度结构的文本条件 VAR 变体 |
| 尺度位置嵌入 | "(scale, row, col) 三元组" | 位置编码同时携带尺度索引和尺度内的空间坐标 |

## 进一步阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，规范参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散比较基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度 tokenizer 扩展的 tokenizer 系列
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像 token 化的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR