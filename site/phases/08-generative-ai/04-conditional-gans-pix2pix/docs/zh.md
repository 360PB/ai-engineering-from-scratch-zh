# 条件 GAN 与 Pix2Pix

> 2014-2017 年的第一个重大突破是控制 GAN 的输出。附加一个标签，或一张图像，或一个句子。Pix2Pix 做了图像版本，它在窄领域图像到图像任务上仍然打败每个通用文生图模型。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 03（GAN），Phase 4 · 06（U-Net），Phase 3 · 07（CNN）
**时间：** 约 75 分钟

## 问题

无条件 GAN 从任意分布采样人脸。对演示有用，在生产中无用。你想要：*将草图映射到照片*、*将地图映射到航拍照片*、*将白天场景映射到夜晚*、*将灰度图像着色*。在所有这些任务中，你得到一个输入图像 `x`，必须输出语义对应的 `y`。每个 `x` 有许多可信的 `y`。均方误差将它们磨成糊状。对抗损失不会，因为"看起来真实"是锐利的。

条件 GAN（Mirza & Osindero，2014）将条件 `c` 作为输入添加到 `G` 和 `D`。Pix2Pix（Isola et al.，2017）将其专门化：条件是完整的输入图像，生成器是 U-Net，判别器是*基于 patch 的*分类器（PatchGAN），损失是 adversarial + L1。这个配方即使在 2026 年也超过从零训练的通用文生图模型在窄领域图像到图像领域的表现，因为它训练在*成对数据*上——你恰好有你所需要的信号。

## 核心概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声被忽略了）。

**条件 D。** `D(x, y) → [0, 1]`。输入是*配对*（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅判断 `y` 看起来是否真实。

**U-Net 生成器。** 带有跨越瓶颈的跳连接的编码器-解码器。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳连接，高频细节会消失。

**PatchGAN 判别器。** D 不是输出单个真/假分数，而是输出一个 `N×N` 网格，其中每个单元格判断一个感受野约 70×70 像素。取平均。这是马尔可夫随机场假设：真实感是局部的。训练更快，参数更少，输出更锐利。

**损失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并推动 G 走向已知目标。L1 比 L2 产生更锐利的边缘（因为 L1 趋向中位数而非均值）。Pix2Pix 默认值 λ = 100。

## CycleGAN — 当你没有配对数据时

Pix2Pix 需要成对的 `(x, y)` 数据。CycleGAN（Zhu et al.，2017）放弃了这个要求，代价是一个额外的损失：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你可以马变斑马，夏天变冬天，无需配对示例。

在 2026 年，非配对图像到图像主要通过扩散完成（ControlNet、IP-Adapter）而不是 CycleGAN，但循环一致性思想在几乎每篇非配对领域适应论文中都存活下来。

## 构建

`code/main.py` 在 1-D 数据上实现了一个微型条件 GAN。条件 `c` 是一个类别标签（0 或 1）。任务：为给定类别从条件分布生成样本。

### 第 1 步：将条件附加到 G 和 D 输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot 编码是最简单的方式。更大的模型使用学习到的 embedding、FiLM 调制或交叉注意力。

### 第 2 步：训练条件模型

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配给定条件的真实分布，而不是边缘分布。

### 第 3 步：验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 陷阱

- **条件被忽略。** G 学习边缘化，D 从不惩罚，因为条件信号弱。修复：更激进地条件化 D（早期层，而不仅仅是晚期），使用投影判别器（Miyato & Koyama 2018）。
- **L1 权重太低。** G 漂移到任意真实感的输出，而不是忠实于输入。从 λ≈100 开始用于 Pix2Pix 风格任务。
- **L1 权重太高。** G 产生模糊输出，因为 L1 仍然是 L_p 范数。一旦训练稳定，渐冷下降。
- **D 中的真实标签泄露。** 将 `(x, y)` 连接作为 D 输入，而不是仅仅 `y`。没有这个，D 无法检查一致性。
- **每个类的模式崩溃。** 每个类可以独立崩溃。运行类别条件多样性检查。

## 使用

2026 年图像到图像任务状态：

| 任务 | 最佳方法 |
|------|---------|
| 草图 → 照片，同领域，配对数据 | Pix2Pix / Pix2PixHD（仍然快速，仍然锐利） |
| 草图 → 照片，非配对 | 带 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 带 IP-Adapter 或 LoRA 的扩散；GAN 方法是遗留的 |
| 深度图 → 照片 | Stable Diffusion 上的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 着色 | ColTran、基于扩散的着色器或 Pix2Pix-color |
| 白天 → 夜间、季节、天气 | CycleGAN 或基于 ControlNet 的方法 |

当（a）你有数千个配对示例，（b）任务窄且可重复，（c）你需要快速推理时，Pix2Pix 仍然是正确的工具。在通用开放领域任务上，扩散胜出。

## 发布

保存为 `outputs/skill-img2img-chooser.md`。Skill 接收任务描述、数据可用性（配对 vs 非配对，N 个样本）和延迟 / 质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本和评估协议（LPIPS、FID、任务特定指标）。

## 练习

1. **简单。** 修改 `code/main.py` 添加第三个类。确认 G 仍然将每个类的噪声映射到正确的模式。
2. **中等。** 在 1-D 环境中用感知风格损失替换 L1（例如，一个小型冻结的 D 作为特征提取器）。它是否改变了条件分布的锐度？
3. **困难。** 在 1-D 环境中草绘一个 CycleGAN：两个分布，两个生成器，循环损失。展示它如何在无配对数据的情况下学习它们之间的映射。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 条件 GAN | "带标签的 GAN" | G(z, c)，D(x, c)。两个网络都看到条件。 |
| Pix2Pix | "图像到图像 GAN" | 配对 cGAN + U-Net G 和 PatchGAN D + L1 损失。 |
| U-Net | "带跳连接的编码器-解码器" | 对称卷积网络；跳连接保留高频。 |
| PatchGAN | "局部真实感分类器" | D 输出每 patch 分数而不是全局分数。 |
| CycleGAN | "非配对图像转换" | 两个 G + 循环一致性损失；无配对数据。 |
| SPADE | "GauGAN" | 用语义图归一化中间激活；分割到图像。 |
| FiLM | "特征级线性调制" | 来自条件的每特征仿射变换；廉价条件化。 |

## 生产笔记：Pix2Pix 作为延迟受限基准

当你有配对数据和窄任务（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的一步推理比扩散快一个数量级。生产比较通常是：

| 路径 | 步数 | L4 单卡 512² 上典型延迟 |
|------|------|------------------------|
| Pix2Pix（U-Net 前向） | 1 | 约 30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | 约 1.2 s |
| SDXL-Turbo Img2Img | 1-4 | 约 0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | 约 3-5 s |

Pix2Pix 在静态 batch 中以吞吐量胜出（每个请求是相同的 FLOPs）。扩散在质量和泛化上胜出。现代策略通常是发布一个 Pix2Pix 风格的蒸馏模型用于窄任务，并为尾部输入提供扩散后备方案。

## 进一步阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影 D。