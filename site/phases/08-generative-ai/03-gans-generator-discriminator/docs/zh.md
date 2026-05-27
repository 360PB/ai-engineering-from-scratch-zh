# GAN — 生成器与判别器

> Goodfellow 2014 年的技巧是完全跳过密度。两个网络。一个生成假货。一个抓住假货。他们斗争直到假货与真品无法区分。这不应该有效。它经常失效。当它有效时，样本在窄领域仍然是文献中最锐利的。

**类型：** Build
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播），Phase 3 · 08（优化器），Phase 8 · 02（VAE）
**时间：** 约 75 分钟

## 问题

VAE 产生模糊样本，因为它的 MSE 解码器损失对于*均值*图像是贝叶斯最优的——而许多可信数字的均值是一张模糊的数字。你想要一个奖励*可信度*而不是与任何一个目标的像素级接近的损失。可信度没有闭式公式。你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和假图像。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号是 `D` 当前认为什么东西看起来真实。这个信号随着 `G` 的改进而更新，追赶一个移动的目标。如果两个网络都收敛，`G` 就学会了数据分布，而无需写出 `log p(x)`。

这就是对抗训练。数学是一个 minimax 博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

在 2026 年，GAN 不再是 SOTA 生成器（扩散和 flow matching 抢走了那顶王冠）。但 StyleGAN 2/3 仍然是有史以来最锐利的人脸模型，GAN 判别器被用作扩散训练中的*感知损失*，对抗训练为快速 1 步蒸馏（SDXL-Turbo、SD3-Turbo、LCM）提供动力，让你能够发布实时扩散。

## 核心概念

![GAN 训练：生成器和判别器的 minimax 博弈](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射到样本 `x̂`。一个解码器形状的网络（密集或转置卷积）。

**判别器 `D(x)`。** 将样本映射到一个标量概率（或分数）。真实 → 1，假 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。二分类交叉熵，真实=1，假=0。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 自信地分类为假时饱和并杀死梯度）。

**训练循环。** 一步 `D`，一步 `G`。重复。

**为什么它有效。** 如果 `G` 完全匹配 `p_data`，那么 `D` 不能比随机好，并在各处输出 0.5；`G` 得不到更多梯度。平衡。

**为什么它崩溃。** 模式崩溃（`G` 找到一个 `D` 无法分类的模式并永远生成它）、梯度消失（`D` 学得太快，`log D` 饱和）、训练不稳定（学习率、batch size，任何东西）。

## 使 GAN 起作用的变体

| 年份 | 创新 | 修复内容 |
|------|------|---------|
| 2015 | DCGAN | 卷积/反卷积、批归一化、LeakyReLU——第一个稳定架构。 |
| 2017 | WGAN、WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失。 |
| 2017 | 谱归一化 | Lipschitz 约束判别器。2026 年的判别器仍在使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，逐步加层。首个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定领域逼真感的 SOTA。 |
| 2021 | StyleGAN3 | 无别名、翻译等变——2026 年仍是人脸金标准。 |
| 2022 | StyleGAN-XL | 条件式、类别感知、更大规模。 |
| 2024 | R3GAN | 以更强正则化重新命名；在 1024² 上无需技巧即可工作。 |

## 构建

`code/main.py` 在 1-D 数据上训练一个微型 GAN：两个高斯混合。生成器和判别器都是单隐藏层 MLP。我们手动实现前向、反向和 minimax 循环。目标是亲眼看到两个关键失败模式（模式崩溃 + 梯度消失）是如何发生的。

### 第 1 步：非饱和损失

原始 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 高度自信地将 G 的假样本分类为假时趋近于 0。这时 G 的梯度基本上为零——G 无法改进。非饱和形式 `-log D(G(z))` 有相反的渐近线：在 D 自信时爆炸，给 G 一个强信号。

```python
def g_loss(d_fake):
    # 最大化 log D(G(z))  <=>  最小化 -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第 2 步：每步一个判别器更新对应一个生成器更新

```python
for step in range(steps):
    # 训练 D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 训练 G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 新鲜的假样本
    update_G(fake_batch)
```

G 的假样本要新鲜，否则梯度是过时的。

### 第 3 步：观察模式崩溃

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] 模式崩溃：一个模式被饿死了")
```

典型症状：两个真实模式中的一个停止被生成。判别器停止纠正它，因为它从未被视为假样本。

## 陷阱

- **判别器太强。** 将 D 的学习率降低 2-5 倍，或添加实例/层噪声。如果 D 达到 >95% 准确率，G 就死了。
- **生成器记忆了一个模式。** 在 D 输入中添加噪声，使用 minibatch 判别器层，或切换到 WGAN-GP。
- **批归一化泄露统计量。** 真实 batch + 假 batch 通过同一个 BN 层混合了它们的统计量。使用实例归一化或谱归一化代替。
- **Inception 分数被游戏。** FID 和 IS 在低样本量时很嘈杂。评估时使用 ≥10k 样本。
- **条件任务的单步采样是个谎言。** 你仍然需要 CFG 尺度、截断技巧和重采样才能获得可用的输出。

## 使用

2026 年 GAN 技术栈：

| 场景 | 选择 |
|------|------|
| 逼真人脸，固定姿态 | StyleGAN3（最锐利、最小） |
| 动漫 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像转换 | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| 快速 1 步文生图 | 扩散的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练中的感知损失 | 图像裁剪上的小型 GAN 判别器 |
| 任何多模态、开放性任务 | 不要——使用扩散或 flow matching |

GAN 锐利但窄。一旦你的领域开放——照片、任意文本提示、视频——切换到扩散。对抗技巧作为组件（感知损失、蒸馏）继续存在，而不是作为独立生成器。

## 发布

保存为 `outputs/skill-gan-debugger.md`。Skill 接收一个失败的 GAN 运行（损失曲线、样本网格、数据集大小）并输出：可能原因的排名列表、每种的一行修复方案，和重新运行协议。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 重新运行。G 的损失以多快速度崩溃为常数？
2. **中等。** 将 Goodfellow BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练更稳定吗？比较墙上时钟收敛时间。
3. **困难。** 将 1-D 示例扩展到 2-D 数据（圆环上 8 个高斯的混合）。跟踪生成器在 1k、5k、10k 步时捕获了多少个 8 个模式。实现 minibatch 判别并重新测量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 生成器 | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器 | "D" | 分类器 `D: x → [0, 1]`，真 vs 假。 |
| Minimax | "博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失 | "修复方法" | 对 G 使用 `-log D(G(z))` 而不是 `log(1 - D(G(z)))`。 |
| 模式崩溃 | "G 记住了一个东西" | 尽管数据多样化，生成器只产生少量不同输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替换 BCE；更平滑的梯度。 |
| 谱归一化 | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个有效的" | 映射网络 + AdaIN；在人脸上最好，2026 年仍然如此。 |

## 生产笔记：单步推理是 GAN 持久优势

GAN 在开放领域生成上不再赢得样本质量，但在推理成本上仍然胜出。在生产推理文献词汇中，GAN 有：

- **没有 prefill，没有 decode 阶段。** 一次 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **没有 KV-cache 压力。** 唯一的状态是权重。Batch size 由激活内存决定，而不是 cache。
- **简单的连续批处理。** 由于每个请求花费相同的固定 FLOPs，在服务器目标占用率下的静态 batch 通常是最优的。不需要飞行中调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文生图的主导技术：它将 20-50 步扩散 pipeline 压缩为 1-4 个 GAN 风格的前向传播，同时保持扩散基础的分布。对抗损失作为训练旋钮继续存在，用于将慢速生成器变成快速生成器。

## 进一步阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。