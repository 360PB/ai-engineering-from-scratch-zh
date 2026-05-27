# 自编码器与变分自编码器（VAE）

> 普通自编码器压缩后重建。它会记忆。它不能生成。只需一个技巧——强制码空间呈高斯分布——你就有了一个采样器。这个技巧，即 `z = μ + σ·ε` 的重参数化，就是 2026 年你使用的每一个潜在扩散和 flow matching 图像模型在输入端都有一个 VAE 的原因。

**类型：** Build
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播），Phase 3 · 07（CNN），Phase 8 · 01（分类体系）
**时间：** 约 75 分钟

## 问题

将一个 784 像素的 MNIST 数字压缩为 16 个数的编码，然后重建。普通自编码器在重建 MSE 上表现优异，但码空间是一团乱码。在码空间中随机取一点解码，得到的是噪声。它没有采样器。它是一个伪装成生成模型的压缩模型。

你真正想要的是：（a）码空间是一个干净、光滑的分布，可以从中采样——比如各向同性高斯 `N(0, I)`；（b）解码任何样本都能产生一个可信的数字；（c）编码器和解码器仍然压缩效果好。三个目标，一个架构，一个损失函数。

Kingma 的 2013 VAE 通过训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚将该分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 采样 `z` 来解决这个问题。在推理时，丢掉编码器，从 `z ~ N(0, I)` 采样，解码。KL 惩罚是迫使码空间结构化的原因。

在 2026 年，VAE 很少单独部署——在原始图像质量上已被扩散超越——但它是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学习 VAE，你就学习了 2026 年你使用的每个图像 pipeline 看不见的第一层。

## 核心概念

![自编码器 vs VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样不可微分。将采样重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上一个非参数噪声——梯度流经 `μ` 和 `σ`。

**损失。** 证据下界（ELBO），两项：

```
loss = 重建 + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||² + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建推动 `x̂` 靠近 `x`。KL 推动 `q(z|x)` 靠近先验。它们相互权衡。小的 β（<1）= 更清晰的样本，码空间不太高斯。大的 β（>1）= 更干净的码空间，更模糊的样本。β-VAE（Higgins 2017）让这个旋钮出名，并开启了 disentanglement 研究。

**采样。** 推理时：从 `z ~ N(0, I)` 采样，前向传播通过解码器。一次前向传播——不像扩散那样迭代采样。

## 构建

`code/main.py` 实现了一个没有 numpy 或 torch 的微型 VAE。输入是从 8 维双分量高斯混合中采样的 8 维合成数据。编码器和解码器都是单隐藏层 MLP。我们实现了 tanh 激活、前向传播、损失和手写反向传播。不是生产级别——是教学演示。

### 第 1 步：编码器前向

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

用 `log σ²` 而不是 `σ`，这样网络输出无约束（用 softplus 处理 σ 是陷阱——当 σ ≈ 0 时梯度会死掉）。

### 第 2 步：重参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第 3 步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个分布都是高斯分布，KL 有精确的闭式解。不要用数值积分。人们在 2026 年仍然在发布带蒙特卡洛 KL 估计的代码——慢 3 倍且没有任何理由。

### 第 4 步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 陷阱

- **后验崩溃。** KL 项过于激进地驱动 `q(z|x) → N(0, I)`，导致 `z` 不携带关于 `x` 的任何信息。修复方法：β 退火（从 β=0 开始，升至 1）、free bits，或跳过不活跃维度的 KL。
- **样本模糊。** 高斯解码器似然意味着 MSE 重建，在 L2（均值）上是贝叶斯最优——一组可信数字的均值是一张模糊的数字。修复方法：离散解码器（VQ-VAE、NVAE），或仅将 VAE 用作编码器，在潜在变量上堆叠扩散（这就是 Stable Diffusion 做的）。
- **β 太大，太早。** 见后验崩溃。从 β≈0.01 开始。
- **潜在维度过小。** 16-D 对 MNIST 够用，256-D 对 ImageNet 256²，2048-D 对 ImageNet 1024²。Stable Diffusion 的 VAE 将 512×512×3 压缩到 64×64×4（空间面积下采样 32 倍，通道 32 倍）。

## 使用

2026 年 VAE 技术栈：

| 场景 | 选择 |
|------|------|
| 用于扩散的图像潜在编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频潜在编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜在变量 | Sora 的时空 patches、Latte VAE、WAN VAE |
| Disentangled 表示学习 | β-VAE、FactorVAE、TCVAE |
| 用于 transformer 建模的离散潜在变量 | VQ-VAE、RVQ（ResidualVQ） |
| 用于生成的连续潜在变量 | 普通 VAE，然后在该潜在空间中条件化 flow/diffusion 模型 |

潜在扩散模型是编码器和解码器之间生活着一个扩散模型的 VAE。VAE 做粗压缩，扩散模型做重活。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）也是同样的模式。

## 发布

保存为 `outputs/skill-vae-trainer.md`。

Skill 接收：数据集画像 + 目标潜在维度 + 下游用途（重建、采样或潜在扩散输入），并输出：架构选择（普通 / β / VQ / RVQ）、β 调度、潜在维数、解码器似然（高斯 vs 分类），以及评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 在 `code/main.py` 中将 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL。你的合成数据上哪个 β 是 Pareto 最优？
2. **中等。** 用伯努利似然（交叉熵损失）替换高斯解码器似然。在相同合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为迷你 VQ-VAE：用 K=32 个条目codebook 的最近邻查找替换连续 `z`。比较重建 MSE，并报告使用了多少个 codebook 条目（codebook 崩溃是真实存在的）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 自编码器 | 编码器-解码器网络 | `x → z → x̂`，学习 MSE。无生成能力。 |
| VAE | 带采样器的 AE | 编码器输出一个分布，KL 惩罚塑造码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ 重建 - KL[q(z|x) || p(z)]`；当 `q = p(z|x)` 时 tight。 |
| 重参数化 | `z = μ + σ·ε` | 将随机节点改写为确定性 + 纯噪声。使通过采样的反向传播成为可能。 |
| Prior | `p(z)` | 潜在变量的目标分布，通常是 `N(0, I)`。 |
| 后验崩溃 | "KL 项赢了" | 编码器忽略 `x`，输出先验；解码器必须幻觉。 |
| β-VAE | 可调 KL 权重 | `loss = 重建 + β·KL`。β 越高越 disentangled 但越模糊。 |
| VQ-VAE | 离散潜在变量 | 用最近的 codebook 向量替换连续 `z`；使 transformer 建模成为可能。 |

## 生产笔记：VAE 是扩散服务器中最热的路径

在 Stable Diffusion / Flux / SD3 pipeline 中，VAE 每次请求被调用两次——一次编码（如果是 img2img / 修复），一次解码。在 1024² 时，解码器通常是整个 pipeline 中最大的激活内存峰值，因为它将 `128×128×16` 潜在变量上采样回 `1024×1024×3`。两个实际后果：

- **分片或平铺解码。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。平铺用小的接缝伪影换取 `O(tile²)` 内存而不是 `O(H·W)`。在消费级 GPU 上 1024²+ 必备。
- **bf16 解码器，fp32 数值用于最终 resize。** SD 1.x VAE 以 fp32 发布，在 1024² 转换为 fp16 时*悄悄产生 NaN*。SDXL 发布了 `madebyollin/sdxl-vae-fp16-fix`——始终优先使用 fp16-fix 变体或使用 bf16。

## 进一步阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — SOTA 图像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准。