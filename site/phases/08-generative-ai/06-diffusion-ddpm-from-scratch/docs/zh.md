# 扩散模型 — 从零实现 DDPM

> Ho、Jain、Abbeel（2020 年）给了这个领域一个无法放弃的配方。在一千个小步骤中用噪声摧毁数据。训练一个神经网络预测噪声。在推理时逆转这个过程。今天，每一个主流图像、视频、3D 和音乐模型都在这个循环上运行，可能叠加了 flow matching 或一致性技巧。

**类型：** Build
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播），Phase 8 · 02（VAE）
**时间：** 约 75 分钟

## 问题

你想要 `p_data(x)` 的采样器。GAN 玩 minimax 博弈，往往发散。VAE 来自高斯解码器产生模糊样本。你真正想要的是：（a）单一稳定损失（无鞍点，无 minimax），（b）`log p(x)` 的下界（所以你有似然），（c）样本达到 SOTA 质量。

Sohl-Dickstein 等人（2015 年）有一个理论答案：定义一个逐步添加高斯噪声的马尔可夫链 `q(x_t | x_{t-1})`，训练一个逆向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020 年）表明损失可以简化为一行——预测噪声——并整理了数学。2020 年这是一个好奇。2021 年产生了 SOTA 样本。2022 年成为 Stable Diffusion。2026 年它是底层基座。

## 核心概念

![DDPM：前向噪声，逆向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步骤中添加高斯噪声。闭式解——这是数学可处理的原因——是累积步骤也是高斯的：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，用于 `β_t` 的调度。在 T=1000 步上从 1e-4 到 0.02 线性选取 `β_t`，`x_T` 近似为 `N(0, I)`。

**逆向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 来预测添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 要么是 `sqrt(β_t)` 要么是学习到的方差。表达式看起来很乱，但它只是代数——给定后验 `q(x_{t-1} | x_t, x_0)` 并用噪声预测估计替换 `x_0`，求解 `x_{t-1}`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，选择一个随机 `t`，采样 `ε ~ N(0, I)`，通过闭式一次计算嘈杂的 `x_t`，并对噪声进行回归。一个损失，无 minimax，无 KL，无重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 迭代逆向步骤到 `t = 1`。完成。

## 为什么有效

三个直觉：

1. **去噪容易；生成难。** 在 `t=T` 时，数据是纯噪声——网络必须解决一个简单问题。在 `t=0` 时，网络只需要清理几个像素。在中间的 `t`，问题很难，但网络从每个噪声水平通过相同的权重获得许多梯度。
2. **分数匹配的伪装。** Vincent（2011 年）证明预测噪声等同于估计 `∇_x log q(x_t | x_0)`，即*分数*。逆向 SDE 使用这个分数来沿着密度梯度向上走——走向高概率区域的引导随机游走。
3. **ELBO 简化为简单 MSE。** 完整的变分下界在每个时间步有一个 KL 项。有了 DDPM 的参数化，这些 KL 项简化为具有特定系数的噪声预测 MSE；Ho 丢弃了这些系数（称之为"简单"损失），质量反而*提升*。

## 构建

`code/main.py` 实现了一个 1-D DDPM。数据是一个双峰混合。"网络"是一个微型 MLP，接收 `(x_t, t)` 并输出预测噪声。训练是一行损失。采样迭代逆向链。

### 第 1 步：前向调度（闭式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 第 2 步：一次采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 第 3 步：一次训练步骤

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 第 4 步：逆向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于 1-D 问题，40 个时间步和 24 单元 MLP，它在大约 200 个 epoch 中学习双峰混合。

## 时间条件化

网络需要知道它正在去噪哪个时间步。两个标准选项：

- **正弦嵌入。** 类似于 Transformer 位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过一个 MLP 传递，广播到网络中。
- **FiLM / 组归一化条件化。** 将嵌入投影为每个块的每通道缩放/偏置（FiLM）。

我们的玩具代码使用正弦 → concat。生产 U-Net 使用 FiLM。

## 陷阱

- **调度非常重要。** 线性 `β` 是 DDPM 默认值，但余弦调度（Nichol & Dhariwal，2021 年）在相同计算量下给出更好的 FID。如果质量停滞，切换调度。
- **时间步嵌入很脆弱。** 传递原始 `t` 作为浮点数对玩具 1-D 有用但对图像失败；始终使用适当的嵌入。
- **V-prediction vs ε-prediction。** 对于窄范围（非常小或非常大的 t），`ε` 信噪比差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 使用它。
- **无分类器引导。** 在推理时，计算条件和非条件 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。在 Lesson 08 中介绍。
- **1000 步太多了。** 生产使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见 Lesson 12。

## 使用

| 角色 | 2026 年典型技术栈 |
|------|-------------------|
| 图像像素空间扩散（小、玩具） | DDPM + U-Net |
| 图像潜在扩散 | VAE 编码器 + U-Net 或 DiT（Lesson 07） |
| 视频潜在扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜在扩散 | Encodec + 扩散 transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用生成 backbone。Flow matching（Lesson 13）是 2024-2026 的竞争者，在相同质量下通常在推理速度上胜出。

## 发布

保存为 `outputs/skill-diffusion-trainer.md`。Skill 接收数据集 + 计算预算并输出：调度（线性 / 余弦 / sigmoid）、预测目标（ε / v / x）、步数、引导尺度、采样器家族和评估协议。

## 练习

1. **简单。** 将 T 从 40 改为 10 在 `code/main.py` 中。样本质量（输出直方图可视化）如何下降？在什么 T 时双峰结构崩溃？
2. **中等。** 从 ε-prediction 切换到 v-prediction。重新推导逆向步骤。比较最终样本质量。
3. **困难。** 添加无分类器引导。条件化在类别标签 `c ∈ {0, 1}` 上，在训练期间 10% 的时间丢弃它，在采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量在 `w = 0, 1, 3, 7` 时的条件模式命中率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 前向过程 | "添加噪声" | 固定马尔可夫链 `q(x_t | x_{t-1})`，摧毁数据。 |
| 逆向过程 | "去噪" | 学习到的链 `p_θ(x_{t-1} | x_t)`，重建数据。 |
| β 调度 | "噪声阶梯" | 每步方差；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的闭式。 |
| 简单损失 | "MSE on noise" | `||ε - ε_θ(x_t, t)||²`；所有变分推导都简化为这个。 |
| ε-prediction | "预测噪声" | 输出是添加的噪声；标准 DDPM。 |
| V-prediction | "预测速度" | 输出是 `α·ε - σ·x`；跨 t 条件化更好。 |
| DDPM | "论文" | Ho 等人 2020；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同训练目标。 |
| 无分类器引导 | "CFG" | 混合条件和非条件噪声预测以放大条件化。 |

## 生产笔记：扩散推理是一个步数问题

DDPM 论文运行 T=1000 逆向步。生产中没有人部署那个。每一个真实推理栈选择三种策略之一——每种都清晰地映射到"延迟来自哪里"的生产推理框架：

1. **更快的采样器，相同模型。** DDIM（20-50 步）、DPM-Solver++（10-20 步）、UniPC（8-16 步）。替换逆向循环的插入式替代品；训练好的 `ε_θ` 权重保持不变。延迟减少 20-50 倍。
2. **蒸馏。** 训练学生以更少步数匹配老师：渐进式蒸馏（2 → 1）、一致性模型（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。再减少 5-10 倍延迟，需要重新训练。
3. **缓存和编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。每步延迟减少约 2 倍。可与（1）和（2）叠加。

对于生产扩散服务器，预算对话与生产文献对 LLM 的描述相同：延迟 = `步数 × 每步成本 + VAE解码`，吞吐量 = `batch_size × (步数 × 每步成本)^-1`。TTFT 很小（一步）；从用户角度看，图像生成是"一次完成"的，所以 TPOT 等价项是完整响应时间。

## 进一步阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散论文，先于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，少步数。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度，学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一记号，最干净的配方。