# Flow Matching 与 Rectified Flow

> 扩散模型需要 20-50 个采样步，因为它们从噪声到数据走一条曲线路径。Flow matching（Lipman 等人，2023 年）和 rectified flow（Liu 等人，2022 年）训练直线路径。更直的路径意味着更少的步意味着更快的推理。Stable Diffusion 3、Flux.1 和 AudioCraft 2 在 2024 年都切换到 flow matching。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 06（DDPM），Phase 1 · Calculus
**时间：** 约 45 分钟

## 问题

DDPM 的逆向过程是从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其压缩为 20-50 步确定性步。你想要更少的步——理想情况下一步。阻塞是求解逆向过程的 ODE 是刚性的；路径是弯曲的。

如果你能训练模型使从噪声到数据的路径是一条*直线*，从 `t=1` 到 `t=0` 的单步 Euler 步就可以工作。Flow matching 直接构建这个：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线路径，训练向量场 `v_θ(x, t)` 匹配其时间导数，在推理时积分。

Rectified flow（Liu 2022）更进一步：通过一种 reflow 程序迭代拉直路径，产生越来越接近线性的 ODE。经过两次 reflow 迭代后，2 步采样器匹配 50 步 DDPM 质量。

## 核心概念

![Flow matching：噪声和数据之间的直线路径插值](../assets/flow-matching.svg)

### 直线路径流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data` 和 `x_1 ~ N(0, I)`。沿这条直线路径的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义一个神经向量场 `v_θ(x_t, t)` 并训练它匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这是**条件流匹配**损失（Lipman 2023）。训练是无模拟的：你永远不需要展开 ODE。只需采样 `(x_0, x_1, t)` 并回归。

### 采样

在推理时，将学习到的向量场*向后*积分：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，Euler 步进到 `t=0`。

### Rectified flow（Liu 2022）

直线路径流有效，但学到的路径*实际上不是直的*——因为许多 `x_0` 可以映射到同一个 `x_1`。Rectified flow 的 reflow 步骤：

1. 用随机配对训练流模型 v_1。
2. 通过从 `x_1` 积分到其着陆 `x_0` 来采样 N 对 `(x_1, x_0)`。
3. 在这些配对示例上训练 v_2。因为配对现在是"ODE 匹配的"，它们之间的直线路径插值实际上是更平坦的。
4. 重复。

实际上，2 次 reflow 迭代让你接近线性，实现 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从 flow matching 模型蒸馏的。

### 为什么这在 2024 年图像上胜出

三个原因：

1. **无模拟训练** — 训练期间没有 ODE 展开，实现简单。
2. **更好的损失几何** — 直线路径有一致的信噪比，而 DDPM ε 损失在调度边缘有糟糕的 SNR。
3. **更快的推理** — SDXL-Turb o 质量下的 4-8 步；一致性蒸馏 1 步。

## Flow matching vs DDPM — 精确联系

Flow matching 与高斯条件路径是扩散*与特定噪声调度*。选择 `x_t = α(t) x_0 + σ(t) x_1` 调度，flow matching 通过 `v = α'·x_0 - σ'·x_1` 恢复 Stratonovich 重构的扩散。两者对于高斯路径是代数等价的。

Flow matching 添加的是：目标的*清晰度*（一个简单的速度）、更干净的损失，以及使用非高斯插值器的许可。

## 构建

`code/main.py` 在双模高斯混合上实现 1-D flow matching。向量场 `v_θ(x, t)` 是一个用直线路径目标训练的微型 MLP。在推理时，积分 1、2、4 和 20 个 Euler 步并比较样本质量。

### 第 1 步：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # 反向传播 + 更新
```

### 第 2 步：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 第 3 步：比较步数

期望 4 步采样器已经匹配 20 步质量——这对延迟来说意义重大。

## 陷阱

- **时间参数化。** Flow matching 使用 `t ∈ [0, 1]`，`t=0` 在数据，`t=1` 在噪声。DDPM 使用 `t ∈ [0, T]`，`t=0` 在数据，`t=T` 在噪声。相同方向，不同尺度。论文经常搞错这个。
- **调度选择。** Rectified flow 的直线是"那个"flow matching 调度，但你可以使用余弦或 logit-normal t 采样（SD3 这样做）以获得更好的尺度覆盖。
- **Reflow 成本。** 生成用于 reflow 的配对数据集是每次采样一次完整推理传递。只有在你真的需要 1-2 步推理时才做 reflow。
- **无分类器引导仍然适用。** 只需在线性组合中用 v 替换 ε：`v_cfg = (1+w) v_cond - w v_uncond`。

## 使用

| 用例 | 2026 年技术栈 |
|------|-------------|
| 文生图，最佳质量 | Flow matching：SD3、Flux.1-dev |
| 文生图，1-4 步 | 蒸馏 flow matching：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 从 flow matching 基础的一致性蒸馏（LCM、PCM） |
| 音频生成 | Flow matching：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | Flow matching 与扩散混合（Sora、Veo、Stable Video） |
| 科学 / 物理（粒子轨迹、分子） | Flow matching + 等变向量场 |

每当一篇论文在 2025-2026 年说"比扩散更快"，它几乎总是 flow matching + 蒸馏。

## 发布

保存为 `outputs/skill-fm-tuner.md`。Skill 接收扩散风格模型规范并转换为 flow matching 训练配置：调度选择、时间采样分布（均匀 / logit-normal）、优化器、reflow 计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py` 并比较 1 步 vs 20 步 MSE vs 真实数据分布。
2. **中等。** 从均匀 `t` 采样切换到 logit-normal（集中在 mid-t）。模型质量有改善吗？
3. **困难。** 实现一次 reflow 迭代：通过积分第一个模型生成配对 `(x_0, x_1)`，在配对上训练第二个模型，并比较 1 步样本质量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Flow matching | "直线路径扩散" | 训练 `v_θ(x, t)` 匹配 `x_1 - x_0` 沿插值器。 |
| Rectified flow | "Reflow" | 迭代拉直学习流的程序。 |
| 向量场 | "v_θ" | 模型输出——移动 `x_t` 的方向。 |
| 直线路径插值器 | "路径" | `x_t = (1-t)·x_0 + t·x_1`；目标导数 trivial。 |
| Euler 采样器 | "一阶 ODE 求解器" | 最简单的积分器；当路径直时效果很好。 |
| Logit-normal t | "SD3 采样" | 将 `t` 采样集中在中间值附近，梯度最强的地方。 |
| 一致性蒸馏 | "1 步采样器" | 训练学生将任何 `x_t` 直接映射到 `x_0`。 |
| 速度上的 CFG | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；相同的技巧，新的变量。 |

## 生产笔记：Flux.1-schnell 是 flow matching 最快的样子

Flow matching 的生产胜利是 Flux.1-schnell——蒸馏到 1-4 推理步同时保持 Flux-dev 级质量的 flow-matched DiT。Niels 的"在 8GB 机器上运行 Flux" notebook 是参考部署配方：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 的 4 步 vs dev 的 50 步），VAE 解码。成本核算：

| 变体 | 步数 | L4 上 1024² 延迟 | 总 FLOPs（相对） |
|------|------|-----------------|-----------------|
| Flux.1-dev（原始） | 50 | 约 15 秒 | 1.0× |
| Flux.1-schnell | 4 | 约 1.2 秒 | 0.08×（12 倍快） |
| SDXL-base | 30 | 约 4 秒 | 0.25× |
| SDXL-Lightning 2 步 | 2 | 约 0.3 秒 | 0.03× |

生产规则：**flow matching 基础 + 蒸馏 = 2026 年快速文生图的默认。** 每个主要供应商都发布这个组合：SD3-Turbo（SD3 + flow + 蒸馏）、Flux-schnell（Flux-dev + rectified flow 拉直）、CogView-4-Flash。纯扩散基础仅用于遗留 checkpoint。

## 进一步阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模 rectified flow。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 涵盖 FM + 扩散的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散 / flow 的一步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的 flow matching。