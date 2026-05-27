# 潜在扩散与 Stable Diffusion

> 在 512×512 图像上进行像素空间扩散是一种计算罪行。Rombach 等人（2022 年）注意到，你不需要所有 786k 维来生成图像——你只需要足够捕获语义结构，另一个解码器处理其余。在 VAE 的潜在空间中运行扩散。就是这一个想法造就了 Stable Diffusion。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 02（VAE），Phase 8 · 06（DDPM），Phase 7 · 09（ViT）
**时间：** 约 75 分钟

## 问题

在 512² 进行像素空间扩散意味着 U-Net 在形状为 `[B, 3, 512, 512]` 的张量上运行。对于 500M 参数 U-Net，每步采样约 100 GFLOPS。50 步是每张图像 5 TFLOPS。在十亿张图像上训练，计算费用是荒谬的。

这些 FLOPs 大部分用于通过网络推送感知上不重要的细节——高频纹理，有损 VAE 可以压缩掉。Rombach 的想法：训练一个 VAE 一次（*第一阶段*），冻结它，完全在 4 通道 64×64 潜在空间（*第二阶段*）中运行扩散。相同的 U-Net。1/16 的像素。约 64 倍更少的 FLOPs，质量相当。

这就是 Stable Diffusion 配方。SD 1.x / 2.x 在 `64×64×4` 潜在变量上使用 860M U-Net，SDXL 在 `128×128×4` 上使用 2.6B U-Net，SD3 用带有 flow matching 的 Diffusion Transformer（DiT）替换了 U-Net。Flux.1-dev（Black Forest Labs，2024 年）发布了一个 12B 参数 DiT-MMDiT。全部运行在相同的双阶段基座上。

## 核心概念

![潜在扩散：VAE 压缩 + 潜在空间中的扩散](../assets/latent-diffusion.svg)

**两个阶段，单独训练。**

1. **阶段 1 — VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴下采样 8 倍 + 调整通道，使总潜在大小约为像素计数的 1/16。损失 = 重建（L1 + LPIPS 感知）+ KL（小权重，这样 `z` 不会被强制太接近高斯，因为我们不需要从 `z` 精确采样）。通常用对抗损失训练，使解码图像锐利。

2. **阶段 2 — 在 `z` 上扩散。** 将 `z = E(x_real)` 作为数据。训练一个 U-Net（或 DiT）来去噪 `z_t`。在推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件化。** 两个额外组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。一个交叉注意力注入：每个 U-Net 块接收 `[Q = 图像特征，K = V = 文本 token]`，并将它们混合。token 是文本影响图像的唯一途径。

**损失函数与 Lesson 06 相同。** 相同的 DDPM / flow matching MSE 在噪声上。你只是换了数据域。

## 架构变体

| 模型 | 年份 | Backbone | 潜在形状 | 文本编码器 | 参数量 |
|------|------|----------|----------|-----------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 token） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏 | 128×128×4 | same | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT 蒸馏 | 128×128×16 | T5-XXL + CLIP-L | 12B，1-4 步 |

趋势：用 DiT 替换 U-Net（在潜在 patches 上运行的 transformer），扩展文本编码器（T5 比 CLIP 更好地遵循提示），增加潜在通道（4 → 16 给出更多细节空间）。

## 构建

`code/main.py` 堆叠了一个玩具 1-D"VAE"（恒等编码器 + 解码器，用于演示；真正的 VAE 将是卷积网络）在 Lesson 06 的 DDPM 之上，并添加了带无分类器引导的类别条件化。它表明相同的扩散损失在原始 1-D 值或编码值上运行都有效——这就是关键洞察。

### 第 1 步：编码器 / 解码器

```python
def encode(x):    return x * 0.5          # 玩具"压缩"到更小尺度
def decode(z):    return z * 2.0
```

真正的 VAE 有训练权重。对于教学来说，这个线性映射足以表明扩散在 `z` 上操作而不关心原始数据空间。

### 第 2 步：在 `z` 空间中扩散

与 Lesson 06 相同的 DDPM。网络看到的数据是 `z = E(x)`。采样 `z_0` 后，用 `D(z_0)` 解码。

### 第 3 步：无分类器引导

训练期间，10% 的时间丢弃类别标签（替换为空 token）。在推理时，计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完整多样性），`w = 3` = 默认，`w = 7+` = 饱和 / 过度锐化。

### 第 4 步：文本条件化（概念，非代码）

用冻结文本编码器输出替换类别标签。将文本嵌入通过交叉注意力馈入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这就是类别条件扩散模型与 Stable Diffusion 之间的唯一实质性区别。

## 陷阱

- **VAE 规模不匹配。** SD 1.x VAE 有一个在编码后应用的缩放常数（`scaling_factor ≈ 0.18215`）。忘记这个会使 U-Net 在方差完全错误的潜在变量上训练。每个 checkpoint 都附带它。
- **文本编码器悄悄错误。** SD3 需要 T5-XXL 且 >=128 token，回退到仅 CLIP 会丢失很多信息。始终检查 `use_t5=True`，否则提示保真度会崩溃。
- **混合潜在空间。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL 潜在变量上训练的 LoRA 在 SD3 上不工作。Hugging Face diffusers 0.30+ 拒绝加载不匹配的 checkpoint。
- **CFG 太高。** `w > 10` 产生饱和、油腻的图像，以多样性为代价过度拟合提示。最优点是 `w = 3-7`。
- **负提示泄露。** 空负提示成为空 token；填充的负提示成为 `ε_uncond`。这些不一样；一些 pipeline 悄悄默认为空。

## 使用

2026 年生产技术栈：

| 目标 | 推荐的 backbone |
|------|----------------|
| 窄领域，配对数据，从头训练模型 | SDXL 微调（LoRA / 全量）——最快上线 |
| 开放领域文生图，开源权重 | Flux.1-dev（12B，Apache / 非商业）或 SD3.5-Large |
| 最快推理，开源权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示遵循，托管 | GPT-Image / DALL-E 3（仍然）、Midjourney v7、Imagen 4 |
| 编辑 workflow | Flux.1-Kontext（2024 年 12 月）——原生接受图像 + 文本 |
| 研究，基准 | SD 1.5——古老但研究充分 |

## 发布

保存 `outputs/skill-sd-prompter.md`。Skill 接收文本提示 + 目标风格并输出：模型 + checkpoint、CFG 尺度、采样器、负提示、分辨率、可选 ControlNet/IP-Adapter 组合，以及每步 QA 检查清单。

## 练习

1. **简单。** 用引导 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每类的平均样本。在什么 `w` 下类均值偏离真实数据均值？
2. **中等。** 将玩具线性编码器替换为带重建损失的 tanh-MLP 编码器/解码器对。在新的潜在变量上重新训练扩散。样本质量有变化吗？
3. **困难。** 使用 diffusers 设置真正的 Stable Diffusion 推理：加载 `sdxl-base`，运行 30 步 Euler 和 CFG=7，计时。现在切换到 `sdxl-turbo` 用 4 步和 CFG=0。相同主题，不同质量——描述变化的内容和原因。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 第一阶段 | "VAE" | 训练的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段 | "U-Net" | 潜在空间上的扩散模型。 |
| CFG | "引导尺度" | `(1+w)·ε_cond - w·ε_uncond`；调整条件化强度。 |
| 空 token | "空提示嵌入" | 用于 `ε_uncond` 的非条件嵌入。 |
| 交叉注意力 | "文本如何进入" | 每个 U-Net 块作为 K 和 V 关注文本 token。 |
| DiT | "扩散 Transformer" | 用潜在 patches 上的 transformer 替换 U-Net；更好的可扩展性。 |
| MMDiT | "多模态 DiT" | SD3 的架构：文本和图像流与联合注意力。 |
| VAE 缩放因子 | "魔法数字" | 将潜在变量除以约 5.4，使扩散在单位方差空间操作。 |

## 生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是"我有消费级 GPU，能上线吗？"的规范配方。技巧与生产推理文献中列出的三个旋钮配方应用于扩散 DiT 相同：

1. **交错加载。** Flux 有三个从不同时需要共存于 VRAM 中的网络：T5-XXL 文本编码器（fp32 约 10 GB）、CLIP-L（小）、12B MMDiT 和 VAE。首先编码提示，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 一次只能容纳一个阶段。
2. **通过 bitsandbytes 的 4-bit 量化。** 在 T5 编码器和 DiT 上使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存减少 8 倍，质量下降在 Aritra 的基准测试中对于文生图来说可以忽略（链接在 notebook 中）。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 在每个前向传播推进时自动在 CPU 和 GPU 之间交换模块。增加 10-20% 延迟，但使 pipeline 能够运行。

内存核算：`10 GB T5 / 8 = 1.25 GB` 量化，`12 B 参数 × 0.5 字节 = ~6 GB` 量化 DiT，加上激活。用 stas00 的术语，这是 TP=1 推理的极端端——无模型并行，最大量化。对于生产，你会在 H100 上运行 TP=2 或 TP=4；对于单个开发笔记本电脑，这就是配方。

## 进一步阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 系列。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 每个上述 checkpoint 的参考实现。