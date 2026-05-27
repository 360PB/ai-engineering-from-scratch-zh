# 修复、扩展绘制与图像编辑

> 文生图创造新事物。修复修复旧事物。在生产中，70% 的计费图像工作是编辑——交换背景、移除标志、扩展画布、再生成一只手。修复是扩散获得回报的地方。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 07（潜在扩散），Phase 8 · 08（ControlNet 与 LoRA）
**时间：** 约 75 分钟

## 问题

客户发送了一张完美的产品照片，背景中有一个分散注意力的标志。你想擦除标志并让其他所有像素完全相同。你不能从零开始运行文生图——结果会有不同的颜色、不同的光照、不同的产品角度。你只想重新生成*被遮挡的区域*，并且你想让重新生成的内容尊重周围上下文。

这就是修复。变体：

- **修复（Inpainting）。** 在遮罩内重新生成，保留外部像素。
- **扩展绘制（Outpainting）。** 在遮罩外重新生成（或画布之外），保留内部。
- **图像编辑。** 重新生成整张图像，但保持与原始图像的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年每个扩散 pipeline 都发布了修复模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于相同的原理工作。

## 核心概念

![修复：遮罩感知的去噪与上下文保留的重新注入](../assets/inpainting.svg)

### 朴素方法（以及为什么它是错的）

使用遮罩运行标准文生图。在每个采样步，将噪声潜在变量的未遮罩区域替换为前向扩散的干净图像。它能工作……但效果很差。边界伪影渗透，因为模型没有关于遮罩内有什么的信息。

### 正确的修复模型

训练一个修改的 U-Net，它接受 9 个输入通道而不是 4：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外通道是 VAE 编码的源图像副本加上单通道遮罩。在训练时，你随机遮罩图像区域，并训练模型仅在遮罩区域内去噪，而未遮罩区域作为干净条件信号给出。在推理时，模型可以"看到"遮罩区域周围的内容，并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这个 9 通道（或类似）输入。Diffusers `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等人，2022 年）—— 免费编辑

将噪声添加到源图像直到某个中间 `t`，然后用新提示从 `t` 到 0 运行逆向链。无需重新训练。起始 `t` 的选择权衡保真度与创意自由度：

- `t/T = 0.3` → 几乎与源相同，小的风格改变
- `t/T = 0.6` → 中等编辑，保留粗略结构
- `t/T = 0.9` → 从接近噪声生成，最小源保留

### InstructPix2Pix（Brooks 等人，2023 年）

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。在推理时，同时以输入图像和文本指令为条件（"让它日落"、"添加一条龙"）。两个 CFG 尺度：图像尺度和文本尺度。

### RePaint（Lugmayr 等人，2022 年）

保持一个标准的无条件扩散模型。在每个逆向步，重新采样——偶尔跳回更嘈杂的状态并重新生成。避免边界伪影。当你没有训练好的修复模型时使用。

## 构建

`code/main.py` 在 5 维数据上实现了一个玩具 1-D 修复方案。我们在一个 5 维混合数据上训练 DDPM，其中每个样本是来自两个簇之一的 5 个浮点数。在推理时，我们"遮罩"5 维中的 2 维，在每一步注入未遮罩 3 维的噪声前向版本，并仅重新生成遮罩维度。

### 第 1 步：5 维 DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 第 2 步：在所有 5 维上训练去噪器

标准 DDPM。网络为 5 维噪声输入输出 5 维噪声预测。

### 第 3 步：在推理时，遮罩感知的逆向

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 用新鲜噪声版本的干净源替换未遮罩维
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...然后在 x_t 上运行正常的逆向步
```

这是朴素方法，在玩具 1-D 数据上有效。真正的图像修复使用 9 通道输入，因为纹理一致性更重要。

### 第 4 步：扩展绘制

扩展绘制是反转遮罩的修复：遮罩新（以前不存在）画布，用原始填充其余。相同的训练目标。

## 陷阱

- **接缝。** 朴素方法留下可见边界，因为梯度信息不流过遮罩。修复：将遮罩扩大 8-16 像素，或使用正确的修复模型。
- **遮罩泄露。** 如果条件图像的未遮罩区域质量低或有噪声，它会污染遮罩内的生成。先去噪或轻微模糊。
- **CFG 与遮罩大小交互。** 小遮罩上的高 CFG = 饱和补丁。减少小编辑的 CFG。
- **SDEdit 保真度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能失去主体身份。扫描并检查点。
- **提示不匹配。** 提示应该描述*整张图像*，而不仅仅是新内容。"一只猫坐在椅子上"而不是"一只猫"。

## 使用

| 任务 | Pipeline |
|------|----------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示 |
| 替换天空 | SD-Inpaint + "日落时蓝色天空" |
| 扩展画布 | SDXL outpaint 模式（8px 羽化）或 Flux-Fill 带 outpaint 遮罩 |
| 重新生成手 / 脸 | SD-Inpaint + 重新描述主体的提示 + ControlNet-Openpose |
| 改变一个区域的风格 | 在遮罩区域上以 `t/T=0.5` 进行 SDEdit |
| "让它日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 遮罩 → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image（托管）用于最难的情况 |

SAM（Meta 的 Segment Anything，2023 年）+ 扩散修复是 2026 年背景去除 pipeline。SAM 2（2024 年）在视频上工作。

## 发布

保存为 `outputs/skill-editing-pipeline.md`。Skill 接收原始图像 + 编辑描述 + 可选遮罩（SAM 提示）并输出：遮罩生成方法、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或修复模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将遮罩维度的比例从 0.2 变化到 0.8。在什么比例下修复质量（遮罩维度的残差）等于无条件生成？
2. **中等。** 实现 RePaint：每 10 个逆向步，跳回 5 步（添加噪声）并重新去噪。测量它是否减少了遮罩边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 在 20 个人脸再生成任务上。分别评分姿态遵循度和身份保留度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 修复（Inpainting） | "填补空洞" | 在遮罩内重新生成；保留外部像素。 |
| 扩展绘制（Outpainting） | "扩展画布" | 在画布外重新生成；保留内部。 |
| 9 通道 U-Net | "正确的修复模型" | 以 `noisy | encoded-source | mask` 作为输入的 U-Net。 |
| SDEdit | "带噪声级别的图到图" | 噪声到时间 `t`，用新提示去噪。 |
| InstructPix2Pix | "仅文本编辑" | 在（图像、指令、输出）三元组上微调的扩散。 |
| RePaint | "无需重新训练" | 在逆向期间定期重新噪声以减少接缝。 |
| SAM | "Segment Anything" | 点击或框选遮罩生成器；与修复配合。 |
| Flux-Kontext | "用上下文编辑" | Flux 变体，接收参考图像 + 指令进行编辑。 |

## 生产笔记：编辑 pipeline 是延迟敏感的

用户编辑图像期望亚 5 秒往返。1024² 上 30 步 SDXL-Inpaint 在 L4 上是 3-4 秒，加上 SAM 遮罩生成（约 200 ms）和 VAE 编码/解码（约 500 ms 合计）。在生产术语中，这是 TTFT 绑定的而不是吞吐量绑定的——batch 1，低并发，最小化每个阶段：

- **SAM-H 是慢的那个。** SAM-H 在 1024² 上约 200 ms；SAM-ViT-B 约 40 ms，质量损失很小。SAM 2（视频）增加时间开销；不要将它用于单图像编辑。
- **可能时跳过编码。** `pipe.image_processor.preprocess(img)` 编码为潜在变量。如果你有来自先前生成的潜在变量（在迭代编辑 UI 中很典型），通过 `latents=...` 直接传递它们以跳过一次 VAE 编码。
- **遮罩扩大对吞吐量也很重要。** 小遮罩意味着大部分 U-Net 前向传播被浪费（未遮罩像素无论如何都被箝位）。`diffusers` 的 `StableDiffusionInpaintPipeline` 运行完整 U-Net；只有 9 通道正确的修复变体利用遮罩计算。
- **Flux-Kontext 是 2025 年的答案。** 跨 `(source_image, instruction)` 的单次前向传播——无单独遮罩，无 SDEdit 噪声扫描。在 H100 上约 1.5 秒发出一条编辑。架构教训：折叠阶段。

## 进一步阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无训练修复。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力级编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年工具。