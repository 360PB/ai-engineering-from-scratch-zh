# ControlNet、LoRA 与条件化

> 文本本身是一个笨拙的控制信号。ControlNet 让你克隆一个预训练扩散模型并用深度图、姿态骨架、涂鸦或边缘图像来引导它。LoRA 让你通过训练 1000 万参数来微调一个 20 亿参数模型。它们一起将 Stable Diffusion 从一个玩具变成了 2026 年每个代理都在部署的图像 pipeline。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 07（潜在扩散），Phase 10（从头构建 LLM —— LoRA 基础）
**时间：** 约 75 分钟

## 问题

像"一个女人穿着红色连衣裙在繁忙街道上遛狗"这样的提示没有给模型关于*狗在哪里*、*女人是什么姿态*或*街道的视角*的信息。文本固定了约 10% 你需要指定图像的内容。其余是视觉的，无法用文字有效描述。

为每个信号（姿态、深度、canny、分割）从头训练新的条件模型是禁止的。你想保持 2.6B 参数 SDXL backbone 冻结，附加一个读取条件化的小型侧网络，让它轻推 backbone 的中间特征。这就是 ControlNet。

你也想教模型新概念（你的脸、你的产品、你的风格）而不重新训练整个模型。你想要小 100 倍的 delta。这就是 LoRA——低秩适配器，插入现有注意力权重。

ControlNet + LoRA + 文本 = 2026 年从业者的工具包。大多数生产图像 pipeline 在 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 核心概念

![ControlNet 克隆编码器；LoRA 添加低秩 delta](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等人，2023 年）

取一个预训练 SD。*克隆* U-Net 的编码器半部分。冻结原始的。训练克隆接受额外的条件化输入（边缘、深度、姿态）。通过*零卷积*跳连接将克隆连接回原始的解码器半部分（初始化为零的 1×1 卷积——开始是 no-op，学习一个 delta）。

```
SD U-Net 解码器：   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 开始时是恒等的——即使在训练前也没有伤害。在 100 万个（提示、条件、图像）三元组上用标准扩散损失训练。

每个模态 ControlNet 作为小型侧模型发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等人，2021 年）

对于模型中任何线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩 delta：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力用 rank 4-16 是标准的，重微调用 rank 64-128。新参数数量：`2 · d · r` 而不是 `d²`。对于 `d=640`、`r=16` 的 SDXL 注意力：每个适配器 20k 参数而不是 410k——减少 20 倍。整个模型：LoRA 通常是 20-200MB vs 基础 5GB。

推理时你可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常的。多个 LoRA 可加法叠加（通常的警告是它们以非线性方式相互作用）。

### IP-Adapter（Ye 等人，2023 年）

一个微型适配器，接收*图像*作为条件化（以及文本）。使用 CLIP 图像编码器生成图像 token，将它们与文本 token 一起注入交叉注意力。每个基础模型约 20MB。让你做"用这个参考图像的风格生成图像"而无需 LoRA。

## 可组合性矩阵

| 工具 | 控制内容 | 大小 | 何时使用 |
|------|---------|------|---------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 参考图像的风格或主体 | 20MB | 文本无法描述的外观 |
| Textual Inversion | 作为新 token 的单个概念 | 10KB | 遗留，大部分被 LoRA 取代 |
| DreamBooth | 主体上的全量微调 | 2-5GB | 强身份，高计算量 |
| T2I-Adapter | 更轻量的 ControlNet 替代 | 70MB | 边缘设备，推理预算 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者都用。

## 构建

`code/main.py` 在 1-D 上模拟两个机制：

1. **LoRA。** 一个预训练线性层 `W`。冻结它。训练一个低秩 `B @ A`，使得 `W + BA` 匹配目标线性层。展示 `r = 1` 足以完美学习 rank-1 校正。

2. **ControlNet-lite。** 一个"冻结基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出由初始化为零的学习标量门控（我们的零卷积版本）。训练并观察门控上升。

### 第 1 步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 冻结的；A、B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第 2 步：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # 门初始化为 0
h = base(x) + gated
```

在步 0 输出与基础完全相同。早期训练缓慢更新 `gate`——没有灾难性漂移。

## 陷阱

- **LoRA 过度缩放。** `α = 2` 或 `α = 3` 是常见的"让它更强"技巧，产生过度风格化 / 损坏的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 在权重 1.0 使用 Pose ControlNet 和在权重 1.0 使用 Depth ControlNet 通常会超调。权重之和 ≈ 1.0 是安全的默认值。
- **LoRA 在错误的基础上。** SDXL LoRA 在 SD 1.5 上悄悄 no-op，因为注意力维度不匹配。Diffusers 会在 0.30+ 中警告。
- **Textual Inversion 漂移。** 在一个 checkpoint 上训练的 token 在另一个上漂移严重。LoRA 更可移植。
- **LoRA 权重合并和存储。** 你可以将 LoRA 合并到基础模型权重中以加快推理（无需运行时加法），但你失去了在运行时缩放 `α` 的能力。保留两个版本。

## 使用

| 目标 | 2026 年 pipeline |
|------|----------------|
| 再现品牌艺术风格 | 在约 30 张精选图像上以 rank 32 训练的 LoRA |
| 将我的脸放入生成的图像中 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考 + 提示 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 修复（Lesson 09） |
| 快速 1 步风格 | SDXL-Turbo 上的 LCM-LoRA |

## 发布

保存为 `outputs/skill-sd-toolkit-composer.md`。Skill 接收任务（输入资产：提示、可选参考图像、可选姿态、可选深度、可选涂鸦）并输出工具栈、权重和可复现种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中变化 LoRA rank `r` 从 1 到 4。在什么 rank 下 LoRA 完全匹配 rank-2 目标 delta？
2. **中等。** 在两个目标变换上训练两个单独的 LoRA。一起加载并展示它们的加法交互。交互何时打破线性？
3. **困难。** 使用 diffusers 堆叠：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。当堆栈权重变化时，测量 FID vs 提示遵循的权衡。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| ControlNet | "空间控制" | 克隆编码器 + 零卷积跳连接；读取条件图像。 |
| 零卷积 | "开始时是恒等" | 初始化为零的 1×1 卷积；ControlNet 开始时是 no-op。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；比全量微调少 100 倍参数。 |
| rank r | "旋钮" | LoRA 压缩；4-16 典型，64+ 用于重度个性化。 |
| α | "LoRA 强度" | LoRA delta 的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP-图像 token 的小型图像条件化适配器。 |
| DreamBooth | "主体全量微调" | 在约 30 张主体图像上训练整个模型。 |
| Textual Inversion | "新 token" | 仅学习新的词嵌入；遗留，大部分被取代。 |

## 生产笔记：LoRA 交换、ControlNet 通道、多租户服务

一个真正的文生图 SaaS 在同一个基础 checkpoint 上提供数百个 LoRA 和十几个 ControlNet。服务问题看起来很像 LLM 多租户（生产文献在连续批处理和 LoRAX / S-LoRA 下涵盖了 LLM 案例）：

- **热交换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础中使每步推理快约 3-5%，但冻结了 `α` 和基础。将 LoRA 保留在 VRAM 中作为 rank-r delta 热备用；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于按请求激活。交换成本是 `2 · d · r · num_layers` 权重——MB 规模，亚秒级。
- **ControlNet 作为第二个注意力通道。** 克隆的编码器与基础并行运行。两个权重 1.0 的 ControlNet = 每步两个额外前向传播，不是一个合并的传播。Batch size 余量按二次方下降。为每个活跃 ControlNet 预算约 1.5 倍每步成本。
- **量化 LoRA 也可以。** 如果你量化了基础（见 Lesson 07，8GB 上 Flux），LoRA delta 也可以干净地量化为 8 位或 4 位。QLoRA 风格加载让你在 4-bit Flux 基础上堆叠 5-10 个 LoRA 而不爆内存。

Flux 特定：Niels 的 8GB 上运行 Flux notebook 将基础量化为 4-bit；在该量化基础上堆叠风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）在 `weight_name="pytorch_lora_weights.safetensors"` 仍然有效。这是 2026 年大多数 SaaS 代理部署的配方。

## 进一步阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；移植到扩散）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的轻量替代。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考 pipeline。