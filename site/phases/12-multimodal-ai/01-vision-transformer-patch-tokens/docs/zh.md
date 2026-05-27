# Vision Transformers 与 Patch-Token 原语

> 在任何多模态处理之前，图像必须先转化为 transformer 能处理的 token 序列。2020 年的 ViT 论文用 16x16 像素块、线性投影和位置编码回答了这个问题。五年后的 2026 年，每一款前沿模型（Claude Opus 4.7 支持 2576px 原生分辨率、Gemini 3.1 Pro、Qwen3.5-Omni）仍以此为起点——编码器从 ViT 演变为 DINOv2 再到 SigLIP 2，新增了 register token，位置编码改为 2D-RoPE，但这一原语始终未变。本节从端到端解读 patch-token 流水线，并用标准库 Python 从头实现，为 Phase 12 的后续内容建立"视觉 token"的具体心智模型。

**类型：** Learn
**语言：** Python（标准库，patch 分词器 + 几何计算器）
**前置知识：** Phase 7 (Transformers)，Phase 4 (Computer Vision)
**时间：** 约 120 分钟

## 学习目标

- 将 HxWx3 的图像转换为带正确位置编码的 patch token 序列。
- 给定 patch 大小、分辨率、隐藏维度和深度，计算 ViT 的序列长度、参数量和 FLOPs。
- 说出将 ViT 从 2020 研究推向 2026 生产的三个升级：自监督预训练（DINO/MAE）、register token 和原生分辨率打包。
- 为下游任务在 CLS pooling、mean pooling 和 register token 之间做出选择。

## 问题背景

Transformer 操作的是向量序列。文本天生是序列（字节或 token）。而图像是带三个颜色通道的二维像素网格——不是序列。如果把每个像素展平，224x224 的 RGB 图像会变成 150,528 个 token，在此长度上做自注意力的计算成本是平方级的，根本不可行。

2020 年之前的做法是在前面接一个 CNN 特征提取器：ResNet 产生一个 7x7 的 2048 维特征图，49 个 token 送入 transformer。这可行，但继承了 CNN 的归纳偏置（平移等变性、局部感受野），也丢失了 transformer 的规模扩展能力。

Dosovitskiy 等人（2020）提出了一个直白的问题：能不能跳过 CNN？把图像切成固定大小的 patch（如 16x16 像素），对每个 patch 做线性投影，加位置编码，送入普通 transformer。当时这是异端——视觉任务不用卷积。但只要数据足够（JFT-300M，然后是 LAION），它就超越了 ResNet 在 ImageNet 上的表现，并持续改进。

到 2026 年，ViT 原语已无疑义地成为基础。每款开源权重 VLM 的视觉塔都是其后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是"要不要用 patch？"，而是"patch 大小多少？分辨率调度怎么安排？预训练目标是什么？位置编码选哪种？"

## 核心概念

### Patch 即 Token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch 大小 `P`，将图像划分为 `(H/P) x (W/P)` 个不重叠的 patch。每个 patch 是 `P x P x 3` 的像素立方体，展平为 `3P²` 维向量，再通过共享线性投影 `W_E`（形状 `(3P², D)`）映射到模型的隐藏维度 `D`。

以 ViT-B/16 标准配置为例：
- 分辨率 224，patch 大小 16 → 14x14 网格 → 196 个 patch token。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` token → 序列长度 197。

Patch 投影在数学上等价于一个卷积核大小为 `P`、步幅为 `P`、输出通道为 `D` 的 2D 卷积。生产代码正是这样实现的：`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"是概念层面的说法，"卷积核"是高效实现的角度。

### 位置编码

Patch 本身没有顺序——transformer 将它们视为一个集合。早期的 ViT 添加可学习的 1D 位置编码（每个位置一个 768 维向量，共 197 个）。有效，但将模型绑定在训练分辨率上：推理时若改变网格大小，必须对位置表做插值。

现代视觉骨干网使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认配置）或分解式 2D 位置编码。2D-RoPE 根据 patch 的（行，列）索引旋转 query 和 key 向量，使模型从旋转角度推断相对 2D 位置。无位置表，模型可在推理时处理任意大小的网格。

### CLS token、池化输出与 Register Token

图像级表示是什么？三种选择并存：

1. `[CLS]` token。在 patch 序列前添加一个可学习向量。经过所有 transformer 块后，CLS token 的隐藏状态即为图像表示。继承自 BERT。原始 ViT 和 CLIP 使用。
2. Mean pool。对 patch token 的输出隐藏状态做平均。SigLIP、DINOv2 和大多数现代 VLM 使用。
3. Register token。Darcelet 等人（2023）观察到，没有显式 sink token 的 ViT 在训练中会形成高范数的"artifact" patch，劫持自注意力。添加 4-16 个可学习的 register token 吸收这一负载，提升密集预测质量（分割、深度）。DINOv2 和 SigLIP 2 都内置了 register。

选择对下游任务很重要。CLS 适合分类。对于向 LLM 输入 patch token 的 VLM，完全跳过池化——每个 patch 都成为 LLM 的输入 token。Register 在移交前被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 用 JFT-300M 做监督分类预训练。很快被以下方法取代：

- CLIP（2021）：在 4 亿对图文对上做对比学习。参见第 12.02 节。
- MAE（2021，He 等）：掩码 75% 的 patch，重建像素。自监督，纯图像可用。
- DINO（2021）/ DINOv2（2023）：师生自蒸馏，无需标签，无需文本描述。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干，也是"密集特征"用例的默认选择。
- SigLIP / SigLIP 2（2023，2025）：用 sigmoid 损失和 NaFlex 原生宽高比的 CLIP。2026 年开源 VLM（Qwen、Idefics2、LLaVA-OneVision）的主导视觉塔。

预训练方式的选择决定了骨干网擅长什么：CLIP/SigLIP 用于与文本的语义匹配，DINOv2 用于密集视觉特征，MAE 作为下游微调的起点。

### 扩展定律

ViT 扩展（Zhai 等，2022）建立了 ViT 质量与模型规模、数据规模和计算量之间的可预测关系。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch 大小是控制序列长度与保真度之间权衡的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型值）比 patch 16 每张图像产生更多 token；对 OCR 和密集任务更好，速度更慢。
- 分辨率是另一个大杠杆。从 224 到 384 再到 512 几乎总有提升，但 FLOPs 成本是平方级的。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 token）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开源 VLM 的两种主力编码器。

### ViT 参数量

完整计算见 `code/main.py`。以 ViT-B/16 @ 224 为例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前，用这个方法估算每个 ViT 的参数量。骨干网大小决定任何下游 VLM 的 VRAM 下限。

### 2026 年生产配置

2026 年大多数开源 VLM 内置的编码器是 SigLIP 2 SO400m/14（原生分辨率 NaFlex）。其配置为：
- 4 亿参数。
- Patch 大小 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 图像级任务用 mean pool；VQA 时所有 729 个 patch 送入 LLM。
- 4 个 register token，移交 LLM 前丢弃。
- 原生宽高比的图像级缩放 + 2D-RoPE。

该配置中的每个决策都可以追溯到一篇可查阅的论文。

## 使用方法

`code/main.py` 是一个 patch 分词器和几何计算器。输入（图像 H、W，patch P，隐藏维度 D，深度 L），输出：

- 分块后的网格形状和序列长度。
- 合成 8x8 像素玩具图像的 token 序列（走一遍展平+投影流程）。
- 参数量分解（patch 嵌入、位置嵌入、transformer 块、注意力头）。
- 目标分辨率下前向传播的 FLOPs 估算。
- 跨 ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的比较表。

运行它。将参数量与已发布数字对照。调整 patch 大小和分辨率，感受 token 数量的成本。

## 输出作品

本节生成 `outputs/skill-patch-geometry-reader.md`。给定 ViT 配置（patch 大小、分辨率、隐藏维度、深度），生成 token 数量、参数量和 VRAM 估算，并附理由说明。每当为 VLM 选视觉骨干网时使用此技能——防止"token 爆炸导致 LLM 上下文溢出"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch 大小 14 下的 patch-token 序列长度。与仅用 CLS 的表示相比如何？

2. 1080p 帧（1920x1080）在 patch 14 下产生多少 token？30 FPS、5 分钟视频共有多少视觉 token？哪项成本节省最显著：池化、帧采样还是 token 合并？

3. 用纯 Python 实现 patch token 上的 mean pooling。验证对 DINOv2 输出的 196 个 token 做 mean pooling 的结果与模型 `forward` 返回的 pooled embedding 一致。

4. 阅读"Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述 register 吸收的是什么 artifact，以及它对下游密集预测的重要性。

5. 修改 `code/main.py` 支持 patch-n'-pack：给定不同分辨率的图像列表，生成一个打包序列和块对角注意力掩码。在学习第 12.06 节时与之对照验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Patch | "16x16 像素方块" | 输入图像的一个固定大小不重叠区域；成为一个 token |
| Patch embedding | "线性投影" | 将展平的 patch 像素映射到 D 维向量的共享学习矩阵（或 stride=P 的 Conv2d） |
| CLS token | "类别 token" | 前置的可学习向量，其最终隐藏状态代表整张图像；在 2026 年已非必需 |
| Register token | "吸收 token" | 额外的可学习 token，用于吸收 ViT 在预训练过程中形成的高范数注意力 artifact |
| Position embedding | "位置信息" | 每个位置的向量或旋转，使序列具有位置感知；2D-RoPE 是现代默认 |
| Grid | "Patch 网格" | 给定分辨率和 patch 大小下，(H/P) x (W/P) 的二维 patch 数组 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 特性：单个模型在不重训练的情况下支持多种宽高比和分辨率 |
| Backbone | "视觉塔" | 预训练的图像编码器，其 patch-token 输出在 VLM 中馈入 LLM |
| Pooling | "图像级摘要" | 将 patch token 汇总为一个向量的策略：CLS、mean、attention pool 或 register |
| Patch 14 vs 16 | "更细粒度 vs 更粗粒度" | Patch 14 每张图像产生更多 token，对 OCR 更好，速度更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy 等 — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT 论文。
- [He 等 — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab 等 — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无需标签。
- [Darcelet 等 — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 和 artifact 分析。
- [Tschannen 等 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认视觉塔。
- [Zhai 等 — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验扩展定律。