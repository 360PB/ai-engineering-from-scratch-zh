# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前的 VLM 答案——将所有图像 resize 到固定正方形——抛弃了 OCR、文档理解和超高分辨率场景解析所依赖的信号。NaViT（Google，2023）展示了可以用块对角掩码将可变分辨率 patch 打包到单个 transformer batch 中。Qwen2-VL 的 M-RoPE（2024）完全丢弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像平铺为 base + 子图像。SigLIP 2 的 NaFlex 变体（2025）现在是想要用单个 checkpoint 服务所有宽高比的开源 VLM 的默认编码器。本节端到端实现 patch-n'-pack。

**类型：** Build
**语言：** Python（标准库，patch 打包器 + 块对角掩码）
**前置知识：** Phase 12 · 01（ViT patch），Phase 12 · 05（LLaVA）
**时间：** 约 120 分钟

## 学习目标

- 将可变分辨率图像 batch 中的 patch 打包到一个序列中，并构建块对角注意力掩码。
- 为给定任务在 AnyRes 平铺（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 计算 OCR、图表和摄影场景下的 token 预算，不做 resize。
- 说出正方形 resize 的三种失败模式：文字被压扁、内容被裁剪、padding 浪费 token。

## 问题背景

Transformer 期望一个序列。一个 batch 是一叠等长的序列。如果图像都是 224x224，每次得到 196 个 patch token，无需 padding，问题解决。

但现实不配合。文档是竖向的（8.5x11 英寸，2:3 左右）。图表截图是横向的（16:9）。收据是又高又窄的（1:3）。医学影像以 2048x2048 或更大尺寸输出。手机设备截图是 1170x2532（0.46:1）。

2024 年之前的三种选项及其各自的失败原因：

1. Resize 到固定正方形（224x224 或 336x336）。压扁会扭曲文字和人脸。下采样会破坏图表标签和 OCR 内容。LLaVA-1.5 之前的标准做法。
2. 裁剪到固定宽高比。你丢弃了图像的大部分内容，选择裁剪位置本身就是另一个视觉问题。
3. Padding 到最长边。修复了扭曲但对竖向图像浪费 50%+ 的 token。所有那些 pad token 上的二次注意力成本。

2024-2025 年的答案：让 transformer 以图像原生分辨率摄入 patch，想办法将异构 batch 打包到一个序列中，不浪费计算。

## 核心概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani 等，2023）是证明了这一方案可以大规模工作的论文。思路很机械：

1. 对 batch 中每张图像，在选定的 patch 大小（假设 14）下计算其原生 patch 网格。
2. 将每张图像的 patch 展平为其自身可变长度序列。
3. 将所有图像的 patch 拼接为一个长序列。
4. 构建块对角注意力掩码，使图像 A 的 patch 只在图像 A 内部相互注意。
5. 携带每个 patch 的位置信息（2D RoPE 或分数位置编码）。

batch 中三张图像：336x336（576 token）、224x224（256 token）、448x336（768 token）变为一个 1600 token 的序列，带 1600x1600 的块对角掩码。无 padding。无浪费计算。Transformer 处理任意宽高比。

NaViT 还引入了训练期间分数 patch 丢弃——在 batch 中随机丢弃 50% 的 patch——既正则化又加速训练。SigLIP 2 继承了这一做法。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一张高分辨率图像和一个固定编码器（336 分辨率的 CLIP 或 SigLIP），平铺图像：

1. 从预定义网格集中选择最适合图像宽高比的网格布局——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等。
2. 将完整图像平铺到网格中；每个图块成为 336x336 裁剪。
3. 同时生成一张缩略图：将整张图像 resize 到 336x336 作为全局上下文 token。
4. 用冻结的 336 编码器编码每个图块。拼接图块 token + 缩略图 token。

672x672 图像在 2x2 网格加缩略图下：4 * 576 + 576 = 2880 个视觉 token。昂贵但有效——LLM 同时看到局部细节和全局上下文。

AnyRes 是在编码器冻结且只支持一种分辨率时的首选。当图像很大时它会爆炸 token 数量（1344x1344 图像在 4x4 网格下是 9216 + 576 ≈ 9800 token，填满大部分 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置编码。不是 NaViT 的分数位置，也不是 AnyRes 的图块+缩略图，每个 patch 携带一个 3D 位置（时间、高度、宽度）。Query/Key 旋转处理任意的 H、W 和时间长度。

M-RoPE 原生支持动态分辨率，无需重训练。推理时喂入任意 HxW 图像，patch 嵌入器产生 H/14 x W/14 个 token，每个 token 获取其 (t=0, r=row, c=col) 位置，RoPE 用正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续这一做法。InternVL3 的 V2PE 是同一思路，但每种模态的编码可变。

与 AnyRes 不同，M-RoPE 在原生分辨率下是 O(H x W / P²) token——无乘法的图块开销。与 NaViT 不同，它仍然期望每次前向一张图像。跨分辨率的 batch 仍需在其上叠加 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的原生灵活模式。单个模型在推理时服务多种序列长度（256、729、1024 token）。内部在训练时使用 NaViT 风格的 patch-n'-pack 和每个 patch 的绝对分数位置。卖点：一个 checkpoint，根据任务在推理时选择 token 预算。

语义任务（分类、检索）用 256 token。OCR 或图表理解用 1024 token。无需重训练。

### 打包掩码

块对角掩码是大多数实现跌倒的地方。对于覆盖图像 `i=0..B-1`、长度为 `n_i` 的打包序列，总长度 `N_total` 的掩码 `M`，形状 `(N_total, N_total)`：当两个索引都落在同一图像的块内时为 1，否则为 0。可以从累积长度列表构建：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 当且仅当存在 b 使得 offsets[b] <= i < offsets[b+1] 且 offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中用 `torch.block_diag` 或显式 gather 是一行。FlashAttention 的变长路径（`cu_seqlens`）完全跳过掩码，直接用累积长度张量在序列内做注意力——对典型 batch 比密集掩码快约 10 倍。

### Token 预算

按任务选策略：

- OCR / 文档：1024-4096 token。SigLIP 2 NaFlex 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：384-448 原生分辨率下 729-1024 token。Qwen2.5-VL 动态分辨率加最大像素上限。
- 自然照片：256-576 token 足够。下游 LLM 看到了足够信息。在内容密度高的地方付费 token。
- 视频：空间池化后每帧 64-128 token，2-8 FPS。第 12.17 节涵盖此内容。

2026 年的生产规则：选一个每任务最大像素上限，在该上限内以原生宽高比编码，打包 batch，跳过 padding。Qwen2.5-VL 暴露了 `min_pixels` 和 `max_pixels` 就是为这个旋钮。

## 使用方法

`code/main.py` 为异构 batch 图像实现 patch-n'-pack（整数像素坐标）。它：

- 接收 (H, W) 图像尺寸列表。
- 计算每张图像在 patch 大小 14 下的 patch 序列长度。
- 将它们打包为一个总长度 `sum(n_i)` 的序列。
- 构建块对角注意力掩码（密集版，便于理解）。
- 比较打包成本与正方形 resize 和 AnyRes 平铺。
- 为混合 batch（收据、图表、截图、照片）打印 token 预算表。

运行它。输出的数字就是为什么每款 2026 年开源 VLM 都使用 patch-n'-pack 的原因。

## 输出作品

本节生成 `outputs/skill-resolution-budget-planner.md`。给定混合宽高比工作负载（OCR、图表、照片、视频帧）和总 token 预算，选择正确策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并发出每请求配置。当为产品评估 VLM 大小时使用此技能——防止悄悄发生的 10 倍 token 爆炸杀死延迟预算。

## 练习

1. 收据是 600x1500（1:2.5）。在 patch 大小 14 下，原生分辨率有多少 token？正方形 resize 到 336 后有多少？哪个在实践中丢失更多 OCR 精度？

2. 为四个长度为 256、576、729、1024 的图像 batch 构建块对角掩码。验证注意力矩阵是 2585x2585，且恰好有 `256² + 576² + 729² + 1024²` 个非零元素。

3. 1792x896 图像在 patch 14 下比较：(a) 正方形 resize 到 336 后编码，(b) AnyRes 2x1 + 缩略图，(c) M-RoPE 原生分辨率。哪个用 token 最少？哪个保留最多细节？

4. 实现分数 patch 丢弃：给定打包序列，随机均匀丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏度变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）第 3.2 节。两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个边界都重要。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Patch-n'-pack | "NaViT 风格打包" | 将不同图像的可变长度 patch 序列拼接为一个 batch 维度 |
| Block-diagonal mask | "打包掩码" | 注意力掩码，将每张图像的 patch 限制为只相互注意，不与包中邻居注意 |
| AnyRes | "LLaVA-NeXT 平铺" | 将高分辨率图像拆分为固定大小图块网格加全局缩略图；用固定编码器编码每个图块 |
| NaFlex | "SigLIP 2 原生灵活" | 单个 SigLIP 2 checkpoint 在推理时服务 256/729/1024 token 预算，无需重训练 |
| M-RoPE | "多模态 RoPE" | 3D 旋转位置编码（时间、行、列），处理任意 H、W、T，无需位置表 |
| cu_seqlens | "FlashAttention 打包" | FlashAttention varlen 路径使用的累积长度张量，替代密集块对角掩码 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 每请求旋钮，对非常小或非常大的输入限制 token 数量 |
| Visual token budget | "每图像多少 token" | 每张图像发出的 patch token 粗略计数；设置 LLM 的 prompt 预算和注意力成本 |

## 延伸阅读

- [Dehghani 等 — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang 等 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon 等 — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen 等 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)