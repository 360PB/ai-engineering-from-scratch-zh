# 视觉 Transformer（ViT）

> 将图像切割成补丁，把每个补丁当作一个词，运行标准 Transformer。不要回头看。

**类型:** Build
**语言:** Python
**前置要求:** Phase 7 Lesson 02 (自注意力), Phase 4 Lesson 04 (图像分类)
**时长:** 约 45 分钟

## 学习目标

- 从头实现补丁嵌入、学习位置嵌入、类别 token 和 transformer 编码器块，构建最小化 ViT
- 解释为什么 ViT 曾被认为需要海量预训练数据，直到 DeiT 和 MAE 证明了并非如此
- 比较 ViT、Swin 和 ConvNeXt 在架构先验（无、局部窗口注意力、卷积主干网）上的差异
- 使用 `timm` 和标准 linear-probe / 微调方案在小型数据集上微调预训练 ViT

## 问题背景

十年来，卷积一直是计算机视觉的代名词。CNN 有强归纳偏置——局部性、平移等变性——没人认为可以被取代。然后 Dosovitskiy 等人（2020）表明，一个应用于展平图像补丁的普通 transformer，在完全没有卷积 machinery 的情况下，在规模上可以匹配或击败最佳 CNN。

陷阱在于"在规模上"。ViT 在 ImageNet-1k 上输给了 ResNet。在 ImageNet-21k 或 JFT-300M 上预训练然后在 ImageNet-1k 上微调的 ViT 击败了它。结论是 transformer 缺乏有用的先验，但可以从足够的数据中学习它们。后续工作（DeiT、MAE、DINO）表明，通过正确的训练配方——强数据增强、自监督预训练、知识蒸馏——ViT 也可以在小数据上很好地训练。

到 2026 年，纯 CNN 在边缘设备上仍有竞争力（ConvNeXt 最强），但 transformer 主宰了其他一切：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 块结构是必须掌握的。

## 核心概念

### 管线

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["补丁嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平为<br/>(196, 768) token"]
    FLAT --> CAT["预置<br/>[CLS] token"]
    CAT --> POS["加上学习<br/>位置嵌入"]
    POS --> ENC["N 个 transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>token 输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤。补丁 -> token -> 注意力 -> 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）改变其中一两个，其余保持不变。

### 补丁嵌入

第一个卷积是秘密所在。核大小 16，步幅 16，因此 224x224 图像变为 14x14 的 16x16 补丁网格，每个补丁投影到 768 维嵌入。这个单卷积同时完成了补丁化和线性投影。

```
输入：  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
输出： (768, 14, 14)
展平空间： (196, 768)
```

196 个补丁 = 196 个 token。每个 token 的特征维度是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### 类别 token

一个学习到的向量预置到序列前面：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

经过 N 个 transformer 块后，`[CLS]` 输出是全局图像表示。分类头只读取这一个向量。

### 位置嵌入

Transformer 没有内置的空间位置概念。给每个 token 加上一个学习到的向量：

```
tokens = tokens + learned_pos_embedding   (也是 shape (197, 768))
```

这个嵌入是模型的一个参数；基于梯度的训练将它适应到 2D 图像结构。存在正弦 2D 替代方案，但实践中很少使用。

### Transformer 编码器块

标准结构。多头自注意力、MLP、残差连接、Pre-LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP 是两层 GELU：Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的块，每个块 12 个注意力头，共 86M 参数。

### 为什么用 Pre-LN

早期 transformer 使用 post-LN（`x = LN(x + sublayer(x))`），在不用 warmup 的情况下训练超过 6-8 层很困难。Pre-LN（`x = x + sublayer(LN(x))`）可以稳定地训练更深的网络而无需 warmup。每个 ViT 和每个现代 LLM 都使用 pre-LN。

### 补丁大小权衡

- 16x16 补丁 -> 196 个 token，标准。
- 32x32 补丁 -> 49 个 token，更快但分辨率更低。
- 8x8 补丁 -> 784 个 token，更精细但 O(n^2) 注意力成本扩展很差。

更大的补丁 = 更少的 token = 更快但空间细节更少。SwinV2 在分层窗口中使用 4x4 补丁。

### DeiT 在 ImageNet-1k 上训练 ViT 的配方

原始 ViT 需要 JFT-300M 才能击败 CNN。DeiT（Touvron et al., 2020）通过四个改动，用 ImageNet-1k alone 将 ViT-B 训练到 81.8% top-1：

1. 重数据增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度（训练时随机丢弃整个块）。
3. 重复增强（同一图像每个批次采样 3 次）。
4. 从 CNN 教师蒸馏（可选，进一步提升精度）。

每个现代 ViT 训练配方都源自 DeiT。

### Swin vs ConvNeXt

- **Swin**（Liu et al., 2021）——基于窗口的注意力。每个块在一个局部窗口内做注意力；交替块平移窗口以跨窗口混合信息。在保持注意力算子的同时恢复了类似 CNN 的局部性先验。
- **ConvNeXt**（Liu et al., 2022）——重新设计的 CNN，匹配 Swin 的架构选择（深度可分离卷积、LayerNorm、GELU、倒置瓶颈）。表明差距不在"注意力 vs 卷积"而在"现代训练配方 + 架构"。

在 2026 年，ConvNeXt-V2 和 Swin-V2 都是生产级的；正确选择取决于推理栈（ConvNeXt 对边缘编译更好）和预训练语料。

### MAE 预训练

掩码自编码器（He et al., 2022）：随机遮罩 75% 的补丁，训练编码器只处理可见的 25%，训练一个小型解码器从编码器输出重建被遮罩的补丁。预训练后，丢弃解码器，微调编码器。

MAE 使 ViT 可在 ImageNet-1k 上单独训练，达到 SOTA，是当前默认的自监督配方。

## 构建过程

### 步骤 1：补丁嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积，一个展平，一个转置。这就是图像到 token 的全部步骤。

### 步骤 2：Transformer 块

Pre-LN，多头自注意力，带 GELU 的 MLP，残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 处理头的分割、缩放点积和输出投影。`batch_first=True` 所以形状是 `(N, seq, dim)`。

### 步骤 3：ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约 2.8M 参数——一个可在 CPU 上运行的微型 ViT。真实 ViT-B 是 86M；相同的类定义，只需 `dim=768, depth=12, num_heads=12`。

### 步骤 4：完整性检查——单张图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应该可以无错误运行。概率和为 1。

## 应用

`timm` 提供了每个 ViT 变体及其 ImageNet 预训练权重。一行代码：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` 是 2026 年视觉 transformer 的生产默认选择。支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 等数十种，在同一 API 下统一。

对于多模态工作（图像 + 文本），`transformers` 提供 CLIP、SigLIP、BLIP-2、LLaVA。其中所有图像编码器都是 ViT 变体。

## 交付物

本课产出：

- `outputs/prompt-vit-vs-cnn-picker.md`——一个提示词，基于数据集大小、计算量和推理栈在 ViT、ConvNeXt 或 Swin 之间选择。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md`——一个技能，验证 ViT 的补丁嵌入和位置嵌入形状是否与模型预期的序列长度匹配，捕获最常见的移植 bug。

## 练习

1. **(简单)** 打印通过上述微型 ViT 前向传播时每个中间张量的形状。确认：输入 `(N, 3, 64, 64)` -> 补丁 `(N, 16, 192)` -> 加 CLS 后 `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **(中等)** 在 Lesson 4 的合成 CIFAR 数据集上微调预训练 `timm` ViT-S/16。与在同一数据上微调的 ResNet-18 比较。报告训练时间和最终精度。
3. **(困难)** 为微型 ViT 实现 MAE 预训练：遮罩 75% 的补丁，训练编码器 + 小型解码器重建被遮罩的补丁。在预训练前后的合成数据上评估 linear-probe 精度。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 补丁嵌入 | "第一个卷积" | 核大小 = 步幅 = 补丁大小的卷积；将图像转换为 token 嵌入网格 |
| 类别 token | "[CLS]" | 学习到的向量预置到 token 序列；其最终输出是全局图像表示 |
| 位置嵌入 | "学习位置" | 加到每个 token 上的学习向量，使 transformer 知道每个补丁来自哪里 |
| Pre-LN | "在子层前做 LayerNorm" | 稳定的 transformer 变体：`x + sublayer(LN(x))` 而非 `LN(x + sublayer(x))` |
| 多头注意力 | "并行注意力" | 标准 transformer 注意力分割到 num_heads 个独立子空间，之后拼接 |
| ViT-B/16 | "Base，补丁 16" | 标准尺寸：dim=768, depth=12, heads=12, patch_size=16, image=224；约 86M 参数 |
| DeiT | "数据高效 ViT" | 仅用 ImageNet-1k 训练并配合强数据增强的 ViT；证明了海量预训练数据集并非严格必需 |
| MAE | "掩码自编码器" | 自监督预训练：遮罩 75% 的补丁，重建；主导的 ViT 预训练配方 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)——ViT 论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877)——如何在 ImageNet-1k 上单独训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377)——MAE 预训练
- [timm 文档](https://huggingface.co/docs/timm)——生产中你会用到的每个视觉 transformer 的参考