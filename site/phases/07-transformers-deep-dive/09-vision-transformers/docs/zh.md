# Vision Transformer（ViT）

> 图像是一组 patches 的网格。句子是一组 token 的网格。同一个 transformer 吞噬两者。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 第 5 课（完整 Transformer）、Phase 4 第 3 课（CNN）、Phase 4 第 14 课（Vision Transformer 简介）
**时长：** ~45 分钟

## 问题

2020 年之前，计算机视觉意味着卷积。每个 ImageNet、COCO 和检测基准的最先进都使用 CNN 主干。Transformer 用于语言。

Dosovitskiy et al.（2020）——"一张图像价值 16×16 个词"——表明你可以完全丢弃卷积。将图像切成固定大小的 patches，线性投影每个 patch 为嵌入，将序列送入普通 transformer 编码器。在足够规模（ImageNet-21k 预训练或更大）下，ViT 匹配或击败基于 ResNet 的模型。

ViT 是 2026 年更广泛模式的开始：一种架构，多种模态。Whisper 将音频分词。ViT 将图像分词。机器人动作 token。视频像素 token。Transformer 不关心——喂给它一个序列，它学习。

截至 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）拥有大部分视觉。CNN 在边缘设备和延迟敏感任务上仍然胜出。其他一切都在栈中某处有 ViT。

## 概念

![图像 → patches → token → transformer](../assets/vit.svg)

### 第一步 — 划分 patches

将 `H × W × C` 图像分割为 `N × (P·P·C)` 的扁平 patches 序列。典型设置：`224 × 224` 图像，`16 × 16` patches → 196 个 patches，每个 768 值。

```
image (224, 224, 3) → 14 × 14 的 16x16x3 patches 网格 → 196 个长度为 768 的向量
```

Patch 大小是杠杆。更小的 patches = 更多 token，更高分辨率，二次注意力成本。更大的 patches = 更粗糙，更便宜。

### 第二步 — 线性嵌入

单个学习矩阵将每个扁平 patch 投影到 `d_model`。等价于核大小 `P` 和步长 `P` 的卷积。在 PyTorch 中这正是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——两行实现。

### 第三步 — 前置 `[CLS]` token，添加位置嵌入

- 前置一个可学习的 `[CLS]` token。其最终隐藏状态是用于分类的图像表示。
- 添加可学习位置嵌入（原始 ViT）或正弦 2D（后续变体）。
- 在 2024+ RoPE 扩展到 2D 位置，有时无显式嵌入。

### 第四步 — 标准 transformer 编码器

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 块。与 BERT 完全相同。无视觉特定层。这是论文的教学冲击。

### 第五步 — head

对于分类：取 `[CLS]` 隐藏状态 → 线性 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用 patch 嵌入。

### 起作用的变体

| 模型 | 年份 | 变化 |
|------|------|------|
| ViT | 2020 | 原创。固定 patch 大小，完整全局注意力。 |
| DeiT | 2021 | 蒸馏；可在 ImageNet-1k 上训练。 |
| Swin | 2021 | 分层带滑动窗口。固定亚二次成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；扩展律适用。 |
| SigLIP | 2023 | ViT + 语言对，sigmoid 对比损失。比匹配计算的 CLIP 更好。 |
| SAM 3 | 2025 | 分割任何；ViT-Large + 可提示掩码解码器。 |

### 为什么花了这么久

ViT 需要*大量*数据才能匹配 CNN，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有 >100M 标注图像或强自监督预训练，CNN 在匹配计算下仍然胜出。DeiT 在 2021 年用蒸馏技巧修复了这个问题；DINOv2 在 2023 年用自监督永久修复了这个问题。

## 构建

见 `code/main.py`。纯标准库划分 + 线性嵌入 + 健全性检查。无训练——ViT 在任何现实规模上需要 PyTorch 和数小时 GPU 时间。

### 第一步：假图像

24 × 24 RGB 图像作为 `(R, G, B)` 元组行的列表。我们使用 6×6 patches → 16 个 patches，每个 108-d 嵌入向量。

### 第二步：划分 patches

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：沿网格按行优先。每个 ViT 使用此顺序。

### 第三步：线性嵌入

将每个扁平 patch 乘以随机 `(patch_flat_size, d_model)` 矩阵。前置 `[CLS]` 后验证输出形状为 `(N_patches + 1, d_model)`。

### 第四步：计算现实 ViT 的参数数

打印 ViT-Base 的参数计数：12 层、12 头、d=768、patch=16。与 ResNet-50（~25M）比较。ViT-Base 落在 ~86M。ViT-Large ~307M。ViT-Huge ~632M。

## 使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768)：[CLS] + 196 个 patches
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2 嵌入是 2026 年图像特征的默认配置。** 冻结主干，训练一个小 head。适用于分类、检索、检测、描述。Meta 的 DINOv2 检查点在每个非文本视觉任务上优于 CLIP。

**Patch 大小选择。** 小型模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 交付

见 `outputs/skill-vit-configurator.md`。该 skill 根据数据集大小、分辨率和计算预算为新的视觉任务选择 ViT 变体和 patch 大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证 patches 数等于 `(H/P) * (W/P)`，扁平 patch 维度等于 `P*P*C`。
2. **中等。** 实现 2D 正弦位置嵌入——两个独立的正弦码，用于每个 patch 的 `row` 和 `col`，拼接。将其送入一个微型 PyTorch ViT 并比较 CIFAR-10 上可学习位置嵌入的准确率。
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上训练，4×4 patches。测量测试准确率。现在在同一 1,000 张图像上添加 DINOv2 预训练（简化：仅训练编码器从掩码 patches 预测 patch 嵌入）。准确率提高了吗？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| Patch | "视觉-transformer token" | 图像 `P × P × C` 区域的扁平像素值向量。 |
| Patchify | "切 + 扁平" | 将图像分割为不重叠的 patches，扁平每个为向量。 |
| `[CLS]` token | "图像摘要" | 前置可学习 token；其最终嵌入是图像表示。 |
| 归纳偏置 | "模型假设什么" | ViT 比 CNN 假设更少；需要更多数据来弥补差距。 |
| DINOv2 | "自监督 ViT" | 无标签训练，使用图像增强 + 动量教师。2026 年最佳通用图像特征。 |
| SigLIP | "CLIP 的继任者" | ViT + 文本编码器，用 sigmoid 对比损失训练；在匹配计算上优于 CLIP。 |
| Swin | "窗口化 ViT" | 带局部注意力 + 滑动窗口的分层 ViT；亚二次。 |
| Register token | "2023 年技巧" | 几个额外的可学习 token，吸收注意力 sink；改善 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的 register-token 修复。