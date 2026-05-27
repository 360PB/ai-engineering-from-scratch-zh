# CLIP 与对比视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足够强大的想法：仅用嘈杂的网页图文对和对比损失，将图像编码器和文本编码器对齐到同一向量空间。不需要任何监督标签。4 亿对图文。得到的嵌入空间可以做零样本分类、图文检索，并作为每款 2026 年 VLM 的视觉塔。SigLIP 2（2025）用 sigmoid 替代 softmax，以更低的成本扩展规模超越 CLIP。本节从 InfoNCE 到 sigmoid 成对损失，完整走读数学并用标准库 Python 实现训练步骤。

**类型：** Build
**语言：** Python（标准库，InfoNCE + sigmoid 损失实现）
**前置知识：** Phase 12 · 01（ViT patch），Phase 7（Transformers）
**时间：** 约 180 分钟

## 学习目标

- 从互信息推导出 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 成对损失（SigLIP）能够扩展到 32768+ 的 batch 而无需 softmax 所需的全局汇聚开销。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练给你的四个杠杆：batch 大小、温度、prompt 模板、数据质量。

## 问题背景

CLIP 出现之前的视觉领域靠的是监督学习。收集带标签数据集（ImageNet：120 万张图像，1000 类），训练 CNN，发布。标签成本高、标签存在标注者共识偏差、没有微调就无法迁移到新任务。

互联网上有数十亿张带弱标签的图片，这些图片自带监督信号——文本描述了图像。问题在于：你能把这些转化为有用的训练信号吗？

CLIP 的答案：将图文对视为匹配任务。给定一批 N 张图像和 N 条文本描述，学习将每张图像匹配到其对应文本，同时排除 N-1 个干扰项。监督信号是"这两个是一对的，其他 N-1 个不是。"无需类别标签，无需人工标注，只需一个对比损失。

由此产生的嵌入空间能做的事超出了 CLIP 的训练目标。ImageNet 零样本有效，因为"a photo of a cat"的嵌入与从未被标注为猫的猫图像接近。这赌注催生了 2026 年的所有 VLM。

## 核心概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，每条文本输出一个 D 维向量。

两个塔将其输出归一化为单位长度。由于都是单位范数，相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`。

对于一批 N 对（图像，文本），构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是学习到的温度（CLIP 初始化为 0.07，在对数空间中学习）。

### InfoNCE 损失

CLIP 使用对称的行列交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是自己的文本
loss_t2i = CE(S^T, labels=identity)   # 每条文本的正样本是自己的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 强制每张图像与其文本的匹配度高于批次中所有其他文本。"负样本"是批次中所有其他样本。更大的 batch = 更多负样本 = 更强的信号。CLIP 在 32k batch 上训练；规模很重要。

### 温度

`tau` 控制 softmax 的锐度。低 tau → 锐利分布，硬负样本挖掘效果。高 tau → 柔和，所有样本都有贡献。CLIP 学习 `log(1/tau)`，裁剪以防止崩溃到接近零的 tau。SigLIP 2 固定初始 tau，使用学习的偏置。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，必须将每个 embedding 汇聚到所有副本，然后做 softmax。通信量随世界规模的平方增长。

SigLIP 将 softmax 替换为逐元素 sigmoid：对于每对 `(i, j)`，损失是二分类——"这是否是一对匹配的 pair？"正类标签在对角线上，其他都是负类。损失为：

```
L = -1/N Σ_{i,j} [ y_ij · log sigmoid(S[i,j]) + (1-y_ij) · log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当且仅当 `i == j`，否则为 0。每对的损失是独立的。无需全局汇聚。每块 GPU 计算本地块并求和。SigLIP 2 能以极低的通信成本扩展到 32k-512k batch，而 CLIP 需要比例上更多的通信。

### 零样本分类

给定 N 个类别名，为每个类别构造文本模板：

```
"a photo of a {class}"
```

用文本编码器编码每个模板。用图像编码器编码你的图像。余弦相似度 argmax = 预测类别。不需要针对目标类别进行训练。

Prompt 模板很重要。CLIP 原论文每个类别使用 80 个模板（普通、艺术、照片、绘画等）并对 embedding 做平均。+3 个 ImageNet 分数。现代用法通常只选一两个模板。

### 线性探测与微调

零样本是一个基准。在冻结的 CLIP 特征上训练一个线性层（目标类别的线性探测）在域内任务上优于零样本。全量微调在域内优于线性探测，但可能损害零样本迁移能力。三个 regime，三种权衡。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2（2025）新增：
- NaFlex：单个模型处理可变的宽高比和分辨率。
- 更好的分割和深度估计密集特征，作为 VLM 中冻结骨干网的目标用途。
- 多语言：在 100 多种语言上训练，而 CLIP 仅限英语。
- 10 亿参数规模，CLIP 最高只有 4 亿。

在 2026 年开源 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。CLIP 仍是纯图文检索的默认选择，前提是 LAION-2B 的特定训练分布与你的查询模式匹配。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同思路，18 亿对，90% 嘈杂。证明了嘈杂数据可以扩展。OpenCLIP（LAION）：在 LAION-400M/2B 上开源复现 CLIP，多种规模，是首选开源 checkpoint。EVA-CLIP：从掩码图像建模初始化；VLM 的强骨干。BASIC：Google 的 CLIP+ALIGN 混合。同一家族，不同数据和调优。

### 零样本天花板

CLIP 类模型在 ImageNet 零样本上约 76% 封顶（CLIP-G、OpenCLIP-G）。超越需要更大数据（SigLIP 2 达到 80%+）或架构改动（监督头、更多参数）。基准在饱和；真正有价值的是下游 VLM 消费的那个嵌入空间。

## 使用方法

`code/main.py` 实现：

1. 一个玩具双编码器（基于哈希的图像特征，文本字符特征），让你不用 numpy 也能看到 InfoNCE 的形状。
2. 纯 Python 中的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定）。
3. 用于对比的 sigmoid 成对损失。
4. 零样本分类例程：计算与一组文本 prompt 的余弦相似度，argmax 得到预测。

运行它，观察损失曲线。绝对数值是玩具级的，但形状与真实 CLIP 训练器输出一致。

## 输出作品

本节生成 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，用 CLIP 模板构建文本 prompt，用指定 checkpoint（如 `openai/clip-vit-large-patch14`）嵌入两端，返回 top-1 / top-5 预测及相似度分数。该技能拒绝对不在 prompt 列表中的类别做出预测。

## 练习

1. 用手算实现 4 对的 InfoNCE。构造 4x4 相似度矩阵，跑 softmax，取对角线，计算交叉熵。用手算验证你的 Python 实现。

2. SigLIP 在温度之外还使用偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 有严重的类别不平衡（每行正样本远少于负样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为猫狗构建零样本分类器。尝试两个 prompt 模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测精度。模板集成是否优于单一模板？

4. 计算 512 GPU、batch 32k 下 softmax InfoNCE 与 sigmoid 成对损失的通信成本。哪个是 O(N)，哪个是 O(N²)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 扩展定律论文（arXiv:2212.07143，Cherti 等）。从图表复现其结论：在固定模型规模下，ImageNet 零样本准确率与训练数据规模之间的对数线性关系是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| InfoNCE | "对比损失" | 批次相似度矩阵上的交叉熵；每项的正样本是其配对项，负样本是其他所有项 |
| Sigmoid 损失 | "SigLIP 损失" | 逐对的二元交叉熵；无需 softmax，无需全局汇聚，分布式训练成本低 |
| Temperature | "tau" | 在 softmax/sigmoid 前缩放 logit 的标量；控制分布的锐度 |
| Zero-shot | "无需微调的分类" | 用文本 prompt 构造类别 embedding，通过余弦相似度分类；无需针对目标类别训练 |
| Prompt template | "a photo of a ..." | 围绕类别名的文本框架；影响零样本精度 1-5 个百分点 |
| Dual encoder | "双塔" | 一个图像编码器 + 一个文本编码器，输出到共享的 D 维空间 |
| Hard negative | "困难干扰项" | 与正样本足够相似的负样本，使模型必须努力区分 |
| Linear probe | "冻结 + 一层" | 只在冻结特征上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力，能以任意宽高比和分辨率摄入图像，无需 resize |
| Temperature scaling | "对数参数化 tau" | CLIP 用 `log(1/tau)` 参数化 tau 使梯度行为合理；裁剪以防止崩溃到接近零的 tau |

## 延伸阅读

- [Radford 等 — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai 等 — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen 等 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia 等 — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用嘈杂网络数据规模化。
- [Cherti 等 — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 扩展定律。