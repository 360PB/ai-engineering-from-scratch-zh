# Transfusion：一个 Transformer 中的自回归文本 + 扩散图像

> Chameleon 和 Emu3 把赌注全押在离散 token 上。它们有效，但量化瓶颈可见——图像质量 plateau 在连续空间扩散模型以下。Transfusion（Meta，Zhou 等，2024 年 8 月）走另一条路：保持图像连续，完全去掉 VQ-VAE，用两个损失训练一个 transformer。文本 token 用下一 token 预测。图像 patches 用流匹配/扩散损失。两者目标优化相同权重。Stable Diffusion 3（MMDiT）的架构是一个近亲。本节解读 Transfusion 论文，构建一个玩具双损失训练器，并追踪让一个 transformer 同时做两件事的注意力掩码。

**类型：** Build
**语言：** Python（标准库，MNIST 规模玩具上的双损失训练器）
**前置知识：** Phase 12 · 11（Chameleon），Phase 8（Generative AI）
**时间：** 约 180 分钟

## 学习目标

- 连接一个运行两个损失（NTP 对文本 token，扩散 MSE 对图像 patches）的 transformer。
- 解释为什么图像 patches 上的双向注意力加上文本 token 上的因果注意力是正确的掩码选择。
- 在计算、质量和代码复杂度上比较 Transfusion 风格（连续图像，扩散损失）和 Chameleon 风格（离散图像，NTP）。
- 说出 MMDiT 的贡献：每个 block 的模态特定权重，联合注意力在残差流上。

## 问题背景

离散 vs 连续图像 token 的争论早于 LLM。连续表示（原始像素、VAE 潜变量）保留细节。离散 token（VQ 索引）契合 transformer 的原生词汇表，但在量化步骤丢失细节。

Chameleon / Emu3 走离散路线：一个损失，一个架构，但图像保真度受分词器质量限制。

扩散模型走连续路线：卓越的图像质量，但与 LLM 分开的模型，复杂的噪声调度工程，与文本生成没有干净集成。

Transfusion 问：能否两者兼得？保持图像连续，仍然训练一个模型，用两个损失缝进一个梯度步。

## 核心概念

### 双损失架构

单个仅解码器 transformer 处理包含以下内容的序列：

- 文本 token（离散的，来自 BPE 词汇表）。
- 图像 patches（连续的，16x16 像素块通过线性嵌入投影到隐藏 dim——与 ViT 编码器输入相同）。
- `<image>` 和 `</image>` 标签标记连续 patches 所在位置。

前向传播运行一次。损失对每个 token 选择两个头之一：

- 对于文本 token：词汇表 logit 头的标准交叉熵。
- 对于图像 patches：连续 patches 上的扩散损失——预测每个 patch 添加的噪声。

梯度通过共享 transformer 主体流动。两个损失同时改善共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——不能让文本 token 注意未来的文本，否则 teacher forcing 崩溃。但是图像 patches 代表一个快照；它们应该在同一图像块内相互双向注意。

掩码：

```
M[i, j] = 1 如果：
  (i 是文本且 j 是文本且 j <= i)   # 文本因果
  或 (i 是图像且 j 是图像且 same_image_block(i, j))   # 图像内双向
  或 (i 是文本且 j 是图像且 j < i_image_end)   # 文本注意前面的图像
  或 (i 是图像且 j 是文本且 j < i_image_start)   # 图像注意前面的文本
```

实现为训练和推理时的块三角掩码。

### Transformer 内的扩散损失

扩散损失是标准的：向图像 patch 添加噪声，要求模型预测噪声（或等效地预测干净 patch）。Transfusion 版本使用流匹配——预测从噪声到干净数据的 velocity 场。

训练期间：
1. 对每个图像 patch x0，采样随机 timestep t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列的文本 NTP 损失一起反向传播。

推理时生成：
- 文本 token：标准自回归采样。
- 图像 patches：扩散采样循环（典型 10-30 步），以先前的文本 token 为条件。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等，2024 年 3 月）在 Transfusion 同期发布了 MMDiT（多模态扩散 Transformer）。架构是兄弟。

MMDiT 的关键区别：

- 每个 block 的模态特定权重。每个 transformer block 对文本 token vs 图像 patches 有单独的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其他都是模态特定的。
- 整流流训练。一种特定的流匹配变体，采样和数学比 DDPM 更简单。
- 规模。MMDiT 是 SD3（20 亿和 80 亿参数变体）的骨干。Transfusion 论文扩展到 70 亿。

两者收敛到相同的核心思想：一个 transformer 对文本运行 NTP，对连续图像表示运行扩散。

### 为什么这胜过 Chameleon 风格

连续-扩散和离散-NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 在 70 亿参数下，在 FID 上击败相同规模的 Chameleon 风格模型 3-5 分。
- 不需要分词器训练——图像编码器更简单（线性投影到隐藏，与 ViT 输入层相同）。
- 推理时可以并行化图像 patch 去噪，不像自回归图像 token。

缺点：Transfusion 是双损失模型，训练动态更棘手。损失权重需要调优。NTP 和扩散之间的调度不匹配可能导致一个头占主导。

### 下游是什么

Janus-Pro（第 12.15 节）通过解耦理解用和生成用的视觉编码器——一个用 SigLIP，一个用 VQ——同时共享 transformer 主体，来提炼 Transfusion 的想法。Show-o（第 12.14 节）将扩散换成离散扩散（掩码预测）。统一生成家族在 Transfusion 之后迅速分支。

2026 年发出图像的生产 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定使用这个家族的某个后代。细节是专有的。

## 使用方法

`code/main.py` 在一个小型类 MNIST 问题上构建玩具 Transfusion：

- 文本标题是描述数字（0-9）的短整数序列。
- 图像是 4x4 字节网格。
- 一对共享权重线性投影作为 transformer 替代品；文本上 NTP 损失，有噪 patches 上 MSE 损失。
- 训练循环交替两个损失，注意力掩码是显式的。
- 生成在一个前向传播中产生文本标题和 4x4 图像。

Transformer 是玩具。双损失的管道、注意力掩码构建和推理循环才是真正的产物。

## 输出作品

本节生成 `outputs/skill-two-loss-trainer-designer.md`。给定新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），设计双损失调度（损失权重、掩码形状、共享 vs 模态特定块）并标记实现风险。

## 练习

1. Transfusion 风格模型训练 70% 文本 token 和 30% 图像 patches。图像扩散损失在幅度上约是文本 NTP 损失的 10 倍。什么损失权重使它们平衡？

2. 实现 `[T, T, <image>, P, P, P, P, </image>, T]` 序列的块三角掩码。标记每个条目 0 或 1。

3. MMDiT 有模态特定的 QKV 权重。这比 Transfusion 的全共享 transformer 增加多少参数量？在 70 亿参数下，值得吗？

4. 生成：给定文本 prompt，模型运行 50 token NTP，然后碰到 `<image>`，然后在 256 patches 上运行 20 步扩散去噪。总共多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述整流流及为什么它比 DDPM 用更少推理步收敛。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Two-loss training | "NTP + 扩散" | 单个 transformer 在同一梯度步优化文本 token 上的交叉熵和连续图像 patches 上的 MSE |
| Flow matching | "整流流" | 预测从噪声到干净数据的 velocity 场的扩散变体；比 DDPM 数学更简单 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 的架构：联合注意力，模态特定的 MLP 和 norm |
| Block-triangular mask | "因果文本 + 双向图像" | 注意力掩码，在文本上因果但在图像区域内双向 |
| Continuous image representation | "无 VQ" | 图像 patches 作为实值向量，而非整数 codebook 索引 |
| Velocity prediction | "v 参数化" | 网络输出是噪声和数据之间的 velocity 场，而非噪声本身 |

## 延伸阅读

- [Zhou 等 — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等 — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等 — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)