# Chameleon 与早期融合仅 Token 多模态模型

> 到目前为止我们看到的所有 VLM 都将图像和文本分开。视觉 token 来自视觉编码器，流经投影器，然后在 LLM 内部与文本相遇。视觉和文本词汇表从不重叠。Chameleon（Meta，2024 年 5 月）问：如果重叠呢？训练一个 VQ-VAE，将图像转换为来自共享词汇表的离散 token 序列。现在每个多模态文档都是一个序列——文本 token 和图像 token 交错，单一自回归损失。附带效果：模型可以生成混合模态输出——在单次推理调用中交替输出文本和图像 token。本节解读早期融合论文并从头构建一个玩具版本。

**类型：** Build
**语言：** Python（标准库，VQ-VAE 分词器 + 交错解码器）
**前置知识：** Phase 12 · 05，Phase 8（Generative AI）
**时间：** 约 180 分钟

## 学习目标

- 解释为什么共享词汇表 + 单一损失改变了模型能做什么。
- 描述 VQ-VAE 如何将图像分词为与 transformer 下一 token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置、LayerNorm 顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，并描述何时选择哪个。

## 问题背景

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两件不同的事。文本 token 通过 `embed(text_token)`；图像通过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中途合并。

三个后果：

1. LLM 只能消费图像，不能输出。输出只能是文本。
2. 混合模态文档（文章中交替的段落和图像）很别扭——你必须在模型外部解析多模态输入或链式生成。
3. 分布不匹配。视觉 token 和文本 token 活在隐藏空间的不同区域，产生微妙的对齐问题。

Chameleon 拒绝这个前提：图像只是来自共享词汇表的离散 token 序列。在交错的文档上训练，一个损失，一个自回归解码器，你就免费解锁了混合模态生成。

## 核心概念

### VQ-VAE 作为图像分词器

分词器是一个向量量化变分自编码器。架构：

- 编码器：CNN + ViT，将图像映射到空间特征图，比如 32x32 个 dim 256 的特征。
- Codebook：K 个向量的学习词汇表（Chameleon 用 8192），维度也是 256。
- 量化：对每个空间特征，通过 L2 距离查找最近的 codebook 条目。用整数索引替换连续特征。
- 解码器：CNN，将量化特征带回像素。

训练：VAE 重建损失 + 承诺损失 + codebook 损失。Codebook 索引形成图像的离散字母表。

对于 Chameleon：一张图像变成 32*32 = 1024 个 token，来自大小为 8192 的词汇表。与文本 token（来自 LLM 的 BPE 词汇表，假设 32000）拼接。最终词汇表：40192。Transformer 看到一个序列，一个损失。

### 共享词汇表

Chameleon 的词汇表结合文本 token、图像 token 和模态分隔符。每个 token 有单一 ID。输入嵌入层将每个 ID 映射到 D 维隐藏向量。输出投影将隐藏向量映射回词汇表 logit。Softmax 选下一个 token，无论什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签包围图像 token 序列。在生成时，如果模型发出 `<image>`，下游软件知道接下来 1024 个 token 是 VQ 索引，发送给解码器渲染像素。

### 混合模态生成

推理是在共享词汇表中的下一 token 预测。示例 prompt："画一只猫并描述它。"Chameleon 发出：

```
<image> 4821 1029 2891 ... (1024 个图像 token) </image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序——可能先生成图像再文本，先文本再图像，或交错。同一个解码器，同一个损失。

对比基于适配器的 VLM，其生成只能是文本。Chameleon 重新开启了模型输出模态的问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 顺序

早期融合训练在大规模下不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在注意力内对 query 和 key 投影应用 LayerNorm，在点积之前。防止深度处的 logit 幅度爆炸。被多个 2024 年后的大型模型使用。
- Dropout 放置。在每个 residual-add 后加 dropout，而不只是注意力和 MLP 后。当图像 token 的梯度可能占主导时需要更多正则化。
- LayerNorm 顺序。主 residual 分支上 Pre-LN（标准），加最后一个块 skip 连接上的额外 LN。稳定最终层梯度流。

没有这些技巧，340 亿参数的 Chameleon 训练在多个 checkpoint 发散。有了它们，模型收敛。训练配方与架构一样是核心贡献。

### 分词器的重建上限

VQ-VAE 是有损的。在 8192 个 codebook 条目和每 512x512 图像 1024 token 下，重建 PSNR 上限约 26-28 dB。这对可识别的图像生成够用，但明显差于连续空间扩散（Stable Diffusion 3 达到 32+ dB）。

分词器是瓶颈。更好的分词器（MAGVIT-v2、IBQ、SBER-MoVQGAN）提升上限。Emu3（第 12.12 节）仅通过更好的分词器达到 SDXL 质量的生成。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词汇表）：
- 一个损失，一个解码器。
- 生成混合模态输出。
- 分词器是质量上限。
- 昂贵：每生成一张图像，推理路径上需要 VQ-VAE 解码器。

BLIP-2 / LLaVA（晚期融合，分离塔）：
- 视觉输入，文本输出。
- 重用预训练 LLM。
- 对理解没有分词器瓶颈。
- 便宜：单次前向传播。

按任务选择。需要图像生成选 Chameleon 家族。仅需要理解选基于适配器的 VLM，更简单且重用更多预训练计算。

### Fuyu 和 AnyGPT

Fuyu（Adept，2023）是相关方法：完全跳过单独的视觉编码器，将原始图像 patches 通过 LLM 的输入投影作为 token 输入，不需要分词器。比 Chameleon 更简单，失去共享词汇表输出生成。

AnyGPT（Zhan 等，2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每个都用相同的 VQ-VAE 技巧，共享 transformer。任意到任意生成。在第 12.16 节更详细介绍。

## 使用方法

`code/main.py` 构建一个玩具端到端早期融合模型：

- 一个小型 VQ-VAE 风格量化器，将 8x8 patches 映射到 codebook 索引（K=16）。
- 共享词汇表（文本 ids 0..31）+（图像 ids 32..47）+（分隔符 48, 49）。
- 一个玩具自回归解码器（二元语法表），在合成标题 + 图像 token 序列上训练。
- 采样循环，给定 prompt 时发出交错的文本 + 图像 token。

代码故意将 transformer 保持很小（二元语法），这样你可以从头到尾追踪信号流。

## 输出作品

本节生成 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅理解 vs 理解 + 生成、所需图像质量、成本预算），在 Chameleon 家族（早期融合）和 LLaVA 家族（晚期融合）之间选择，并用定量经验法则说明理由。

## 练习

1. Chameleon 使用 K=8192 codebook 条目和每 512x512 图像 1024 token。估算与 24 位 RGB 图像的压缩比。是有损的吗？损多少？

2. 4K 图像（3840x2160）以相同 VQ-VAE 密度产生多少图像 token？Chameleon 风格的模型能在一次推理调用中生成 4K 图像吗？最先坏的是什么——上下文、分词器质量还是 KV 缓存？

3. 用纯 Python 实现 QK-Norm。给定 64 维 query 和 key，显示 LayerNorm 前后的点积。为什么深度处的幅度控制重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性。在 340 亿参数下没有 QK-Norm 时描述论文观察到的确切失败模式。"范数爆炸"签名是什么？

5. 将玩具解码器扩展为在纯文本 prompt 下发出混合模态响应。测量模型在训练数据分布 60% 文本优先 / 40% 图像优先下选择图像优先 vs 文本优先的频率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Early fusion | "统一 token" | 图像转换为共享 transformer 词汇表的离散 token，从第一步起 |
| VQ-VAE | "图像分词器" | CNN + ViT + codebook，将图像映射到 transformer 可预测的整数索引 |
| Shared vocabulary | "一个字典" | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "注意力稳定器" | 在 query 和 key 的点积之前应用 LayerNorm，防止深度处范数爆炸 |
| Mixed-modality generation | "文本 + 图像输出" | 在单次传递中自主产生交错的文本和图像 token 的推理 |
| Codebook size | "K 条目" | VQ-VAE 可量化的离散向量数量；权衡压缩率和保真度 |
| Tokenizer ceiling | "重建上限" | 解码 VQ token 可达到的最佳 PSNR；限制模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan 等 — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu 等 — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan 等 — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)