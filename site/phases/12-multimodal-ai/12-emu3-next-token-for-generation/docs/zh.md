# Emu3：图像和视频生成的下一 Token 预测

> BAAI 的 Emu3（Wang 等，2024 年 9 月）是 2024 年本应结束扩散与自回归之争的结果。一个单一的类 Llama 仅解码器 transformer，仅用下一 token 预测目标，在文本 + VQ 图像 token + 3D VQ 视频 token 的统一词汇表上训练，在图像生成上击败 SDXL，在感知上匹配 LLaVA-1.6。没有 CLIP 损失。没有扩散调度。推理时使用无分类器引导提高质量，但核心训练目标是带 teacher forcing 的下一 token 预测。发表于 Nature。本节解读 Emu3 论文——为什么更好的分词器加规模化就是你需要的全部——并与扩散方法对比。

**类型：** Learn
**语言：** Python（标准库，3D 视频分词器数学 + 自回归采样器骨架）
**前置知识：** Phase 12 · 11（Chameleon）
**时间：** 约 120 分钟

## 学习目标

- 解释为什么 Emu3 的单一损失下一 token 目标有效，尽管人们长期以来认为图像质量需要扩散。
- 描述 3D 视频分词器：时空 VQ codebook 是什么样子的，为什么 patches 跨越时间。
- 比较 Emu3 与 Stable Diffusion XL 在（训练计算、推理成本、质量上限）。
- 说出同一 Emu3 模型扮演的三个角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题背景

直到 2024 年的 conventional wisdom：图像生成需要扩散。论点：离散图像 token 丢失太多信息无法重建细节，自回归采样在数千 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的扩散。Chameleon（第 12.11 节）在小规模上部分证伪了这一点，但质量上没有匹配 SDXL。

Emu3 正面攻击这个论点。主张：更好的视觉分词器 + 足够规模 + 下一 token 损失 = 在同一个也能做感知的模型中击败扩散的图像生成。

论文发表时赌注有争议。两年过去，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）是研究的默认路径；生产前沿模型似乎使用某种变体。

## 核心概念

### Emu3 分词器

关键成分是视觉分词器。Emu3 在 8x8 分辨率缩减下训练自定义 IBQ 类分词器（逆向瓶颈量化器，SBER-MoVQGAN 族）。512x512 图像在 codebook 大小 32768 下变为 64x64 = 4096 token。

这比 Chameleon 的每 512x512 1024 token（K=8192）更大，但每 token 更便宜（更小的 codebook 查找，更简单的编解码器）。关键指标：重建 PSNR 30.5 dB，与 Stable Diffusion 连续潜空间 32 dB 持平。

对于视频：3D VQ 分词器将时空 patch（4x4x4 像素）编码为一个整数。4 秒片段在 8 FPS 有 32 帧；在 256x256 和 4x 空间缩减及 4x 时间缩减下，token 数是 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768。

分词器质量是上限。Emu3 的贡献部分是"我们训练了一个非常好的分词器"。

### 单一损失训练

Emu3 用一个目标：在文本 token、2D 图像 token 和 3D 视频 token 的共享词汇表上进行下一 token 预测。训练期间权重乘以模态特定因子以平衡贡献，但损失函数相同。

在以下混合数据上训练：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布中学习何时发出图像 token vs 文本 token。生成从模型在 `<image>` 标签后预测图像 token 而涌现。

### 无分类器引导和温度

自回归图像生成在推理时配合无分类器引导（CFG）效果好很多。Emu3 使用它：生成两次，一次用完整标题，一次用空标题，用引导权重混合 logit（典型 3.0-7.0）。这与扩散使用的 CFG 技巧相同，借用到自回归设置。

温度重要：太高，有伪影；太低，模式崩溃。Emu3 推荐感知用温度 1.0，图像生成用 0.8。

### 三个角色，一个模型

Emu3 作为三个功能不同的 API 发货，但底层权重集是一个：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和标题生成。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

无任务特定 head。只是不同的 prompt 模板。相同 checkpoint。

### 基准

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上击败 SDXL（5.4 vs 5.6），GenEval 总体（0.54 vs 0.55——统计持平），Deep-Eval 复合指标相当。
- 图像感知：在 VQAv2 上击败 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致匹配。
- 视频生成：4 秒片段质量与 Sora 时代公开基准模型竞争 FVD。

数字不总是赢——Emu3 在这里丢一分，在那里得一分——但"下一 token 预测就是你需要的全部"的主张在各模态上是站得住脚的。

### 计算成本

Emu3 在约 3000 亿多模态 token 上用 70 亿参数模型训练。GPU 小时大致与 Llama-2-7B 预训练相当（A100 类芯片上 2000-4000 GPU 年）。Stable Diffusion 3 等扩散模型训练预算相似，但需要单独的文本编码器和更复杂的流水线。

推理时，Emu3 比 SDXL 每张图像慢：每 512x512 图像 4096 token 在 30 tok/s 下约 2 分钟，而 SDXL 是 2-5 秒。投机解码和 KV 缓存优化缩小差距但不关闭。自回归图像生成是计算密集型的；这是持续的权衡。

### 为什么重要

Emu3 的深层贡献是概念性的。如果下一 token 预测能在图像生成上扩展到与扩散持平，统一模型路径（一个损失、一个骨干、任意模态）是可行的。未来的模型不需要单独的文本编码器、单独的扩散调度器、单独的 VAE。一个 transformer，每模态一个分词器，规模化。

Show-o、Janus-Pro 和 InternVL-U 都建立在或挑战这个论文上。到 2025 年，中国实验室（BAAI、DeepSeek）比美国实验室在这一方向上更积极地发表。

## 使用方法

`code/main.py` 构建两个玩具组件：

- 2D vs 3D VQ 分词器计数计算器：给定（分辨率、patch、片段长度、FPS），计算图像 vs 视频的 token 数。
- 带无分类器引导的自回归图像 token 采样器。

CFG 实现与 Emu3 配方匹配——用引导权重混合条件和非条件 logit。

## 输出作品

本节生成 `outputs/skill-token-gen-cost-analyzer.md`。给定生成产品规格（图像或视频、目标分辨率、质量层级、延迟预算），计算 token 数、推理成本，并在 Emu3 家族和扩散之间选择。

## 练习

1. Emu3 每 512x512 图像产生 4096 token，8x8 缩减。计算 1024x1024 和 2048x2048 的等效值。推理延迟会发生什么？

2. 阅读 Emu3 第 3.3 节关于视频分词器。描述 3D VQ patch 形状，为什么是 4x4x4 而不是 8x8x1。

3. 无分类器引导权重 5.0 vs 3.0：什么视觉效果？追踪 `code/main.py` 中的数学。

4. 计算 Emu3-7B 在 300B token 上的训练 FLOPs 并与 Stable Diffusion 3 比较。哪个训练更贵？

5. Emu3 在 FID 上击败 SDXL，但在 VQAv2 上未击败专门的 VLM。解释为什么统一损失方法在不同基准上对不同专家表现不同。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Next-token prediction | "NTP" | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；当 token 化后适用于所有模态 |
| IBQ tokenizer | "逆向瓶颈量化器" | 一类 VQ-VAE，具有更大的 codebook（32768+）和比 Chameleon 更好的重建 |
| 3D VQ | "时空量化器" | 由（时间、行、列）索引的 codebook；一个 token 覆盖 4x4x4 像素立方体 |
| Classifier-free guidance | "CFG" | 用权重 gamma 混合条件和非条件 logit；在推理时提高图像质量 |
| Unified vocabulary | "共享 token" | 文本 + 图像 + 视频都从相同的整数空间抽取；模型预测下一个出现的任何模态 |
| MJHQ-30K | "图像生成基准" | Midjourney 质量基准，30k prompts；Emu3 在此报告 FID |

## 延伸阅读

- [Wang 等 — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun 等 — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu 等 — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu 等 — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian 等 — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)