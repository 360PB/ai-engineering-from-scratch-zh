# 视频-语言模型：时间 Token 与时间定位

> 视频不是一堆照片。5 秒片段有因果顺序、动作动词和事件时间，图像模型无法表示。Video-LLaMA（Zhang 等，2023 年 6 月）发布了首个带音视觉接地的开源视频-LLM。VideoChat 和 Video-LLaVA 规模化了这一模式。到 2025 年 Qwen2.5-VL 的 TMRoPE 缩小了与前沿专有模型的差距。每个系统在时间 token 上有不同解法——每片段 Q-former、逐帧 concat 池化、逐 token M-RoPE。本节解读这些模式，构建统一 vs 动态帧采样器，并在时间定位任务上评估。

**类型：** Build
**语言：** Python（标准库，帧采样器 + 时间定位评估器）
**前置知识：** Phase 12 · 08（LLaVA-OneVision）
**时间：** 约 180 分钟

## 学习目标

- 解释为什么时间位置编码独立于视觉编码器改变视频 VLM 性能。
- 在每分钟 token 数 vs 定位准确率上比较均匀、动态 FPS 和事件驱动帧采样。
- 描述每片段 Q-former（Video-LLaMA）vs 逐帧池化（Video-LLaVA）vs 逐 token M-RoPE（Qwen2.5-VL）设计。
- 说出四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题背景

30 FPS 的 1 分钟视频是 1800 帧。在每帧 196 视觉 token（ViT-B @ 224）下，这是 352k token——大于任何 2024 年时代 LLM 上下文。

三种缩减策略：

1. 子采样帧（1-8 FPS，取决于内容）。
2. 激进池化每帧的 patch token（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩，将 16 帧片段输出 64 token。

每种权衡不同。子采样丢失时间细节。池化丢失空间细节。Q-former 两者都丢失一点但节省 token。

时间位置编码是另一轴：模型如何知道帧 5 在帧 6 之前？选项包括简单 1D 时间 RoPE（Video-LLaMA）、学习时间 embedding（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，全 3D）。

## 核心概念

### Video-LLaMA：每片段 Q-former + 音频分支

Video-LLaMA（2023）是首个开源视频-LLM。架构：

- 16 帧片段在 2 FPS（8 秒）。
- 每帧 ViT 特征 → 视频 Q-former 交叉关注所有 16 帧 → 32 个学习 queries → LLM。
- 并行音频分支：波形 → ImageBind 音频编码器 → 音频 Q-former → 32 queries → LLM。

优势：音视觉联合推理。弱点：固定片段长度，无任意时间定位。

### VideoChat 和 Video-LLaVA

VideoChat 保留 Video-LLaMA 思路但去掉音频并简化。Video-LLaVA（Lin 等，2023）在图像和视频帧上训练单一视觉编码器（"投影前对齐"），给出统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

两者都不处理长视频。都是 8-16 帧系统。

### Qwen2.5-VL 和 TMRoPE

Qwen2.5-VL 引入 TMRoPE——时间-模态旋转位置 embedding。每个 patch token 携带 (t, h, w) 位置，其中 t 是实际时间戳（不是帧索引）。

与简单时间 embedding 的关键区别：

- 绝对时间，不是索引。模型看到"在 4.2 秒"而不是"在帧 15"。
- 逐 token 旋转，不是逐片段。每个视觉 token 按其时间戳独立旋转。
- 与动态 FPS 兼容。如果在这里以 2 FPS 采样，在那里以 4 FPS 采样，TMRoPE 原生地处理不均匀间隔。

TMRoPE 支持"猫在第几秒跳跃？"查询。模型可以输出"在 4.2 秒"。Video-LLaMA 只能说"在片段早期"。

### 帧采样策略

均匀：均匀采样 N 帧覆盖持续时间。简单，丢失运动峰值。

动态 FPS：基于运动强度自适应采样。光流或帧差分选择在更密集采样的高运动段。Qwen2.5-VL 在此基础上训练。

事件驱动：运行轻量检测器，在有动作的地方采样更多。VideoAgent 使用。

关键帧 + 上下文：在镜头边界采样 + 一些相邻帧。用于电影内容。

### 逐帧池化

在 1 FPS 和每帧 576 token 下，5 分钟片段是 172,800 token。用 Qwen2.5-VL-72B 的 128k 上下文可行但昂贵。

3x3 双线性池化缩减到每帧 64 token → 5 分钟 19,200 token。大多数任务的最佳点。

更激进池化（6x6 → 每帧 16 token）用于 agent 工作流，那里空间细节不太重要。

### 四个视频基准

- VideoMME：全面视频理解，短 + 中 + 长。
- TempCompass：细粒度时间推理，"之前" / "之后"问题。
- EgoSchema：长期第一人称视频。
- Video-MMMU：多学科多模态视频问题。

完整视频-VLM 评估需覆盖全部四个。它们强调不同轴——TempCompass 全是排序，EgoSchema 是 3+ 分钟推理，VideoMME 跨度持续时间。

### 时间定位输出格式

时间定位的输出格式：

- 自由文本："猫在约 4 秒处跳跃。" 容易解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练这个。
- 基于 token：特殊 `<time>4.1</time>` token 与答案交错。Qwen2.5-VL 内部格式。

基于 token 对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026 年最佳实践

2026 年视频 VLM：

- 编码器：SigLIP 2 带 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（1-4，取决于运动）加最大帧数上限。
- 逐帧池化：3x3 双线性。
- 输出：带时间 + 事件字段的结构化 JSON。
- 基准：VideoMME + TempCompass 用于通用；EgoSchema 用于长期。

## 使用方法

`code/main.py` 包括：

- 均匀和动态 FPS 帧采样器。
- 玩具时间定位评估器：给定时间 T 的"真实"事件和模型输出，以容差评分准确率。
- 跨 Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）的比较。

## 输出作品

本节生成 `outputs/skill-video-vlm-frame-planner.md`。给定视频任务（监控、动作识别、时间定位、摘要），选择帧采样器、池化因子、输出格式和预期准确率层级。

## 练习

1. 对于 3 分钟烹饪演示，选均匀 vs 动态 FPS。用 token 计数提供理由。

2. TMRoPE 具体添加了什么简单时间 embedding 表无法做到的事？

3. 为时间定位编写一个 VLM 可以学习发出的 JSON schema。包含错误情况。

4. 阅读 Video-LLaVA 第 3 节关于"投影前对齐"。为什么这比训练单独的图像和视频编码器更好？

5. 给定 VideoMME 排行榜，2026 年顶级开源模型与顶级专有模型之间的差距是什么？该差距有多少归因于时间编码 vs 基础 LLM 规模？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Temporal grounding | "时间定位答案" | VLM 输出事件发生的特定时间戳范围 |
| TMRoPE | "时间-多模态 RoPE" | 带绝对时间戳的 3D 旋转位置，Qwen2.5-VL 使用 |
| Dynamic FPS | "运动感知采样" | 在高运动段采样更多帧，在静态段更少 |
| Frame pooling | "逐帧空间压缩" | 在送入 LLM 前用双线性插值减少每帧 patches |
| Video Q-former | "片段压缩器" | 将 N 帧映射到 K 个学习 queries 的交叉注意力瓶颈 |
| VideoMME | "视频基准" | 全面短/中/长视频基准，2500+ 样本 |

## 延伸阅读

- [Zhang 等 — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li 等 — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin 等 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin 等 — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)