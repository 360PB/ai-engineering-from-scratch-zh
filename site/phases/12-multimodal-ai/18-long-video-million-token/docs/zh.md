# 百万 Token 上下文的长视频理解

> 24 FPS 的 1 小时 4K 视频，打 patch 并 embedding，产生约 6000 万 token。2 小时播客转录是 30,000 token。即使以激进池化压缩，一部完整蓝光电影也是数十万 token。Google 的 Gemini 1.5（2024 年 3 月）开启了这个时代，1000 万 token 上下文，在小时级视频上进行可靠的海底捞针召回。LWM（Liu 等，2024 年 2 月）展示了 ring attention 的扩展路径。LongVILA 和 Video-XL 进一步规模化 ingestion。VideoAgent 用 agentic 检索换原始上下文。每个方法是计算、召回和工程复杂性上的不同权衡。本节并列解读它们。

**类型：** Build
**语言：** Python（标准库，海底捞针模拟器 + agentic 检索路由器）
**前置知识：** Phase 12 · 17（视频时间 token）
**时间：** 约 180 分钟

## 学习目标

- 计算不同 FPS 和池化下长篇视频的总视觉 token 数。
- 解释三条扩展路径：暴力上下文（Gemini 1.5）、ring attention（LWM）、token 压缩（LongVILA / Video-XL）。
- 在准确率和延迟上比较原始上下文视频 VLM 与 agentic 检索视频 VLM（VideoAgent）。
- 为 30 分钟视频设计海底捞针测试并在特定分钟测量召回率。

## 问题背景

单帧 Qwen2.5-VL 大小 patches 在 384 原生分辨率下约 729 token。3x3 池化后每帧 81 token。30 分钟片段在 1 FPS 下 = 1800 帧 = 145,800 token。2025 年开源 VLM 可行，紧。在 2 FPS 下 291,600 token——只有最大的上下文能容纳。

2 小时电影在 1 FPS 下是 583k token。超出大多数 2026 年开源模型；需要更激进池化。

出现了三条扩展路径。

## 核心概念

### 路径 1：暴力上下文（Gemini 1.5，Claude Opus）

用硬件解决这个问题。将上下文扩展到数百万 token，一次前向传播处理一切。

Gemini 1.5 Pro 以 100 万 token 启动；Gemini 1.5 Ultra 到 1000 万；Gemini 2.5 Pro 在 2026 年可靠处理小时级视频。论文（arXiv:2403.05530）记录了高达约 950 万 token 的海底捞针召回率 99.7%。

工程：带内存层次（局部 + 全局 + 稀疏）加上 MoE 专家路由的自定义注意力实现，用于长上下文效率。细节未完全公开。不是开源的。

### 路径 2：Ring attention（LWM，LongVILA）

Ring attention 在"环形"中将长序列分布到设备上，每个设备持有一个块。全序列上的注意力通过每个设备将其块发送到下一个设备，计算部分注意力，聚合。

LWM（Liu 等，2024）用这种方式训练 100 万 token 上下文模型。训练计算随上下文线性扩展，不二次扩展——注意力的二次冲击在环形设备间摊销。

LongVILA（arXiv:2408.10188）将模式适配到 VLM。1400 帧视频每帧 192 token = 268k 上下文，在 8 路并行上用 ring attention 训练。

### 路径 3：Token 压缩（Video-XL，LongVA）

比暴力上下文便宜：在 LLM 看到序列前激进压缩。

Video-XL（arXiv:2409.14485）使用视觉摘要 token：每 N 帧片段产生一个"摘要" token，该 token 关注这 N 帧。推理时，LLM 每个片段看到一个摘要 token，大幅缩小上下文。

LongVA 通过"长上下文迁移"技术将 LLM 上下文从 200k 扩展到 200 万。在长上下文文本上训练，迁移到长上下文视频通过共享表示。

Token 压缩在特定时间戳召回上权衡可扩展性。模型大致知道发生了什么，但有时错过确切帧。

### 路径 4：Agentic 检索（VideoAgent）

不要将完整视频喂给 LLM。相反，将视频视为数据库并用 LLM 查询。

VideoAgent（arXiv:2403.10517）：
1. LLM 读取问题。
2. LLM 向检索工具询问相关片段（"显示有猫的段"）。
3. 工具返回匹配的片段时间戳。
4. LLM 通过 VLM 读取那些片段。
5. LLM 组合答案或提出后续查询。

这是 LLM-as-agent 模式应用于长视频。更便宜的推理（只编码相关片段），更难的工程（检索质量成为瓶颈）。

### 海底捞针基准

标准长上下文测试：在视频随机点插入唯一视觉或文本标记，然后问一个需要召回它的查询。

指标：跨视频长度和标记位置的 Recall@k。

Gemini 2.5 Pro 在长达 90 分钟视频上得分 >99% 召回。开源 72B 模型（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟约 85-90% 且在 60 分钟后退化。

如果工具好，VideoAgent 可以在 2+ 小时匹配或击败原始上下文模型。

### 选哪条路径

15 分钟片段前沿准确率：开源 72B + 原生上下文通常可行。选 Qwen2.5-VL-72B。

30 分钟到 1 小时内容：开源用 LongVILA 或 Video-XL；封闭用 Gemini 2.5 Pro。质量 bar 重要——前沿走封闭。

2+ 小时内容：VideoAgent 或类似检索模式。或者总结为更小的块并喂入分层摘要。

### 2026 年生产模式

实际上，生产长视频流水线是混合的：

1. 在整个视频上运行动态 FPS 采样 + 激进池化（获得 100k token 全局表示）。
2. 传给 72B VLM 获取全局摘要。
3. 如果用户问详细问题，使用摘要作为索引运行 agentic 检索。

这结合了全局理解的暴力上下文和局部细节的检索。

## 使用方法

`code/main.py`：

- 计算 1 分钟到 3 小时视频在不同 FPS + 池化下的 token 预算。
- 模拟海底捞针运行：在随机时间戳注入标记，问问题，评分召回。
- 包含 agentic 检索路由器模拟器，选择特定片段喂入下游 VLM。

运行预算表，感受规模差距。

## 输出作品

本节生成 `outputs/skill-long-video-strategy-planner.md`。给定视频持续时间和查询复杂度，在暴力上下文、压缩和 agentic 检索之间选择，并计算延迟 + 质量期望。

## 练习

1. 45 分钟讲座在 1 FPS、每帧 81 token。总 token 数？适合哪个模型的上下文？

2. 设计一个海底捞针测试：在第几分钟注入标记，确切的查询格式是什么？

3. 在 1 小时视频上比较暴力上下文 Qwen2.5-VL-72B（80k 上下文）与 VideoAgent（Claude 3.5 + 检索）。哪个在召回上赢？哪个在延迟上赢？

4. Ring attention 的内存成本随序列长度线性扩展且随设备数量线性扩展。解释为什么，如果在 drop ring-rotation 阶段会出什么故障？

5. 阅读 Gemini 1.5 第 5 节关于海底捞针。论文在 100 万 vs 1000 万 token 边界发现什么召回情况？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Brute context | "更多 token" | 将 LLM 上下文扩展到数百万 token；一次前向传播处理一切 |
| Ring attention | "LWM 风格并行" | 分布式注意力模式，每个设备持有一个块并轮转 |
| Token compression | "摘要 token" | 通过学习压缩器在 LLM 前减少每片段 token |
| Needle-in-haystack | "NIH 测试" | 在随机点插入唯一标记，在测试时要求模型召回 |
| Agentic retrieval | "LLM 作为查询规划器" | LLM 向检索工具询问相关片段，通过 VLM 读取，组合答案 |
| VideoAgent | "视频检索模式" | 规范 agentic 检索设计：问题 → 工具 → 片段 → 答案 |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu 等 — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue 等 — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu 等 — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang 等 — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)