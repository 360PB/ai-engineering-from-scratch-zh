# 开源权重 VLM 配方：什么真正重要

> 2024-2026 年的开源权重 VLM 文献是一片消融表森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合组合。Allen AI 的 Molmo 证明了详细人工标题胜过 GPT-4V 蒸馏。Cambrian-1 跑了 20+ 种编码器比较。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上比较了 27 种训练配方。在所有这些噪声中，一小组结果在论文间一致成立：图像编码器比连接器架构重要，数据混合比前两者都重要，详细人工标题胜过蒸馏合成数据。本节解读这些表格，让你不必自己去读。

**类型：** Learn + lab
**语言：** Python（标准库，消融表解析器 + 配方选择器）
**前置知识：** Phase 12 · 05（LLaVA 基线）
**时间：** 约 180 分钟

## 学习目标

- 说出五轴 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 消融表并预测哪个旋钮移动给定基准。
- 给定计算预算和任务组合，为新 VLM 选配方（编码器、连接器、数据、分辨率）。
- 解释为什么详细人工标题在相同 token 数量下优于 GPT-4V 蒸馏。

## 问题背景

数百个开源权重 VLM 存在。"好"和"最先进的"之间的差距大部分不在架构。差距在数据、分辨率调度和编码器选择。当模型表现不佳时，知道先转哪个旋钮可以让你避免 500 万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）在标题对预训练 + LLaVA-Instruct-150k 上运行。良好基线。在 MMMU 上约 35% 封顶。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）跑了穷举消融。结果出人意料且实用。

## 核心概念

### 五轴设计空间

Idefics2（Laurençon 等，2024）命名了这些轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上不同。
2. 连接器。MLP（2-4 层）、Q-Former（32 个 query + 交叉注意力）、Perceiver Resampler（64 个 query）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B/70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主要的参数成本。
4. 训练数据。标题对（CC3M、LAION）、交错的（OBELICS、MMC4）、指令（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度。固定 224/336/448、AnyRes、原生动态。在训练中 ramp 或保持恒定。

每个生产 VLM 在每个轴上都有选择。MMMU 分数的大部分方差由轴 1、4 和 5 解释——不是你选了哪个连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节显示：从 CLIP ViT-L/14 切换到 SigLIP SO400m/14 增加 3+ MMMU 分。从 MLP 切换连接器到 Perceiver Resampler 增加不到 1 分。Idefics2 复现：SigLIP > CLIP，Q-Former ≈ MLP ≈ Perceiver 在相同 token 数量下。

Cambrian-1 的"Cambrian 视觉编码器对决"（Tong 等，2024）在视觉中心基准（CV-Bench）上跑了 20+ 个编码器。排行榜顶端是 DINOv2 和 SigLIP 的混合；CLIP 在中间；ImageBind 和 ViT-MAE 较低。从 CLIP ViT-L 到 DINOv2 ViT-g/14 在 CV-Bench 上差距约 5-7 分。

2026 年开源 VLM 的默认编码器是 SigLIP 2 SO400m/14（语义 + 密集特征），有时与 DINOv2 ViT-g/14 特征拼接（Cambrian 的"Spatial Vision Aggregator"）。

### 轴 2：连接器设计是 wash

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出相同结论：在固定视觉 token 数量下，连接器架构几乎不重要。在 mean-pooled patches 上运行 2 层 MLP 与在相同 token 预算下运行 32-query Q-Former 性能相差在 1 分以内。

重要的是 token 数量。更多视觉 token = 更多 LLM 计算 = 更好的性能，到某点后边际递减。64 token 每图像对 OCR 太少。576-1024 token 是大多数开源 VLM 的最佳点。2048+ 只对文档和图表有帮助。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 将 token 上限限制在 32-64，不受图像分辨率影响；MLP 发出所有 patch token。对于高分辨率输入，Q-Former 节省 LLM 上下文；对于低分辨率，差异是噪声。

### 轴 3：LLM 大小设定上限

LLM 从 7B 加倍到 13B 在每篇 VLM 论文中可靠地增加 2-4 分 MMMU。在 70B 时，大多数基准饱和。VLM 的多模态推理上限是 LLM 的文本推理上限——视觉编码器只能喂它，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上碾压对手：语言脑很大。一个 7B VLM 不能通过精巧的连接器设计替代 70B VLM。

### 轴 4：数据——详细人工标题优于蒸馏

Molmo + PixMo（Deitke 等，2024）是 2024 年每个人都应该读的结果。Allen AI 让人类标注员在 1-3 分钟的密集语音转录中对图像进行描述，产生 71.2 万张密集标题图像。没有 GPT-4V 蒸馏在任何训练数据中。

Molmo-72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。差距不在架构——是标题质量。详细人工标题每张图像包含的信息是网页 alt 文本的 5-10 倍，且保持事实正确，而 GPT-4V 蒸馏会产生幻觉。

ShareGPT4V（Chen 等，2023）和 Cauldron（Idefics2）用人工 + GPT-4V 混合标题遵循了相同 playbook。趋势清晰：对于 2026 年的前沿，标题密度 > 标题数量 > 蒸馏便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融：384 → 448 增加 1-2 分。用图像分割从 448 → 980（AnyRes）在 OCR 基准上再加 3-5 分。平面分辨率训练在中等精度 plateau；分辨率 ramp（从 224 开始，到 448 或原生）训练更快且最终更高。

Cambrian-1 跑了分辨率 vs token 权衡：在固定计算量下，你可以选择低分辨率更多 token 或高分辨率更少 token。对 OCR，高分辨率赢；对通用场景理解，低分辨率更多 token 赢。

2026 年生产配方：Stage 1 在 384 固定训练，Stage 2 动态分辨率高达 1280 用于 OCR 繁重任务。

### Prismatic 受控比较

Prismatic VLMs（Karamcheti 等，2024）是控制所有轴的论文。相同 13B LLM、相同指令数据、相同评估——一次只改变一个轴。结果：

- 每图像视觉 token 数量解释约 60% 的方差。
- 编码器选择解释约 20%。
- 连接器架构解释约 5%。
- 其他一切（数据混合、调度器、LR）剩余约 15%。

这是一个粗略的分解，但这是文献中对"我应该先消融哪个"最清晰的回答。

### 2026 年配方选择器

根据证据，2026 年新项目的默认开源 VLM 配方：

- 编码器：SigLIP 2 SO400m/14 原生分辨率加 NaFlex，需要分割/接地时与 DINOv2 ViT-g/14 拼接用于密集特征。
- 连接器：patch token 上的 2 层 MLP。除非你在 token 受限，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，成本用 7B，质量用 70B，按目标延迟选择。
- 数据：PixMo + ShareGPT4V + Cauldron，外加任务特定指令数据补充。
- 分辨率：动态（长边每边 256-1280 像素）。
- 调度：Stage 1 对齐（仅投影器），Stage 2 全量微调，Stage 3 任务特定微调。

每个默认值都可以追溯到本节末尾引用的论文中经过测量的消融。

## 使用方法

`code/main.py` 是一个消融表解析器和配方选择器。它编码 MM1 和 Idefics2 消融表（精简版），让你查询：

- "给定预算 X 和任务 Y，哪个配方赢？"
- "如果我在 7B Llama 上将 SigLIP 换成 CLIP，预期 MMMU delta 是多少？"
- "哪个轴应该首先消融以获得 80% 置信度答案？"

输出是带预期基准 delta 和"首先消融"推荐的排名配方列表。

## 输出作品

本节生成 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、计算预算和延迟目标，发出完整配方（编码器、连接器、LLM、数据混合、分辨率调度），附引用的消融以证明每个选择。让工程师不必在每个新 VLM 项目开始时重新发明 Idefics2 消融表。

## 练习

1. 阅读 MM1 第 3.2 节。在固定 2B LLM、5000 万图像预算下，哪个编码器赢？在 13B LLM 下答案会翻转吗？为什么？

2. Cambrian-1 发现拼接 DINOv2 + SigLIP 在视觉中心基准上优于单独任一个，但在 MMMU 上没有增加信号。预测哪些基准增加，哪些持平。

3. 目标是移动 UI 代理，用 2B LLM。选编码器、连接器、分辨率和数据混合。用特定消融表为每个选择提供理由。

4. Molmo 发布 4B 和 72B 模型。4B 与封闭的 7B VLM 竞争；72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。这告诉你关于 LLM 大小 plateau 假说的什么？

5. 设计一个消融表，从 7B VLM 上的数据混合质量和编码器质量中隔离出来。最少需要多少次训练运行？提议四个轴的设置。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Ablation | "转一个旋钮" | 训练多次运行，只在恰好一个设计空间轴上不同，其他全部保持恒定 |
| Connector | "桥" / "投影器" | 将视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption | "密集标题" | 人类撰写的多句描述（通常 80-300 token），比网页 alt 文本丰富得多 |
| Distillation | "GPT-4V 标题" | 由更强的专有 VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / dynamic res | "高分辨率路径" | 通过平铺或 M-RoPE 将大于编码器原生分辨率的图像送入的策略 |
| Resolution ramp | "课程" | 从低分辨率开始增加的训练调度，加速对齐学习 |
| Vision-centric bench | "CV-Bench / BLINK" | 强调细粒度视觉感知而非语言重推理的评估 |
| PixMo | "Molmo 的数据" | Allen AI 的 71.2 万张密集标题图像数据集；人类语音转录为密集标题 |

## 延伸阅读

- [McKinzie 等 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等 — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等 — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)