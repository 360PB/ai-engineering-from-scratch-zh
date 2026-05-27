# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用 2 层 MLP 替换了 BLIP-2 的 Q-Former，用朴素的 token 拼接替换了 Flamingo 的门控交叉注意力，并在 GPT-4 生成的 158k 视觉指令轮次上训练（从纯文本标题生成）。2023 到 2026 年间任何构建 VLM 的从业者都构建了 LLaVA 的某种变体。LLaVA-1.5 添加了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 统一了单图像、多图像和视频的单一方案。本节解读该方案，实现投影器，并解释为什么"更简单赢了"。

**类型：** Build
**语言：** Python（标准库，投影器 + 指令模板构建器）
**前置知识：** Phase 12 · 02（CLIP），Phase 11（LLM Engineering — 指令微调）
**时间：** 约 180 分钟

## 学习目标

- 构建一个 2 层 MLP 投影器，将 ViT patch embedding（dim 1024）映射到 LLM 的 embedding dim（dim 4096）。
- 走读 LLaVA 两阶段方案：（1）在 55.8 万对标题对上做投影器对齐，（2）在 15.8 万 GPT-4 生成的指令轮次上做视觉指令微调。
- 构造一个 LLaVA 格式的 prompt，包含图像 token 占位符、系统 prompt 和用户/助手轮次。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上占优。

## 问题背景

BLIP-2 的 Q-Former（第 12.03 节）将图像压缩到 32 个 token。干净、高效、基准好。但有两个问题。

第一，Q-Former 是可训练的，但其损失不是最终任务。第一阶段训练 ITC+ITM+ITG。第二阶段训练 LM 损失。queries 学习某种中间表示，LLM 再将其解码。信息在瓶颈中丢失了。

第二，Q-Former 有 1.88 亿参数，在 LLaVA 2023 年的规模下必须与目标 LLM 共同设计。换 LLM，重训 Q-Former。换视觉编码器，重训。每个组合都是单独的研发项目。

LLaVA 的答案简单得令人尴尬：取 ViT 的 576 个 patch token，每个通过 2 层 MLP（`1024 → 4096 → 4096`），全部 576 个送入 LLM 的输入序列。无瓶颈。无第一阶段预训练的奇怪目标。只在直接 LM 损失上训练 MLP。

数据从哪来？LLaVA 的第二个洞见：用 GPT-4（纯文本）生成指令数据。将图像的 COCO 标题和边界框数据喂给 GPT-4，让它生成对话、描述和复杂推理问题。免费获得 15.8 万指令-响应轮次。无需人工标注。

结果：一个在 8 块 A100 上训练一天、在 MMMU 上击败 Flamingo、发布了社区可以扩展的开源 checkpoint 的 VLM。到 2023 年底，它催生了 50 多个分支。

## 核心概念

### 架构

LLaVA-1.5 @ 13B：
- 视觉编码器：CLIP ViT-L/14 @ 336（第一阶段冻结，第二阶段可选解冻）。
- 投影器：带 GELU 激活的 2 层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

图像 + 文本 prompt 的前向传播：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占据 LLM 上下文的 576 个 token。在 2048 上下文下，剩 1472 个 token 给文本。在 32k 上下文中，这只是舍入误差。

### 第一阶段：投影器对齐

冻结 ViT。冻结 LLM。只训练 2 层 MLP。数据集：55.8 万对图文（COCO）。损失：以上述投影 token 为条件的标题语言建模。

在 batch 128 的单个 epoch 中，几小时完成。投影器学习从 ViT 空间到 LLM 空间的映射。无任务特定监督。

### 第二阶段：视觉指令微调

解冻投影器（仍然可训练）。解冻 LLM（通常全量，有时 LoRA）。在 15.8 万视觉指令轮次上训练。

指令数据是关键。Liu 等人的生成方式：
1. 取一张 COCO 图像。
2. 提取文本描述（5 个人工标题 + 边界框列表）。
3. 用三个 prompt 模板发送给 GPT-4：
   - 对话："生成用户和助手之间关于这张图像的来回对话。"
   - 详细描述："给出图像的丰富详细描述。"
   - 复杂推理："问一个需要推理图像的问题，然后回答它。"
4. 解析 GPT-4 输出为（指令，响应）对。

这一切不直接接触图像——只有文本描述。GPT-4 幻觉出合理的图像内容。有噪声，但有效：15.8 万轮足以解锁对话能力。

### 为什么社区复制了它

- 无需调第一阶段特定损失。全部用 LM 损失。
- 投影器几小时训练完成，不是几天。
- LLM 可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重训投影器。
- 视觉指令数据流水线用 GPT-4，换新领域重生成成本低。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）添加：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的系统 prompt。
- 2048 → 32k 上下文。

LLaVA-NeXT（2024 年 1 月）添加：
- AnyRes：将高分辨率图像分成 2x2 或 1x3 的 336x336 裁剪网格，加一张全局低分辨率缩略图。每个裁剪成为 576 个 token；每张图像共约 2880 个视觉 token。OCR 和图表任务大幅提升。
- 更好的指令数据混合，用 ShareGPT4V（高质量 GPT-4V 标题）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 节深入讲解 OneVision。简短版本：相同投影器，但用涵盖单图像、多图像和视频的统一课程训练，一个模型共享视觉 token 预算。

### 与 Q-Former 的比较

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每图像视觉 token | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 1.88 亿 + LM | 4000 万 + LM |
| 第一阶段损失 | ITC+ITM+ITG | 纯 LM |
| LLM 热插拔 | 需要重训 | 最小重训可替换 |
| 多图像 | 麻烦 | 自然（拼接） |
| 视频 | 麻烦 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简单性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底，token 预算不再是约束（LLM 上下文扩展到 32k-128k+），简单性主导。

### Prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是占位符 token。在 token 化之前，替换为 N 个投影视觉 token。分词器看到比训练时略长的序列，但 LLM 处理这个新输入，因为第一阶段教会了它。

### 参数经济性

LLaVA-1.5-7B 分解：
- CLIP ViT-L/14 @ 336：3.03 亿（第一阶段冻结，第二阶段常不解冻）。
- 投影器（2x 线性）：约 2200 万可训练。
- Llama-7B：70 亿。
- 总计：73 亿参数。第二阶段可训练：完整 70 亿 + 2200 万投影器。

第二阶段训练成本：8 块 A100 约 20 小时。这是关键数字——一天、一个节点、可复现。这就是 LLaVA 传播的原因。

## 使用方法

`code/main.py` 实现：

1. 2 层 MLP 投影器（在玩具规模下 dim 16 → 32 → 32），纯 Python。
2. prompt 构建流水线：系统 prompt + `<image>` 替换为 N 个投影 token + 用户轮次 + 助手生成占位符。
3. 可视化工具：展示 576 token 视觉块在 LLM 上下文中占的比例（2k / 32k / 128k 的百分比）。

## 输出作品

本节生成 `outputs/skill-llava-vibes-eval.md`。给定 LLaVA 族 checkpoint，运行 10-prompt vibes-eval 套件（3 个标题、3 个 VQA、2 个推理、2 个拒绝），输出人类可读评分卡。不是基准；是冒烟测试，确认投影器和 LLM 连接良好。

## 练习

1. 计算 2 层 MLP 投影器在 `1024 → 4096 → 4096` 下的可训练参数量。带 GELU 和 bias，在 LLaVA-13B 中占比多少？

2. 为"拒绝"案例构造 LLaVA prompt——图像包含私人个体。写出预期的助手响应。为什么 LLaVA 应该零样本拒绝，需要什么训练数据来强化拒绝？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的视觉 token 数量。与 336x336 基础 576 token 比较。

4. LLaVA 第一阶段投影器用标题上的 LM 损失训练。如果跳过第一阶段直接进入第二阶段（视觉指令微调）会怎样？引用 Prismatic VLMs 消融实验（arXiv:2402.07865）的答案。

5. LLaVA-Instruct-150k 用 COCO 标题生成 GPT-4 指令。对于新领域（医学 X 光片、卫星图像），描述生成领域指令的四步数据流水线。每步可能出什么问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Projector | "MLP 桥" | 带 GELU 的 2 层 MLP，将 ViT dim 映射到 LLM dim |
| Image token | "<image> 占位符" | Prompt 标记，在推理前替换为 N 个投影视觉 token |
| Visual instruction tuning | "LLaVA 第二阶段" | 在 GPT-4 生成的（图像，指令，响应）三元组上训练 |
| Stage 1 alignment | "投影器预训练" | 冻结 ViT 和 LLM，在标题上用 LM 损失训练投影器 |
| AnyRes | "多裁剪平铺" | 将高分辨率图像拆分为图块网格，拼接每个图块的视觉 token |
| LLaVA-Instruct | "GPT-4 生成" | 15.8 万对从 COCO 标题 + GPT-4 合成的指令-响应对 |
| Vision encoder freeze | "骨干网锁定" | CLIP 权重在第一阶段不更新，有时第二阶段也不更新 |
| ShareGPT4V | "更好的标题" | GPT-4V 生成的 100 万密集标题，用于更高质量的对齐 |
| VQA | "视觉问答" | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 消融实验，系统测试投影器和数据选择 |

## 延伸阅读

- [Liu 等 — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu 等 — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen 等 — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集标题数据集。
- [Karamcheti 等 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融实验。
- [Li 等 — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图像、多图像、视频。