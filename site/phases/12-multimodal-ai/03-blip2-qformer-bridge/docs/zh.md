# 从 CLIP 到 BLIP-2 — Q-Former 作为模态桥

> CLIP 能对齐图像和文本，但无法生成标题、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个小型可训练桥解决了这个问题：32 个可学习 query 向量通过交叉注意力关注冻结 ViT 的特征，然后直接插入冻结 LLM 的输入流。1.88 亿参数的桥将 11B LLM 连接到 ViT-g/14。所有 2026 年基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的表亲们——都是它的后裔。本节解读 Q-Former 的架构，解释两阶段训练，并构建一个玩具版本将视觉 token 喂入冻结的文本解码器。

**类型：** Build
**语言：** Python（标准库，交叉注意力 + 可学习 query 演示）
**前置知识：** Phase 12 · 02（CLIP），Phase 7（Transformers）
**时间：** 约 180 分钟

## 学习目标

- 解释为什么在冻结视觉编码器和冻结 LLM 之间放置一个可训练瓶颈在成本和稳定性上优于端到端微调。
- 实现一个交叉注意力块，其中固定的可学习 query 集合关注外部图像特征。
- 走读 BLIP-2 的两阶段预训练：表示学习（ITC + ITM + ITG）然后生成学习（用冻结解码器的 LM 损失）。
- 将 Q-Former 与 LLaVA 中更简单的 MLP projector 比较，讨论各自的适用场景。

## 问题背景

你有一个冻结的 ViT，每张图像产生 256 个 dim 1408 的 patch token。你有一个冻结的 7B LLM，期望 dim 4096 的 token embedding。明显的桥——从 1408 到 4096 的线性层——可行，但将所有 256 个 patch token 送入 LLM 的上下文每张图像多消耗 256 个 token。32 张图像的 batch，仅视觉模态就消耗 8192 个 token。

BLIP-2 的问题是：能否将 256-token 的图像表示压缩到少得多的 token（如 32 个），同时保留足够的信息让 LLM 生成标题、回答问题、推理图像？能否在不触碰冻结骨干网的情况下训练这个桥，保持训练成本仅与桥的参数相关？

答案是 Q-Former。32 个可学习的"query"向量对 ViT 的 patch token 做交叉注意力，产生 32 个 token 的视觉摘要供 LLM 消费。总计 1.88 亿参数。在接触 LLM 之前，用对比、匹配和生成目标训练。

## 核心概念

### 可学习的 queries

Q-Former 的核心技巧：不让 LLM 的文本 token 关注图像 patch，而是引入一组新的 32 个可学习 query 向量 `Q`，让*它们*关注图像 patch。queries 是模型的参数——在训练中学习，对每张图像使用相同的 32 个 queries。

交叉注意力后，每个 query 持有图像的压缩摘要——"描述主要物体"、"描述背景"、"数物体"等。queries 并不真正按语义标签专化；它们学习任何能让下游损失下降的编码。

### 架构

Q-Former 是一个小型 transformer（12 层，约 1 亿参数），有两条路径：

1. Query 路径：32 个 query 向量通过自注意力（相互之间）、然后对冻结 ViT 的 patch token 做交叉注意力、然后 FFN。
2. Text 路径：类似 BERT 的文本编码器与 query 路径共享自注意力和 FFN 权重。文本路径的交叉注意力被禁用。

训练时两条路径都运行。queries 和文本通过共享自注意力交互，这意味着 queries 可以根据需要文本的条件（ITM、ITG）。在 VLM 移交的推理时，只有 queries 流动，产生 32 个视觉 token。

### 两阶段训练

BLIP-2 分两阶段预训练：

第一阶段：表示学习（无 LLM）。三个损失：
- ITC（图文对比）：CLIP 风格的对比，在汇集的 query token 和文本 CLS token 之间。
- ITM（图文匹配）：二分类器——这对图文是否匹配？经硬负样本挖掘。
- ITG（图 grounded 文本生成）：文本上的因果 LM head，以 queries 为条件。强制 queries 编码可生成文本的内容。

只有 Q-Former 在训练。ViT 冻结。没有 LLM 参与。

第二阶段：生成学习。接入一个冻结的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。用小型线性层将 32 个 query 输出投影到 LLM 的 embedding 维度。将其 prepend 到文本 prompt。用 LM 损失在连接的 prompt + 图像 + caption 序列上训练线性投影和 Q-Former。

第二阶段后，Q-Former + 投影是完整的视觉适配器。推理时：图像 → ViT → Q-Former → 线性投影 → prepend 到文本 → 冻结 LLM 发出输出。

### 参数经济性

BLIP-2 + ViT-g/14（11 亿，冻结）+ OPT-6.7B（67 亿，冻结）+ Q-Former（1.88 亿，训练中）= 总计 80 亿，训练 1.88 亿。Q-Former 单独约占总参数量的 2.4%。训练成本反映这一点：在几块 A100 上训练几天 vs 端到端的几周。

质量：BLIP-2 在零样本 VQA 上与 Flamingo-80B 匹配或超越，同时体积小 50 倍。桥方案有效。

### InstructBLIP 与指令感知的 Q-Former

InstructBLIP（2023）向 Q-Former 引入了一个额外输入：指令文本本身。在交叉注意力时，queries 现在同时访问图像 patch 和指令。queries 可以按指令专化（"数汽车"、"描述情绪"），而不是学习单一的固定摘要。在留出任务上有基准提升。

### MiniGPT-4 与仅投影器方法

MiniGPT-4 保留了 Q-Former，但只训练输出线性投影，其他全部冻结。成本低，但代价是质量——queries 是 BLIP-2 的，不是你自己的。适合快速迭代，不是最优架构。

### 为什么 LLaVA 走向更简单的方案

LLaVA（2023，第 12.05 节）将 Q-Former 替换为普通 2 层 MLP，将每个 ViT patch token 投影到 LLM 空间——每张图像 576 个 token（24x24 网格），全部喂入 LLM。压缩效果差但让 LLM 能关注原始 patch。当时这有争议；到 2023 年底它成为主流，因为视觉指令数据（LLaVA-Instruct-150k）证明了 MLP 可以训练到保留足够信号。权衡：LLaVA 的上下文填充更快，但可以自然扩展到多图像和视频。

到 2026 年，领域分化：Q-Former 在 token 预算紧张的地方存活（长视频、多图像）；MLP projector 在每 token 质量优先的地方占主导。

### 门控交叉注意力：Flamingo，先祖

Flamingo（第 12.04 节）早于 BLIP-2，使用了相同的交叉注意力思路，但应用在每个冻结 LLM 层上，而不是作为单一桥。BLIP-2 证明了只需压缩到输入层也能工作。Gemini 和 Idefics 结合两者：交错输入 token + 可选的带门交叉注意力用于上下文中的 few-shot。

### 2026 年的后裔们

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，大多数视频-语言模型因 token 预算原因使用。
- Perceiver resampler：Flamingo 的变体（第 12.04 节）；Idefics 家族、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四种方案都有效。决定性的问题是：你受 token 预算约束还是受每 token 质量约束。

## 使用方法

`code/main.py` 构建一个类 Q-Former 的标准库交叉注意力：

1. 模拟 256 个 dim 128 的图像 patch token。
2. 实例化 32 个 dim 128 的可学习 queries。
3. 运行缩放点积交叉注意力（Q 来自 queries，K/V 来自 patches）。
4. 通过线性层投影到 LLM-dim（512）。
5. 输出 32 个 LLM 就绪的视觉 token。

全部数学用纯 Python（嵌套循环向量化）。是玩具但形状正确。打印注意力权重矩阵，可以看到每个 query 从哪些 patch 提取了信息。

## 输出作品

本节生成 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（视觉编码器 token 数量、LLM 上下文预算、部署约束、质量目标），推荐 Q-Former vs MLP vs Perceiver resampler，附简短理由和每个桥的参数量估算。

## 练习

1. 用 PyTorch 实现交叉注意力块。验证 32 个 queries 和 256 个 keys/values 下，注意力权重矩阵是 32 x 256，且每行在 softmax 后和为 1。

2. 在 BLIP-2 第一阶段，Q-Former 同时运行三个损失：ITC、ITM、ITG。写出每个前向签名的伪代码。哪个需要文本编码器路径处于活跃状态？

3. 比较参数量：Q-Former（12 层，768 隐藏）vs 2 层 MLP projector（1408 → 4096，两层）。在什么 LLM 规模下 1.88 亿 Q-Former 的成本能在训练效率上回本？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 初始化的部分。解释为什么从 BERT-base 初始化（而非随机）加速收敛。

5. 对 10 分钟视频以 1 FPS 采样到 60 帧，计算每帧 token 成本：(Q-Former → 每帧 32 tokens) vs (MLP projector → 每帧 576 tokens)。哪个能放入 128k-token LLM 上下文窗口？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Q-Former | "Querying transformer" | 带 32 个可学习 query 向量的小型 transformer，通过交叉注意力关注冻结 ViT 特征 |
| Learnable queries | "视觉软 prompt" | 固定的一组参数，作为交叉注意力的 query 侧；按模型学习，对所有输入共享 |
| Cross-attention | "Q 来自这里，K/V 来自那里" | query、key、value 来自不同源的注意力；queries 如何从 ViT patch 中提取信息 |
| ITC | "图文对比" | CLIP 风格的损失，应用在 Q-Former 汇集的 queries 和文本 CLS 之间 |
| ITM | "图文匹配" | 在硬负样本挖掘的对上做二分类器；强制 queries 区分细粒度的不匹配 |
| ITG | "图 grounded 文本生成" | 因果 LM 损失，文本以 queries 为条件生成；强制 queries 编码可解码文本的内容 |
| Two-stage pretraining | "表示学习然后生成" | 第一阶段只训练 Q-Former（ITC/ITM/ITG）；第二阶段接入冻结 LLM，只训练投影 + Q-Former |
| Frozen backbone | "不微调" | 视觉编码器和 LLM 权重固定；只有桥在训练 |
| Projection head | "线性到 LLM dim" | 将 Q-Former 输出映射到 LLM embedding 维度的最终线性层 |
| Perceiver resampler | "Flamingo 版本" | 类似的可学习 query 交叉注意力，Flamingo 在每层使用而非单一桥 |

## 延伸阅读

- [Li 等 — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li 等 — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 前身，包含 ITC/ITM/ITG 三合一。
- [Li 等 — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "对齐然后融合"——第一阶段训练的概念祖先。
- [Dai 等 — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 指令感知的 Q-Former。
- [Zhu 等 — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 仅投影器方法。
- [Jaegle 等 — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 可学习 query 交叉注意力的通用架构。