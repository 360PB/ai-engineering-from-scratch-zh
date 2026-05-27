# Jamba — 混合 SSM-Transformer

> 状态空间模型（SSM）和 Transformer 各有所图。Transformer 以二次方成本购买注意力质量。SSM 通过递归购买线性时间推理和常数内存，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）将两者放入同一模型：每 7 个 Mamba 层配 1 个 Transformer 层，交替块上使用 MoE，256k 上下文窗口可以容纳在单张 80GB GPU 上。Mamba-3（ICLR 2026）用复数值状态空间和 MIMO 投影收紧了 SSM 侧。本课从头到尾解读两种架构，解释为什么混合配方在纯 SSM 和纯 Transformer 长上下文尝试失败后存活了三年。

**类型：** Learn
**语言：** Python（标准库、层混合计算器）
**前置要求：** Phase 10 · 14（开放模型架构）、Phase 10 · 17（原生稀疏注意力）
**时长：** 约 60 分钟

## 学习目标

- 解释 Jamba 块的三个原语——Transformer 层、Mamba 层、MoE——以及 1:7:交替的交错配方。
- 简述 SSM 递归在高层的样子，以及为什么它能实现常数内存推理。
- 计算 Jamba 模型在 256k 上下文时的 KV 缓存占用，与纯 Transformer 模型的需求对比。
- 说出 Mamba-3 的三个创新（指数-梯形离散化、复数值状态更新、MIMO）及每个创新针对的问题。

## 问题背景

注意力在序列长度上是二次方的。状态空间模型是线性的。这个差异会累积：在 256k token 时，Transformer 注意力图每个头有 65B 个条目；SSM 的递归状态大小固定，不随序列长度变化。

纯 SSM 模型（Mamba、Mamba-2）在小规模上匹配 Transformer 困惑度，但在状态追踪任务上落后，在某些类别的上下文内检索上失败。直觉：SSM 将历史压缩到固定状态，当历史很长时，信息会泄漏。注意力精确记住一切，但付出二次方成本。

显而易见的修复：两者都用。在需要精确回忆的地方用 Transformer 层。其他地方用 SSM 层。调比例。Jamba 是第一个大规模交付此混合配方的生产级模型（总计 52B，激活 12B，256k 上下文，单张 80GB GPU）。Jamba 1.5 将系列扩展到 398B 总计 / 94B 激活。Mamba-3（ICLR 2026）是当前最佳纯 SSM 基线，可以在其上重建混合模型。

本课解读全部三篇论文，形成"选择正确比例"的思维模型。

## 核心概念

### 一页纸的 SSM

状态空间模型通过固定大小状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步状态通过线性动力学 `A` 演化，接收输入 `B x_t`，发出输出 `C h_t`。`A, B, C` 可以学习。注意关键属性：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要更早的 `x`。内存恒定。推理每个 token O(1)。

建模质量的关键在于 `A` 的结构。S4（Gu 2021）使用了高度结构化的矩阵，可以在训练时作为长卷积高效求值。Mamba（Gu, Dao 2023）将固定的 `A, B, C` 替换为数据依赖的（"选择性"部分）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置重新引入了复数。

关键属性：对于解码器 LLM，SSM 层是注意力层的替代品，用固定大小的逐层状态替代增长的 KV 缓存。

### Jamba 块

Jamba 块按两个数字交错层：

- `l`：注意力与 Mamba 的比例。Jamba 使用 `l = 8`，意味着每 7 个 Mamba 层配 1 个 Transformer 层（7 个 Mamba + 1 个 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，意味着每隔一层应用 MoE。

块内的层序列：

```
M  M  M  M  M  M  M  A    （7 个 Mamba + 1 个 Attention）
|  M  |  M  |  M  |  M    （| 标记 MoE 应用的位置）
```

每个 Jamba 块为 8 层。在 4 块深度（32 层总计）时，得到 28 个 Mamba 和 4 个 Attention 层。其中 16 个使用 MoE。

### 为什么是 1:7 比例

AI21 做了消融实验：什么样的注意力: Mamba 比例能给出最佳困惑度-每参数比 AND 在他们的长上下文评估上的上下文内回忆？

- 注意力太多（1:1）：质量上升但内存和速度下降。
- 注意力太少（1:15）：内存很好但上下文内检索失败。
- 最佳点：1:7 或 1:8。

直觉：Transformer 层处理精确回忆和状态追踪。Mamba 层处理廉价的大部分处理。

### 位置编码

Mamba 层本身是位置感知的（通过递归）。原始基于 Mamba 的混合模型中，注意力层不使用 RoPE——SSM 层提供了位置信息。Jamba 1.5 为注意力层添加了 RoPE 以实现更长上下文的泛化，这是基于经验性长上下文评估的事后改进。

### 内存预算

对于 Jamba-1 形状（32 层：28 个 Mamba + 4 个 Attention，隐藏 4096，32 个注意力头）：

- KV 缓存（仅限注意力层）：在 256k BF16 时 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个注意力层贡献。
- SSM 状态：每 token 前缀 `28 * hidden * state_size`，但这是固定大小的逐层，不随序列长度缩放。典型 Mamba 状态每特征 16，隐藏 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

对比相同形状、32 层、全 MHA 32 头的纯 Transformer：在 256k BF16 时需要 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV 缓存减少 8 倍。即使对比大多数 2024 模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合在 16 GB 仍是 2 倍小。

这就是 AI21 所说的"单张 80GB GPU 上 256k 上下文"。全 MHA 纯 Transformer 的 KV 缓存放不下；即使 GQA 基线也为权重和激活留不出空间；Jamba 可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 侧引入三个创新：

1. **指数-梯形离散化。** 用更表达性的递归替换 Mamba-2 中的欧拉方法离散化。在核心递归内部而非作为对 `x_t` 的外层卷积，对状态-输入应用类似卷积的操作。

2. **复数值状态更新。** 之前的 Mamba 将状态矩阵从复数（S4）简化为实对角（Mamba）再简化为缩放单位阵（Mamba-2）。Mamba-3 在特定位置重新引入复数值——相当于在状态上的数据依赖旋转嵌入。这恢复了之前实值简化所造成损失的状态追踪能力。

3. **多输入多输出（MIMO）投影。** 不使用每特征标量投影，而是使用矩阵值投影。在不增加解码延迟的情况下提高建模能力和推理时硬件利用率。

在 1.5B 参数下，Mamba-3 在平均下游准确率上比 Gated DeltaNet 高 0.6 点；MIMO 变体额外增加 1.2 点，总计 1.8 点增益。在相同状态大小下，Mamba-3 用一半状态匹配 Mamba-2。

Mamba-3 尚未在规模化生产混合模型中发货——但它是下一代 Jamba 类模型 SSM 侧的明显候选。

### 何时选择混合

混合模型在以下情况胜出：

- 上下文足够长，使得纯 Transformer KV 缓存变得痛苦（64k+）。
- 任务混合短程结构（SSM 擅长）和长程回忆（需要 Transformer）。
- 你想部署在单 GPU 内存预算下，仅 Transformer KV 缓存就放不下。

混合模型在以下情况失利：

- 上下文短（低于 16k）。SSM 开销被浪费；纯 Transformer 足够。
- 任务需要处处到处的注意力（深度推理、多文档交叉引用）。混合中注意力层的稀疏性伤害性能。
- 你正在扩展到万亿参数前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前在能力竞赛中领先。

### 竞争格局

| 模型 | 系列 | 规模 | 独特主张 |
|------|------|------|---------|
| Mamba-2 | 纯 SSM | 3B | 线性时间，常数内存 |
| Jamba | 混合 | 52B/12B | 256k 在 80GB 上 |
| Jamba 1.5 Large | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯 SSM | 1.5B（论文） | 状态追踪恢复 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿能力 |

2026 年格局：纯 Transformer MoE 在前沿占主导，但混合模型拥有 256k 以上上下文细分市场。Mamba-3 的状态追踪胜利可能推动下一代混合比例更低（更多 SSM，更少注意力）。

## 使用它

`code/main.py` 是混合架构的内存计算器。给定 SSM-Transformer 比例和隐藏大小/层数配置，它计算：

- 目标上下文时的 KV 缓存。
- SSM 状态内存。
- 在上下文 N 时一系列模型形状的总内存。

计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV 缓存）。

数字直接来自 Jamba-1 和 Jamba-1.5 论文的已发布形状，并对假设变体进行外推。

真实部署的集成注意事项：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k 上下文时，Jamba 的内存优势体现在并发请求吞吐上。在相同 VRAM 下，Jamba 序列比 Transformer 序列容纳更多。
- Mamba-3 作为独立模型尚未在生产中发货——1.5B 研究预览。

## 交付它

本课产出 `outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度分布、任务混合、内存预算），它在纯 Transformer、Jamba 风格混合和纯 SSM 之间推荐，并附有内存和质量权衡的明确推理。

## 练习

1. 运行 `code/main.py` 计算 32 层纯 Transformer（隐藏 4096，32 头）和同形状 Jamba-1 混合在 256k 上下文时的 KV 缓存。验证 AI21 论文声称的约 8 倍内存减少。

2. 修改计算器以建模 1:3 混合（4 个 Mamba : 1 个 Attention）和 1:15 混合（14 个 Mamba : 1 个 Attention）。绘制 KV 缓存 vs 比例图。在什么比例下 KV 缓存等于 SSM 状态内存？

3. 阅读 Jamba 论文（arXiv:2403.19887）第 3 节。解释为什么 AI21 使用 Mamba-1 而非 Mamba-2，尽管 Mamba-2 更快。提示：混合消融部分记录了这一点。

4. 计算 Jamba 1.5 Large（398B 总计，94B 激活）每隔一层应用 MoE 的参数开销。将激活比例与 DeepSeek-V3（37B/671B）对比，解释为什么 Jamba 的架构将激活比例推得更高。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）第 3 节。用三句话解释为什么复数值状态更新等价于数据依赖旋转嵌入。将答案与 Phase 7 · Lesson 04 的 RoPE 推导联系起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 状态空间模型（SSM） | "固定状态的递归" | 一层具有学习递归 `h_t = A h_{t-1} + B x_t`；每个 token 常数内存 |
| 选择性 SSM | "Mamba 的技巧" | 数据依赖的 A、B、C 参数，以线性时间给予模型类似门控的选择性 |
| 注意力-Mamba 比例 | "多少注意力层" | 在 Jamba 中，`l = 8` 意味着每 7 个 Mamba 层配 1 个注意力层 |
| Jamba 块 | "8 层组" | 1 个 Attention + 7 个 Mamba + 交替位置 MoE |
| SSM 状态 | "隐藏缓冲区" | 固定大小的逐层状态，替代 Mamba 层的 KV 缓存 |
| 256k 上下文 | "Jamba 的旗舰数字" | Jamba-1 在单张 80GB GPU 上容纳的序列长度；纯 Transformer 在该大小无法容纳 |
| Mamba-3 | "2026 纯 SSM" | 当前最佳纯 SSM 架构，复杂状态 + MIMO；混合模型重建所围绕的基线 |
| MIMO | "多输入多输出" | Mamba-3 创新，使用矩阵值投影而非每特征标量 |
| 指数-梯形离散化 | "Mamba-3 的递归" | 更表达性的递归，包含了 Mamba-2 的欧拉方法离散化 |
| 混合架构 | "混合注意力和 SSM" | 任何交错 Transformer 和 SSM 层的模型；Jamba 是生产原型 |

## 扩展阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba 论文，比例消融，256k 上下文声明
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 规模化家族，398B/94B 和 12B/52B 公开版本
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所基于的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化的结构化状态空间后继
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 复数值状态、MIMO、2026 纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 论文，LLM SSM 谱系的起点