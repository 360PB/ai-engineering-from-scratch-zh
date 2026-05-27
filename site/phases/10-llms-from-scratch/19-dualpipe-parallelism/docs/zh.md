# DualPipe 并行化

> DeepSeek-V3 在 2,048 张 H800 GPU 上训练，MoE 专家分散在多个节点上。跨节点专家 all-to-all 通信每消耗 1 GPU 小时算力就产生 1 GPU 小时通信。GPU 一半时间处于空闲。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，重叠前向与反向计算以及它们触发的 all-to-all 通信。气泡消失，吞吐上升，而保留两份模型参数拷贝（"Dual"之名的由来）在专家并行已将专家分散到各 rank 的情况下成本很低。本课是类型为"学习"的讲解，解析 DualPipe 实际做了什么，以及 Sea AI Lab 的 DualPipeV 改进如何在以 2 倍参数复制为代价换取略大气泡。

**类型：** Learn
**语言：** Python（标准库、流水线模拟器）
**前置要求：** Phase 10 · 05（分布式训练、FSDP、DeepSpeed）、Phase 10 · 14（开放模型架构与 MoE）
**时长：** 约 60 分钟

## 学习目标

- 说出 DualPipe 前向-反向 chunk 的四个组成部分，以及每个组件为何有自己独立的重叠窗口。
- 解释流水线气泡问题在规模化时的表现，以及"无气泡"在实际中与营销说法的区别。
- 手工追踪 DualPipe 调度在 8 个 PP rank 和 16 个微批次上的情况，确认前向和反向流如何填满彼此的空闲槽。
- 陈述 DualPipeV（Sea AI Lab，2025）的权衡：放弃 2 倍参数复制，代价是当专家并行未激活时气泡略大。

## 问题背景

在 2k 张 H800 GPU 上训练 671B MoE 模型面临三个叠加瓶颈：

1. **内存压力。** 每块 GPU 持有模型的一个切片。在序列长度 8k、61 层、128 头时，激活内存非常庞大。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）让 GPU 在等待本阶段的输入或梯度时处于空闲。在 8 个阶段，即使 1F1B 调度也约有 12% 的 GPU 时间是气泡。
3. **跨节点 all-to-all。** MoE 配合专家并行将专家分散到各节点。每次前向传递触发一次 all-to-all 将令牌分发到各自的专家，再触发一次 all-to-all 合并结果。在 2k GPU 上这很容易变成 1:1 的算力-通信比。

每个问题都有独立解决方案：梯度检查点对应内存、Zero Bubble（Sea AI Lab，2023）对应流水线气泡、专家并行 all-to-all 内核对应跨节点通信。DualPipe 所做的是让它们协同运作。调度将计算和通信在单个前向-反向 chunk 内重叠，从流水线两端同时注入微批次，用得到的调度将 all-to-all 隐藏进计算窗口。

报告的结果：流水线气泡近乎消除，DeepSeek-V3 的 14.8T token 训练运行中 GPU 利用率超过 95%。

## 核心概念

### 流水线并行回顾

将 N 层模型切分到 P 个设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个微批次从前向后流经设备 0 到 P-1，然后从后向前反向传播。每个设备只有在前一个设备发送其输出后才能开始前向阶段，只有在下游设备发送上游梯度后才能开始反向阶段。

GPipe（Huang et al., 2019）一次调度一个微批次，浪费了大部分 GPU 时间。1F1B（Narayanan et al., 2021）将多个微批次的前向和反向传递交错排列。Zero Bubble（Qi et al., 2023）将反向传递拆分为两部分——反向输入梯度（B）和反向权重梯度（W）——并调度它们填满气泡。经过 Zero Bubble 优化后，流水线几乎紧密。

DualPipe 是下一步。它在此基础上增加了两个思路：

### 思路 1：chunk 分解

每个前向 chunk 被分解为四个组件：

- **注意力。** Q/K/V 投影、注意力、输出投影。
- **All-to-all 分发。** 将令牌发送到各专家的跨节点通信。
- **MLP。** MoE 专家计算。
- **All-to-all 合并。** 将专家输出取回的跨节点通信。

反向 chunk 添加这些组件各自的梯度版本。DualPipe 调度它们使得 all-to-all 分发与下一个 chunk 的注意力计算并行发生，all-to-all 合并与下下一个 chunk 的 MLP 计算并行发生。

### 思路 2：双向调度

大多数流水线调度从 stage 0 注入微批次并流向 stage P-1。DualPipe 从**两端同时**注入微批次。Stage 0 看到源自该处的前向微批次；stage P-1 也看到源自该处的前向微批次。两股流在中间相遇。

为此，设备 `i` 必须同时持有**早流水线层 `i` 和晚流水线层 `P - 1 - i`**。这就是 DualPipe 中"dual"的含义：每个设备为它需要服务的层保留两份拷贝（每个方向一份）。在 DeepSeek-V3 的规模上，这是 2 倍参数复制成本。但这可以承受，因为专家并行已经将 MoE 专家分散得很薄，再复制一份非专家层是小头。

关键在于：在一个方向上的前向流与另一方向上的反向流恰好在单方向调度中会出现气泡的位置重叠。气泡消失了。

### 手绘调度追踪

考虑 P = 4 个 rank、8 个微批次，分为 4 个前向 / 4 个反向。时间从左到右；行是设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

"F4/F5R" 标记的读法：rank 1 在同一时间槽内运行微批次 4 的前向（从左到右流经流水线）和微批次 5 的前向（从右到左流经流水线）。这就是"双向"的实际操作含义。

在 rank 2，跨流重叠出现得更早；在 rank 0 和 P-1，重叠出现得最晚。在调度的稳定中期阶段，每个 rank 运行某方向的前向重叠于另一方向的反向。计算忙碌。前向传递的 all-to-all 分发隐藏在反向计算中。All-to-all 合并隐藏在反向计算中。气泡被挤出。

### 气泡核算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进降低了气泡，但不到零。DualPipe 在稳定阶段，如果微批次数量可被流水线深度乘以 2 整除，则气泡为零。在稳定阶段之外（预热和冷却），存在一些气泡，但气泡不随微批次数量增长——这是论文强调的一个关键特性。

营销说法："无气泡"。技术说法：气泡不随微批次数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）显示，只有当专家并行不是瓶颈时才完全零气泡；在 EP 驱动的 all-to-all 下，总是存在一些调度上的妥协。

### DualPipeV——改进版

Sea AI Lab（2025 年）观察到，当 EP 通信重叠不是重点时，2 倍参数复制是浪费。他们的 DualPipeV 调度将双向注入折叠为"V 形"调度，只运行一份参数拷贝。气泡略大于 DualPipe，但节省的内存相当可观。DeepSeek 在其开源 DualPipe 实现中采用 DualPipeV 作为 EP 关闭模式。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|------|---------|-----------|------|------------|
| 每设备参数拷贝数 | 2 | 1 | 1 | 1 |
| 气泡 vs 微批次 | 常数 | 略微增长 | 增长 | 增长 |
| 计算-通信重叠 | 全部 | 部分 | 极少 | 部分 |
| 适用场景 | EP 重度 MoE | 密集或 EP 轻度 | 基线 | 任意流水线 |

### 对 14.8T token 运行意味着什么

DeepSeek-V3 的预训练在约 2.8M GPU 小时内消耗了 14.8T token。使用朴素的 1F1B，他们会损失其中的 12-15% 给流水线气泡——340-420k GPU 小时，足够训练一个完整的 70B 模型。DualPipe 收回了大部分。没有内部日志直接量化贡献很难，但论文声称训练期间平均 GPU 利用率超过 95%。

对于较小规模运行（低于 1k GPU），DualPipe 过于杀鸡用牛刀——流水线气泡相对于总成本较小，密集模型训练很少遇到 all-to-all 瓶颈。对于多千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 它在技术栈中的位置

- 与 **FSDP** 互补（Phase 10 · 05）。FSDP 跨 rank 分片模型参数；DualPipe 跨 rank 调度计算。两者结合。
- 与 **ZeRO-3** 梯度分片兼容。两份参数复制的簿记需要与 ZeRO 的分片梯度合作。
- 需要针对特定集群拓扑调优的**自定义 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

## 使用它

`code/main.py` 是一个流水线调度模拟器。它接收 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 的稳定阶段利用率。这是一个教学工具——数字与论文中的定性声明一致，不代表对生产测速提升的声明。

模拟器的价值：用不同 P 和微批次计数运行，观察 1F1B 的气泡占比如何增长，但 DualPipe 不会。

真实训练运行的集成注意事项：

- 选择能整除微批次计数的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的内核是参考。
- 第一次花一周时间调试调度本身。簿记很繁琐。
- 按 rank 监控 GPU 利用率，而非仅看总量。DualPipe 的收益来自收紧落后者。

## 交付它

本课产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑、互连、模型形状），它推荐流水线并行策略、要使用的调度算法，以及在目标规模下的预期气泡占比。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异，以每百万 token 训练回收的 GPU 小时数表示。

2. 手绘 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。标记每个时间槽的微批次 ID 和方向。找出气泡消失的第一个时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）中的图 5。找出 DualPipe 前向 chunk 内 all-to-all 分发的重叠窗口。解释计算调度如何隐藏它。

4. 计算 70B 密集模型（P=8 流水线阶段）和 671B MoE 模型（P=16 流水线阶段）的 DualPipe 2 倍参数开销。说明为什么 MoE 情况下开销占比更小（大部分参数是专家，分散在一个大的 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年的竞争性双向调度器）进行比较。找出 DualPipe 添加而 Chimera 没有的两个特定属性，以论文第 3.4 节为参考。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 流水线气泡 | "每个 rank 的空闲时间" | GPU 周期因流水线阶段等待输入或梯度而浪费 |
| 1F1B | "默认流水线调度" | 一个前向 / 一个反向交错调度；DualPipe 的对比基线 |
| Zero Bubble | "Sea AI Lab 2023" | 将反向拆分为 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | "DeepSeek-V3 调度" | 双向流水线 + 计算-通信重叠；气泡不随微批次数量增长 |
| DualPipeV | "减半" | V 形改进，放弃 2 倍参数复制，代价是气泡略大 |
| Chunk | "流水线工作单元" | 一个微批次通过一个流水线阶段的一次前向或反向传递 |
| All-to-all 分发 | "发送令牌到专家" | 将令牌路由到其指定 MoE 专家的跨节点通信 |
| All-to-all 合并 | "取回专家输出" | MLP 之后收集专家输出的跨节点通信 |
| 专家并行（EP） | "专家分散在 GPU 上" | 将 MoE 专家分片到各 rank，使不同 GPU 持有不同的专家 |
| 流水线并行（PP） | "层分布在 GPU 上" | 将模型层分片到各 rank；这是 DualPipe 调度的维度 |
| 气泡占比 | "浪费的 GPU 时间" | bubble_time / total_time；DualPipe 趋向零的指标 |

## 扩展阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)，第 3.3.2 节和图 5](https://arxiv.org/abs/2412.19437) — 主要 DualPipe 参考
- [DeepSeek — DualPipe GitHub 仓库](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包含 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 前驱
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 启发 DeepSeek EP 关闭模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 对比的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 原始流水线并行论文和气泡问题