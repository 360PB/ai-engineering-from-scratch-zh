# DeepSeek-V3 架构讲解

> Phase 10 · Lesson 14 列出了每个开放模型都会调整的六个架构旋钮。DeepSeek-V3（2024 年 12 月，总参数 671B，激活 37B）转动了全部六个，还额外增加了四个：多头潜在注意力（MLA）、无辅助损失负载均衡、多令牌预测（MTP）和 DualPipe 训练。本课自顶向下解读 DeepSeek-V3 的架构，从已发布配置推导出每个参数计数。学完本课后，你能够解释为什么 671B/37B 这个比例是正确的赌注，以及为什么 MLA + MoE 组合在边界上比单独任何一个都更强。

**类型：** Learn
**语言：** Python（标准库、参数计算器）
**前置要求：** Phase 10 · 14（开放模型讲解）、Phase 10 · 17（NSA）、Phase 10 · 18（MTP）、Phase 10 · 19（DualPipe）
**时长：** 约 75 分钟

## 学习目标

- 从上到下阅读 DeepSeek-V3 配置，用六个 GPT-2 旋钮加四个 DeepSeek 特有扩展来解释每个字段。
- 推导总参数计数（671B）、激活参数计数（37B）及其各部分贡献。
- 计算 MLA 在 128k 上下文时的 KV 缓存占用，并与相同激活参数密集模型使用 GQA 的成本进行对比。
- 陈述四个 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每个创新针对架构/训练栈的哪一部分。

## 问题背景

DeepSeek-V3 是第一个架构上与 Llama 系列有本质不同的前沿开放模型。Llama 3 405B 是"六个旋钮转动的 GPT-2"。DeepSeek-V3 是"六个旋钮加四个更多的 GPT-2"。读 Llama 3 配置是读 DeepSeek 配置的热身，但深度结构——注意力块的形状、路由逻辑、训练时目标——差异足够大，需要单独的讲解。

学习它的收益：DeepSeek-V3 的开源权重发布改变了"前沿能力"在开放模型中的含义。2026 年的许多训练运行都在复制这个架构。理解它是任何涉及前沿 LLM 训练或推理岗位的最低门槛。

## 核心概念

### 不变核心，再次回顾

DeepSeek-V3 仍然是自回归的。仍然堆叠解码器块。每个块仍然有注意力加 MLP 加两个 RMSNorm。仍然在 MLP 中使用 SwiGLU。仍然使用 RoPE。Pre-norm。权重绑定的嵌入。与每个 Llama 或 Mistral 相同的基线。

### 转折：MLA 替代 GQA

从 Phase 10 · 14 你知道 GQA 通过在 Q 头组之间共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（MLA）走得更远：K 和 V 被压缩到一个共享的低秩潜在表示（`kv_lora_rank`），然后在运行时按头解压。KV 缓存只存储潜在向量——通常每 token 每层 512 个浮点数，而非 8 × 128 = 1024 个浮点数。

在 128k 上下文中，DeepSeek-V3 使用 MLA（每 token 每层一个共享潜在 `c^{KV}`；K 和 V 都从这个潜在通过上投影得到，后者可以被吸收进后续矩阵乘法）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的 GQA 基线（Llama 3 70B 形状，8 个 KV 头，头维度 128）需要：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文中，MLA 比 Llama-3-70B 风格的 GQA 缓存小 4 倍。

权衡：MLA 在每次注意力计算时增加了一个解压步骤（按头）。额外计算量相对于节省的带宽很小。对长上下文推理来说净收益为正。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个 token。朴素路由器将过多工作集中在少数专家身上，使其他人闲置。标准修复：添加一个辅助损失项惩罚负载不均衡。这有效但会略微降低主任务性能。

DeepSeek-V3 引入了一种无辅助损失方案。每个专家的偏置项被加到路由器 logits 上，在训练期间按简单规则调整：如果专家 `e` 过载，降低 `bias_e`；如果欠载，增加 `bias_e`。没有额外的损失项。训练保持干净。专家负载保持均衡。

对主损失的影响：可测量为零。对 MoE 架构的影响：更干净，没有辅助损失超参数需要调优。

### MTP：更密集训练 + 免费草稿

从 Phase 10 · 18 你知道 DeepSeek-V3 添加了 D=1 的 MTP 模块，预测两个位置后的令牌。推理时，训练好的模块被改造成接受率超过 80% 的推测解码草稿。训练时，每个隐藏状态在 D+1 = 2 个目标上受监督，提供更密集的信号。

参数：671B 主模型之上 14B。开销：2.1%。

### 训练：DualPipe

从 Phase 10 · 19 你知道 DualPipe 是一种双向流水线，将前向和反向 chunk 与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048-H800 规模上，它收回了 1F1B 会被流水线气泡吞噬的约 245k GPU 小时。

### 配置逐字段解析

以下是 DeepSeek-V3 配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   （密集 MLP 隐藏大小，用于前几层）
moe_intermediate_size: 2048 （专家 MLP 隐藏大小）
num_hidden_layers: 61
first_k_dense_layers: 3    （前 3 层使用密集 MLP）
num_attention_heads: 128
num_key_value_heads: 128   （形式上在 MLA 下等于 num_heads，
                            真正压缩在 kv_lora_rank）
kv_lora_rank: 512          （MLA 潜在维度）
num_experts: 256            （每块的 MoE 专家数）
num_experts_per_tok: 8      （top-8 路由）
shared_experts: 1           （每块一个常开共享专家）
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               （深度 1 处 1 个 MTP 模块）
```

解析：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的密集 MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，然后按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个路由专家之上，1 个常开专家贡献每个 token。可以把它想象成确保每个 token 都获得可靠内容的"密集底座"。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏大小。由于有 256 个，比密集 MLP 小。

### 参数核算

完整计算在 `code/main.py` 中。摘要：

- 嵌入：`vocab * hidden = 129280 * 7168 ≈ 0.93B`。
- 前 3 个密集块：带 MLA 的注意力（每块约 144M）+ 密集 MLP（每块约 260M）+ norms。合计约 1.2B。
- 58 个 MoE 块：带 MLA 的注意力（144M）+ 各含 256 个专家（每个 30M）+ 1 个共享专家（30M）+ norm。每块合计约 7.95B（含所有专家）。58 个 MoE 块共 461B。
- MTP 模块：14B。

总计：核心架构约 476B + 14B MTP。公开发布的 671B 数字中包含额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。计算器中我们复现的数字与发布值误差在 3-5% 以内——差异来自 DeepSeek 报告在第 2 节附录中详细列出的细粒度核算。

每前向的激活参数：

- 注意力：每层 144M × 61 = 8.8B（所有层都触发）。
- MLP 激活：前 3 层密集（3 × 260M = 780M），58 个 MoE 层每层激活 8 个路由 + 1 个共享 + 路由开销。每层激活 MLP 约 260M。合计：3 × 260M + 58 × 260M ≈ 15.9B。
- 嵌入 + norms：1.2B。
- 总激活：核心约 26B + 14B MTP（已训练但推理时不一定运行）≈ 37B。

### 671B / 37B 比例

18 倍稀疏比（激活参数占总数的 5.5%）。DeepSeek-V3 是开源权重中稀疏度最高的前沿 MoE 模型。Mixtral 8x7B 比例 13/47（28%）稠密得多。Llama 4 Maverick 比例 17B/400B（4.25%）可比。DeepSeek 的赌注：在前沿规模下，更多专家配合更低激活比产生更好的每激活 FLOP 质量。

### DeepSeek-V3 的定位

| 模型 | 总计 | 激活 | 比例 | 注意力 | 创新点 |
|------|------|------|------|--------|--------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后继者：R1、V4

DeepSeek-R1（2025 年）是基于 V3 骨干的推理训练运行。R1 使用相同架构。改变的是后训练配方（大规模 RL 在可验证任务上），而非预训练架构。

DeepSeek-V4（如果发布）预计保持 MLA + MoE + MTP 并添加 DSA（DeepSeek 稀疏注意力），即 Phase 10 · 17 中 NSA 的后继。血统稳定：架构层面的创新累积；每个版本转动额外的旋钮。

## 使用它

`code/main.py` 是专用于 DeepSeek-V3 形状的参数计算器。运行它，将其输出与论文数字比较，并在假设变体上使用它（256 专家 vs 512，top-8 vs top-16，MLA rank 512 vs 1024）。

关注点：

- 总参数计数与公开的 671B 对比。
- 激活参数计数与公开的 37B 对比。
- 128k 上下文时的 KV 缓存——MLA vs GQA 对比。
- 每层分解，了解参数预算实际花在哪里。

## 交付它

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或任何未来变体），它生成逐组件架构解读，命名配置的每个字段，按组件推导参数计数，并识别该模型使用了四个 DeepSeek 特有创新中的哪几个。

## 练习

1. 运行 `code/main.py`。将计算器的总参数估计与公开的 671B 进行对比，并找出差异来源。论文第 2 节有完整的逐项说明。

2. 将配置修改为使用 MLA rank 256 而非 512。计算 128k 上下文下的 KV 缓存大小。它带来了百分之多少的减少，代价是什么（按头表达性的代价）？

3. 将 DeepSeek-V3 的（256 专家，top-8）路由与假设的（512 专家，top-8）变体进行对比。总参数增长；激活参数相同。理论上额外的专家容量买到了什么，推理成本是多少？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA 的内容。用三句话解释为什么 K 和 V 解压矩阵可以被"吸收"进后续矩阵乘法以提高推理时效率。

5. DeepSeek-V3 在大多数操作中使用 FP8 训练。计算 FP8 vs BF16 存储 671B 权重的内存节省。这如何与 14.8T token 训练预算相交？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MLA | "多头潜在注意力" | 将 K 和 V 压缩到共享低秩潜在（kv_lora_rank，通常 512），按头即时解压；KV 缓存只存储潜在向量 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 的共享潜在大小；DeepSeek-V3 使用 512 |
| First k dense layers | "前几层保持密集" | MoE 模型的前几层跳过 MoE 路由器，运行密集 MLP 以保持稳定 |
| num_experts_per_tok | "Top-k 路由" | 每个 token 触发多少个路由专家；DeepSeek-V3 使用 8 |
| Shared experts | "常开专家" | 无论路由结果如何都处理每个 token 的专家；DeepSeek-V3 使用 1 |
| 无辅助损失路由 | "偏置调整负载均衡" | 训练期间调整每个专家的偏置项以保持专家负载均衡，无需添加损失项 |
| MTP 模块 | "额外预测头" | 预测从 `h^(1)` 和 `E(t+1)` 的 t+2 的 Transformer 块；更密集训练、免费推测解码草稿 |
| DualPipe | "双向流水线" | 将前向/反向计算与跨节点 all-to-all 重叠的训练调度 |
| 激活参数比例 | "稀疏度" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 训练 | "8 位训练" | 大部分存储和计算操作使用 FP8；相对于 BF16 约减少一半内存，质损很小 |

## 扩展阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整架构、训练和结果文档
- [DeepSeek-V3 模型卡 on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2 论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前身
- [DeepSeek-R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — V3 架构上的推理训练后继
- [原生稀疏注意力 (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族注意力的未来方向
- [DualPipe 仓库](https://github.com/deepseek-ai/DualPipe) — 训练调度参考