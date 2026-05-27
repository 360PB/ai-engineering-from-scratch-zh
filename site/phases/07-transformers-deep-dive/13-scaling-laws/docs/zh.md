# 扩展律

> 2020 年 Kaplan 论文说：更大的模型，更低的损失。2022 年 Hoffmann 论文说：你训练不足。计算进入两个桶——参数和 token——分割不明显。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 第 5 课（完整 Transformer）、Phase 7 第 7 课（GPT）
**时长：** ~45 分钟

## 问题

当你有 C FLOPs 训练计算且想要最佳模型时，你面临两个旋钮：

1. **多少参数（N）？** 更大的模型，更高的容量。
2. **多少训练 token（D）？** 更多数据，更好地利用容量。

FLOPs 约按 `6 × N × D` 缩放。你可以推高 N 并压低 D，或推高 D 并压低 N。哪个更好？

2022 年之前，答案是"使劲推 N"。GPT-3（2020）是 175B 参数训练在约 300B token 上。约 1.7 token 每参数。Kaplan 扩展律支持这一点。

Hoffmann 等人（2022），训练一个称为 Chinchilla 的小型模型家族，发现了不同的东西：最优比率接近 **每参数 20 token**。GPT-3 训练不足 10×。Chinchilla（70B 参数，1.4T token）在每个基准上击败 GPT-3（175B，300B token），同时计算成本低 2.5 倍。

2026 是 Chinchilla 的世界——带一个重要转折。Llama 3 8B 训练在 15 万亿 token 上，每参数 1,875 token 的比率。超过 Chinchilla 最优 94 倍。推理成本对将大规模使用的模型比训练成本更重要，所以超过训练（在 Chinchilla 之后）是小模型可部署足迹的 2026 年默认。

## 概念

![Chinchilla 曲线：各种 N/D 比率下损失 vs 计算](../assets/scaling-laws.svg)

### Hoffmann 定律

从 Chinchilla 论文，损失遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数（非嵌入）。
- `D` = 训练 token。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约损失上限。
- `A ≈ 406`，`B ≈ 411`。

两个项在你扩展时相互竞争。取固定计算（C = 6ND）下对 N 的导数并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

计算最优：每参数 20 token。

### 为什么超过训练

Chinchilla 最优最小化每训练 FLOP 的训练损失。但你支付训练成本一次；推理成本永远。

对于每月服务一万亿 token 的聊天机器人，推理主导总成本。Llama 的方法：训练更小，更久。8B 在 15T token 上深度推理优化：

- 适合消费级 GPU。
- 延迟是 70B Chinchilla 最优的一小部分。
- 质量对大多数任务足够接近。

DeepMind 的 2024 年论文（"超过训练是新的最优"）形式化了这个。对于推理主导工作负载，正确的比率取决于服务量，接近每参数 100–500 token。

### 涌现 vs 平滑

主张：某些能力（算术、多步推理、思维链跟随）在某个规模突然"涌现"。

Schaeffer 等人（2023）认为这是测量伪影：涌现指标使用不连续评分（精确匹配、阈值准确度）隐藏了底层 logit 中平滑改进。连续指标（交叉熵）显示平滑曲线。

2026 年共识：通过连续损失预测是可靠的。基准跳跃通常是评分器伪影。根据连续指标计划预算。

### 2026 年图景

扩展律仍然有效，但：

| 因素 | 改变方式 |
|------|---------|
| 数据质量 | 策划"好"token（Phi 风格）通过 >2× 有效计算移动曲线 |
| MoE | 总参数从激活 FLOPs 解耦；按激活 FLOPs 的扩展律 |
| 后训练 | 某些能力（指令遵循、代码）随 SFT+RLHF 移动多于预训练 |
| 多模态 | 图像 + 文本 token 一起扩展；每模态单独曲线 |
| 合成数据 | 模型生成训练数据；有效计算可以复合 |

Muon 优化器（Kimi Moonlight，2024）在匹配数据上显示比 AdamW ~2× 有效计算增益。一些 2026 年训练运行默认使用 Muon。改变扩展律中的绝对常数，不改变其形状。

## 构建

见 `code/main.py`。我们实现 Chinchilla 损失方程，并在多个计算预算下求解计算最优 `(N, D)`。

### 第一步：Chinchilla 损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 下将 `L` 绘制为 `(N, D)` 的等高线。找最小值。

### 第二步：计算最优前沿

对于从 `1e17` 到 `1e25` FLOPs 的计算预算，找到在 `6ND = C` 条件下使损失最小的 `(N, D)`。验证比率 `D/N ≈ 20`。

### 第三步：超过训练成本

计算训练 10× 更小模型（1/10 最优 N，10× 最优 D）所支付的额外损失。报告换取的推理 FLOPs 节省（与 N 成比例）。

### 第四步：与真实模型比较

输入 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（激活参数）的已知 `(N, D)` 对，并比较预测 vs 报告损失。

## 使用

你不太可能自己训练前沿模型。但扩展律告诉你：

1. **你的微调是否有足够数据。** 如果你的任务特定数据低于基础模型每参数 20 token，预期在某个损失地板饱和。
2. **是否选更大的基础模型。** 如果你在推理上花费全部预算，偏好更小、更长时间训练的模型。
3. **收益递减在哪里。** 超过 1000× Chinchilla 最优，log-loss 变化变为噪声。

**2026 年研究轨迹：**

- **数据受限体制。** 网络有有限数量的高质量 token（过滤后约 5–10 万亿英语）。前沿预训练接近这个天花板。合成数据、多语言、多模态和 RLHF 扩展微调是下一个杠杆。
- **计算倍增器技巧。** Muon 优化器、MoE、更好的数据策划——每个移动绝对常数，不改变渐近线。
- **RL 扩展律。** 开放问题。早期证据表明 RL 样本中的幂律，但指数与预训练非常不同。

## 交付

见 `outputs/skill-training-budget-estimator.md`。该 skill 根据计算预算、部署约束和目标损失为新训练运行选择 `(N, D, hours, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印计算预算 `1e20`、`1e22`、`1e24` 的 Chinchilla 最优 `(N, D)`。与真实模型表比较。
2. **中等。** 实现 Hoffmann 损失作为计算函数曲线。在计算最优前沿上绘制损失 vs `log10(C)`。确定定律预测我们需要 `>10^28` FLOPs 用于下一个 0.1 交叉熵降低的时间。
3. **困难。** 在同一数据集上训练 5 个微型模型（100K 到 10M 参数）来拟合你自己的扩展律。估计 `α` 和 `E`。你的指数与已发布指数匹配程度如何？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| 参数（N） | "模型大小" | 非嵌入权重计数；决定容量。 |
| Token（D） | "训练数据" | 看到的训练 token 数；决定参数被利用得有多好。 |
| 计算（C） | "花费的 FLOPs" | 约 `6 × N × D` 用于标准 transformer。 |
| Chinchilla 最优 | "D/N ≈ 20" | 最小化每预训练 FLOP 损失的比率。 |
| 超过训练 | "在 Chinchilla 之后" | 花额外训练 FLOPs 以节省推理 FLOPs；D/N >> 20。 |
| 不可约损失 | "地板" | 扩展律中的 `E` 项；数据本身的熵。 |
| 涌现能力 | "规模上突然跳跃" | 通常是评分器伪影；连续损失是平滑的。 |
| 有效计算 | "训练效率倍增器" | 更好的数据/优化器/架构乘以一个 FLOP 能走多远。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一个扩展律论文；训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 涌现作为测量伪影。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 为什么 Llama 的超过训练对其工作负载是正确的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× 计算倍增器。