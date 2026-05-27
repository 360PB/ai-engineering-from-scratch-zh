# 推测解码和 EAGLE-3

> Phase 7 · Lesson 16 证明了数学：Leviathan 拒绝规则精确保留验证器的分布。这节课是 2026 年生产推测解码的训练栈视角。EAGLE-3 将草稿模型从廉价近似变成目的构建的微型网络，在验证器自己的隐藏状态上训练，然后添加了训练时测试循环对齐其训练和推理分布。结果：端到端加速 3× 到 6.5×，聊天上每个 Token 接受率超过 0.9，无分布权衡。2026 年每个生产推理栈默认发版它。

**类型：** 构建
**语言：** Python（stdlib）
**先修内容：** Phase 7 · 16（推测解码数学），Phase 10 · 12（推理优化）
**学习时间：** 约 75 分钟

## 学习目标

- 用一句话陈述 Leviathan 定理并证明推测循环产生与验证器完全相同分布的样本。
- 走过从朴素推测解码（Leviathan 2023）到 EAGLE、EAGLE-2 和 EAGLE-3 的两年进程并命名每步移除的确切限制。
- 从接受率 `α` 和草稿-验证器成本比 `c` 计算预期加速，并选择每个 regime 的最优草稿长度 `N`。
- 从零实现完整推测循环：草稿、验证、从残差拒绝-采样、在拒绝时回滚 KV 缓存、在完全接受时发出奖励 Token。

## 问题所在

自回归 70B 模型在 H100 上可能以每秒 35 Token 运行。GPU 远未饱和。内存带宽是天花板：每 Token 从 HBM 加载 70B 权重，做一步算术，产生一个浮点数。计算单元大部分时间空闲。

推测解码将其变成你可以实际解决的吞吐量问题。廉价草稿在 N 个小型前向传播中提议 N 个 Token。验证器在前缀加所有 N 个草稿上运行一次。如果验证器在位置 `i` 同意草稿（在统计意义上我们将精确化），我们接受；如果不同意，我们拒绝并从残差分布采样纠正。单个大模型前向产生最多 N+1 个接受 Token 而非一个。

重要定理是 Leviathan、Kalman、Matias（ICML 2023）：输出分布与直接从验证器采样完全相同。不是近似。完全相同。这就是推测解码在生产中可接受的全部原因——它是纯延迟优化，无质量权衡。

Phase 7 · Lesson 16 给你的数学。这节课给你的是训练栈。好草稿比廉价草稿多值 2 倍加速。EAGLE、EAGLE-2 和 EAGLE-3（Li et al., 2024-2025）将"草稿 = 同模型更小版本"变成精确工程学科。2026 年生产推理服务器默认 EAGLE-3。

## 核心概念

### 不变量：Leviathan 拒绝采样

设 `p(t)` 是给定某前缀的下一个 Token 的草稿分布，`q(t)` 是验证器的。采样草稿 Token `d ~ p`。以概率 `min(1, q(d) / p(d))` 接受。在拒绝时，从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 采样。结果样本根据 `q` 分布。无论 `p` 多糟这都成立——它越糟你拒绝越频繁，但输出保持精确。

将 N 个这样的调用堆叠回接，使用一次验证器前向传播在前缀 + d_1 + ... + d_N 上。验证器同时返回 q_1, q_2, ..., q_{N+1}。从左走到右。在第一个拒绝位置 `j`，从 `residual(q_j, p_j)` 采样并停止。在完全接受时，从 `q_{N+1}` 采样一个奖励 Token。

### 什么决定加速

设 `α` 为每个起草 Token 的预期接受率。设 `c = cost(draft) / cost(verifier)` 为成本比。预期每次验证器前向的接受 Token 数：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每次接受 Token 的预期墙上时间是 `(N * c + 1) / E[accepted]`。最小化它关于 `N` 你得到最佳点。对于 `α = 0.8, c = 0.05`：最优 `N` 约 5-7，加速 3.2×。对于 `α = 0.95, c = 0.02`：最优 `N` 约 8-10，加速接近 5×。

最大单一杠杆是 `α`。在固定 `N = 5` 下从 `α = 0.6`（朴素草稿）到 `α = 0.9`（EAGLE-3）让你从每次验证器前向 2.2 预期接受 Token 到 4.1。相同验证器上近 2 倍更多吞吐量。

### 两年进程

**朴素推测（Leviathan，2023）。** 草稿模型是同族独立训练的更小 LLM。易于接入，`α ≈ 0.6`，最好约 2× 加速。

**EAGLE-1（Li et al., 2024）。** 草稿是一个微型 transformer——通常一两层——将验证器最后一层隐藏状态作为输入并直接预测下一个 Token。因为草稿看到验证器的特征表示，其分布更接近验证器。`α` 攀升至 0.7-0.8。

**EAGLE-2（Li et al., 2024）。** 添加动态草稿树：不是提议单个 N Token 序列，而是提议候选小树，用验证器一次前向（树注意力）评分每个，并走最高概率路径。草稿长度变为每步自适应。`α` 每个接受路径 Token 攀升超过 0.85。

**EAGLE-3（Li et al., 2025, NeurIPS）。** 再两个变化。第一，完全丢弃特征预测损失——EAGLE-1/2 训练草稿匹配验证器隐藏状态，这限制了数据帮助多少。EAGLE-3 直接在 Token 预测上训练。第二，训练时测试（TTT）：在草稿训练期间，将草稿自己的之前预测作为输入喂回多步，与推理时运行方式相同。这对齐了训练和测试分布并阻止误差累积。测量加速：聊天上高达 6.5×，SGLang 在 H100 上批处理 64 时吞吐量提高 38%。

### KV 缓存回滚

验证在一次传递中将验证器 KV 缓存扩展 N 个条目。如果在位置 `j` 发生拒绝，位置 j-1 之后的缓存内容现在错误。两种常见实现：写入草稿缓冲区并在接受时提交（vLLM、TensorRT-LLM），或保持物理 KV 缓存加逻辑长度并在拒绝时截断。任一方式，回滚成本是每层每头字节数，与前向传播成本相比可忽略。

对于 EAGLE-2 树搜索，验证器运行带非因果掩码的注意力，遵守树拓扑。工程上繁琐但计算是带自定义掩码的标准 flash-attention 调用。

### 2026 年草稿架构

| 策略 | 草稿类型 | `α` | 加速 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| 朴素 | 独立小 LLM | 0.55-0.70 | 1.8-2.3× | 无（重用现有小模型）|
| Medusa | 验证器上额外 LM 头 | 0.65-0.75 | 2-3× | ~1B SFT Token |
| EAGLE-1 | 隐藏状态上的 1 层 transformer | 0.70-0.80 | 2.5-3× | ~60B Token |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | ~60B Token |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B Token |
| Lookahead | 无草稿（Jacobi 迭代）| N/A | 1.3-1.6× | 无 |

2026 年生产：vLLM 和 SGLang 默认在可用时使用 EAGLE-3，否则 EAGLE-2。TensorRT-LLM 有 Meta 和 NVIDIA 公共模型最快 Medusa 路径。llama.cpp 在 CPU 部署上发版朴素草稿。

## 构建

见 `code/main.py`。这是完整 Leviathan 推测循环，包含所有组件：草稿-N、验证器并行传递、逐位置拒绝、残差采样、奖励 Token、KV 回滚，以及经验验证输出分布匹配直接采样自 `q`。

### 步骤 1：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 步骤 2：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 步骤 3：完整推测步骤

`spec_step` 函数从 `p` 起草 N 个 Token，然后在一次并行 `q` 求值中验证所有。对每个起草 Token 应用拒绝规则，在第一个拒绝时从残差采样。如果全部接受，从 `q_{N+1}` 发出奖励 Token。

### 步骤 4：KV 回滚记账

模拟器追踪每工作者的逻辑 `kv_length`。在 k 个草稿接受时，`kv_length += k`。在位置 j 拒绝时，缓存已写入超过 j，但逻辑长度设为 `prefix_length + j + 1`——纠正 Token 之后一个。后续读取截断到逻辑长度。

### 步骤 5：Leviathan 检查

运行 50,000 次推测步骤。计数接受 Token 的经验分布。与 50,000 次直接从 `q` 采样比较。卡方统计量应远低于临界值。定理在实践中通过。

### 步骤 6：加速 vs α

通过以不同振幅扰动 `p` 偏离 `q` 来扫描草稿质量。测量 `α`，然后绘制作为 `α` 和 `N` 函数的每次验证器调用的预期 Token。代码打印表格显示 EAGLE-3 类草稿质量（`α ≈ 0.9`）解锁每次验证器调用 4-5 个 Token。

## 使用

生产级 `vllm serve` 带 EAGLE-3：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang 带 EAGLE-3 在 H100 批处理 64：据 EAGLE-3 论文约比批处理 64 朴素解码高 1.38 倍吞吐量。

何时使用推测解码：

- 任何 p50 延迟比峰值吞吐量更重要的交互式聊天工作负载。
- 代码生成和结构化输出（JSON、SQL）。`α` 超过 0.9 因为目标分布高度可预测。
- 长篇生成（数千 Token）。摊销加速持续支付。

何时不用：

- 非常小的模型（< 3B）。草稿不比验证器便宜多少。
- 微小批处理-1 CPU 部署。草稿模型内存开销可能不值得。
- 非常高三温度创意采样，`α` 崩溃。

## 发货

这节课产出 `outputs/skill-eagle3-tuner.md`。给定推理工作负载（模型、批次大小、目标延迟、任务画像），它推荐推测解码策略和调优参数（草稿家族、`N`、树深度、温度感知切换）。

## 练习

1. 运行 `code/main.py`。确认 Leviathan 分布检查的卡方统计在 50,000 样本上保持在 95% 临界值以下。

2. 在 `α` 保持在 0.9 且 `c` 保持在 0.04 下扫描 `N` 从 1 到 10。绘制每次验证器调用的预期 Token 和实际墙上时间/Token。找到最小化墙时间的 `N`。解释曲线形状。

3. 修改代码模拟 EAGLE-2 树搜索：每步草稿提议形状 `[2, 2, 2]` 的树（8 个候选路径）。验证器运行一次，并行路径中最优概率接受路径获胜。计算每叶 `α` 和每次验证器调用的总 Token。与等价计算下线性链推测解码比较。

4. 为两个并发序列实现批处理 KV 回滚模拟器。序列 A 全部接受；序列 B 在位置 2 拒绝。显示正确 `kv_length` 是每序列更新的，没有工作被浪费。

5. 阅读 EAGLE-3 论文第 4 节（训练时测试）。用两句话解释为什么没有 TTT 的朴素草稿训练遭受暴露偏差，以及为什么在训练期间将草稿自己的预测喂给它修复了它。连接到 seq2seq 中的计划采样文献。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Leviathan 规则 | "min(1, q over p)" | Bernoulli 接受/拒绝，概率 `min(1, q(d)/p(d))`，在拒绝时从残差采样时精确保留验证器分布 |
| 残差分布 | "(q 减 p) 正部，归一化" | `(q - p)_+` 钳制到零并重新归一化——拒绝时采样的正确分布 |
| 接受率 α | "草稿多久对一次" | 拒绝规则下每个 Token 的预期 Bernoulli 成功概率；管理所有加速数学 |
| EAGLE-1 | "隐藏状态草稿" | 微型 transformer 草稿，以验证器最后一层隐藏状态为条件（Li et al., 2024）|
| EAGLE-2 | "动态草稿树" | EAGLE-1 加上在一验证器前向中树注意力评分的候选延续树 |
| EAGLE-3 | "训练时测试" | 丢弃特征预测损失，用草稿在前向传播期间自己的输出训练直接 Token 预测 |
| 训练时测试（TTT） | "暴露偏差修复" | 在训练期间自回归运行草稿，使训练和测试输入分布匹配——计划采样的直接类比 |
| KV 回滚 | "撤销拒绝草稿" | 拒绝后将验证器 KV 缓存重置为接受前缀长度的记账 |
| 奖励 Token | "免费的那个" | 当所有 N 个草稿接受时，从 `q_{N+1}` 采样一个额外 Token，无额外验证器成本 |
| 树注意力 | "一次验证多个候选" | 遵守草稿树拓扑的非因果掩码注意力；一次前向传播计算树中每个节点的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 基础论文和等价定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — 并行独立介绍，带干净证明
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1，隐藏状态条件草稿
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — 2026 年生产默认
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — 替代无草稿方法
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 接入所有策略的生产规范参考