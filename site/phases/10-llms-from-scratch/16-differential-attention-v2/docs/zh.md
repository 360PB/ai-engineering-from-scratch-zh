# 差分注意力（V2）

> Softmax 注意力将少量概率散布在每个不匹配 Token 上。在 100k Token 上那噪声累积淹没信号。差分 Transformer（Ye et al., ICLR 2025）通过计算为两个 softmax 的差来修复，减去共享噪声基线。DIFF V2（Microsoft，2026 年 1 月）是生产栈重写：匹配基线 Transformer 解码延迟，无自定义内核，FlashAttention 兼容。这节课是 V1 到 V2 端到端，用纯 stdlib Python 实现你可运行的工作差分操作。

**类型：** 构建
**语言：** Python（stdlib）
**先修内容：** Phase 7 · 02（自注意力），Phase 7 · 15（注意力变体），Phase 10 · 14（架构详解）
**学习时间：** 约 60 分钟

## 学习目标

- 精确陈述为什么 softmax 注意力有噪声基线以及为什么它随上下文长度增长。
- 推导差分注意力公式并解释为什么减法抵消共享噪声分量同时保留信号。
- 走过 V1 到 V2 diff：什么变快了，什么变简单了，什么变稳定了，以及为什么每个变化对生产预训练都是必要的。
- 在纯 Python 中从零实现差分注意力，并在合成信号加噪声查询上经验验证噪声取消属性。

## 问题所在

标准 softmax 注意力有一个在规模上变成操作头痛的数学属性。对于查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不能产生精确零——每个不匹配 Token 获得一些正质量。那残差质量是噪声，它随上下文长度缩放。在 128k Token，即使每个不匹配 Token 只获得 0.001% 概率，127,999 个合计贡献约 12%。模型必须学会绕过随上下文增长的噪声基线。

经验上这表现为注意力头干扰：长上下文 RAG 中幻觉引用、100k Token 检索任务的中间丢失失败，以及 32k 后大海捞针基准上微妙的精度下降。差分 Transformer 论文（arXiv:2410.05258，ICLR 2025）测量了差距：DIFF Transformer 在困惑度、长上下文精度上更低，幻觉更少。

DIFF V1 有三个让它离开前沿预训练流水线的问题。它的值缓存每解码步骤必须加载两次，需要打破 FlashAttention 兼容性的自定义 CUDA 内核，以及其每头 RMSNorm 在 70B+ 规模上使长期训练不稳定。DIFF V2（Microsoft unilm 博客，2026 年 1 月 20 日）修复了所有三个。这节课走两个版本，构建差分算子，并在玩具查询上基准测试噪声取消。

## 核心概念

### Softmax 的噪声基线

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重是：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有 `w_i` 是零。如果 `k_i` 与 `q` 完全无关，score `q . k_i` 不是 0——它围绕零波动，方差 `||q||^2 / d`。Softmax 归一化后，每个无关 Token 仍贡献 `O(1/N)` 到加权和。无关 Token 的总贡献是 `O((N-1)/N) = O(1)`——不是小量。

模型想要的是类似硬 top-k 的东西：对匹配 Token 高权重，到处接近零。Softmax 太平滑无法直接做到。

### 差分想法

将每头的 Q 和 K 投影分成两个：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法抵消两个图共享的任何噪声分布。如果两个图在 127k 无关 Token 上大致均匀权重（它们会，在随机初始化时），那些抵消。信号——在少数实际相关 Token 上的峰值权重——只有当它以相同幅度出现在两个图中时才抵消，一旦模型训练就不会。

`lambda` 是每头学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负的。`lambda_init` 默认为如 0.8 的小正数。

### 为什么这匹配有头噪声消除

想两个嘈杂麦克风录制相同声音。两者都拾取说话者加相关背景噪声。相减一个和另一个，共享噪声下降。声音存活因为两个信号在相位或振幅上差异足以防止完全抵消。每头 `lambda` 精确学习这种平衡。

### V1 vs V2：diff

V1 保持参数量等于基线 Transformer。为每头获得两个查询它将头维度减半。那牺牲了头表达能力，更痛苦的是每头值缓存减半。Decode 每步必须加载值缓存两次。结果：尽管匹配参数量，decode 比基线慢。

V2 增加查询头数同时保持 KV 头相同（从 up-projection 借用参数）。头维度保持与基线相同。在减法后，额外维度在进入 O_W 投影前向下投影回匹配。一次发生三件事：

1. Decode 速度匹配基线（KV 缓存加载一次）。
2. FlashAttention 不变运行（无自定义内核）。
3. Decode 时算术强度上升（每次从 HBM 加载字节有更多计算）。

V2 还去除了 V1 用来稳定减法的每头 RMSNorm。在 70B 类预训练规模，那使后期训练不稳定。V2 用更简单的初始化方案替换它，无需额外模块即可保持训练稳定。

### 何时使用它

| 工作负载 | 收益 |
|----------|------|
| 长上下文 RAG（64k+）| 更清晰的注意力图，更少幻觉引用 |
| 大海捞针基准 | 32k 以上精度大幅提升 |
| 多文档 QA | 更少跨文档干扰 |
| 8k 代码补全 | 边际，不值得架构变化 |
| 短聊天（< 4k）| 与基线基本无法区分 |

价值随上下文长度增长。4k Token 时噪声基线足够小标准注意力没问题。128k 时它在伤害你。

### 如何与其他 2026 旋钮堆叠

| 特性 | 与 DIFF V2 兼容？ |
|---------|------------------------|
| GQA | 是（V2 增加 Q 头，非 KV 头）|
| MLA（DeepSeek）| 原则上是，未发表组合论文 |
| MoE | 是（注意力独立于 MLP 块）|
| RoPE | 是（不变）|
| YaRN / 长上下文缩放 | 是（正是 DIFF 最大帮助的地方）|
| FlashAttention | V2 是（V1 否）|
| 推测解码 | 是（注意力变化对推测解码循环不可见）|

## 构建

`code/main.py` 在纯 Python 中实现差分注意力。带已知信号加噪声结构的玩具查询让你直接测量噪声取消比率。

### 步骤 1：标准 softmax 注意力

Stdlib 矩阵运算：列表的列表，手动 matmul，带数值稳定性减去最大值的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤 2：将 Q、K 分裂为两半

V1 风格：将头维度减半。V2 风格：保持头维度并加倍头数。玩具实现使用 V1 为教学清晰——数学相同，只是记账不同。

### 步骤 3：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。那没问题——值缓存仍处理有符号贡献。后续 V 投影吸收符号。

### 步骤 4：噪声取消测量

构建长度 1024 的合成序列。将信号 Token 放在已知位置，其余填充噪声。计算 (a) 信号位置的标准 softmax 注意力权重和 (b) 差分注意力权重。测量每个中信噪比。DIFF 注意力可靠地产生高 3x-10x 的信噪比，取决于两个分支被训练差异多少。

### 步骤 5：V1 vs V2 参数量计算

给定配置（hidden=4096，heads=32，d_head=128），打印：

- 基线 Transformer：Q、K、V 各 `hidden * hidden`，MLP 在 4 * hidden。
- DIFF V1：Q、K 各 `hidden * hidden`，V `hidden * hidden`（不变），头维度内部减半。添加每头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q `2 * hidden * hidden`，K `hidden * hidden`，V `hidden * hidden`。额外维度在 O_W 前投影回。添加相同 `lambda` 参数。

玩具测量 V2 的额外参数成本（约每注意力块额外 `hidden * hidden`）并打印。

## 使用

截至 2026 年 4 月，DIFF V2 尚未在每个生产推理服务器中发版，但 vLLM 和 SGLang 中正在集成。同时出现在：

- Microsoft 内部长上下文生产模型。
- 多个瞄准 256k+ 上下文的开源模型训练研究复现。
- 结合 DIFF 注意力和交替层滑动窗口的混合架构。

你在 2026 年会伸手用它的时候：

- 从头训练瞄准 64k+ 有效上下文的新模型。从一开始就加差分注意力；之后重新训练昂贵。
- 微调长上下文模型，其中中间丢失失败主导你的评估。LoRA 在 Q 投影上可近似 DIFF 结构。

你不会的时候：

- 你在服务预训练有稳定长上下文性能的密集模型。重新训练成本很少在现有权重上回本。
- 你的上下文始终在 16k 以下。噪声基线可忽略。

## 发货

这节课产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉画像和训练预算，它为将差分注意力添加到新预训练运行或 LoRA 微调生成集成计划。

## 练习

1. 运行 `code/main.py`。验证在合成查询上报告的差分注意力信噪比高于标准 softmax 注意力。改变噪声振幅并显示标准注意力变得不可用的交叉点。

2. 计算从基线到 DIFF V1 和从基线到 DIFF V2 对于 7B 类模型（hidden=4096，heads=32，d_head=128，32 层）的参数量 delta。显示哪些组件增加了参数哪些保持不变。

3. 阅读 DIFF V1 论文第 3 节和 DIFF V2 HuggingFace 博客第 2 节。用两句话解释为什么 V1 每头 RMSNorm 是必要的，以及为什么 V2 可以移除它而不导致训练分歧。

4. 实现消融：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全减法）计算差分注意力。在合成查询上测量信噪比如何随 sweep 变化。找到最大化信噪比的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。显示 KV 缓存大小匹配具有相同（8, 32）配置的基线 GQA 模型。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 差分注意力 | "两个 softmax 相减" | 将 Q、K 分裂为两半，计算两个 softmax 图，减去第二个（按 lambda 缩放），然后乘以 V |
| 噪声基线 | "softmax 的非零尾" | softmax 在每个无关 Token 上放置的 O(1/N) 权重，在长上下文上累积为 O(1) |
| lambda | "减法标度" | 每头学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | "ICLR 2025 版本" | 原始差分 Transformer；为保持参数量减半头维度，需要自定义内核，decode 更慢 |
| DIFF V2 | "2026 年 1 月修复" | 加倍 Q 头保持 KV 头；匹配基线 decode 速度并与 FlashAttention 兼容 |
| 每头 RMSNorm | "V1 稳定器 | V1 在差分后应用的额外归一化；V2 为防止训练后期不稳定而移除 |
| 信噪比 | "浪费了多少注意力 | 真正信号位置上的权重与无关 Token 平均权重之比 |
| 中间丢失 | "长上下文失败模式 | 准确率在长上下文中间文档处下降的经验现象——DIFF 注意力减少这个 |
| 算术强度 | "每字节 FLOPs | V2 在 decode 时通过每次 KV 加载加倍查询增加；在内存受限 decode 中重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，带噪声消除理论和长上下文消融
- [Microsoft unilm — Differential Transformer V2 (Hugging Face 博客，2026 年 1 月)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产栈重写，匹配基线 decode，FlashAttention 兼容
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 为什么减法恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 减法所基于的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF 注意力瞄准的长上下文基准