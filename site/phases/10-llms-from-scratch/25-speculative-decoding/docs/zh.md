# 投机解码与 EAGLE

> 前沿 LLM 每生成一个 token 都需要对数十亿参数做一次完整的前向传播。这个前向传播严重过度配置：大多数时候，一个小得多的模型可以正确猜测下一个 3-5 个 token，而大模型只需要*验证*这些猜测。当猜测正确时，你用一个 token 的代价获得了 5 个 token。投机解码（Leviathan et al., 2023）将这一思想形式化，而 EAGLE-3（2025）将接受率提升至每个验证约 4.5 个 token——在输出分布不变的前提下实现 4-5 倍加速。

**类型：** Build（构建）
**语言：** Python（含 numpy）
**前置要求：** Phase 10 Lesson 12（推理优化）、Phase 10 Lesson 04（从零训练 Mini-GPT）
**时长：** 约 75 分钟

## 问题背景

在 H100 上，一个 70B 类模型的解码吞吐量通常为 40-80 tokens/秒。每个 token 都需要一次完整的前向传播，从 HBM 读取所有模型权重。你不能在不改变输出的情况下让模型变小。你无法在超出内存限制的情况下增大批大小。你被困住了——除非你能让模型每次前向传播输出多个 token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这其中存在并发机会。如果你有一个廉价的预测器说"接下来 4 个 token 可能是 [a, b, c, d]"，你可以在大模型的**单次前向传播**中并行验证所有 5 个位置，然后接受最长匹配的前缀。

Leviathan, Kalai, Matias（2023，《通过投机解码实现快速推理》）通过一个巧妙的接受/拒绝规则使这一想法精确化，同时保留了目标模型的采样分布。相同的输出分布，2-4 倍的速度提升。

## 核心概念

### 双模型架构

- **目标模型** `M_p`：那个又大又慢但高质量的模型，你真正想从它那里采样的对象。分布：`p(x)`。
- **草稿模型** `M_q`：一个小巧、快速但质量较低的模型。分布：`q(x)`。比目标模型小 5-30 倍。

每个步骤：

1. 草稿模型自回归地提出 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型在所有 `K+1` 个位置上运行**一次前向传播**并行计算，产生每个被提议 token 的 `p(x_k)`。
3. 通过下面修改后的拒绝采样规则从左到右接受/拒绝每个 token。接受最长匹配的前缀。
4. 如果任何 token 被拒绝，从修正后的分布中采样替换 token 并停止。否则从 `p(· | x_1...x_K)` 中采样一个额外 token。

如果草稿与目标完美匹配，你每次目标前向传播获得 K+1 个 token。如果草稿在位置 1 就错了，你只获得 1 个 token。

### 精确性规则

投机解码**在分布上与从 p 采样严格等价**。拒绝规则：

```
对于每个草稿 token x_t：
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t)：
        接受 x_t
    else：
        从残差分布中采样替换：(p - q)+ / ||(p - q)+||_1
        停止
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿与目标一致时（`p ≈ q`），接受率接近 1。当它们不一致时，残差分布被构造使得最终采样仍然严格等于 `p`。

**贪婪情况。** 对于 temperature=0 采样，只需检查 `argmax(p) == x_t`。是则接受；否则输出 `argmax(p)` 并停止。

### 预期加速比

如果草稿模型的 token 级接受率为 `α`，每次目标前向传播的预期产出 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 草稿长度，α ∈ [0, 1]
```

当 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` tokens/前向传播。单一目标前向的成本大致为 `cost_q * K + cost_p`（K 步草稿 + 一步目标验证）。如果 `cost_p >> cost_q * K`，吞吐量加速比为 `3.36× / 1 = 3.36×`。

唯一真正的参数是 `α`，它完全取决于草稿-目标的匹配程度。一个好的草稿就是一切。

### 训练草稿：蒸馏

随机的小模型是糟糕的草稿。标准配方是从目标模型蒸馏：

1. 选取一个小架构（对于 70B 目标约 1B，对于 7B 目标约 500M）。
2. 在大量文本语料上运行目标模型；存储其下一个 token 的分布。
3. 用 KL 散度在目标分布（而非真实 token）上训练草稿。

结果：在代码任务上 `α` 通常为 0.6-0.8，在自然语言聊天上为 0.7-0.85。生产环境中的加速比为 2-3 倍。

### EAGLE：树形草稿 + 特征复用

Li, Wei, Zhang, Zhang（2024，《EAGLE：投机采样需要重新思考特征不确定性》）观察到标准投机解码的两个低效之处：

1. 草稿执行 K 步串行操作，每步都是完整计算。但草稿可以复用目标模型在最新一次验证时的特征（隐藏状态）——目标已经计算了丰富的表示，而草稿是从零重新推导的。
2. 草稿输出一条线性链。如果草稿可以输出一棵**候选树**（每个节点多个猜测），目标的单次前向传播可以通过树注意力掩码并行验证多条候选路径，并选择最长接受的分支。

EAGLE-1 的改变：
- 草稿输入 = 目标在位置 t 的最终隐藏状态，而非原始 token。
- 草稿架构 = 1 层 transformer 解码器层（而非独立的小模型）。
- 输出 = 每层深度 K = 4-8 个候选，分支深度 4-6。

EAGLE-2（2024）添加了动态树拓扑结构：树在草稿不确定时扩展得更宽，在确定时保持狭窄。在不增加验证成本的情况下提高了 `α_effective`。

EAGLE-3（Li et al., 2025，《EAGLE-3：通过训练时测试扩大 LLM 推理加速规模》）去除了对顶层特征的固定依赖，并在新的"测试时模拟"损失下训练草稿——草稿在匹配目标测试时分布的输出上训练，而非教师强制训练分布。接受率从 0.75（EAGLE-2）提升至 0.82（EAGLE-3），每验证的平均 token 数从 3.0 提升至 4.5。

### 树注意力验证

当草稿输出一棵树时，目标模型使用**树注意力掩码**在单次前向传播中验证它——这是一种编码树拓扑而非纯线条的因果掩码。每个 token 只注意树中它的祖先。验证前向传播仍然是一次前向、一次矩阵乘法；拓扑掩码只增加几个额外的 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争性的第一个 token 候选，`c, d, e, f` 是第二个 token 候选，则所有六个位置在一次前向传播中被验证。输出是任意接受路径上的最长前缀。

### 何时有效，何时无效

**有效场景：**
- 可预测文本的聊天/补全（代码、常用英语、结构化输出）。`α` 较高。
- 解码阶段有未使用 GPU 算力的设置（内存受限阶段）。树形草稿利用可用的 FLOP。

**无效/无提升场景：**
- 高随机性输出（高温创造性写作）。`α` 下降到接近 `1/|vocab|`。
- 高并发批量服务——批处理已经消耗了 FLOP，没有剩余空间进行树验证。
- 非常小的目标模型，此时草稿并不比目标小多少。

生产团队通常报告聊天场景下墙钟时间加速 2-3 倍，代码生成场景 3-5 倍，创造性写作场景几乎为零。

## 构建

`code/main.py`：

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确的拒绝规则，并验证它保持目标分布（经验 KL < 0.01 与普通目标采样相比）。
- 一个 EAGLE 风格的树形草稿器，用 top-p 分支构建深度 K 的树。
- 一个树注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试工具，在一个小型 LM 上同时运行（从一个 GPT-2-medium 目标蒸馏一个 GPT-2-small 草稿）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一轮投机解码。返回接受的 token 列表。"""
    # 1. 草稿 K 个 token
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标在每个草稿位置 + 1 个额外位置上计算 p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 所有 K 个都被接受 → 从目标采样一个额外 token
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用

- **vLLM** 和 **SGLang** 提供原生投机解码支持。参数：`--speculative_model`、`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 起草）、`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 起草）。
- **Medusa heads**（Cai et al., 2024，《Medusa：一个简单 LLM 推理加速框架》）不是使用独立的草稿模型，而是在目标本身添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 发布

本课生成 `outputs/skill-speculative-tuning.md`——一份技能文档，对目标模型的工作负载进行性能分析并选择：草稿模型、K（草稿长度）、树宽度、temperature，以及何时回退到普通解码。

## 练习

1. 实现精确的拒绝规则并经验验证。运行 10K 次 `speculative_decode` 和普通目标采样；计算两个输出分布之间的 TV 距离。应小于 0.01。

2. 计算加速比公式。给定固定的 `α` 和 `K`，绘制每次目标前向传播的预期 token 数。为 α ∈ {0.5, 0.7, 0.9} 找到最优 K。

3. 训练一个小草稿。用 KL 损失将一个 124M GPT-2 目标蒸馏为一个 30M GPT-2 草稿（在 100M tokens 上）。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格树形草稿。与链式不同，让草稿在每个深度输出 top-3 分支。构建树注意力掩码。验证目标接受最长正确分支。

5. 测试失效模式。在 temperature=1.5（高随机性）下运行投机解码。显示 `α` 崩溃，由于草稿开销算法比普通解码更慢。

## 关键术语

| 术语 | 别人怎么说 | 实际含义 |
|------|-----------|----------|
| Target model | "大模型" | 你想从中采样的慢速、高质量模型（p 分布） |
| Draft model | "推测器" | 小型快速预测器（q 分布）；比目标小 5-30 倍 |
| K / draft length | "前瞻" | 每次验证中推测的 token 数 |
| α / acceptance rate | "命中率" | 草稿提议被接受的单个 token 概率 |
| Exact rejection rule | "接受测试" | r < p/q 的比较，保持目标分布不变 |
| Residual distribution | "修正后的 p-q" | (p - q)+ / ||(p - q)+||_1，在拒绝时采样的分布 |
| Tree drafting | "分支推测" | 草稿输出一棵候选树，在一次前向传播中用树结构注意力掩码验证 |
| Tree attention mask | "拓扑掩码" | 编码树拓扑的因果掩码，使每个节点只注意其祖先 |
| Medusa heads | "并行头" | 目标本身上的 K 个额外预测头；无需独立草稿模型 |
| EAGLE feature reuse | "隐藏状态草稿" | 草稿输入是目标的最后隐藏状态，而非原始 token，减小草稿规模 |
| Test-time simulation loss | "EAGLE-3 训练" | 在匹配目标测试时分布的输出上训练草稿，而非教师强制分布 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 并发投机采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 并行头的替代草稿模型方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用和树形草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时-测试时匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，无需独立推测器