# 投机解码 — 草稿、验证、重复

> 自回归解码是串行的。每个 token 等待前一个。投机解码打破链条：便宜模型草稿 N 个 token，昂贵模型一次验证所有 N。当草稿正确时，你用一个大的前向传递支付 N 次生成。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 第 7 课（GPT 因果 LM）、Phase 7 第 12 课（KV 缓存 & Flash Attention）
**时长：** ~60 分钟

## 问题

70B LLM 在 H100 上采样一个 token 约需 30 ms。3B 草稿模型约需 3 ms。如果我们让 3B 草稿领先 5 个 token，然后运行 70B 一次验证所有 5，总计 `5×3 + 30 = 45 ms` 用于最多 5 个接受的 token——而直线生成是 `5×30 = 150 ms`。这是完整投机解码的卖点：用少量额外 GPU 内存（草稿模型）换取 2–4× 更低解码延迟。

这个技巧必须保留分布。Leviathan 等人（2023）引入的投机采样及陈等人同时引入，保证输出序列与大型模型自己采样时产生的**完全相同分布**。无质量权衡。只是更快。

2026 年四种草稿-验证对主导推理：

1. **朴素投机（Leviathan 2023）。** 分离草稿模型（例如 Llama 3 1B）+ 验证器（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上的多个解码头并行预测位置 `t+1..t+k`。无分离草稿模型。
3. **EAGLE 家族（Li 2024，2025）。** 轻量级草稿重用验证器的隐藏状态；比朴素更高的接受率；典型 3–4×。
4. **Lookahead 解码（Fu 2024）。** 雅可比迭代；根本不需要草稿模型。自投机。利基但无依赖。

2026 年每个生产推理栈默认发货投机解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都支持至少朴素 + EAGLE-2。

## 概念

### 核心算法

给定验证器 `M_q` 和更便宜的草稿 `M_p`：

1. 令 `x_1..x_k` 为已解码的前缀。
2. **草稿**：用 `M_p` 自回归提议 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，草稿概率 `p_1..p_N`。
3. **并行验证**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上运行 `M_q` 一次，得到位置 `k+1..k+N+1` 的验证概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个草稿 token**：对于每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次拒绝时：从"残差"分布 `(q_j - p_j)_+` 归一化采样 `t_j`。丢弃 `j` 之后的所有草稿。
6. 接受全部 N 时：从 `q_{N+1}` 采样一个额外 token `t_{N+1}`（免费奖励 token）。

残差分布技巧是保持输出完全与 `M_q` 自己采样相同分布的数学洞察。

### 什么决定加速

令 `α` = 每个草稿 token 的预期接受率。令 `c` = 草稿到验证器成本比率。每步：

- 朴素生成每个 token 做 1 次大模型调用。
- 投机每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` token 做 1 次大模型调用，当 `α` 高时。

`α = 0.75` 和 `N = 5` 的典型经验法则：减少 3× 大模型调用。草稿成本 5× 便宜。总墙钟下降约 2.5×。

**α 取决于：**

- 草稿近似验证器的程度。相同家族 / 相同训练数据显著提升 α。
- 解码策略。朴素草稿对朴素验证器：高 α。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由形式创意写作接受更少。

### Medusa — 无草稿模型的草稿

Medusa 用验证器上的额外输出头替换草稿模型。在位置 `t`：

```
共享 trunk → 隐藏 h_t
    ├── head_0: 预测 t+1 位置的 token（标准 LM head）
    ├── head_1: 预测 t+2
    ├── head_2: 预测 t+3
    ├── head_3: 预测 t+4
```

每个头输出自己的 logit。在推理时从每个头采样得到候选序列，然后用一次前向传递使用树注意力方案验证所有候选延续。

优点：无第二个模型。缺点：添加可训练参数；需要监督微调阶段（约 1B token）；与好草稿的朴素投机相比接受率略低。

### EAGLE — 通过重用隐藏状态获得更好的草稿

EAGLE-1/2/3（Li 等人，2024–2025）使草稿模型成为一个tiny transformer（通常 1 层），摄取验证器的最后一层隐藏状态。因为草稿看到验证器的特征表示，其预测与验证器输出分布强相关。接受率从 ~0.6（朴素）攀升到 0.85+。

EAGLE-3（2025）添加了对候选延续的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认 spec 路径。

### KV 缓存舞蹈

验证在一次前向传递中将 N 个草稿 token 送入验证器。这将验证器的 KV 缓存扩展 N 个条目。如果某些草稿被拒绝，你必须将缓存回滚到接受前缀长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用临时 KV 缓冲区处理这个。先写入，接受时提交。它在概念上不难，但很繁琐。

## 构建

见 `code/main.py`。我们实现核心投机采样算法（拒绝步骤 + 残差分布），包含：

- 一个"大模型"，它是手工编码分布上的确定性 softmax（这样我们可以分析验证接受数学）。
- 一个"草稿模型"，它是该大模型的扰动。
- 接受/拒绝循环，产生与直接采样相同的边际分布。

### 第一步：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对草稿 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理是：这个伯努利决策，然后在拒绝时从残差采样，完全保留验证器分布。

### 第二步：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 减去 `p`，将负值钳位为零，重新归一化。在任何拒绝时从中采样。

### 第三步：一次投机步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

5 个接受 → 一个奖励 → 一次验证器传递产生 6 个 token。

### 第四步：测量接受率

在各种草稿质量水平上运行 10,000 次投机步骤。绘制接受率 vs 草稿和验证器分布之间的 KL 散度。你应该看到一个干净的单调关系。

### 第五步：验证分布等价

经验上：投机循环产生的 token 直方图应与直接从验证器采样产生的直方图匹配。这是 Leviathan 定理在实践中的应用。卡方检验确认在采样误差范围内。

## 使用

生产：

```bash
# vLLM + EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM + 朴素草稿模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM 有截至 2026 年中最快的 Medusa 路径。`faster-whisper` 用小型草稿包装 Whisper-large 的投机解码。

**选择草稿：**

| 策略 | 何时选 | 加速 |
|------|--------|------|
| 朴素草稿（1B/3B Llama 家族） | 快速原型，无训练 | 1.8–2.3× |
| Medusa 头 | 你可以微调验证器 | 2–3× |
| EAGLE-2 / 3 | 生产，最大速度 | 3–4× |
| Lookahead | 无草稿，无训练，无额外参数 | 1.3–1.6× |

**何时不投机解码：**

- 1–5 token 的单序列生成。开销占主导。
- 极度创意/高温采样（α 下降）。
- 内存受限部署（草稿模型增加 VRAM）。

## 交付

见 `outputs/skill-spec-decode-picker.md`。该 skill 为新推理工作负载选择投机解码策略（朴素 / Medusa / EAGLE / lookahead）和调优参数（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。在 50,000 token 上确认投机 token 分布与验证器直接采样分布在卡方 p > 0.05 范围内匹配。
2. **中等。** 绘制加速（每次大模型前向的 token）作为 `N` 的函数，用于 `α = 0.5, 0.7, 0.85`。确定每个 α 的最优 `N`。（提示：每次验证调用的预期 token = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个微型 Medusa：取第 14 课的顶点 GPT，添加 3 个额外 LM head 预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用联合多头损失训练。比较接受率 vs 由截断同一模型制成的朴素草稿。
4. **困难。** 实现回滚：从 10-token 前缀 KV 缓存开始，喂入 5 个草稿 token，模拟在位置 3 拒绝。验证你的缓存读取在下次迭代中正确匹配"前缀 + 前 2 个接受草稿"。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| 草稿模型 | "便宜的那个" | 提出候选 token 的较小模型；通常比验证器便宜 10–50×。 |
| 验证器 | "大的那个" | 我们保留其分布的目标模型；每次投机步骤运行一次。 |
| 接受率（α） | "草稿多久正确一次" | 验证器接受草稿的每个 token 概率。典型 0.7–0.9。 |
| 残差分布 | "拒绝后备" | `(q - p)_+` 归一化；在拒绝时采样保留验证器分布。 |
| 奖励 token | "免费的那个" | 当全部 N 个草稿被接受时，从验证器的下一步分布采样一个。 |
| Medusa | "无草稿投机" | 验证器上的多个 LM head 并行预测位置 t+1..t+k。 |
| EAGLE | "隐藏状态草稿" | tiny transformer 草稿以验证器最后一层隐藏状态为条件。 |
| Lookahead 解码 | "雅可比迭代" | 使用不动点迭代的自投机；无草稿模型。 |
| 树注意力 | "一次验证多个候选" | 分支验证，同时考虑多个草稿延续。 |
| KV 回滚 | "撤销拒绝的草稿" | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法和等价定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 并发介绍；干净的伯努利-拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；以隐藏状态为条件的草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，无草稿方法。
- [vLLM 文档 — 投机解码](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 带全部四种策略接入的标准生产参考。
- [SafeAILab / EAGLE 参考实现](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。