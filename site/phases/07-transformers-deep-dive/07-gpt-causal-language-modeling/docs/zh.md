# GPT — 因果语言建模

> BERT 看到两边。GPT 只看过去。三角掩码是现代 AI 中最意义重大的单行代码。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 第 2 课（自注意力）、Phase 7 第 5 课（完整 Transformer）、Phase 7 第 6 课（BERT）
**时长：** ~75 分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个 token，token `t` 上的概率分布是什么？在此信号上训练——下一个 token 预测——你得到一个可以一次一个 token 生成任意文本的模型。

要端到端在完整序列上并行训练，你需要每个位置的预测只依赖于更早的位置。否则模型通过查看答案轻易作弊。

因果掩码做到了。它是在 softmax 之前添加到注意力分数的单一上三角 `-inf` 矩阵。softmax 后，这些位置变为 0。每个位置只能关注自己和更早的位置。并且因为你一次性应用于整个序列，你在一个前向传播中得到 N 个并行的下一个 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是纯解码器因果 transformer，核心循环相同。只是更大、更好的数据、更好的 RLHF。

## 概念

![因果掩码创建三角注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前将 `M` 加到原始注意力分数。`exp(-inf) = 0`，所以掩码位置贡献零权重。注意力矩阵的每行是仅在先前位置上的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒。对该领域的影响：一切。

### 并行训练，串行推理

训练：一次前向传播完整 `(N, d_model)` 序列，计算 N 个交叉熵损失（每位置一个），求和，反向传播。沿序列并行。这是 GPT 训练可扩展的原因——你在一次 GPU 传递中处理 1M token。

推理：你逐个 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）保存 `t1…tn` 的隐藏状态，这样你不必在每步重新计算它们。但推理时串行深度 = 输出长度。那是自回归税，也是为什么解码是每个 LLM 的延迟瓶颈。

### 损失——移位一

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对于每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的交叉熵。

你听说的每个 transformer LM 都在此损失上训练。预训练、微调、SFT——相同损失，不同数据。

### 解码策略

训练后，采样选择比人们想象的更重要。

| 方法 | 做什么 | 何时使用 |
|------|--------|---------|
| 贪心 | 每步 argmax | 确定性任务、代码补全 |
| 温度 | 除以 T 后采样 | 创造性任务，T 越高越多样 |
| Top-k | 仅从 top-k token 采样 | 杀死低概率尾 |
| Top-p（核） | 从累计概率 ≥ p 的最小集合采样 | 2020+ 默认；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024+；比 top-p 更好地拒绝长尾 |
| 投机解码 | 小模型提议 N 个 token，大模型验证 | 在相同质量下 2–3× 延迟减少 |

2026 年，min-p + 温度 0.7 是开放权重模型的合理默认。投机解码是任何生产推理栈的必备条件。

### 是什么让"GPT 配方"有效

1. **纯解码器。** 无编码器开销。每层一次注意力 + FFN。
2. **扩展。** 124M → 1.5B → 175B → 万亿。Chinchilla 扩展律（第 13 课）告诉你如何花费计算。
3. **上下文学习。** 在约 6B–13B 时出现。模型可以遵循 few-shot 示例而不微调。
4. **RLHF。** 在人类偏好上后训练，将原始预训练文本转换为聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 大规模稳定训练。

核心架构自 GPT-2 以来没有太大变化。所有有趣的事都发生在数据、规模和后训练中。

## 构建

### 第一步：因果掩码

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将其加到注意力分数。这就是整个机制。

### 第二步：2 层类 GPT 模型

堆叠两个解码器块（带掩码自注意力 + FFN，无交叉注意力）。添加 token 嵌入、位置编码和一个解嵌入（绑定到 token 嵌入矩阵——自 GPT-2 以来的标准技巧）。

### 第三步：端到端的下一个 token 预测

在 20-token 玩具词汇上，在每个位置产生 logit。相对于移位一目标计算交叉熵损失。无梯度——这是前向传播健全性检查。

### 第四步：采样

实现贪心、温度、top-k、top-p、min-p。在固定提示上运行每个并比较输出。一个采样函数 10 行。

## 使用

PyTorch，2026 年习惯：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在引擎盖下，`generate()` 运行前向传播，拉取最终位置 logit，采样下一个 token，附加它，重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）用重度优化实现相同循环——批处理预填充、连续批处理、KV 缓存分页、投机解码。

**GPT vs BERT，各一行：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定模型是否能生成。

## 交付

见 `outputs/skill-sampling-tuner.md`。该 skill 为新的生成任务选择采样参数，并在需要确定性解码时标记。

## 练习

1. **简单。** 运行 `code/main.py` 并验证因果注意力矩阵在 softmax 后是下三角的。抽查：第 3 行应该只在列 0–3 有权重。
2. **中等。** 为宽度 4 实现束搜索。在 10 个短提示上比较束-4 vs 贪心的困惑度。束总是赢吗？（提示：翻译通常如此，开放聊天则不然。）
3. **困难。** 实现投机解码：用一个微型 2 层模型作为草稿，用 6 层模型作为验证器。在 100 个长度 64 的完成上测量墙上时钟加速。确认输出与验证器的贪心匹配。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| 因果掩码 | "三角" | 上三角 `-inf` 矩阵，在注意力分数上添加，使位置 `i` 只看到位置 ≤ i。 |
| 下一个 token 预测 | "损失" | 模型分布在每个位置上相对于真实下一个 token 的交叉熵。 |
| 自回归 | "一次生成一个" | 将输出作为输入反馈；仅在训练时并行，生成时不并行。 |
| Logit | "softmax 前的分数" | LM head 在 softmax 之前的原始输出；采样在此进行。 |
| 温度 | "创造力旋钮" | 除以 T；T→0 = 贪心，T→∞ = 均匀。 |
| Top-p | "核采样" | 将分布截断到累计概率 ≥p 的最小集合；从剩余中采样。 |
| Min-p | "优于 top-p" | 保留 `p ≥ min_p × max_p` 的 token；适应截断到分布锐度。 |
| 投机解码 | "草稿 + 验证" | 便宜模型提议 N 个 token；大模型并行验证。 |
| 教师强制 | "训练技巧" | 在训练时输入真实前一个 token，而非模型的预测。每个 seq2seq LM 的标准。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 和上下文学习。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准的因果 LM 参考代码。