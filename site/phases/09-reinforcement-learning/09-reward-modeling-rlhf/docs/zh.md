# 奖励建模与 RLHF

> 人类无法写出"好的助手回复"的奖励函数，但他们可以比较两个回复并选择更好的那个。用这些比较来拟合奖励模型，然后用 RL 优化语言模型。Christiano 2017。InstructGPT 2022。将 GPT-3 变成 ChatGPT 的配方。2026 年它大多正被 DPO 取代——但思维模型留下来。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 05（情感分析），Phase 9 · 08（PPO）
**时间：** 约 45 分钟

## 问题

你在下一个 token 预测目标上训练了一个语言模型。它写语法正确的英语。但它也撒谎、啰嗦、拒绝拒绝。更多预训练无法修复它——网页文本是问题，不是解药。

你需要一个*标量奖励*说"对指令 X，回复 A 比回复 B 好"。手工写那个奖励函数是不可能的。"有帮助性"不是 token 上的闭式表达式。但人类可以比较两个输出并标记偏好。这很便宜，可以大规模收集。

RLHF（Christiano 等人 2017；Ouyang 等人 2022）将偏好转换为奖励模型，然后用 PPO 针对该奖励优化 LM。三步：SFT → RM → PPO。这是 2023–2025 年 ChatGPT、Claude、Gemini 和每个对齐 LLM 发货的配方。

2026 年 PPO 步骤大多被 DPO（Phase 10 · 08）取代，因为它更便宜，对齐调优质量几乎相同。但*奖励模型*部分仍然支撑着每个 Best-of-N 采样器、每个来自可验证奖励的 RL 流程、以及每个使用过程奖励模型的推理模型。理解 RLHF 就理解了整个对齐栈。

## 概念

![三阶段 RLHF：SFT、成对偏好上的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调（SFT）。** 从预训练基础模型开始。在目标行为（指令跟随回复、有帮助的回复等）的人工演示上微调。结果：一个模型 `π_SFT`，*偏向好的行为*但仍有无界动作空间。

**阶段 2：奖励模型训练。**

- 收集对提示 `x` 的回复对 `(y_+, y_-)`，由人类标记为"y_+ 优于 y_-"。
- 训练奖励模型 `R_φ(x, y)` 给 `y_+` 打更高分。
- 损失：**Bradley-Terry 成对 logistic**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励差异暗示偏好的赔率。BT 自 1952 年（Bradley-Terry）以来是标准，也是现代 RLHF 的主要选择。

- `R_φ` 通常从 SFT 模型初始化，上面加一个标量 head。同一个 transformer 主干；单层线性层输出奖励。

**阶段 3：针对 RM 运行带 KL 惩罚的 PPO。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保持一个冻结的*参考* `π_ref = π_SFT`。
- 响应 `y` 末端奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是*正则化器*，不是硬信任域。`β` 通常为 `0.01`-`0.05`。
- 用这个奖励运行 PPO（Lesson 08）。优势在 token 级轨迹上计算，但 RM 只对完整回复评分。

**为什么需要 KL？** 没有它，PPO 会乐意找到奖励黑客策略——RM 只在分布内补全上训练。一个分布外回复可能比任何人写的分数都高。KL 使 `π_θ` 保持在 RM 训练所在流形附近。这是 RLHF 中最重要的旋钮。

**2026 年状态：**

- **DPO**（Rafailov 2023）：封闭形式代数将阶段 2+3 崩溃为对偏好数据的单个监督损失。无 RM，无 PPO。在对齐基准上质量相同，计算量的一小部分。见 Phase 10 · 08。
- **GRPO**（DeepSeek 2024–2025）：用组相对基线代替 critic 的 PPO，奖励来自*验证器*（代码运行 / 数学答案匹配）而不是人工训练的 RM。对推理模型占主导。见 Phase 9 · 12。
- **过程奖励模型（PRM）：** 对部分解（每个推理步骤）评分，用于推理的 RLHF 和 GRPO 变体。
- **Constitutional AI / RLAIF：** 用对齐的 LLM 生成偏好而不是人类。扩展偏好预算。

## 构建

本课使用极简合成"提示"和"回复"，表示为字符串。RM 是词袋表示上的线性评分器。没有真正的 LLM——管道的*形状*重要，不是规模。见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中这被人类标注者替换。形状——`(prompt, 优选回复, 拒绝回复)`——是相同的。

### 步骤 2：Bradley-Terry 奖励模型

线性评分：`R(x, y) = w · bag(y)`。训练最小化 BT 成对 log 损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百步更新后，`w` 给好词 token 分配正权重，给坏词分配负权重。

### 步骤 3：在 RM 上运行类 PPO 策略

我们的玩具策略产生一个来自词汇表的单个 token。我们在 RM 上评分 token，计算 `log π_θ(token | prompt)`，加上对参考的 KL 惩罚，然后应用裁剪的 PPO 替代。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 在 theta 上类 PPO 更新，将 reward 视为回报
    ...
```

### 步骤 4：监控 KL

每次更新跟踪平均 `KL(π_θ || π_ref)`。如果超过 `~5-10`，策略已从 `π_SFT` 漂移很远——要么 `β` 在上升要么奖励黑客开始了。这是真实 RLHF 中的顶级诊断。

### 步骤 5：使用 TRL 的生产配方

一旦你理解了玩具管道，这里是同一个循环作为真实库用户编写的方式。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于阶段 2，`PPOTrainer`（内置 KL 到参考）用于阶段 3。

```python
# 阶段 2：从成对偏好训练奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行：{"prompt", "chosen", "rejected"} — Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 阶段 3：针对 RM 的 PPO，带对 SFT 参考的 KL 惩罚
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats 包括：mean_kl, clip_frac, value_loss — 三个 PPO 诊断
```

库为你做三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观察到的 KL 超过 `target_kl`，β 加倍；如果低于一半，β 减半。参考模型按惯例冻结——你不能与 `policy` 意外共享参数。值 head 与策略在同一主干上（`AutoModelForCausalLMWithValueHead` 附加一个标量 MLP head），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## 陷阱

- **过度优化 / 奖励黑客。** RM 不完美；`π_θ` 找到得分高但实际差的对抗补全。症状：奖励无限攀升而人类评估分数持平或下降。修复：提前停止，提高 `β`，扩大 RM 训练数据。
- **长度黑客。** 在有帮助回复上训练的 RM 通常隐式奖励长度。策略学会填充回复。补救：长度归一化奖励，或带长度感知 RM 的 RLAIF。
- **RM 太小。** RM 至少需要与策略一样大。小的 RM 无法忠实评分策略的输出。
- **KL 调优。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不变。标准技巧是针对每步固定 KL 的*自适应* β。
- **偏好数据噪声。** ~30% 的人类标签是噪声或模糊的。通过在一致过滤数据上训练 RM 来校准，或对 BT 使用温度。
- **离线问题。** PPO 数据在第一轮 epoch 后略微离线。监控 Lesson 08 中的裁剪比例。

## 使用

2026 年 RLHF 是分层的：

| 层 | 目标 | 方法 |
|-------|--------|--------|
| 指令跟随、有帮助、无害 | 对齐 | DPO（Phase 10 · 08）优于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（Phase 9 · 12）。 |
| 长视野多步任务 | Agentic | 带过程奖励模型的 PPO / GRPO。 |
| 安全 / 拒绝行为 | 安全 | 带独立安全 RM 的 RLHF-PPO，或 Constitutional AI。 |
| 推理时 Best-of-N | 快速对齐 | 在解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 上训练小"奖励 head"。 |

RLHF 在 2022–2024 年是*那个*方法。2026 年，生产对齐管道以 DPO 为先，PPO 仅用于 RM 密集或安全关键步骤。

## 交付

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: 为语言模型设计 RLHF / DPO / GRPO 对齐管道，包括 RM、KL 和数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定基础 LM、目标行为（对齐 / 推理 / 拒绝 / agent）和偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？附理由。
2. 偏好或验证器来源。人类、AI 反馈、基于规则、单元测试通过、或奖励蒸馏。
3. KL 策略。固定 β、自适应 β、或 DPO（隐式 KL）。
4. 诊断。平均 KL、奖励稳定性、过度优化保护（盲人类评估集）。
5. 安全门。红队集、拒绝率、独立于有帮助 RM 的安全 RM。

拒绝在无 KL 监控的情况下发布 RLHF-PPO。拒绝使用小于目标策略的 RM。拒绝纯长度奖励。标记任何没有保留盲人类评估集作为过度优化保护的管道。
```

## 练习

1. **简单。** 在 `code/main.py` 中的 500 个合成偏好对上训练 Bradley-Terry 奖励模型。在保留的 100 对上测量成对准确率。应超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对于每个，绘制 RM 分数 vs 跨更新的 KL 到参考。哪个运行会奖励黑客？
3. **困难。** 在相同偏好数据上实现 DPO（封闭形式偏好似然损失），并与 RLHF-PPO 管道在所用计算和达到的最终 RM 分数上比较。

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO 管道（Christiano 2017，Ouyang 2022）。 |
| 奖励模型（RM） | "评分网络" | 学到的标量函数，通过 Bradley-Terry 拟合成对偏好。 |
| Bradley-Terry | "成对 logistic 损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | "保持在参考附近" | 奖励中的 `β · KL(π_θ || π_ref)`；抗奖励黑客正则化器。 |
| 奖励黑客 | "Goodhart 定律" | 策略利用 RM 缺陷；症状：奖励上升，人类 eval 持平。 |
| RLAIF | "AI 标记的偏好" | RLHF 其中标签来自另一个 LM 而不是人类。 |
| PRM | "过程奖励模型" | 对部分推理步骤评分；用于推理管道。 |
| Constitutional AI | "Anthropic 的方法" | AI 生成的偏好由显式规则引导。 |

## 拓展阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开启 RLHF 的论文。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 更早的摘要 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年后 RLHF 默认。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 和自我批评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源码以了解自适应 KL 和值 head 细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 带图表的三阶段管道规范 walk-through。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 库；`examples/` 有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假设视角；思考奖励黑客的必备前提。