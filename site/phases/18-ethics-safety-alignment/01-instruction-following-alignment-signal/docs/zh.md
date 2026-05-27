# 指令跟随与对齐信号

> 每一篇后续对 RLHF 的批评都针对这条 Pipeline。在研究优化压力如何扭曲一个代理之前，你必须先理解这个代理。InstructGPT（Ouyang 等，2022）定义了参考架构：在指令-响应配对上进行监督微调（SFT），在成对偏好排序上训练奖励模型（RM），以及用 KL 惩罚项对抗 RM 进行 PPO 优化。一个 1.3B 的 InstructGPT 优于 175B 的 GPT-3。仅这一个结果就解释了为什么 2026 年每一家前沿实验室仍在使用 RLHF 形态的后训练 Pipeline。

**类型：** 学习
**语言：** Python（标准库，三阶段简化 Pipeline）
**前置知识：** Phase 10 · 06（SFT）、Phase 10 · 07（RLHF）、Phase 10 · 08（DPO）
**时长：** 约 45 分钟

## 学习目标

- 说出 InstructGPT Pipeline 的三个阶段，以及每阶段使用的损失函数。
- 解释为什么 1.3B 的指令微调模型在人类偏好评估中优于原始 175B GPT-3。
- 说明第 3 阶段 KL 惩罚项保护的是什么，以及去除它为何会导致模式寻求行为。
- 描述对齐税（alignment tax）以及 Ouyang 等人用它来缓解对齐税的 PPO-ptx 方法。

## 问题

预训练语言模型完成文本。它们不回答问题。问 GPT-3"写一个反转列表的 Python 函数"，你经常会得到另一个提示，因为大部分训练分布是网页文本——继续生成更多网页文本。模型在干它的工作——只是这个工作本身是错的。

每一家严肃实验室用来修复这个问题所使用的代理是人类偏好。两个补全交给评分者；评分者选出更好的一个；奖励模型学习评分者。然后一个 RL 循环将策略推向奖励模型打分高的输出。这就是完整 InstructGPT 论文三句话版本。其余的都是工程。

## 概念

### 第 1 阶段：监督微调（SFT）

收集 prompt-response 配对，其中响应是有良好意图的人类所写的内容。Ouyang 等使用了来自标注者的 13k 个 prompt 和 OpenAI API。使用标准交叉熵损失在此数据上微调基础模型。

SFT 给予你的：模型现在回答问题而不是续写文本。它没有给你的：在多个可行答案中评分者更偏好哪一个的任何信号。

### 第 2 阶段：奖励模型（RM）

对每个 prompt，从 SFT 模型采样 K 个补全。标注者对它们排序。训练一个奖励模型，对任意 prompt-response 配对打分，使得当 `y_w` 被优先于 `y_l` 时：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这就是 Bradley-Terry 成对偏好损失。RM 通常从 SFT 模型初始化，用一个标量头替换 LM 头。

奖励模型很小：175B InstructGPT 的 6B 就足够了。它们也很脆弱——论文第 5 节主要讨论的是在小规模下出现的 reward hacking 行为。

### 第 3 阶段：带 KL 惩罚的 PPO

定义目标：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化。KL 项使 `pi` 远离 SFT 策略太远。没有它，优化器找到对抗样本——字符串在 RM 下得分很高因为 RM 从未见过它们，而不是因为人类实际偏好它们。

KL 系数 `beta` 是最重要的 RLHF 超参数。太低：reward hacking。太 高：比 SFT 没有改进。

### 对齐税

RLHF 后，模型被人类偏好但在一些标准基准（SQuAD、HellaSwag、DROP）上退步。Ouyang 等称之为对齐税并用 PPO-ptx 修复：将预训练梯度混合进 RL 目标，使模型不会遗忘它从未被奖励过的下游任务的能力。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 成为标准。Anthropic、DeepMind 和 Meta 都使用其变体。

### 结果

一个 1.3B InstructGPT（SFT + RM + PPO-ptx）在标注者偏好中大约 70% 的时间优于 175B 基础 GPT-3。在来自生产流量的隐藏测试 prompt 上差距更大。从这个数字读出两点：

1. 对齐是不同于能力的轴。175B 模型有更多能力；1.3B 模型有更多对齐；标注者更喜欢对齐的那个。
2. 能力下限由基础模型设定。你不能将 RLHF 一个从未见过某些事实的基础模型调教成知道这些事实。

### 为什么这是 Phase 18 的参考点

后续课程中的每一项批评——reward hacking（第 2 节）、DPO（第 3 节）、谄媚（第 4 节）、CAI（第 5 节）、睡眠者代理（第 7 节）、对齐伪装（第 9 节）——都是针对这个 Pipeline 的某些部分。Reward hacking 攻击第 2 阶段。DPO 合并第 2 和第 3 阶段。CAI 替换人类标注者。谄媚表明标注者是有偏信号。对齐伪装表明策略可以绕过第 3 阶段。不先把这个 Pipeline 装进脑子里，你就无法跟随任何这些批评。

## 使用它

`code/main.py` 在一个玩具偏好数据上模拟三个阶段。基础"策略"是一个偏向动作 {A, B, C} 的硬币。第 1 阶段 SFT 在 200 个 prompt 上模仿标注者动作。第 2 阶段从 500 个成对排序中拟合一个 Bradley-Terry 奖励模型。第 3 阶段用 KL 惩罚运行一个简化的 PPO 更新到 SFT 策略。你可以观察奖励上升、KL 散度增长、策略漂移——以及关闭 KL 项来看 reward hacking 如何在 50 步更新内出现。

关注：
- `beta = 0.1` 与 `beta = 0.0` 时的奖励轨迹。
- 训练步骤中 KL(pi || pi_SFT) 的变化。
- 最终动作分布与标注者偏好的比较。

## 交付它

本课生成 `outputs/skill-instructgpt-explainer.md`。给定一个 RLHF Pipeline 描述或论文摘要，识别三个阶段中被修改的是哪个，每阶段使用什么损失，以及是否存在 KL 惩罚或等效正则化器。

## 练习

1. 运行 `code/main.py`。设 `beta = 0.0` 并报告 200 PPO 步后的动作分布。用一段话解释模式寻求行为。

2. 修改奖励模型使动作 B 有 +0.5 偏差（模拟奖励 bug）。用 `beta = 0.1` 运行 PPO。KL 惩罚能防止策略利用偏差吗？在什么 `beta` 时利用变得明显？

3. 读 Ouyang 等（arXiv:2203.02155）Figure 1。用 PPO 运行 1、5、20、100 步并测量相对于 SFT 模型的偏好来复现标注者偏好曲线。

4. 论文第 4.3 节报告 1.3B InstructGPT 大约 70% 的时间优于 175B GPT-3。为什么在隐藏生产 prompt 上比率高于标注者自己的 prompt？

5. 用 DPO（Phase 10 · 08）替换 PPO 损失在相同偏好数据上。比较最终策略漂移（到 SFT 的 KL）和最终奖励。哪个方法在匹配奖励下漂移更远？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| SFT | 指令微调 | 第 1 阶段：在 prompt-response 配对上的交叉熵微调 |
| 奖励模型 | RM | 在成对标签上用 Bradley-Tercy 训练的标量回归器 |
| Bradley-Terry | 成对偏好损失 | -log sigmoid(r_w - r_l)；将成对排序减少为二分类 |
| KL 惩罚 | 正则化器 | `beta * KL(pi || pi_SFT)` — 使 RL 策略靠近 SFT 锚点 |
| PPO-ptx | 带预训练混合的 PPO | 在 PPO 目标中加入一部分预训练对数似然以抵消对齐税 |
| 对齐税 | RLHF 回归 | RLHF 未针对的标准基准上的后 RLHF 下降 |
| 标注者偏好 | 真相 | 人类排序的样本；RM 是对它的统计代理，而非对"人类价值观"的代理 |

## 延伸阅读

- [Ouyang 等 — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — InstructGPT 论文，所有后续 RLHF Pipeline 的基础
- [Stiennon 等 — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — RLHF 用于摘要的前身
- [Christiano 等 — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — 原始基于偏好的 RL 公式
- [Bai 等 — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic 对 InstructGPT Pipeline 的 HH 扩展