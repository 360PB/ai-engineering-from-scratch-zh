# 直接偏好优化家族

> Rafailov 等（2023）表明 RLHF 的最优解在偏好数据方面有一个闭合形式，所以你可以跳过显式奖励模型直接优化策略。这个洞察产生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每一个修复一种 DPO 的失败模式。到 2026 年，直接对齐算法比 PPO 发行更多前沿后训练运行。但第 2 课的过度优化曲线仍然适用：DAAs 不能逃脱 Goodhart，只是移动了它咬的地方。

**类型：** 学习
**语言：** Python（标准库，六变体偏好损失比较器）
**前置知识：** Phase 18 · 01（InstructGPT），Phase 18 · 02（Reward hacking），Phase 10 · 08（DPO 基础）
**时长：** 约 75 分钟

## 学习目标

- 从 RLHF 带 KL 的最优解推导出 DPO 闭合形式。
- 陈述 IPO、KTO、SimPO、ORPO、BPO 各自修复 DPO 的哪个失败模式。
- 区分"隐式奖励差距"与"偏好强度"，并解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov 等（NeurIPS 2024）证明 DAAs 过度优化尽管没有显式 RM。

## 问题

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

所以奖励由最优策略与参考的比例隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然，分区函数 `Z(x)` 消去因为它只依赖于 `x`。剩下的是一个仅在策略参数上的损失——不需要奖励模型。这就是 DPO。

麻烦：推导假设最优可达、偏好数据在分布内、参考策略是真正的模式锚点。没有一个完全成立。每个家族成员修复一个不同的违反假设。

## 概念

### DPO（Rafailov 等，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个微弱的偏好可以产生任意大的差距。
- 损失驱动被选和被拒的对数概率朝相反方向。只要被拒的下降更快，它可以把被选的对数概率向下推。这是退化被选响应（Degraded Chosen Response）现象。
- 分布外的偏好（罕见被选对罕见被拒）产生任意的隐式奖励。

### IPO（Azar 等，2024）

恒等偏好优化用对偏好概率的恒等映射替换 log-sigmoid。损失变成对有界目标的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边界由 `1/(2 beta)` 有界。偏好强度和隐式奖励差距成比例。没有爆炸。

### KTO（Ethayarajh 等，2024）

Kahneman-Tversky 优化完全放弃成对结构。给定一个单一标注输出和一个二元" desirable"或"undesirable"信号，它映射到一个前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对增益和损失有不同权重（损失厌恶）。好处：你可以使用不成对数据，这更丰富。

### SimPO（Meng 等，2024）

简单偏好优化使训练信号与生成对齐。完全移除参考策略并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

有边界 `gamma` 以稳定。长度归一化移除了利用 DPO 长度偏差失败模式的动机（更长的 `y_w` 凭构造产生更大的对数概率差距）。

### ORPO（Hong 等，2024）

几率比偏好优化将偏好项添加到标准 SFT 负对数似然：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT 项是正则化器。从基础模型到对齐模型单阶段训练。没有单独的 SFT 检查点。

### BPO（ICLR 2026 提交，OpenReview id=b97EwMUWu7）

识别退化被选响应问题：DPO 保留排名 `y_w > y_l` 但 `y_w` 的绝对对数概率可能下降。BPO 添加一个单行校正，对被选响应向下移动进行惩罚。在 Llama-3.1-8B-Instruct 的数学推理上比 DPO 报告 +10.1% 准确率。

### 通用结果：DAAs 仍然过度优化

Rafailov 等"Direct Alignment Algorithms 中 Reward Model 过度优化的扩展律"（NeurIPS 2024）在多个数据集上用 DPO、IPO、SLiC 训练策略，跨越 KL 预算。金牌-奖励-vs-KL 曲线具有相同的 Gao 等峰值-和-崩溃形状。隐式奖励在训练期间查询分布外样本；KL 正则化不能稳定这个。

DAAs 不能逃脱 Goodhart。它们将受咬的表面从"奖励模型过度优化"变为"参考策略比过度优化"。通用修复——更好的数据、集成、早停——适用于两者。

### 在它们之间选择（2026）

- 如果你有大配对偏好数据：DPO 与保守 beta；如果长度偏差明显用 SimPO。
- 如果你有不成对二元反馈：KTO。
- 如果你想要从基础模型的单阶段 Pipeline：ORPO。
- 如果在 DPO 日志中看到退化被选对数概率：BPO。
- 如果偏好强度变化很大且 DPO 饱和：IPO。

每个实验室在电池上运行全部五种并为每项任务选择赢家。没有理由最优对数学推理和安全是一样的。

## 使用它

`code/main.py` 比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO）在偏好强度变化的玩具偏好数据集上。每个损失在相同 500 对样本上用一个小 softmax 策略优化。绘制每种方法的最终胜率、被选对数概率漂移和隐式奖励扩散。

## 交付它

本课生成 `outputs/skill-preference-loss-selector.md`。给定数据集统计（成对 vs 不成对、变量 vs 均匀偏好强度、长度分布）和目标（单阶段或 SFT 然后偏好），推荐一种偏好损失并报告它防止的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终被选对数概率下降。BPO 应保留更高的被选绝对概率——验证这一点。

2. 修改偏好数据使所有配对强度相等。六种方法中哪个最鲁棒？哪个退化？解释 IPO 这里的优势。

3. 使被拒响应平均比被选长 2 倍。在不改变其他任何东西的情况下，数值上展示 DPO 的长度利用和 SimPO 的修复。

4. Rafailov 等（NeurIPS 2024）声称 DAAs 过度优化。复现一个单点版本：绘制被选减被拒的 KL 差距并在大的 beta 处观察 DPO 过度优化。

5. 读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 添加到 DPO 的一行校正。根据 `code/main.py` 中的实现确认。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| DPO | 无奖励模型的 RLHF | 从 RLHF 最优闭合形式导出的损失；仅策略参数 |
| 隐式奖励 | 对数比 | `beta * log(pi(y|x) / pi_ref(y|x))` — DPO 隐含的奖励 |
| IPO | 有界 DPO | 用恒等替换 log-sigmoid；隐式奖励差距由 `1/(2 beta)` 上限 |
| KTO | 不成对 DPO | 对带损失厌恶的单一标签的前景理论效用 |
| SimPO | 无参考 DPO | 长度归一化对数似然 + 边界；无参考策略 |
| ORPO | 单阶段 DPO | NLL + 几率比偏好项；从基础模型一次通过训练 |
| BPO | 保留行为的 DPO | DPO 加上对被选响应绝对对数概率下降的惩罚 |
| 退化被选 | 被选下降 | DPO 使被选对数概率下降只要被拒的下降更快 |
| DAA | 直接对齐算法 | 跳过显式 RM 的任何偏好损失方法 |

## 延伸阅读

- [Rafailov 等 — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar 等 — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh 等 — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等 — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)