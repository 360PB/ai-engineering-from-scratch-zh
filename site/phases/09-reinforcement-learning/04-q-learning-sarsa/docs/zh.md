# 时序差分——Q-Learning 与 SARSA

> 蒙特卡洛要等到 episode 结束才能更新。TD 每步之后都通过引导下一个值估计来更新。Q-learning 是离策略且乐观的；SARSA 是在线策略且保守的。两者都只有一行代码。两者都是本 Phase 所有深度 RL 方法的基础。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 9 · 01（MDP），Phase 9 · 02（动态规划），Phase 9 · 03（蒙特卡洛）
**时间：** 约 75 分钟

## 问题

蒙特卡洛有效，但有两个昂贵的需求。它需要能终止的 episode，而且只能在最终回报到手后才更新。如果你的 episode 是 1000 步，MC 要等 1000 步才能更新任何东西。它高方差、低偏差，实践中很慢。

动态规划有相反的特点——零方差的引导 backup——但需要一个模型。

时序差分（TD）学习在两者之间取折中。从单个转移 `(s, a, r, s')` 构成一个一步目标 `r + γ V(s')`，然后推动 `V(s)` 向它靠近。不需要模型。不需要完整 episode。偏差来自在 RHS 上使用近似的 `V`，但方差比 MC 低得多，而且从第一步起就可以在线更新。

这就是现代 RL——DQN、A2C、PPO、SAC——所围绕的支点。本 Phase 的其余内容都是在你将在本课写出的一步 TD 更新上叠加函数近似和技巧的层次。

## 概念

![Q-learning vs SARSA：离策略 max vs 在线策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号里的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛需要 `α` 满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`）且所有状态被无限频繁访问。

**Q-learning。** 一个用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

这个 `max` 假设从 `s'` 开始将遵循*贪婪*策略，不管智能体实际采取什么动作。这种解耦使 Q-learning 在智能体通过 ε-greedy 探索时学习 `Q*`。Mnih et al. (2015) 将其转化为 Atari 上的深度 Q-learning（Lesson 05）。

**SARSA。** 一个在线策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称来自元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*执行的下一个动作 `a'`，而不是贪婪的 `argmax`。收敛到运行中任意 ε-greedy `π` 的 `Q^π`，在 `ε → 0` 的极限下成为 `Q*`。

**悬崖行走的差异。** 在经典的悬崖行走任务上（掉下悬崖 = 奖励 -100），Q-learning 学习沿着悬崖边缘的最优路径，但探索时偶尔会承受惩罚。SARSA 学习的是离悬崖一步远的更安全路径，因为它将探索噪声纳入其 Q 值。随着训练，两者都在 `ε → 0` 时达到最优。实践中这很关键：当探索在实际部署时仍在发生时，SARSA 的行为更保守。

**期望 SARSA。** 用 `π` 下 `Q(s', a')` 的期望值替换 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（没有 `a'` 的采样），相同在线策略目标。现代教材中通常是默认选项。

**n 步 TD 和 TD(λ)。** 在引导之前等待 `n` 步，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 做平均。大多数深度 RL 使用 3 到 20 之间的 `n`。

## 构建

### 步骤 1：ε-greedy 策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。*唯一*与 Q-learning 的区别在目标行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号是在线策略和离策略之间的区别。

### 步骤 3：学习曲线

跟踪每 100 个 episode 的平均回报。Q-learning 在简单确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 的 4×4 GridWorld 上，两者在约 2,000 个 episode 后都接近最优，`α=0.1, ε=0.1`。

### 步骤 4：与 DP 真值比较

运行值迭代（Lesson 02）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格 TD 智能体在 4×4 GridWorld 上 10,000 个 episode 后落在 `~0.5` 以内。

## 陷阱

- **初始 Q 值很重要。** 乐观初始化（对负奖励任务 `Q = 0`）鼓励探索。悲观初始化可能使贪婪策略永远陷入。
- **α 调度。** 常数 `α` 对非平稳问题可以。衰减 `α_n = 1/n` 在理论上给出收敛但实践中太慢——将 `α` 固定在 `[0.05, 0.3]`，监控学习曲线。
- **ε 调度。** 开始高（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（无限探索下极限贪婪）是收敛条件。
- **Q-learning 中的 max 偏差。** 当 `Q` 有噪声时，`max` 算子有向上偏差。导致过估计——Hasselt 的双 Q-learning（Lesson 05 的 DDQN 中使用）用两个 Q 表修复这个问题。
- **非终止 episode。** TD 可以在没有终止的情况下学习，但你需要要么限制步数要么在限制处正确引导。标准做法：将限制视为非终止，继续引导。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（元组，不是列表；舍入的浮点数元组，不是原始值）。

## 使用

2026 年 TD 格局：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在线策略安全关键 | SARSA / 期望 SARSA | 探索期间更保守。 |
| 高维状态 | DQN（Phase 9 · 05） | 神经网络 Q 函数配合重放缓冲区和目标网络。 |
| 连续动作 | SAC / TD3（Phase 9 · 07） | Q 网络上的 TD 更新；策略网络输出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO（Phase 9 · 08、12） | 通过 GAE 的 TD 风格优势计算的 Actor-Critic。 |
| 离线 RL | CQL / IQL（Phase 9 · 08） | 带保守正则化的 Q-learning。 |

2026 年论文中你读到的"RL"有 90% 是 Q-learning 或 SARSA 的某种变体。在读更深入内容之前，先在指尖理解这个表格更新。

## 交付

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: 为表格或小型特征 RL 任务在 Q-learning、SARSA、期望 SARSA 之间选择。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定一个表格或小型特征环境，输出：

1. 算法。Q-learning / SARSA / 期望 SARSA / n 步变体。一句话理由，与在线策略 vs 离线策略和方差相关。
2. 超参数。α、γ、ε、衰减调度。
3. 初始化。Q_0 值（乐观 vs 零）及理由。
4. 收敛诊断。目标学习曲线，如果有 DP 则做 `|Q - Q*|` 检查。
5. 部署警告。推理时探索将如何表现？需要 SARSA 的保守性吗？

拒绝将表格 TD 应用于状态空间 > 10⁶ 的情况。拒绝在未标注 max 偏差警告的情况下发布 Q-learning 智能体。标记任何训练期间 ε 始终保持 1.0 的智能体（无利用阶段）。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2,000 个 episode 的学习曲线（每 100 个 episode 的平均回报）。谁收敛更快？
2. **中等。** 构建悬崖行走环境（4×12，最后一行是悬崖，奖励 -100，重置到起点）。比较 Q-learning 和 SARSA 的最终策略。截图每个走的路径。哪个更靠近悬崖？
3. **困难。** 实现双 Q-learning。在带噪声奖励的 GridWorld 上（每步奖励加 Gaussian 噪声 σ=5），展示 Q-learning 过估计 `V*(0,0)` 一个有意义的量，而双 Q-learning 不会。

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| TD 误差 | "更新信号" | `δ = r + γ V(s') - V(s)`，引导残差。 |
| TD(0) | "一步 TD" | 每步转移后使用仅来自下一个状态估计的更新。 |
| Q-learning | "离线策略 RL 101" | 使用 `max` 跨下一状态动作的 TD 更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "在线策略 Q-learning" | 使用实际下一个动作的 TD 更新；学习当前 ε-greedy π 的 `Q^π`。 |
| 期望 SARSA | "低方差 SARSA" | 用 `π` 下采样的期望替换 `a'`。 |
| GLIE | "正确的探索调度" | 无限探索下极限贪婪；Q-learning 收敛所需。 |
| Bootstrapping | "在目标中使用当前估计" | 区分 TD 和 MC 的东西。偏差来源但大幅减少方差。 |
| 最大化偏差 | "Q-learning 过估计" | 对噪声估计取 `max` 是向上偏的；双 Q-learning 修复。 |

## 拓展阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文及收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、期望 SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏差修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — 期望 SARSA 动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 发明 SARSA 的论文（当时称为"改进的连接主义 Q-learning"）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，是从 Q-learning 到 eligibility traces 再到后来 PPO 中 GAE 的路径。