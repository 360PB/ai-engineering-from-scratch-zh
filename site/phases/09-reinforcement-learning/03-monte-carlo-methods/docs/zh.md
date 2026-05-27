# 蒙特卡洛方法——从完整 episode 中学习

> 动态规划需要一个模型。蒙特卡洛只需要 episode。运行策略，观察回报，平均它们。RL 中最简单的思想——也是解锁下游一切的那一个。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 9 · 01（MDP），Phase 9 · 02（动态规划）
**时间：** 约 75 分钟

## 问题

动态规划很优雅，但它假设你可以对每个状态和动作查询 `P(s' | s, a)`。现实世界几乎没有任何东西这样工作。机器人无法解析关节扭矩之后相机像素的分布。定价算法无法对每种可能的客户反应做积分。LLM 无法枚举一个 token 之后所有可能的续写。

你需要一种只需要能够从环境*采样*的方法。运行策略。得到一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用这个轨迹估计值。这就是蒙特卡洛。

从 DP 到 MC 的转变具有哲学上的重要性：我们从*已知模型 + 精确 backup* 移动到*采样 rollout + 平均回报*。方差增大了，但适用性爆炸式增长。本课之后的所有 RL 算法——TD、Q-learning、REINFORCE、PPO、GRPO——在本质上都是一个蒙特卡洛估计器，有时加上 bootstrapping 层。

## 概念

![蒙特卡洛：rollout，计算回报，平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想，一行：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问 `s` 时观察到的回报。

**首次访问 vs 每次访问 MC。** 给定一个多次访问状态 `s` 的 episode，首次访问 MC 只计算第一次访问的回报；每次访问 MC 计算所有访问。两者在极限下都是无偏的。首次访问更简单易分析（iid 样本）。每次访问每个 episode 使用更多数据，实践中通常收敛更快。

**增量均值。** 不存储所有回报，而是更新运行均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新整理：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成常数步长 `α ∈ (0, 1)`，你就得到了一个跟踪 `π` 变化的非平稳 MC 估计器。这一步就是从 MC 到 TD 再到每个现代 RL 算法的全部跳跃。

**探索现在是一个问题。** DP 通过枚举遍历每个状态。MC 只看到策略访问的状态。如果 `π` 是确定性的，整个状态空间区域永远不会被采样，它们的值估计永远停留在零。三种修复方法，按历史顺序：

1. **探索性起始。** 从随机 (s, a) 对开始每个 episode。保证覆盖率；实践中不现实（你不能将机器人"重置"到任意状态）。
2. **ε-greedy。** 以当前 Q 贪婪行动，但以概率 `ε` 随机选择动作。所有状态-动作对在渐近期都能被采样。
3. **离线 MC。** 在行为策略 `μ` 下收集数据，通过重要性采样学习目标策略 `π`。高方差，但这是通向 DQN 等重放缓冲区方法的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代一样，但评估是基于采样的：

1. 运行 `π`，得到一个 episode。
2. 从观察到的回报更新 `Q(s, a)`。
3. 使 `π` 对 `Q` ε-greedy。
4. 重复。

在温和条件下（每个状态-动作对被无限频繁访问，`α` 满足 Robbins-Monro）以概率 1 收敛到 `Q*` 和 `π*`。

## 构建

### 步骤 1：rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有模型，只有 `env.reset()` 和 `env.step(s, a)`。与 gym 环境接口相同，但简化了。

### 步骤 2：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次扫描，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免重复求和。

### 步骤 3：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行做全部工作：首次访问时标记状态已见、增加计数、更新运行均值。

### 步骤 4：ε-greedy MC 控制（在线策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步骤 5：与 DP 金牌标准比较

你的 MC 对 `V^π` 的估计应该在 episodes → ∞ 时与 Lesson 02 的 DP 结果一致。实践中：4×4 GridWorld 上 50,000 个 episode 能让你与 DP 答案的差距在约 `~0.1` 以内。

## 陷阱

- **无限 episode。** MC 要求 episode 必须*终止*。如果你的策略可能永远循环，用 `max_steps` 设置上限并将上限视为隐式失败。随机策略下的 GridWorld 经常超时——那是正常的，只需确保你正确地计数了。
- **方差。** MC 使用完整回报。在长 episode 上，方差巨大——结束时的单个不幸奖励会以相同量级移动 `V(s_0)`。TD 方法（Lesson 04）通过 bootstrapping 削减这个。
- **状态覆盖率。** 在全新的 Q 上做贪婪 MC 时遇到平局只会尝试一个动作。你*必须*探索（ε-greedy、探索性起始、UCB）。
- **非平稳策略。** 如果 `π` 变化（如在 MC 控制中），旧的回报来自不同的策略。常数-α MC 可以处理；样本平均 MC 不能。
- **离线重要性采样。** 权重 `π(a|s)/μ(a|s)` 在轨迹上相乘。方差随视野爆炸。用每步加权 IS 截断或切换到 TD。

## 使用

2026 年蒙特卡洛方法的作用：

| 用例 | 为什么用 MC |
|----------|--------|
| 短视野游戏（21点、扑克） | Episode 自然终止；回报干净。 |
| 离线评估已记录策略 | 对存储的轨迹平均折扣回报。 |
| 蒙特卡洛树搜索（AlphaZero） | MC rollout 从树叶引导选择。 |
| LLM RL 评估 | 对给定策略的采样完成计算平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC `G_t`。 |
| RL 教学 | 最简单且实际有效的算法——去掉 bootstrapping 看核心。 |

现代深度 RL 算法（PPO、SAC）通过 n 步回报或 GAE 在纯 MC（完整回报）和纯 TD（一步 bootstrapping）之间插值。两个端点都是同一估计器的实例。

## 交付

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: 通过蒙特卡洛 rollout 评估策略，并在有条件时产出与 DP 比较的收敛报告。
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定一个环境（episodic，有 reset+step API）和一个策略，输出：

1. 方法。首次访问 vs 每次访问 MC。理由。
2. Episode 预算。目标数量、方差诊断、预期标准误差。
3. 探索计划。ε 调度（如果需要）或探索性起始。
4. 金牌标准比较。如果可表格化则用 DP 最优 V*；否则用 Q-learning / PPO 基线给出界。
5. 终止检查。最大步数上限、超时、非终止轨迹的处理。

拒绝在非 episodic 任务上没有有限视野上限的情况下运行 MC。拒绝报告表格任务每个状态少于 100 个 episode 的 V^π 估计。标记任何零方差动作的策略为探索风险。
```

## 练习

1. **简单。** 对 4×4 GridWorld 上的均匀随机策略实现首次访问 MC 评估。运行 10,000 个 episode。将 `V(0,0)` 作为 episode 数量的函数绘制，并与 DP 答案比较。
2. **中等。** 实现 ε-greedy MC 控制，`ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个 episode 后的平均回报。曲线是什么样的？偏差-方差权衡在哪里？
3. **困难。** 用重要性采样实现*离线* MC：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较朴素 IS vs 每步 IS vs 加权 IS。哪个方差最低？

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| 蒙特卡洛（Monte Carlo） | "随机采样" | 通过从分布中平均 iid 样本来估计期望。 |
| 回报 `G_t` | "未来奖励" | 从步骤 `t` 到 episode 结束的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC | "每个状态只计数一次" | 只有 episode 中第一次访问才贡献值估计。 |
| 每次访问 MC | "使用所有访问" | 每次访问都贡献；略微有偏但样本效率更高。 |
| ε-greedy | "探索噪声" | 以概率 `1-ε` 选贪婪动作；以概率 `ε` 选随机动作。 |
| 重要性采样 | "纠正从错误分布采样" | 用 `π(a|s)/μ(a|s)` 乘积权重来重新加权回报，以从 `μ` 数据估计 `V^π`。 |
| 在线策略（On-policy） | "从自己的数据中学习" | 目标策略 = 行为策略。朴素 MC、PPO、SARSA。 |
| 离线策略（Off-policy） | "从别人的数据中学习" | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 拓展阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 标准处理。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 首次访问 vs 每次访问分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 离线 MC 和方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自对弈收敛到超人类水平的首次大规模实证演示；本 Phase 第二半各课的概念先驱。