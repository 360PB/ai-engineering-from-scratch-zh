# 多智能体强化学习

> 单智能体 RL 假设环境是平稳的。在同一个世界中放置两个学习智能体，这个假设就失效了：每个智能体是另一个环境的一部分，而且两者都在变化。当马尔可夫假设不再成立时，多智能体 RL 是一套让学习收敛的技巧。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 9 · 04（Q-learning），Phase 9 · 06（REINFORCE），Phase 9 · 07（Actor-Critic）
**时间：** 约 45 分钟

## 问题

学习导航一个房间的机器人是一个单智能体 RL 问题。一支足球队不是。AlphaStar vs StarCraft 对手不是。投标智能体的市场不是。两辆车在四路停车标志谈判不是。许多对许多的现实世界问题不是。

在每个多智能体设置中，从任何一个智能体的角度看，其他智能体*是*环境的一部分。随着它们学习和改变行为，环境变得非平稳。马尔可夫性质——"下一状态仅取决于当前状态和我的动作"——被违反了，因为下一状态还取决于*其他*智能体选择了什么，而他们的策略是移动的目标。

这打破了表格收敛证明（Q-learning 的保证假设平稳环境）。也打破了朴素的深度 RL：智能体在循环中互相追逐，永远不会收敛到稳定策略。你需要多智能体特定技术：中心化训练 / 分散执行、反事实基线、联盟学习、自对弈。

2026 年应用：机器人群体、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（Phase 16），以及任何有多于一个智能玩家的游戏。

## 概念

![四种 MARL 机制：独立、中心化 critic、自对弈、联盟](../assets/marl.svg)

**形式化：马尔可夫博弈。** MDP 的推广：状态 `S`、联合动作 `a = (a_1, …, a_n)`、转移 `P(s' | s, a)`、以及每智能体奖励 `R_i(s, a, s')`。每个智能体 `i` 在自己的策略 `π_i` 下最大化自己的回报。如果奖励相同，是**完全合作**的。如果零和，是**对抗**的。如果混合，是**一般和**的。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看 `P(s' | s, a_i)` 依赖于 `π_{-i}`，而它在变化。
- **信用分配。** 有共享奖励时，哪个智能体造成了它？
- **探索协调。** 智能体必须探索互补策略，而不是冗余地探索相同状态。
- **可扩展性。** 联合动作空间在 `n` 中指数增长。
- **部分可观测。** 每个智能体只看到自己的观测；全局状态是隐藏的。

**四种主要机制：**

**1. 独立 Q-learning / 独立 PPO（IQL、IPPO）。** 每个智能体学习自己的 Q 或策略，将其他智能体视为环境的一部分。简单，有时有效（特别是经验回放作为平滑智能体建模技巧时）。理论收敛：无。实践中：对松散耦合任务好，对紧耦合任务坏。

**2. 中心化训练，分散执行（CTDE）。** 最常见的现代范式。每个智能体有自己的*策略* `π_i`，条件于局部观测 `o_i`——部署时标准分散执行。在*训练*期间，中心化 critic `Q(s, a_1, …, a_n)` 条件于完整全局状态和联合动作。例子：
- **MADDPG**（Lowe 等人 2017）：带中心化 critic 的 DDPG。
- **COMA**（Foerster 等人 2017）：反事实基线——问"如果我采取了动作 `a'` 而不是，我的奖励会是什么？"——隔离我的贡献。
- **MAPPO** / **IPPO** 带共享 critic（Yu 等人 2022）：带中心化值函数的 PPO。2026 年合作 MARL 的主导。
- **QMIX**（Rashid 等人 2018）：值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，单调混合。

**3. 自对弈。** 同一智能体的两个副本互相对弈。对手的策略*就是*我过去快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合零和游戏；训练信号是对称的。

**4. 联盟学习。** 自对弈在一般和 / 对抗环境上的扩展：保留过去和当前策略的人口，从联盟中采样对手，训练它们。添加探索者（专门击败当前最佳）和主探索者（专门击败探索者）。AlphaStar（StarCraft II）。当游戏允许"石头剪刀布"策略循环时需要。

**通信。** 允许智能体相互发送学到的消息 `m_i`。在合作设置中有效。Foerster 等人（2016）表明可微分的智能体间通信可以端到端训练。今天的基于 LLM 的多智能体系统（Phase 16）本质上用自然语言通信。

## 构建

本课使用 6×6 GridWorld 和两个合作智能体。它们从对角开始，必须到达共享目标。共享奖励：任一智能体仍在移动时每步 -1，两者都到达时 +10。见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 两个智能体

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 步骤 2：独立 Q-learning

每个智能体运行自己的以联合状态为键的 Q 表。每步：两者都选 ε-greedy 动作，收集联合转移，每个用自己的共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在这个任务上有效，因为奖励密集且对齐。在紧耦合任务上失效（例如，一个智能体必须*等待*另一个）。

### 步骤 3：中心化 Q 与分解值更新

用一个联合动作的 Q `Q(s, a_1, a_2)`。从共享奖励更新。执行时通过边缘化分散：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数联合动作空间换取*正确的*全局视图。

### 步骤 4：简单自对弈（对抗 2 智能体）

同一智能体，两个角色。训练智能体 A 对抗智能体 B；每 `K` 个 episode 后将 A 的权重复制到 B。对称训练，一致进展。AlphaZero 配方的微型版。

## 陷阱

- **非平稳回放。** 独立智能体的经验回放比单智能体更糟，因为旧转移是由现在已经过时的对手生成的。修复：重新标注或按新鲜度加权。
- **信用分配模糊。** 长期 episode 后的共享奖励；没有办法清楚地说哪个智能体贡献了。修复：反事实基线（COMA），或每个智能体的奖励塑造。
- **策略漂移 / 追逐。** 每个智能体的最佳响应随彼此更新而变化。修复：中心化 critic、慢学习率、或冻结一个。
- **通过协调的奖励黑客。** 智能体找到设计者未预料到的协调利用。拍卖智能体收敛到出价为零。修复：仔细的奖励设计、行为约束。
- **探索冗余。** 两个智能体探索相同状态-动作对。修复：每个智能体的熵奖励，或角色条件化。
- **联盟循环。** 纯自对弈可能陷入主导循环。修复：带多样化对手的联盟学习。
- **样本爆炸。** `n` 智能体 × 状态空间 × 联合动作。用函数近似近似；分解动作空间（每个智能体一个策略输出 head）。

## 使用

2026 年 MARL 应用地图：

| 领域 | 方法 | 备注 |
|--------|--------|-------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享 critic + 分散 actor。 |
| 两人零和游戏（象棋、围棋、扑克） | 带 MCTS 的自对弈（AlphaZero） | 零和；对称训练。 |
| 复杂多人（Dota、StarCraft） | 联盟学习 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | 带注意力的 CTDE MAPPO / PPO | 部分可观测；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时的均值场 RL。 |
| LLM 多智能体系统（Phase 16） | 自然语言通信 + 角色条件化 | RL 循环在智能体规划层。 |

2026 年 MARL 最大的增长领域是基于 LLM 的：语言模型智能体协商、辩论、构建软件的网络。RL 表现为对*轨迹级*输出的偏好优化，而不是 token 级（Phase 16 · 03）。

## 交付

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: 为给定任务选择正确的多智能体 RL 机制（IPPO、CTDE、自对弈、联盟）。
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

给定 `n` 个智能体的任务，输出：

1. 机制分类。合作 / 对抗 / 一般和。说明理由。
2. 算法。IPPO / MAPPO / QMIX / 自对弈 / 联盟。与耦合紧密度和奖励结构相关的理由。
3. 信息访问。中心化训练（什么全局信息给 critic）？分散执行？
4. 信用分配。反事实基线、值分解、或奖励塑造。
5. 探索计划。每个智能体熵、基于种群的训练、或联盟。

拒绝在紧耦合合作任务上用独立 Q-learning。拒绝为有循环风险的了一般和推荐自对弈。标记任何没有固定对手评估的 MARL 管道（自对弈 ELO 是未校准的）。
```

## 练习

1. **简单。** 在 2 智能体合作 GridWorld 上训练独立 Q-learning。平均回报 > 0 需要多少个 episode？绘制联合学习曲线。
2. **中等。** 添加一个"协调"任务：只有在两智能体在同一回合踏上目标时才到达目标。独立 Q 仍然收敛吗？什么打破了？
3. **困难。** 为 MAPPO 风格的训练实现中心化 critic，并在协调任务上比较与独立 PPO 的收敛速度。

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| 马尔可夫博弈 | "多智能体 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "中心化训练，分散执行" | 训练时联合 critic；每个智能体的策略只用局部观测。 |
| IPPO | "独立 PPO" | 每个智能体单独运行 PPO。简单基线；经常被低估。 |
| MAPPO | "多智能体 PPO" | 带中心化值函数的 PPO，条件于全局状态。 |
| QMIX | "单调值分解" | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许分散 argmax。 |
| COMA | "反事实多智能体" | 优势 = 我的 Q 减去对我动作边缘化的期望 Q。 |
| 自对弈 | "智能体 vs 过去的自己" | 单智能体，两个角色；零和标准。 |
| 联盟学习 | "基于种群训练" | 缓存过去策略，从池中采样对手；处理策略循环。 |

## 拓展阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带中心化 critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性的值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 对 MARL 出人意料地强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联盟学习。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和游戏中的纯自对弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包括教材对多智能体设置的简短处理和 CTDE 设计的非平稳性问题。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 覆盖合作、竞争和混合 MARL 与收敛结果的调查。