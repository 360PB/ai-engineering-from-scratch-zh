# MDP、状态、动作与奖励

> 马尔可夫决策过程由五部分构成：状态、动作、转移、奖励、折扣。RL 中的所有算法——Q-learning、PPO、DPO、GRPO——都在这个结构上做优化。学一遍，剩余的强化学习内容就能免费阅读。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 1 · 06（概率与分布），Phase 2 · 01（机器学习分类）
**时间：** 约 45 分钟

## 问题

你要写一个象棋 bot，或者一个库存规划器，或者一个交易智能体，或者训练推理模型的 PPO 循环。四个不同的领域，却有一个令人惊讶的事实：它们全部可以归约到同一个数学对象。

监督学习给你 (x, y) 对，让你拟合一个函数。强化学习不给你标签——只有状态流、你采取的动作和一个标量奖励。这步棋赢了吗？这次补货决策省钱了吗？这笔交易盈利了吗？LLM 刚才输出的 token 是否让评判器给出了更高的奖励？

在你能从这个流中学习之前，必须先将其形式化。"我看到了什么""我做了什么""接下来发生了什么""那有多好"——每一项都要成为一个可以推理的对象。这个形式化就是马尔可夫决策过程（MDP）。本 Phase 的所有 RL 算法，包括末尾的 RLHF 和 GRPO 循环，都在这个结构上做优化。

## 概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。智能体做决策所需的全部信息。在 GridWorld 中是格子，在象棋中是棋盘，在 LLM 中是上下文窗口加上任意记忆。
- **动作** `A`。可选的操作。上/下/左/右移动、走一步棋、输出一个 token。
- **转移** `P(s' | s, a)`。给定状态 `s` 和动作 `a` 后，下一状态的分布。象棋中 deterministic，库存中 stochastic，LLM 解码中近乎 deterministic。
- **奖励** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减去成本。GRPO 中的对数似然比项。
- **折扣** `γ ∈ [0, 1)`。未来奖励相对于当前奖励的权重。`γ = 0.99` 购买约 100 步视野；`γ = 0.9` 购买约 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来仅取决于当前状态。如果不满足，状态表示就不完整——不是方法的问题，是状态的问题。

**策略与回报。** 策略 `π(a | s)` 将状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣总和。值函数 `V^π(s) = E[G_t | s_t = s]` 是从状态 `s` 在策略 `π` 下开始的期望回报。Q 值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从特定动作开始的期望回报。所有 RL 算法都估计这两个量之一，然后据此改进 `π`。

**Bellman 方程。** 所有本 Phase 算法都在使用的定点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报分解为"这一步的奖励"加上"到达位置的折扣值"。递归的。本 Phase 的所有算法要么迭代这个方程至收敛（动态规划），要么从中采样（蒙特卡洛），要么引导一步（时序差分）。

## 构建

### 步骤 1：一个极简确定性 MDP

4×4 的 GridWorld。智能体从左上角开始，终点在右下角，每步奖励 -1，动作为 `{上、下、左、右}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行。这就是一个完整的环境。确定性转移、恒定步惩罚、吸收型终止状态。

### 步骤 2：展开一个策略

策略是状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。该 4×4 棋盘的平均回报约为 -60 至 -80。最优回报是 -6（直线向右下的路径）。弥合这个差距就是 Phase 9 的全部内容。

### 步骤 3：通过 Bellman 方程精确计算 `V^π`

对于小型 MDP，bellman 方程是一个线性方程组。枚举状态，应用期望，迭代直到值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。它是 Sutton & Barto 的第一个算法，也是所有后续 RL 方法的理论基础。

### 步骤 4：`γ` 是一个有物理意义的超参数

有效视野大致为 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低则智能体短视。太高则信用分配变得噪声大，因为很多早期步骤共同分担远期奖励的责任。LLM RLHF 通常使用 `γ = 1`，因为 episode 短且有界。控制任务使用 `0.95–0.99`。长视野策略游戏使用 `0.999`。

## 陷阱

- **非马尔可夫状态。** 如果你需要最近三个观测才能决策，"状态"就不只是当前观测。修复方法：堆叠帧（DQN 在 Atari 上堆叠 4 帧）或使用循环状态（LSTM/GRU 处理观测序列）。
- **稀疏奖励。** 只在胜利时给奖励，在大状态空间中几乎不可能学习。塑造奖励（中间信号）或用模仿学习引导（Phase 9 · 09）。
- **奖励黑客。** 优化代理奖励往往产生病态行为。OpenAI 的赛船智能体绕圈永远收集能量而不完成比赛。始终从目标结果出发定义奖励，而不是从代理出发。
- **折扣误设。** 无限视野任务上 `γ = 1` 会使所有值无穷大。始终用有限视野或 `γ < 1` 来约束。
- **奖励尺度。** {+100, -100} 与 {+1, -1} 的奖励给出相同的最优策略，但梯度量级差异巨大。在接入 PPO/DQN 之前归一化到 `[-1, 1]` 范围。

## 使用

2026 年的技术栈在写任何训练循环之前先将每个 RL 流程归约为 MDP：

| 场景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（ locomotion、manipulation） | 关节角度 + 速度 | 连续力矩 | 任务相关的塑造奖励 | 0.99 |
| 游戏（象棋、围棋、扑克） | 棋盘 + 历史 | 合法走子 | 赢 = +1 / 输 = -1 | 1.0（有限） |
| 库存/定价 | 库存 + 需求 | 订货量 | 收入 - 成本 | 0.95 |
| LLM 的 RLHF | 上下文 token | 下一个 token | 结束时奖励模型分数 | 1.0（episode 约 200 token） |
| GRPO 推理 | 提示 + 部分回复 | 下一个 token | 结束时验证器 0/1 | 1.0 |

写五元组在前，写任何训练循环在后。大多数"RL 不 work"的 bug 报告都能追溯到 MDP 公式化在纸上就是坏的。

## 交付

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: 给定任务描述，产出马尔可夫决策过程规范并在训练前标记公式化风险。
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

给定一个任务（控制 / 游戏 / 推荐 / LLM 微调），输出：

1. 状态。精确的特征向量或张量规格。说明马尔可夫性质的依据。
2. 动作。离散集合或连续范围。维度。
3. 转移。确定性、已知模型的随机、或仅可采样。
4. 奖励。函数和来源。稀疏 vs 塑造。终止 vs 每步。
5. 折扣。数值和视野依据。

拒绝发布任何状态非马尔可夫且未明确提及帧堆叠或循环状态的 MDP。拒绝任何奖励未从目标结果出发定义的 reward。标记任何无限视野任务上 `γ ≥ 1.0` 的情况。标记奖励范围超过典型步奖励 100 倍的情况为可能的梯度爆炸源。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 GridWorld 和随机策略 rollout。运行 10,000 个 episode。报告均值和回报标准差。与最优回报（-6）比较。
2. **中等。** 对均匀随机策略运行 `policy_evaluation`，`γ ∈ {0.5, 0.9, 0.99}`。将 `V` 打印为每个 γ 对应的 4×4 网格。解释为什么靠近终止点的状态值随 γ 增大而增长更快。
3. **困难。** 将 GridWorld 随机化：每个动作以概率 `p = 0.1` 滑到相邻方向。重新评估均匀随机策略。`V[start]` 变好还是变坏？为什么？

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| MDP | "强化学习设置" | 满足马尔可夫性质的五元组 `(S, A, P, R, γ)`。 |
| 状态（State） | "智能体看到什么" | 在选定策略类下未来动态的充分统计量。 |
| 策略（Policy） | "智能体的行为" | 条件分布 `π(a | s)` 或确定性映射 `s → a`。 |
| 回报（Return） | "总奖励" | 从当前步开始的折扣奖励和 `Σ γ^t r_t`。 |
| 值函数（Value） | "一个状态有多好" | 从 `s` 开始在 `π` 下的期望回报。 |
| Q 值（Q-value） | "一个动作有多好" | 从 `s` 开始且第一个动作为 `a` 在 `π` 下的期望回报。 |
| Bellman 方程 | "动态规划递归" | 值 / Q 的定点分解，分解为一步奖励加上折扣后继值。 |
| 折扣 `γ` | "未来 vs 现在" | 远期奖励的几何权重；有效视野 `~1/(1-γ)`。 |

## 拓展阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教材。第 3 章涵盖 MDP 和 Bellman 方程；第 1 章解释作为所有后续课程基础的奖励假设。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Bellman 方程的起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度 RL 角度撰写的简明 MDP 入门。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP 和精确求解方法的运筹学参考文献。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将 MDP 作为动态规划特例的最清晰推导。