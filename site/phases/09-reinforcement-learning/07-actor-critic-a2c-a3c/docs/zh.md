# Actor-Critic——A2C 和 A3C

> REINFORCE 噪声很大。添加一个学习 `V̂(s)` 的 critic，从回报中减去它，你就得到了一个优势——期望相同但方差低得多。这就是 actor-critic。A2C 同步运行它；A3C 跨线程运行它。两者都是每个现代深度 RL 方法的思维模型。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 9 · 04（TD 学习），Phase 9 · 06（REINFORCE）
**时间：** 约 75 分钟

## 问题

朴素 REINFORCE 有效，但方差很糟糕。蒙特卡洛回报 `G_t` 在 episodes 之间可能摇摆超过一个数量级。将那个噪声乘以 `∇ log π` 并平均，产生一个梯度估计器，它需要数千个 episode 才能将策略移动同样的距离，而用少得多的 DQN 更新就可以做到。

方差来自使用原始回报。如果你减去一个基线 `b(s_t)`——任意状态函数，包括一个学到的值——期望不变，方差下降。最可处理的基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是*优势*：

`A(s, a) = G - V̂(s)`

如果一个动作产生了高于平均的回报就是好的；低于平均就是差的。带学到的 critic 的 REINFORCE 就是 *actor-critic*。Critic 给 actor 一个低方差的老师。这就是 2015 年之后的所有深度策略方法（A2C、A3C、PPO、SAC、IMPALA）。

## 概念

![Actor-critic：策略网络 + 值网络，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **Actor** `π_θ(a | s)`：策略。采样来行动。用策略梯度训练。
- **Critic** `V_φ(s)`：从状态估计期望回报。训练来最小化 `(V_φ(s) - target)²`。

**优势。** 两种标准形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差低得多。也称为 *TD 残差* `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现在 Atari 上用 `n = 5`，在 MuJoCo 上用 PPO 时用 `n = 2048`。

**广义优势估计（GAE）。** Schulman 等人（2016）提出了对所有 n 步优势指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值——调整直到偏差/方差刻度盘到你想要的位置。

**A2C：同步优势 Actor-Critic。** 在 `N` 个并行环境上收集 `T` 步。为每步计算优势。在合并的 batch 上更新 actor 和 critic。重复。A3C 更简单、可扩展性更好的兄弟。

**A3C：异步优势 Actor-Critic。** Mnih 等人（2016）。生成 `N` 个工作线程，每个运行一个环境。每个工作线程在自己的 rollout 上本地计算梯度，然后异步应用到共享参数服务器。不需要重放缓冲区——工作线程通过运行不同轨迹去相关。A3C 证明了你可以大规模在 CPU 上训练。2026 年，基于 GPU 的 A2C（批量并行环境）占主导地位，因为 GPU 需要大批量。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、值回归、熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是标准的起始点。

## 构建

### 步骤 1：一个 critic

线性 critic `V_φ(s) = w · features(s)`，用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境上 critic 在几百个 episode 内收敛。在 Atari 上，将线性 critic 替换为共享 CNN 主干 + value head。

### 步骤 2：n 步优势

给定长度为 `T` 的 rollout 和一个引导的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 目标。`advantages` 是乘以 `∇ log π` 的量。

### 步骤 3：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

在线策略，每个 rollout 更新一次，actor 和 critic 学习率分开。

### 步骤 4：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个运行自己的环境和自己的前向传播。定期将梯度更新推送到共享主节点。主节点无锁——竞态是可以的，它们只是增加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠为 `[N, obs_dim]` batch，批量前向传播，批量反向传播。更高的 GPU 利用率，确定性，更容易推理。2026 年的默认值。

我们的玩具代码是单线程的以保持清晰；重写为批量 A2C 只需要三行 numpy。

## 陷阱

- **在 actor 梯度之前 critic 有偏。** 如果 critic 是随机的，其基线无信息，你是在纯粹噪声上训练。在开启策略梯度之前将 critic 预热几百步，或者使用慢的 actor 学习率。
- **优势归一化。** 每个 batch 归一化优势为零均值/单位方差。以接近零的成本大规模稳定训练。
- **共享主干。** 在图像输入上对 actor 和 critic 使用共享特征提取器。分离的 heads。共享特征在两个损失上自由搭车。
- **在线策略合约。** A2C 将数据复用恰好一次更新。更多则梯度有偏（重要性采样校正是 PPO 添加的内容）。
- **熵崩溃。** 没有 `c_e > 0`，策略在几百个更新内变成近确定性，停止探索。
- **奖励尺度。** 优势量级取决于奖励尺度。在任务间获得一致梯度量级前归一化奖励（例如，用运行标准差除）。

## 使用

A2C/A3C 在 2026 年很少是最终选择，但它们是所有后续改进的架构基础：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 裁剪重要性比率用于多轮更新 |
| IMPALA | A3C + V-trace 离线修正 |
| SAC（Phase 9 · 07） | 带软值 critic 的离线 A2C（下一课） |
| GRPO（Phase 9 · 12） | 无 critic 的 A2C——组相对优势 |
| DPO | 崩溃为偏好排序损失的 A2C，无需采样 |
| AlphaStar / OpenAI Five | 带联盟训练 + 模仿预训练的 A2C |

如果你在 2026 年论文中看到"优势"，想想 actor-critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: 为给定环境产出 A2C / A3C / GAE 配置，指定优势估计和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定一个环境和计算预算，输出：

1. 并行性。A2C（GPU 批量）vs A3C（CPU 异步）及工作线程数。
2. Rollout 长度 T。每次更新每个环境的步数。
3. 优势估计器。n 步或 GAE(λ)；指定 λ。
4. 损失权重。`c_v`（值）、`c_e`（熵）、梯度裁剪。
5. 学习率。Actor 和 critic（如果使用分开的话）。

拒绝在视野 > 1000 的环境上用单工作线程 A2C（太在线策略，太慢）。拒绝不启用优势归一化就发布。标记任何 `c_e = 0` 且观察到熵 < 0.1 的运行为熵崩溃。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 MC 优势（`G_t - V(s_t)`）训练 actor-critic。与 Lesson 06 的带运行均值基线的 REINFORCE 比较样本效率。
2. **中等。** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势 batch 的方差。下降了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率。在这个任务上偏差/方差最佳点在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| Actor | "策略网络" | `π_θ(a|s)`，用策略梯度更新。 |
| Critic | "值网络" | `V_φ(s)`，用对回报 / TD 目标的 MSE 回归更新。 |
| 优势 | "比平均水平好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计量。乘以 `∇ log π`。 |
| TD 残差 | "δ" | `δ_t = r + γ V(s') - V(s)`；一步优势估计。 |
| GAE | "插值旋钮" | n 步优势的指数加权平均，参数化为 `λ`。 |
| A2C | "同步 actor-critic" | 批量跨环境；每个 rollout 一次梯度步。 |
| A3C | "异步 actor-critic" | 工作线程将梯度推送到共享参数服务器。原始论文；2026 年较少见。 |
| Bootstrap | "在视野处使用 V" | 截断 rollout，添加 `γ^n V(s_{t+n})` 来封闭求和。 |

## 拓展阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步 actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；当 critic 是神经网络时配合第 9 章函数近似阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带 V-trace 离线修正的可扩展分布式 actor-critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 生产 A2C/PPO 实现，值得阅读。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 双时间尺度 actor-critic 分解的原始收敛结果。