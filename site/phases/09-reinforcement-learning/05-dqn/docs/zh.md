# 深度 Q 网络（DQN）

> 2013 年：Mnih 在原始像素上训练一个 Q-learning 网络，在七款 Atari 游戏上击败所有经典 RL 智能体。2015 年：扩展到 49 款游戏，发表在 Nature，掀起了深度 RL 时代。DQN = Q-learning + 三个让函数近似稳定的技巧。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 3 · 03（反向传播），Phase 9 · 04（Q-learning、SARSA）
**时间：** 约 75 分钟

## 问题

表格 Q-learning 需要为每个（状态，动作）对维护一个单独的 Q 值。象棋棋盘约有 10⁴³ 个状态。一帧 Atari 是 210×160×3 = 100,800 个特征。表格 RL 在数千个状态时就失效了，更不用说数十亿个。

解决方案在事后看来很明显：用神经网络替换 Q 表，`Q(s, a; θ)`。但这个"事后很明显"的方案花了几十年才出现。朴素地将函数近似与 Q-learning 结合会在"致命三角"下发散——函数近似 + bootstrapping + 离线学习。Mnih 等人（2013，2015）确定了三个工程技巧来稳定学习：

1. **经验回放** 去相关转移。
2. **目标网络** 冻结 bootstrap 目标。
3. **奖励裁剪** 归一化梯度量级。

Atari 上的 DQN 是首次用一个架构和一套超参数从原始像素解决了数十个控制问题。此后所有"深度 RL"——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都堆叠在这个三技巧基础上。

## 概念

![DQN 训练循环：环境、回放缓冲区、在线网络、目标网络、Bellman TD 损失](../assets/dqn.svg)

**目标函数。** DQN 在神经 Q 函数上最小化一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，定期从 `θ` 复制（约每 ~10,000 步）。`D` = 过去转移的回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个约 `~10⁶` 转移的环形缓冲区。每次训练步随机均匀采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络可以多次从稀有奖励转移中学习，并去相关了连续梯度更新。没有它，带神经网络的在线策略 TD 在 Atari 上会发散。

**目标网络。** 在 Bellman 方程两边使用同一个网络 `Q(·; θ)` 会使目标在每次更新时都移动——"追自己的尾巴"。修复：保留第二个权重冻结的网络 `Q(·; θ^-)`。每 `C` 步将 `θ → θ^-` 复制一次。这在数千个梯度步上稳定了回归目标。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是更平滑的变体。

**奖励裁剪。** Atari 奖励量级从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可阻止任何单一游戏主导梯度。当奖励量级本身很重要时这是错误的；对 Atari 来说只有符号重要，所以没问题。

**双 DQN。** Hasselt (2016) 修复了最大化偏差：用在线网络来*选择*动作，目标网络来*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用，一致更好。默认使用。

**其他改进（Rainbow，2017）：** 优先回放（更频繁采样高 TD 误差的转移）、双流架构（分离 `V(s)` 和 advantage head）、噪声网络（学习型探索）、n 步回报、分布式 Q（C51/QR-DQN）、多步 bootstrapping。每个增加几个百分点；收益大致可加。

## 构建

这里的代码是纯标准库无 numpy——我们用手写单隐藏层 MLP 在一个微型连续 GridWorld 上运行，所以每个训练步在微秒级内完成。算法与大规模 Atari DQN 完全相同。

### 步骤 1：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 用约 50,000 容量；我们的玩具环境 5,000 就够了。

### 步骤 2：一个微型 Q 网络（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：linear → ReLU → linear。这就是整个网络。

### 步骤 3：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

与 Lesson 04 的 Q-learning 相比有两点不同：(a) 通过可微的 `Q(·; θ)` 反向传播而不是查表，(b) 目标使用 `Q(·; θ^-)`。

### 步骤 4：外层循环

每个 episode，在 `Q(·; θ)` 上做 ε-greedy，将转移推入缓冲区，采样一个小批量，执行梯度步，定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的微型 GridWorld 上用 16 维 one-hot 状态，智能体在约 500 个 episode 内学到接近最优的策略。在 Atari 上，将这个扩展到 200M 帧并加上 CNN 特征提取器。

## 陷阱

- **致命三角。** 函数近似 + 离线 + bootstrapping 可能发散。DQN 用目标网络 + 回放缓解；不要移除任何一个。
- **探索。** ε 必须衰减，通常从 1.0 到 0.01 在训练的前约 10%。早期探索不足会使 Q 网络收敛到一个局部盆地。
- **过估计。** 对噪声 Q 取 `max` 是向上偏的。生产中始终使用双 DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度量级与奖励量级成正比。
- **回放缓冲区冷启动。** 在缓冲区有几千个转移之前不要训练。早期在约 20 个样本上的梯度会过拟合。
- **目标网络同步频率。** 太频繁 ≈ 没有目标网络；太不频繁 ≈ 目标过时。Atari DQN 每 10,000 步环境步同步。经验规则：每训练视野的约 1/100 同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧来使状态马尔可夫。任何包含速度信息的环境需要帧堆叠或循环状态。

## 使用

2026 年 DQN 很少是最先进的，但仍然是参考离线策略算法：

| 任务 | 首选方法 | 为什么不用 DQN |
|------|------------------|--------------|
| 离散动作 Atari 类 | Rainbow DQN 或 Muesli | 同一框架，更多技巧。 |
| 连续控制 | SAC / TD3（Phase 9 · 07） | DQN 没有策略网络。 |
| 在线策略 / 高吞吐量 | PPO（Phase 9 · 08） | 无回放缓冲区；更容易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，无 bootstrapping 爆炸。 |
| 大离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | 可以；装饰更重要。 |
| LLM RL | PPO / GRPO | 序列级，不是步级；损失不同。 |

这些经验仍然适用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自对弈缓冲区和每个离线 RL 方法中。奖励裁剪作为 PPO 中的 advantage 归一化延续了下来。这个架构是蓝图。

## 交付

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: 为离散动作 RL 任务产出 DQN 训练配置（缓冲区、目标同步、ε 调度、奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定一个离散动作环境（观测形状、动作数量、视野、奖励尺度），输出：

1. 网络。架构（MLP / CNN / Transformer）、特征维数、深度。
2. 回放缓冲区。容量、小批量大小、预热大小。
3. 目标网络。同步策略（每 C 步硬同步或软 τ）。
4. 探索。ε 起始 / 结束 / 调度长度。
5. 损失。Huber vs MSE、梯度裁剪值、奖励裁剪规则。
6. 双 DQN。默认开启，除非有明确原因禁用。

拒绝发布没有目标网络、没有回放缓冲区、或 ε 始终为 1 的 DQN。拒绝连续动作任务（路由到 SAC / TD3）。标记任何奖励范围超过步均奖励 10 倍的情况为需要裁剪或尺度归一化。
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每 episode 回报曲线。运行均值超过 -10 需要多少个 episode？
2. **中等。** 禁用目标网络（对 Bellman 目标两边都使用在线网络）。测量训练不稳定性——回报是否振荡或发散？
3. **困难。** 添加双 DQN：用在线网络选择 `argmax a'`，目标网络评估。在带噪声奖励的 GridWorld 上，比较 1,000 个 episode 后有双 DQN 和无双 DQN 的 `Q(s_0, best_a)` 与真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| DQN | "深度 Q 学习" | Q-learning + 神经 Q 函数 + 回放缓冲区 + 目标网络。 |
| 经验回放 | "打乱的转移" | 每个梯度步均匀采样的环形缓冲区；去相关数据。 |
| 目标网络 | "冻结的 bootstrap" | Bellman 目标中使用的 Q 的定期复制；稳定训练。 |
| 致命三角 | "为什么 RL 发散" | 函数近似 + bootstrapping + 离线 = 无收敛保证。 |
| 双 DQN | "修复最大化偏差" | 在线网络选动作，目标网络评估动作。 |
| Dueling DQN | "V 和 A head" | 分解 Q = V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | "所有技巧合一" | DDQN + PER + dueling + n-step + noisy + distributional。 |
| PER | "优先回放" | 按 TD 误差量级比例采样转移。 |

## 拓展阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 2013 年 NeurIPS 研讨会论文，开启了深度 RL。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 游戏 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 堆叠技巧论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代表述。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 教材对"致命三角"（函数近似 + bootstrapping + 离线）的处理，DQN 的目标网络和回放缓冲区正是为应对它而设计的。
- [CleanRL DQN 实现](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 消融研究中使用的参考单文件 DQN；适合与本课从零实现版本对照阅读。