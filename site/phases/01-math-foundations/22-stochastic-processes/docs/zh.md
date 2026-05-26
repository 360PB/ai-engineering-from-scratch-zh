# 随机过程

> 有结构的随机性。随机游走、马尔可夫链和扩散模型的数学基础。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 1，课程 06-07（概率、贝叶斯）
**时间：** 约 75 分钟

## 学习目标

- 模拟一维和二维随机游走并验证位移的 sqrt(n) 标度律
- 构建马尔可夫链模拟器并通过特征分解计算其平稳分布
- 实现 Metropolis-Hastings MCMC 和 Langevin 动力学，从目标分布采样
- 将前向扩散过程与布朗运动联系起来，解释反向过程如何生成数据

## 问题所在

许多 AI 系统涉及随时间演化的随机性。不是静态随机——而是结构化的、顺序的随机，每一步依赖于前一步。

语言模型一次生成一个 token。每个 token 依赖于先前的上下文。模型输出一个概率分布，从中进行采样，然后继续。这是一个随机过程。

扩散模型逐步向图像添加噪声，直到它变成纯静态。然后它反转这个过程，逐步去噪直到生成新图像。前向过程是一条马尔可夫链。反向过程是一条学习的马尔可夫链，向后运行。

强化学习智能体在环境中执行动作。每个动作以一定概率导致新状态。智能体在随机的世界中遵循随机策略。整个过程是一个马尔可夫决策过程。

MCMC 采样——贝叶斯推断的支柱——构建一条马尔可夫链，其平稳分布是你想要采样的后验。

所有这些都建立在四个基础思想上：
1. 随机游走——最简单的随机过程
2. 马尔可夫链——带转移矩阵的结构化随机性
3. Langevin 动力学——带噪声的梯度下降
4. Metropolis-Hastings——从任意分布采样

## 核心概念

### 随机游走

从位置 0 开始。每一步，抛一枚均匀硬币。正面：向右移动 (+1)。反面：向左移动 (-1)。

经过 n 步后，你的_position 是 n 个随机 +/-1 值之和。期望位置是 0（游走是无偏的）。但期望距离原点的距离增长为 sqrt(n)。

这违反直觉。游走是公平的——两个方向都没有漂移。但随着时间推移，它会越来越远离起点。n 步后的标准差是 sqrt(n)。

```
第 0 步：位置 = 0
第 1 步：位置 = +1 或 -1
第 2 步：位置 = +2, 0, 或 -2
...
第 100 步：距原点期望距离 ~ 10（sqrt(100)）
第 10000 步：距原点期望距离 ~ 100（sqrt(10000)）
```

**在 2D 中**，游走向上、下、左、右以相等概率移动。相同的 sqrt(n) 标度律适用于到原点的距离。路径描绘出类似分形的图案。

**为什么是 sqrt(n)？** 每一步以相等概率为 +1 或 -1。n 步后，位置 S_n = X_1 + X_2 + ... + X_n，其中每个 X_i 为 +/-1。每一步的方差为 1，且各步独立，所以 Var(S_n) = n。标准差 = sqrt(n)。根据中心极限定理，S_n / sqrt(n) 收敛到标准正态分布。

这个 sqrt(n) 标度律在机器学习中无处不在。SGD 噪声按 1/sqrt(batch_size) 缩放。嵌入维度按 sqrt(d) 缩放。平方根是独立随机加法的签名。

**与布朗运动的联系。** 取步长为 1/sqrt(n)、每单位时间 n 步的随机游走。当 n 趋于无穷时，游走收敛到布朗运动 B(t)——一个连续时间过程，其中 B(t) 服从均值为 0、方差为 t 的正态分布。

布朗运动是扩散的数学基础。它模拟流体中粒子的随机抖动、股价的波动，以及——关键地——扩散模型中的噪声过程。

**赌徒破产。** 从位置 k 开始的随机游走者，在 0 和 N 处有吸收壁。到达 N 而非 0 的概率是多少？对于公平游走：P(到达 N) = k/N。这出奇地简单优雅。它与鞅理论有关——公平随机游走是一个鞅（期望未来值 = 当前值）。

### 马尔可夫链

马尔可夫链是一个根据固定概率在状态之间转换的系统。关键性质：下一状态仅取决于当前状态，而非历史。

```
P(X_{t+1} = j | X_t = i, X_{t-1} = ...) = P(X_{t+1} = j | X_t = i)
```

这就是马尔可夫性质。它意味着你用一个转移矩阵 P 描述整个动力学：

```
P[i][j] = 从状态 i 转到状态 j 的概率
```

P 的每行之和为 1（你必须去某个地方）。

**示例——天气：**

```
状态：晴天 (0)、雨天 (1)、阴天 (2)

P = [[0.7, 0.1, 0.2],    （如果晴天：70% 晴天，10% 雨天，20% 阴天）
     [0.3, 0.4, 0.3],    （如果雨天：30% 晴天，40% 雨天，30% 阴天）
     [0.4, 0.2, 0.4]]    （如果阴天：40% 晴天，20% 雨天，40% 阴天）
```

从任何状态开始。经过多次转移后，状态分布收敛到平稳分布 pi，其中 pi * P = pi。这是 P 的特征值 1 的左特征向量。

对于这个天气链，平稳分布可能是 [0.53, 0.18, 0.29]——长期来看，无论起点如何，53% 的时间是晴天。

```mermaid
graph LR
    S["晴天"] -->|0.7| S
    S -->|0.1| R["雨天"]
    S -->|0.2| C["阴天"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C
```

**计算平稳分布。** 有两种方法：

1. **幂法**：将任意初始分布反复乘以 P。经过足够多次迭代后它收敛。
2. **特征值方法**：求 P 的特征值 1 的左特征向量。这是 P^T 的特征值 1 的特征向量。

两种方法都要求链满足收敛条件。

**收敛条件。** 马尔可夫链收敛到唯一平稳分布的条件是：
- **不可约**：每个状态可从每个其他状态到达
- **非周期**：链不是以固定周期循环

你在机器学习中遇到的大多数链都满足这两个条件。

**吸收态。** 如果一旦进入某个状态就永远不会离开（P[i][i] = 1），则该状态是吸收态。吸收马尔可夫链模拟有终态的过程——结束的游戏、流失的客户、达到文本结束 token 的 token 序列。

**混合时间。** 链需要多少步才能"接近"平稳分布？形式上，是总变差距离从平稳性降到某个阈值以下的步数。快速混合 = 需要很少步。P 的谱间隙（1 减去第二大特征值）控制混合时间。间隙越大 = 混合越快。

### 与语言模型的联系

语言模型中的 token 生成大约是一个马尔可夫过程。给定当前上下文，模型输出下一个 token 的分布。温度控制锐度：

```
P(token_i) = exp(logit_i / temperature) / sum(exp(logit_j / temperature))
```

- 温度 = 1.0：标准分布
- 温度 < 1.0：更尖锐（更确定性）
- 温度 > 1.0：更平坦（更随机）
- 温度 -> 0：argmax（贪婪）

Top-k 采样截断到概率最高的 k 个 token。Top-p（nucleus）采样截断到累积概率超过 p 的最小 token 集。两者都修改马尔可夫转移概率。

### 布朗运动

随机游走的连续时间极限。位置 B(t) 有三个性质：
1. B(0) = 0
2. B(t) - B(s) 服从均值为 0、方差为 t - s 的正态分布（t > s 时）
3. 非重叠区间的增量是独立的

布朗运动连续但无处可微——它在每个尺度上都抖动。路径在平面中具有分形维数 2。

在离散模拟中，你这样近似布朗运动：

```
B(t + dt) = B(t) + sqrt(dt) * z,    其中 z ~ N(0, 1)
```

sqrt(dt) 标度很重要。它来自应用于随机游走的中心极限定理。

### Langevin 动力学

梯度下降找到函数的最小值。Langevin 动力学找到与 exp(-U(x)/T) 成正比的概率分布，其中 U 是能量函数，T 是温度。

```
x_{t+1} = x_t - dt * gradient(U(x_t)) + sqrt(2 * T * dt) * z_t
```

两种力作用在粒子上：
1. **梯度力**（-dt * gradient(U)）：推向低能量（像梯度下降）
2. **随机力**（sqrt(2*T*dt) * z）：推向随机方向（探索）

在温度 T = 0 时，这是纯梯度下降。在高温时，它几乎是一个随机游走。在合适的温度下，粒子探索能量景观，在低能量区域花费更多时间。

**与扩散模型的联系。** 扩散模型的前向过程是：

```
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise
```

这是一条逐渐将数据与噪声混合的马尔可夫链。经过足够步后，x_T 是纯高斯噪声。

反向过程——从噪声回到数据——也是一条马尔可夫链，但其转移概率由神经网络学习。网络学习预测每一步添加的噪声，然后减去它。

```mermaid
graph LR
    subgraph "前向过程（加噪声）"
        X0["x_0（数据）"] -->|"+ 噪声"| X1["x_1"]
        X1 -->|"+ 噪声"| X2["x_2"]
        X2 -->|"..."| XT["x_T（纯噪声）"]
    end
    subgraph "反向过程（去噪）"
        XT2["x_T（噪声）"] -->|"神经网络"| XR2["x_{T-1}"]
        XR2 -->|"神经网络"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0（生成的数据）"]
    end
```

### MCMC：马尔可夫链蒙特卡洛

有时你需要从一个分布 p(x) 采样，但你只能求值（直到一个常数）而不能直接采样。贝叶斯后验是典型例子——你直到似然乘先验，但归一化常数难解。

**Metropolis-Hastings** 构建一条平稳分布为 p(x) 的马尔可夫链：

1. 从某个位置 x 开始
2. 从提议分布 Q(x'|x) 提议一个新位置 x'
3. 计算接受率：a = p(x') * Q(x|x') / (p(x) * Q(x'|x))
4. 以概率 min(1, a) 接受 x'。否则留在 x。
5. 重复。

如果 Q 是对称的（例如，Q(x'|x) = Q(x|x') = N(x, sigma^2)），比率简化为 a = p(x') / p(x)。你只需要概率比——归一化常数抵消。

在温和条件下，链保证收敛到 p(x)。但如果提议太小（随机游走）或太大（高拒绝率），收敛可能很慢。调整提议是 MCMC 的艺术。

**为什么有效。** 接受率确保详细平衡：在 x 处移动到 x' 的概率等于在 x' 处移动到 x 的概率。详细平衡意味着 p(x) 是链的平稳分布。所以经过足够步后，样本来自 p(x)。

**实际考虑：**
- **预烧期**：丢弃前 N 个样本。链需要时间从起始点到达平稳分布。
- **稀疏化**：每 k 个样本保留一个以减少自相关。
- **多链**：从不同起始点运行几条链。如果它们收敛到相同分布，你有收敛的证据。
- **接受率**：对于 d 维高斯提议，最优接受率约为 23%（Roberts & Rosenthal, 2001）。太高意味着链几乎不动。太低意味着它拒绝一切。

### 人工智能中的随机过程

| 过程 | 人工智能应用 |
|---------|---------------|
| 随机游走 | RL 中的探索、Node2Vec 嵌入 |
| 马尔可夫链 | 文本生成、MCMC 采样 |
| 布朗运动 | 扩散模型（前向过程） |
| Langevin 动力学 | 基于分数的生成模型、SGLD |
| 马尔可夫决策过程 | 强化学习 |
| Metropolis-Hastings | 贝叶斯推断、后验采样 |

## 构建它

### 第 1 步：随机游走模拟器

```python
import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # 向右
    dx[directions == 1] = -1  # 向左
    dy[directions == 2] = 1   # 向上
    dy[directions == 3] = -1  # 向下
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y
```

一维游走存储累积和。每一步为 +1 或 -1。n 步后，位置是累加和。方差随 n 线性增长，所以标准差增长为 sqrt(n)。

### 第 2 步：马尔可夫链

```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
```

平稳分布是 P 的特征值 1 的左特征向量。我们通过计算 P^T 的特征向量来找到它（转置将左特征向量变为右特征向量）。

### 第 3 步：Langevin 动力学

```python
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)
```

梯度将 x 推向低能量。噪声防止它陷入局部极小。在平衡时，样本的分布与 exp(-U(x)/temperature) 成正比。

### 第 4 步：Metropolis-Hastings

```python
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate
```

算法提议一个新点，检查它是否有更高概率（或者按比例接受），然后重复。接受率应该在 23-50% 之间以获得良好混合。

## 使用它

实践中，你使用成熟的库实现这些算法。但理解机制对于调试和调优很重要。

```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"最终位置：{walk[-1]}")
print(f"期望距离：{np.sqrt(10000):.1f}")
print(f"实际距离：{abs(walk[-1])}")
```

### numpy 用于转移矩阵

```python
import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"平稳分布：{np.round(distribution, 4)}")
```

反复将初始分布乘以 P。经过足够多次迭代后，它收敛到平稳分布，无论你从哪里开始。这是求主导左特征向量的幂法。

### 与真实框架的联系

- **PyTorch 扩散：** Hugging Face `diffusers` 中的 `DDPMScheduler` 实现了前向和反向马尔可夫链
- **NumPyro / PyMC：** 使用 MCMC（NUTS 采样器，改进了 Metropolis-Hastings）进行贝叶斯推断
- **Gymnasium（RL）：** 环境 step 函数定义了一个马尔可夫决策过程

### 验证马尔可夫链收敛

```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"特征值：{eigenvalues}")
print(f"谱间隙：{spectral_gap:.4f}")
print(f"近似混合时间：{1/spectral_gap:.1f} 步")
```

谱间隙告诉你链忘记其初始状态的速度。间隙为 0.2 意味着大约 5 步混合。间隙为 0.01 意味着大约 100 步。在运行长模拟前总是检查这个——混合慢的链浪费计算。

## 交付它

本课程产出：
- `outputs/prompt-stochastic-process-advisor.md` -- 一个帮助识别哪种随机过程框架适用于给定问题的提示

## 联系

| 概念 | 出现位置 |
|---------|------------------|
| 随机游走 | Node2Vec 图嵌入、RL 中的探索 |
| 马尔可夫链 | LLM 中的 token 生成、MCMC 采样 |
| 布朗运动 | DDPM 中的前向扩散过程、基于 SDE 的模型 |
| Langevin 动力学 | 基于分数的生成模型、随机梯度 Langevin 动力学（SGLD） |
| 平稳分布 | MCMC 收敛目标、PageRank |
| Metropolis-Hastings | 贝叶斯后验采样、模拟退火 |
| 温度 | LLM 采样、RL 中的 Boltzmann 探索、模拟退火 |
| 混合时间 | MCMC 收敛速度、谱间隙分析 |
| 吸收态 | 序列结束 token、RL 中的终态 |
| 详细平衡 | MCMC 采样器正确性保证 |

扩散模型值得特别提及。DDPM（Ho et al., 2020）定义了一个前向马尔可夫链：

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

其中 beta_t 是噪声调度。经过 T 步后，x_T 近似为 N(0, I)。反向过程由一个预测噪声的神经网络参数化：

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

生成的每一步都是学习马尔可夫链的一步。理解马尔可夫链意味着理解扩散模型如何以及为何生成数据。

SGLD（随机梯度 Langevin 动力学）将小批量梯度下降与 Langevin 噪声结合。不是计算完整梯度，而是使用随机估计并添加校准噪声。随着学习率衰减，SGLD 从优化过渡到采样——你免费获得近似贝叶斯后验样本。这是从神经网络获取不确定性估计的最简单方法之一。

贯穿所有这些联系的核心洞察：随机过程不仅仅是理论工具。它们是现代人工智能系统内部的计算机制。当你调整 LLM 的温度时，你是在调整一条马尔可夫链。当你训练扩散模型时，你是在学习反转一个类似布朗运动的过程。当你运行贝叶斯推断时，你是在构建一条收敛到后验的链。

## 练习

1. **模拟 1000 条 10000 步的随机游走。** 绘制最终位置的分布。验证它近似高斯分布，均值为 0，标准差为 sqrt(10000) = 100。

2. **用马尔可夫链构建文本生成器。** 在一个小语料库上训练：对每个词，统计到下一个词的转移。构建转移矩阵。通过从链中采样生成新句子。

3. **用 Metropolis-Hastings 实现模拟退火。** 从高温开始（几乎接受一切）并逐渐冷却（只接受改进）。用它找到一个有许多局部最小值的函数的最小值。

4. **比较不同温度下的 Langevin 动力学。** 从双井势 U(x) = (x^2 - 1)^2 采样。在低温下，样本聚集在一个井中。在高温下，它们散布在两个井之间。找出链在井之间混合的临界温度。

5. **实现前向扩散过程。** 从一个一维信号（例如，正弦波）开始。用线性噪声调度在 100 步中逐步添加噪声。展示信号如何退化为纯噪声。然后实现一个简单的去噪器来反转这个过程（即使是只是减去估计噪声的天真去噪器）。

## 核心术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 随机游走 | "抛硬币移动" | 每一步位置按随机增量变化的过程 |
| 马尔可夫性质 | "无记忆" | 未来仅取决于当前状态，而非历史 |
| 转移矩阵 | "概率表" | P[i][j] = 从状态 i 移动到状态 j 的概率 |
| 平稳分布 | "长期平均" | 分布 pi 其中 pi*P = pi —— 链的均衡 |
| 布朗运动 | "随机抖动" | 随机游走的连续时间极限，B(t) ~ N(0, t) |
| Langevin 动力学 | "带噪声的梯度下降" | 结合确定性梯度和随机扰动的更新规则 |
| MCMC | "走向目标" | 构建平稳分布为你想要分布的马尔可夫链 |
| Metropolis-Hastings | "提议并接受/拒绝" | 使用接受率确保收敛的 MCMC 算法 |
| 温度 | "随机性旋钮" | 控制探索与利用之间权衡的参数 |
| 扩散过程 | "噪声进，噪声出" | 前向：逐步加噪声。反向：逐步去噪。生成数据。 |

## 延伸阅读

- **Ho, Jain, Abbeel (2020)** -- "去噪扩散概率模型"。引发扩散模型革命的 DDPM 论文。清晰推导了前向和反向马尔可夫链。
- **Song & Ermon (2019)** -- "通过估计数据分布梯度进行生成建模"。使用 Langevin 动力学采样的基于分数的方法。
- **Roberts & Rosenthal (2004)** -- "一般状态空间马尔可夫链和 MCMC 算法"。MCMC 何时以及为何有效的理论。
- **Norris (1997)** -- "马尔可夫链"。标准教材。覆盖收敛、平稳分布和命中时。
- **Welling & Teh (2011)** -- "通过随机梯度 Langevin 动力学进行贝叶斯学习"。将 SGD 与 Langevin 动力学结合以实现可扩展贝叶斯推断。