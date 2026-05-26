# 概率与分布

> 概率是 AI 表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置知识：** 第 1 阶段，第 01~04 课
**耗时：** ~75 分钟

## 学习目标

- 从零实现伯努利、分类、泊松、均匀和正态分布的 PMF 和 PDF
- 计算期望值、方差，并用中心极限定理解释为什么高斯分布无处不在
- 构建 softmax 和 log-softmax 函数，掌握数值稳定性技巧（减去最大 logit）
- 从 logits 计算交叉熵损失，并将其与负对数似然联系起来

## 问题所在

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 50,000 个候选词中挑选下一个词。扩散模型通过从学习到的分布中采样来生成图像。这些都是概率在行动。

模型做出的每一个预测都是一个概率分布。每一个损失函数都在衡量预测分布与真实分布之间的差距。每一步训练都在调整参数，使一个分布更接近另一个。没有概率，你无法读懂任何一篇 ML 论文，无法调试任何一个模型，也无法理解为什么你的训练损失是 NaN。

## 核心概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到 0 到 1 之间的数字。

```
抛硬币：
  S = {正面, 反面}
  P(正面) = 0.5,  P(反面) = 0.5

掷骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了全部概率：
1. 对任何事件 A，P(A) >= 0
2. P(S) = 1（总有事情发生）
3. 当 A 和 B 不能同时发生时，P(A 或 B) = P(A) + P(B)

其他一切（贝叶斯定理、期望、分布）都从这三条规则推导出来。

### 条件概率与独立性

P(A|B) 是在 B 已经发生的条件下 A 发生的概率。

```
P(A|B) = P(A 且 B) / P(B)

示例：一副扑克牌
  P(King | 人头牌) = P(King 且 人头牌) / P(人头牌)
                   = (4/52) / (12/52)
                   = 4/12 = 1/3
```

当知道一个事件对另一个事件毫无影响时，二者独立：

```
独立：  P(A|B) = P(A)
等价于：P(A 且 B) = P(A) * P(B)
```

抛硬币是独立的。不放回抽牌则不是。

### 概率质量函数 vs 概率密度函数

离散随机变量有**概率质量函数（PMF）**。每个结果都有可以直接读出的特定概率。

```
PMF: P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有**概率密度函数（PDF）**。单点的密度不是概率。概率来自在区间上对密度积分。

```
PDF: f(x)

P(a <= X <= b) = f(x) 从 a 到 b 的积分

f(x) 可以大于 1（密度，不是概率）
f(x) 从 -inf 到 +inf 的积分 = 1
```

这个区别在 ML 中很重要。分类输出是 PMF（离散选择）。VAE 的隐空间用 PDF（连续）。

### 常见分布

**伯努利：** 一次试验，两个结果。用于二元分类建模。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p,  方差 = p(1-p)
```

**分类：** 一次试验，k 个结果。用于多类分类建模（softmax 输出）。

```
P(X = i) = p_i,  其中 p_i 之和 = 1
示例：P(猫) = 0.7,  P(狗) = 0.2,  P(鸟) = 0.1
```

**均匀：** 所有结果等可能。用于随机初始化。

```
离散：P(X = k) = 1/n，k 属于 {1, ..., n}
连续：f(x) = 1/(b-a)，x 属于 [a, b]
```

**正态（高斯）：** 钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态：mu = 0, sigma = 1
  68% 数据在 1 个 sigma 内
  95% 在 2 个 sigma 内
  99.7% 在 3 个 sigma 内
```

**泊松：** 固定区间内稀有事件的计数。用于建模事件发生率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda,  方差 = lambda
```

### 期望值与方差

期望值是加权平均结果。

```
离散：  E[X] = x_i * P(X = x_i) 之和
连续：  E[X] = x * f(x) dx 的积分
```

方差衡量围绕均值的散布程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

在 ML 中，期望值体现为损失函数（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度方差高意味着训练噪声大。

### 联合分布与边缘分布

联合分布 P(X, Y) 同时描述两个随机变量。

联合 PMF 示例（X = 天气，Y = 带伞）：

| | Y=0（不带伞） | Y=1（带伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨天） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过对另一个变量求和得到：

```
P(X = x) = 对所有 y 求和 P(X = x, Y = y)
```

上表中的行合计和列合计就是边缘分布。

### 为什么正态分布无处不在

**中心极限定理：** 许多独立随机变量的和（或平均）趋向于正态分布，无论原始分布是什么。

```
掷 1 个骰子：  均匀分布（平的）
2 个骰子的平均：三角分布（有峰）
30 个骰子的平均：近乎完美的钟形曲线

这对**任何**初始分布都成立。
```

这就是为什么：
- 测量误差近似正态（许多小的独立来源）
- 神经网络权重初始化用正态分布
- SGD 中的梯度噪声近似正态（许多样本梯度的和）
- 正态分布是给定均值和方差下最大熵的分布

### 对数概率

原始概率会引起数值问题。将许多小概率相乘很快就会下溢为零。

```
P(句子) = P(词1) * P(词2) * ... * P(词n)
        = 0.01 * 0.003 * 0.02 * ...
        -> 0.0（约 30 项后下溢）
```

对数概率解决这个问题。乘法变成加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词n)
            = -4.6 + -5.8 + -3.9 + ...
            -> 有限数字（不会下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0（因为 0 < P <= 1）
- 越负 = 越不可能
- 交叉熵损失就是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将它们转换为合法的概率分布。

```
softmax(z_i) = exp(z_i) / 对所有 j 求和 exp(z_j)

性质：
  - 所有输出在 (0, 1) 之间
  - 所有输出之和为 1
  - 保持输入的相对顺序
  - exp() 放大了 logits 之间的差距
```

**Softmax 技巧：** 指数化前先减去最大 logit，防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1（安全）

结果相同，无溢出。
```

Log-softmax 结合 softmax 和 log 以获得数值稳定性。PyTorch 内部将其用于交叉熵损失。

### 采样

采样意味着从分布中抽取随机值。在 ML 中：
- Dropout 随机采样哪些神经元置零
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要逆变换采样、拒绝采样或重参数化技巧（VAE 中使用）等技术。

## 动手实现

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    """阶乘：n! = 1 * 2 * ... * n"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    """组合数：C(n,k) = n! / (k! * (n-k)!)"""
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    """条件概率：P(A|B) = P(A且B) / P(B)"""
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | 人头牌) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    """伯努利分布 PMF：一次试验，两个结果"""
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    """分类分布 PMF：一次试验，k 个结果"""
    return probs[k]

def poisson_pmf(k, lam):
    """泊松分布 PMF：稀有事件的计数"""
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    """均匀分布 PDF：区间内等可能"""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    """正态分布 PDF：钟形曲线"""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值与方差

```python
def expected_value(values, probabilities):
    """期望值：概率加权的平均结果"""
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    """方差：围绕均值的期望平方偏差"""
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"骰子: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    """从伯努利分布采样"""
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    """从分类分布采样（累积分布法）"""
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    """Box-Muller 变换：从均匀分布生成正态分布样本"""
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 与对数概率

```python
def softmax(logits):
    """Softmax：将 logits 转换为概率分布"""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]  # 数值稳定性技巧
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    """Log-softmax：softmax 的对数，数值稳定"""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    """交叉熵损失：负对数似然"""
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    """演示中心极限定理：样本平均趋向正态"""
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现（含所有可视化）在 `code/probability.py` 中。

## 用现成库

用 NumPy 和 SciPy，上面的所有东西都是一行代码：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零实现了这些。现在你知道库函数在做什么了。

## 练习题

1. 实现指数分布的逆变换采样。采样 10,000 个值，将直方图与真实 PDF 对比验证。

2. 为两个灌铅骰子构建联合分布表。计算边缘分布，并检验骰子是否独立。

3. 一个 5 类分类器输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别是索引 3。计算交叉熵损失，然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 写一个函数，接收一组对数概率，返回最可能的序列、总对数概率和对应的原始概率。用一个 50 词的句子测试，每个词概率 0.01。

## 关键术语

| 术语 | 大家怎么说的 | 实际上是什么意思 |
|------|-------------|---------------|
| 样本空间 | "所有可能性" | 实验所有可能结果的集合 S |
| PMF | "概率函数" | 给出每个离散结果确切概率的函数，所有概率之和为 1 |
| PDF | "概率曲线" | 连续变量的密度函数。在区间上积分得到概率 |
| 条件概率 | "给定某事的概率" | P(A\|B) = P(A 且 B) / P(B)。贝叶斯思维和贝叶斯定理的基础 |
| 独立性 | "互不影响" | P(A 且 B) = P(A) * P(B)。知道一个事件对另一个毫无信息 |
| 期望值 | "平均值" | 所有结果的概率加权和。损失函数就是一种期望值 |
| 方差 | "有多分散" | 与均值期望平方偏差。方差高 = 估计噪声大、不稳定 |
| 正态分布 | "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。因 CLT 无处不在 |
| 中心极限定理 | "平均变正态" | 许多独立样本的均值趋向正态分布，无论来源分布是什么 |
| 联合分布 | "两个变量一起" | P(X, Y) 描述 X 和 Y 每种组合的概率 |
| 边缘分布 | "把另一个变量求和掉" | P(X) = sum_y P(X, Y)。从联合分布恢复单个变量的分布 |
| 对数概率 | "概率的对数" | log P(x)。把乘积变求和，防止长序列数值下溢 |
| Softmax | "把分数变概率" | softmax(z_i) = exp(z_i) / sum(exp(z_j))。将实值 logits 映射为合法概率分布 |
| 交叉熵 | "损失函数" | -sum(p_true * log(p_predicted))。衡量两个分布的差异。越低越好 |
| Logits | "原始模型输出" | Softmax 之前的未归一化分数。得名于 logistic 函数 |
| 采样 | "抽取随机值" | 按照概率分布生成值。模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown：什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) —— 可视化证明为什么平均趋向正态
- [Stanford CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) —— 涵盖此处一切及更多的精炼参考
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) —— 为什么数值稳定性重要以及如何实现
