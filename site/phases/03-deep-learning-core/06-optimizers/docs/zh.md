# 优化器

> 梯度下降告诉你向哪个方向移动。它不说走多远或多快。SGD 是指南针。Adam 是带交通数据的 GPS。

**类型：** 构建
**语言：** Python
**前置要求：** Lesson 03.05（损失函数）
**时长：** 约 75 分钟

## 学习目标

- 从零在 Python 中实现 SGD、带动量的 SGD、Adam 和 AdamW 优化器
- 解释 Adam 的偏差校正如何在早期训练步骤补偿零初始化的矩估计
- 证明 AdamW 在相同任务上比带 L2 正则化的 Adam 产生更好的泛化
- 为 Transformer、CNN、GAN 和微调选择适当的优化器和默认超参数

## 问题背景

你计算了梯度。你知道第 4721 个权重应该减少 0.003 来减少损失。但 0.003 是什么单位？按什么缩放？在第 1 步和第 1000 步应该走相同的量吗？

普通梯度下降在每一步对每个参数应用相同的学习率：w = w - lr * gradient。这产生了三个问题，使神经网络训练在实践中很痛苦。

第一，振荡。损失景观很少像光滑的碗。它更像一个长的狭窄山谷。梯度指向山谷对面（陡峭方向），而不是沿着它（浅方向）。梯度下降在狭窄维度上来回弹跳，同时在有用的方向上取得微小的进展。你见过这个：损失快速下降然后 plateau，不是因为模型收敛了，而是因为它在振荡。

第二，一个学习率对所有参数是错误的。一些权重需要大更新（它们在早期、欠拟合阶段）。其他需要小更新（它们接近最优值）。对前者有效学习率会摧毁后者，反之亦然。

第三，鞍点。在高维空间中，损失景观有大片梯度接近零的平坦区域。普通 SGD 以梯度的速度爬过这些区域，有效速度为零。模型看起来卡住了。它没有卡住——它在有用下降的另一边有一个平坦区域。但 SGD 没有机制推动通过。

Adam 解决了所有三个。它为每个参数维护两个运行平均值——梯度均值（动量，处理振荡）和梯度平方均值（自适应率，处理不同尺度）。结合前几步的偏差校正，它给你一个单一的优化器，在 80% 的问题上用默认超参数工作。本课从零构建它，这样你就精确地理解为什么它在另外 20% 上失败。

## 核心概念

### 随机梯度下降（SGD）

最简单的优化器。在小批量上计算梯度，然后向相反方向迈步。

```
w = w - lr * gradient
```

"随机"意味着你使用数据的随机子集（小批量）来估计梯度，而不是整个数据集。这种噪声实际上是有用的——它帮助逃离尖锐的局部最小值。但噪声也会导致振荡。

学习率是唯一的旋钮。太 高：损失发散。太 低：训练永远持续。在普通 SGD 上对现代网络，典型值从 0.01 到 0.1。但即使在单一训练运行中，理想学习率也会变化。

### 动量

球滚下坡的类比被过度使用但很准确。不是仅按梯度迈步，而是维护一个累积过去梯度的速度。

```
m_t = beta * m_{t-1} + gradient
w = w - lr * m_t
```

Beta（通常为 0.9）控制保留多少历史。Beta = 0.9 时，动量大约是最后 10 个梯度的平均值（1 / (1 - 0.9) = 10）。

为什么这修复了振荡：方向相同的梯度累积。方向翻转的梯度相互抵消。在那个狭窄的山谷中，"穿过"分量每步翻转符号并被抑制。"沿着"分量保持一致并被放大。结果是在有用方向上的平滑加速。

真实数字：普通 SGD 在条件不好的损失景观上可能需要 10,000 步。带动量的 SGD（beta=0.9）在相同问题上通常需要 3,000-5,000 步。加速不是边缘的。

### RMSProp

第一个真正有效的每参数自适应学习率方法。由 Hinton 在 Coursera 讲座中提出（从未正式发表）。

```
s_t = beta * s_{t-1} + (1 - beta) * gradient^2
w = w - lr * gradient / (sqrt(s_t) + epsilon)
```

s_t 追踪梯度平方的运行平均值。一致大梯度的参数被一个大数除（更小的有效学习率）。小梯度的参数被一个小数除（更大的有效学习率）。

这解决了"一个学习率对所有参数"的问题。一个已经持续获得大更新的权重可能接近其目标——减慢它。一个持续获得小更新的权重可能训练不足——加速它。

Epsilon（通常为 1e-8）防止当参数尚未更新时除以零。

### Adam：动量 + RMSProp

Adam 结合了两个想法。它为每个参数维护两个指数移动平均值：

```
m_t = beta1 * m_{t-1} + (1 - beta1) * gradient        （一阶矩：均值）
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2       （二阶矩：方差）
```

**偏差校正** 是大多数解释跳过的关键细节。在第 1 步，m_1 = (1 - beta1) * gradient。Beta1 = 0.9 时，那是 0.1 * gradient——小 10 倍。移动平均还没有热身。偏差校正补偿：

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

在第 1 步且 beta1 = 0.9 时：m_hat = m_1 / (1 - 0.9) = m_1 / 0.1 = 实际梯度。在第 100 步时：(1 - 0.9^100) 约等于 1.0，所以校正消失。偏差校正对前 ~10 步重要，~50 步后无关。

更新：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Adam 默认值：lr = 0.001，beta1 = 0.9，beta2 = 0.999，epsilon = 1e-8。这些默认值在 80% 的问题上有效。当它们不起作用时，首先改 lr。然后改 beta2。几乎从不改 beta1 或 epsilon。

### AdamW：正确实现权重衰减

L2 正则化将 lambda * w^2 加到损失上。在普通 SGD 中，这等效于权重衰减（每步从权重中减去 lambda * w）。在 Adam 中，这种等效性被打破。

Loshchilov & Hutter 的洞察：当你在损失上加上 L2 然后 Adam 处理梯度时，自适应学习率缩放正则化项。大梯度方差的参数获得更少的正则化。小方差的参数获得更多。这不是你想要的——你想要均匀的正则化，无论梯度统计如何。

AdamW 通过直接应用到权重（在 Adam 更新之后）应用权重衰减来修复：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * lambda * w
```

权重衰减项（lr * lambda * w）不被 Adam 的自适应因子缩放。每个参数获得相同的比例收缩。

这看起来像是一个小细节。不是的。AdamW 实际上在几乎每个任务上都比 Adam + L2 正则化收敛到更好的解决方案。它是训练 Transformer、扩散模型和大多数现代架构的 PyTorch 默认优化器。BERT、GPT、LLaMA、Stable Diffusion——全部用 AdamW 训练。

### 学习率：最重要的超参数

```mermaid
graph TD
    LR["学习率"] --> TooHigh["太高 (lr > 0.01)"]
    LR --> JustRight["正好"]
    LR --> TooLow["太低 (lr < 0.00001)"]

    TooHigh --> Diverge["损失爆炸<br/>NaN 权重<br/>训练崩溃"]
    JustRight --> Converge["损失稳定下降<br/>达到好的最小值<br/>泛化良好"]
    TooLow --> Stall["损失缓慢下降<br/>卡在次优最小值<br/>浪费计算"]

    JustRight --> Schedule["通常需要调度"]
    Schedule --> Warmup["预热：从 0 逐渐升到最大值<br/>训练的前 1-10%"]
    Schedule --> Decay["衰减：随时间减少<br/>余弦或线性"]
```

如果你只调一个超参数，调学习率。学习率改变 10 倍比你做出的任何架构决策都更重要。常见默认值：

- SGD：lr = 0.01 到 0.1
- Adam/AdamW：lr = 1e-4 到 3e-4
- 微调预训练模型：lr = 1e-5 到 5e-5
- 学习率预热：前 1-10% 步线性斜升

### 优化器比较

```mermaid
flowchart LR
    subgraph "优化路径"
        SGD_P["SGD<br/>在山谷中振荡<br/>慢但找到平坦最小值"]
        Mom_P["SGD + 动量<br/>更平滑的路径<br/>比 SGD 快 3 倍"]
        Adam_P["Adam<br/>每参数自适应<br/>快速收敛"]
        AdamW_P["AdamW<br/>Adam + 正确衰减<br/>最佳泛化"]
    end
    SGD_P --> Mom_P --> Adam_P --> AdamW_P
```

### 何时每个优化器胜出

```mermaid
flowchart TD
    Task["你在训练什么？"] --> Type{"模型类型？"}

    Type -->|"Transformer / LLM"| AdamW["AdamW<br/>lr=1e-4, wd=0.01-0.1"]
    Type -->|"CNN / ResNet"| SGD_M["SGD + 动量<br/>lr=0.1, 动量=0.9"]
    Type -->|"GAN"| Adam2["Adam<br/>lr=2e-4, beta1=0.5"]
    Type -->|"微调"| AdamW2["AdamW<br/>lr=2e-5, wd=0.01"]
    Type -->|"还不知道"| Default["从 AdamW 开始<br/>lr=3e-4, wd=0.01"]
```

## 从零构建

### 步骤 1：普通 SGD

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

### 步骤 2：带动量的 SGD

```python
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

### 步骤 3：Adam

```python
import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
```

### 步骤 4：AdamW

```python
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]
```

### 步骤 5：训练比较

在 lesson 05 的圆形数据集上用所有四种优化器训练相同的两层网络。比较收敛。

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, 准确率={accuracy:.1f}%")
        return losses
```

## 使用 PyTorch

PyTorch 优化器处理参数组、梯度裁剪和学习率调度：

```python
import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

模式总是一样的：zero_grad、forward、loss、backward、（clip）、step、（schedule）。记住这个顺序。搞错顺序（如在 optimizer.step() 之前调用 scheduler.step()）是微妙 bug 的常见来源。

对于 CNN，许多从业者仍然喜欢 SGD + 动量（lr=0.1，动量=0.9，权重衰减=1e-4）配合步长或余弦调度。SGD 找到更平坦的最小值，通常泛化更好。对于 Transformer 和 LLM，AdamW 加预热 + 余弦衰减是通用默认值。没有有充分理由测量的共识就不要对抗。

## 发布

本课产出：
- `outputs/prompt-optimizer-selector.md` ——为任何架构选择正确优化器和学习率的决策提示词

## 练习

1. 实现 Nesterov 动量，在"前瞻"位置（w - lr * beta * v）而不是当前位置计算梯度。在圆形数据集上比较标准动量的收敛。

2. 实现学习率预热调度：在前 10% 训练步从 0 线性升到 max_lr，然后余弦衰减到 0。用预热 + 余弦的 Adam vs 没有预热的 Adam 训练。测量在圆形数据集上达到 90% 准确率需要多少轮。

3. 在 Adam 训练期间追踪每个参数的有效学习率。有效率是 lr * m_hat / (sqrt(v_hat) + eps)。在 10、50 和 200 步后绘制有效率分布。所有参数以相同速度更新吗？

4. 实现梯度裁剪（按全局范数裁剪）。将最大梯度范数设置为 1.0。在高学习率（lr=0.01 用于 Adam）下训练有裁剪和无裁剪。在 10 个随机种子中计算有多少次发散（损失变为 NaN）。

5. 在有大权重的网络上比较 Adam vs AdamW。将所有权重初始化为 [-5, 5]（远大于正常）的随机值。用 weight_decay=0.1 训练 200 轮。为两个优化器绘制训练期间的权重 L2 范数。AdamW 应该显示更快的权重收缩。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 学习率 | "步长" | 梯度更新上的标量乘数；训练中影响最大的单一超参数 |
| SGD | "基本梯度下降" | 随机梯度下降：在小批量上计算梯度后减去 lr * gradient 更新权重 |
| 动量 | "滚球类比" | 过去梯度的指数移动平均；抑制振荡并加速一致方向 |
| RMSProp | "自适应学习率" | 将每个参数的梯度除以其最近梯度的运行 RMS；均衡学习率 |
| Adam | "默认优化器" | 结合动量（一阶矩）和 RMSProp（二阶矩）并在初始步骤进行偏差校正 |
| AdamW | "正确做的 Adam" | 带解耦权重衰减的 Adam；直接应用到权重而不是通过梯度应用正则化 |
| 偏差校正 | "运行平均值的预热" | 除以 (1 - beta^t) 以补偿 Adam 矩估计的零初始化 |
| 权重衰减 | "收缩权重" | 每步减去权重值的分数；惩罚大权重的正则化器 |
| 学习率调度 | "随时间改变 lr" | 在训练期间调整学习率的函数；预热 + 余弦衰减是现代默认 |
| 梯度裁剪 | "限制梯度范数" | 当其范数超过阈值时向下缩放梯度向量；防止梯度爆炸更新 |

## 扩展阅读

- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) ——原始 Adam 论文，包含收敛性分析和偏差校正推导
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017) ——证明在 Adam 中 L2 正则化和权重衰减不相等，并提出 AdamW
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) ——引入 LR 范围测试和周期性调度，消除了调优固定学习率的需要
- Ruder, "An Overview of Gradient Descent Optimization Algorithms" (2016) ——所有优化器变体的最佳单一综述，包含清晰的比较和直觉