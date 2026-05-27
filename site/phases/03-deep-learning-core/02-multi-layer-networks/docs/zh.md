# 多层网络与前向传播

> 一个神经元画一条直线。把它们堆叠起来，你就能画任何形状。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 01（数学基础），Lesson 03.01（感知机）
**时长：** 约 90 分钟

## 学习目标

- 从零构建一个多层网络，包含 Layer 和 Network 类，执行完整的前向传播
- 追踪矩阵维度在网络的每一层中的变化，识别维度不匹配的问题
- 解释堆叠非线性激活函数如何使网络学习弯曲的决策边界
- 使用 2-2-1 架构手工调参的 Sigmoid 权重解决 XOR 问题

## 问题背景

单个神经元是一个画直线的工具。就这样。一条穿过数据的直线。每个真实的人工智能问题——图像识别、语言理解、围棋——都需要曲线。将神经元堆叠成层是获得曲线的方法。

1969 年，Minsky 和 Papert 证明了这一局限是致命的：单层网络无法学习 XOR。不是"学习困难"——而是数学上不可能。XOR 真值表将 [0,1] 和 [1,0] 放在一边，[0,0] 和 [1,1] 放在另一边。没有一条直线能将它们分开。

这导致神经网络研究经费停滞了十多年。解决方案后来显而易见：停止使用单层。把神经元堆叠成层。让第一层将输入空间分割成新的特征，让第二层将这些特征组合成单条直线无法做出的决策。

这个堆叠就是多层网络。它是当今所有生产环境中深度学习模型的基础。前向传播——数据从输入流经隐藏层到输出——是你在构建任何其他功能之前首先需要构建的东西。

## 核心概念

### 层：输入层、隐藏层、输出层

多层网络有三种类型的层：

**输入层**——其实不是一个真正的层。它保存你的原始数据。两个特征意味着两个输入节点。这里不发生计算。

**隐藏层**——工作在这里发生。每个神经元接收上一层的每一个输出，应用权重和偏置，然后将结果通过激活函数。"隐藏"是因为你永远不会在训练数据中直接看到这些值。

**输出层**——给出最终答案。对于二分类，一个 Sigmoid 神经元。对于多分类，每个类一个神经元。

```mermaid
graph LR
    subgraph Input["输入层"]
        x1["x1"]
        x2["x2"]
    end
    subgraph Hidden["隐藏层（3 个神经元）"]
        h1["h1"]
        h2["h2"]
        h3["h3"]
    end
    subgraph Output["输出层"]
        y["y"]
    end
    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    h1 --> y
    h2 --> y
    h3 --> y
```

这是一个 2-3-1 网络。两个输入，三个隐藏神经元，一个输出。每个连接带有一个权重。除了输入节点外，每个神经元都带有一个偏置。

每层产生一个称为隐藏状态的数字向量。对于文本，隐藏状态增加维度——将一个词编码为 768 个数字以捕捉语义含义。对于图像，它们减小维度——将数百万像素压缩为可管理的表示。隐藏状态是学习发生的地方。

### 神经元与激活函数

每个神经元做三件事：

1. 将每个输入乘以对应的权重
2. 求所有乘积的和并加上偏置
3. 将和通过激活函数

目前，激活函数使用 Sigmoid：

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Sigmoid 将任意数字压缩到 (0, 1) 范围内。大正输入趋向 1。大负输入趋向 0。零映射到 0.5。这个平滑曲线使学习成为可能——与感知机的硬阶跃不同，Sigmoid 处处有梯度。

### 前向传播：数据如何流动

前向传播将输入数据逐层推过网络，直到到达输出。前向传播过程中不发生学习。它是纯计算：乘法、加法、激活、重复。

```mermaid
graph TD
    X["输入: [x1, x2]"] --> WH["乘以权重矩阵 W1 (2x3)"]
    WH --> BH["加偏置向量 b1 (3,)"]
    BH --> AH["对每个元素应用 Sigmoid"]
    AH --> H["隐藏输出: [h1, h2, h3]"]
    H --> WO["乘以权重矩阵 W2 (3x1)"]
    WO --> BO["加偏置向量 b2 (1,)"]
    BO --> AO["应用 Sigmoid"]
    AO --> Y["输出: y"]
```

在每一层，三个操作按顺序发生：

```
z = W * input + b       （线性变换）
a = sigmoid(z)          （激活）
```

一层的输出成为下一层的输入。这就是整个前向传播。

### 矩阵维度

追踪维度是深度学习中最重度的调试技能。这里是 2-3-1 网络：

| 步骤 | 操作 | 维度 | 结果形状 |
|------|-----------|------------|-------------|
| 输入 | x | -- | (2,) |
| 隐藏层线性 | W1 * x + b1 | W1: (3, 2), b1: (3,) | (3,) |
| 隐藏层激活 | sigmoid(z1) | -- | (3,) |
| 输出层线性 | W2 * h + b2 | W2: (1, 3), b2: (1,) | (1,) |
| 输出层激活 | sigmoid(z2) | -- | (1,) |

规则：第 k 层权重矩阵 W 的形状为 (第 k 层神经元数, 第 k-1 层神经元数)。行对应当前层，列对应上一层。如果形状对不上，就有一个 bug。

### 万能逼近定理

1989 年，George Cybenko 证明了一个了不起的结论：一个带有单个隐藏层和足够多神经元的神经网络可以以任意精度逼近任何连续函数。

这并不意味着单个隐藏层总是最好的。它意味着这个架构在理论上是可行的。实际上，更深的网络（更多层、每层更少神经元）以比宽浅网络少得多的总参数学习相同的函数。这就是深度学习有效的原因。

直觉：隐藏层中的每个神经元学习一个"凸起"或特征。将足够多的凸起放在正确的位置，可以逼近任何平滑曲线。神经元越多，凸起越多，逼近越精确。

```mermaid
graph LR
    subgraph FewNeurons["4 个隐藏神经元"]
        A["粗糙逼近"]
    end
    subgraph MoreNeurons["16 个隐藏神经元"]
        B["接近逼近"]
    end
    subgraph ManyNeurons["64 个隐藏神经元"]
        C["近乎完美拟合"]
    end
    FewNeurons --> MoreNeurons --> ManyNeurons
```

### 可组合性

神经网络是可组合的。你可以堆叠它们、链接它们、并行运行它们。Whisper 模型使用一个编码器网络处理音频和一个独立的解码器网络生成文本。现代 LLM 是纯解码器。BERT 是纯编码器。T5 是编码器-解码器。架构选择决定了模型能做什么。

## 从零构建

纯 Python。不使用 numpy。每个矩阵操作都从零手写。

### 步骤 1：Sigmoid 激活函数

```python
import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
```

将值限制在 [-500, 500] 范围内以防止溢出。`math.exp(500)` 是一个很大的数但有限。`math.exp(1000)` 是无穷大。

### 步骤 2：Layer 类

深度学习中最重要的操作是矩阵乘法。每一层、每个注意力头、每次前向传播——都是矩阵乘法。一个线性层接收一个输入向量，乘以权重矩阵，再加上偏置向量：y = Wx + b。这个等式是神经网络中 90% 计算量的来源。

一个层持有一个权重矩阵和一个偏置向量。它的 forward 方法接收一个输入向量并返回激活后的输出。

```python
class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output
```

权重矩阵形状为 (n_neurons, n_inputs)。每一行是一个神经元对所有输入的权重。forward 方法遍历神经元，计算加权和加上偏置，应用 Sigmoid，收集结果。

### 步骤 3：Network 类

网络是一个层的列表。前向传播将它们链接起来：第 k 层的输出流入第 k+1 层。

```python
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current
```

这就是整个前向传播。四行逻辑。数据进入，流经每一层，从另一端出来。

### 步骤 4：用手动调参的权重解决 XOR

在 Lesson 01 中，我们通过组合 OR、NAND 和 AND 感知机解决了 XOR。现在用我们的 Layer 和 Network 类做同样的事情。2-2-1 架构：两个输入，两个隐藏神经元，一个输出。

```python
hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (四舍五入: {predicted}, 期望: {expected})")
```

大权重（20, -20）使 Sigmoid 的行为像阶跃函数。第一个隐藏神经元近似 OR。第二个近似 NAND。输出神经元将它们组合成 AND，也就是 XOR。

### 步骤 5：圆形分类

一个更难的问题：将二维点分类为在半径 0.5、圆心在原点的圆内或圆外。这需要一个弯曲的决策边界——单个感知机无法做到。

```python
import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])
```

使用随机权重，网络分类效果不会好。但前向传播仍然能运行。这才是重点——前向传播只是计算。学习正确的权重是反向传播，在 Lesson 03 中讲解。

```python
correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"随机权重准确率: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
```

随机权重给出很低的准确率——通常比猜测多数类更差。训练后（Lesson 03），相同架构加上 8 个隐藏神经元将画出一条弯曲的边界，将圆内和圆外分开。

## 使用 PyTorch

PyTorch 用四行代码完成上述所有操作：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
output = model(x)
print(output)
```

`nn.Linear(2, 8)` 就是你的 Layer 类：形状为 (8, 2) 的权重矩阵，形状为 (8,) 的偏置向量。`nn.Sigmoid()` 是你的 Sigmoid 函数，逐元素应用。`nn.Sequential` 就是你的 Network 类：按顺序链接层。

区别是速度和规模。PyTorch 运行在 GPU 上，处理数百万样本的批次，自动计算反向传播的梯度。但前向传播的逻辑和你从零构建的是完全相同的。

## 发布

本课产出一个可用于设计网络架构的提示词：

- `outputs/prompt-network-architect.md`

当你需要为一个给定问题决定层数、每层神经元数和使用哪些激活函数时使用它。

## 练习

1. 构建一个 2-4-2-1 网络（两个隐藏层），用随机权重在 XOR 数据上运行前向传播。打印中间隐藏层的输出，观察表示在每层如何转换。

2. 将圆形分类器的隐藏层大小从 8 改为 2，再改为 32。每次用随机权重运行前向传播。隐藏神经元数量是否改变了输出范围或分布？为什么？

3. 在 Network 类上实现一个 `count_parameters` 方法，返回可训练权重和偏置的总数。在 784-256-128-10 网络（经典 MNIST 架构）上测试。它有多少参数？

4. 为 3-4-4-2 网络构建前向传播。将 RGB 颜色值（归一化到 0-1）输入其中，观察两个输出。这是一个带有两个类的简单颜色分类器的架构。

5. 将 Sigmoid 替换为"泄漏阶跃"函数：如果 z < 0 返回 0.01 * z，否则返回 1.0。用步骤 4 中相同的手动调参权重在 XOR 上运行前向传播。它还能工作吗？为什么平滑的 Sigmoid 比硬截断更好？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 前向传播 | "运行模型" | 将输入推过每一层——乘以权重、加偏置、激活——产生输出 |
| 隐藏层 | "中间部分" | 输入层和输出层之间的任何层，其值不会直接在数据中观察到 |
| 多层网络 | "深度神经网络" | 按顺序堆叠的神经元层，其中每层的输出作为下一层的输入 |
| 激活函数 | "非线性部分" | 在线 性变换之后应用的函数，将曲线引入决策边界 |
| Sigmoid | "S 曲线" | sigma(z) = 1/(1+e^(-z))，将任意实数压缩到 (0,1)，处处平滑可微 |
| 权重矩阵 | "参数" | 形状为 (当前层神经元数, 上一层神经元数) 的矩阵 W，包含可学习的连接强度 |
| 偏置向量 | "偏移量" | 在矩阵乘法之后加上的向量，使神经元即使在所有输入为零时也能激活 |
| 万能逼近 | "神经网络能学习任何东西" | 带有足够神经元的单个隐藏层可以逼近任何连续函数——但"足够"可能意味着数十亿 |
| 线性变换 | "矩阵乘法步骤" | z = W * x + b，激活前的计算，将输入映射到一个新空间 |
| 决策边界 | "分类器切换的地方" | 网络输出穿过分类阈值的输入空间中的曲面 |

## 扩展阅读

- Michael Nielsen, "Neural Networks and Deep Learning"，第 1-2 章 (http://neuralnetworksanddeeplearning.com/) ——关于前向传播和网络结构的最清晰的免费解释，带有交互式可视化
- Cybenko, "Approximation by Superpositions of a Sigmoidal Function" (1989) ——原始万能逼近定理论文，出人意料地易读
- 3Blue1Brown, "But what is a neural network?" (https://www.youtube.com/watch?v=aircAruvnKk) ——20 分钟的层、权重和前向传播视觉讲解，建立正确的心理模型
- Goodfellow, Bengio, Courville, "Deep Learning"，第 6 章 (https://www.deeplearningbook.org/) ——多层网络的标准参考，免费在线阅读