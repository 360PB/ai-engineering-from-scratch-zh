# 链式法则与自动求导

> 链式法则是每个神经网络学习背后的引擎。

**类型：** 动手实现
**语言：** Python
**前置知识：** 第 1 阶段，第 04 课（导数与梯度）
**耗时：** ~90 分钟

## 学习目标

- 从零构建一个极简自动微分引擎（Value 类），能记录运算并通过反向模式自动求导计算梯度
- 利用拓扑排序实现计算图的前向传播与反向传播
- 仅使用从零实现的自动微分引擎，构建并训练一个多层感知器解决 XOR 问题
- 利用梯度校验（与数值有限差分对比）验证自动微分的正确性

## 问题所在

你可以计算简单函数的导数。但神经网络不是简单函数。它是数百个函数的复合：矩阵乘法、加偏置、施加激活函数、再次矩阵乘法、softmax、交叉熵损失。输出是函数的函数的函数。

为了训练网络，你需要损失函数对每一个权重的梯度。手动计算对数百万个参数来说是不可能的。数值法（有限差分）又太慢。

链式法则给了你数学基础。自动求导给了你算法。二者结合，让你能在一次前向传播的时间里，精确计算任意函数复合的梯度。

这就是 PyTorch、TensorFlow 和 JAX 的工作原理。你将从零构建一个迷你版。

## 核心概念

### 链式法则

若 `y = f(g(x))`，则 `y` 对 `x` 的导数为：

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

沿链相乘每个局部导数。

示例：`y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

对于更深的复合：

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

神经网络中的每一层都是这条链上的一个环节。

### 计算图

计算图让链式法则可视化。每个运算成为一个节点。数据正向流过图，梯度反向流过。

**前向传播（计算数值）：**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (乘法)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (加法)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["输出 y"]
```

**反向传播（计算梯度）：**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1，因为 c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

反向传播在每个节点应用链式法则，将梯度从输出传播到输入。

### 前向模式 vs 反向模式

有两种方式将链式法则应用于计算图。

**前向模式**从输入开始，将导数向前推进。它计算 `dx/dx = 1`，然后传播经过每个运算。适用于**输入少、输出多**的场景。

```
前向模式：种子 dx/dx = 1，向前传播

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**反向模式**从输出开始，将梯度向后拉。它计算 `dy/dy = 1`，然后反向传播经过每个运算。适用于**输入多、输出少**的场景。

```
反向模式：种子 dy/dy = 1，向后传播

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

神经网络有数百万个输入（权重）和一个输出（损失）。反向模式一次反向传播就能算出所有梯度。这就是反向传播使用反向模式的原因。

| 模式 | 种子 | 方向 | 最佳场景 |
|------|------|------|---------|
| 前向 | `dx_i/dx_i = 1` | 输入→输出 | 输入少，输出多 |
| 反向 | `dy/dy = 1` | 输出→输入 | 输入多，输出少（神经网络） |

### 对偶数实现前向模式

前向模式可以用对偶数优雅地实现。对偶数形式为 `a + b*epsilon`，其中 `epsilon^2 = 0`。

```
对偶数：(值, 导数)

(2, 1) 表示：值为 2，对 x 的导数为 1

运算规则：
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

将输入变量种子的导数设为 1，导数会自动传播经过每个运算。

### 构建自动微分引擎

自动微分引擎需要三样东西：

1. **数值包装。** 将每个数字包装成对象，存储其值和梯度。
2. **图记录。** 每个运算记录其输入和局部梯度函数。
3. **反向传播。** 对图做拓扑排序，然后反向遍历，在每个节点应用链式法则。

这正是 PyTorch 的 `autograd` 做的事情。`torch.Tensor` 类包装数值，在 `requires_grad=True` 时记录运算，并在调用 `.backward()` 时计算梯度。

### PyTorch Autograd 底层原理

当你写 PyTorch 代码时：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch 内部：

1. 为 `x` 创建一个带 `requires_grad=True` 的 `Tensor` 节点
2. 每个运算（`**`, `*`, `+`）创建一个新节点并记录反向函数
3. `y.backward()` 触发已记录图的反向模式自动求导
4. 每个节点的 `grad_fn` 计算局部梯度并传递给父节点
5. 梯度通过**累加**（而非替换）累积到 `.grad` 属性

图是动态的（define-by-run）。每次前向传播都会构建一个新图。这就是 PyTorch 支持模型内控制流（if/else、循环）的原因。

## 动手实现

### 第 1 步：Value 类

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data        # 当前数值
        self.grad = 0.0         # 梯度（初始为 0）
        self._backward = lambda: None  # 反向传播闭包
        self._prev = set(children)     # 父节点
        self._op = op           # 产生该节点的运算

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

每个 `Value` 存储其数值、梯度（初始为零）、反向函数，以及指向产生它的子节点的指针。

### 第 2 步：带梯度追踪的算术运算

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            # 加法导数：dy/da = 1, dy/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # 乘法导数：d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            # ReLU 导数：正值为 1，负值为 0
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

每个运算创建一个闭包，知道如何计算局部导数并乘以上游梯度（`out.grad`）。`+=` 处理了某个值被用于多个运算的情况。

### 第 3 步：反向传播

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # dy/dy = 1
        for v in reversed(topo):
            v._backward()
```

拓扑排序确保每个节点的梯度在其传播给子节点之前已经完全计算。种子梯度为 1.0（dy/dy = 1）。

### 第 4 步：更多运算以构成完整引擎

基本的 Value 类处理了加法、乘法和 relu。一个真正的自动微分引擎需要更多。以下是构建神经网络所需的运算：

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            # x^n 导数：n * x^(n-1)
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            # exp 导数：exp(x)
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            # log 导数：1/x
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            # tanh 导数：1 - tanh^2(x)
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**每个运算为何重要：**

| 运算 | 反向规则 | 用途 |
|------|---------|------|
| `__sub__` | 复用 add + neg | 损失计算（预测 - 目标） |
| `__pow__` | n * x^(n-1) | 多项式激活、MSE（误差^2） |
| `__truediv__` | 复用 mul + pow(-1) | 归一化、学习率缩放 |
| `exp` | exp(x) * 上游 | Softmax、对数似然 |
| `log` | (1/x) * 上游 | 交叉熵损失、对数概率 |
| `tanh` | (1 - tanh^2) * 上游 | 经典激活函数 |

精妙之处：`__sub__` 和 `__truediv__` 通过现有运算定义。因为链式法则会复合到底层的 add/mul/pow 运算，它们天然就能得到正确的梯度。

### 第 5 步：从零构建迷你 MLP

有了完整的 Value 类，你就可以构建神经网络了。不需要 PyTorch，不需要 NumPy。只需要 Value 和链式法则。

```python
import random

class Neuron:
    """单个神经元：加权求和 + 激活函数"""
    def __init__(self, n_inputs):
        # 随机初始化权重 [-1, 1]，偏置为 0
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        # 计算：tanh(w1*x1 + w2*x2 + ... + b)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    """一层神经元"""
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """多层感知器"""
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

一个 `Neuron` 计算 `tanh(w1*x1 + w2*x2 + ... + b)`。一个 `Layer` 是一组神经元。一个 `MLP` 堆叠多层。每个权重都是一个 `Value`，所以调用 `loss.backward()` 就会把梯度传播给每个参数。

**在 XOR 上训练：**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 输入，4 隐藏神经元，1 输出

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR 模式（用 -1/1 适配 tanh）

for step in range(100):
    preds = [model(x) for x in xs]
    # MSE 损失
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    # 每次迭代前清零梯度（避免累加）
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # 梯度下降
    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"第 {step:3d} 步  loss = {loss.data:.4f}")

print("\n训练后的预测：")
for x, y in zip(xs, ys):
    print(f"  输入={x}  目标={y:2d}  预测={model(x).data:6.3f}")
```

这就是 micrograd。一个用纯 Python 写的完整神经网络训练循环，带自动微分。每个商业深度学习框架都在大规模上做同样的事情。

### 第 6 步：梯度校验

怎么知道你的自动微分是否正确？和数值导数对比。这就是梯度校验。

```python
def gradient_check(build_expr, x_val, h=1e-7):
    """梯度校验：对比自动微分 vs 数值差分"""
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

在复杂表达式上测试：

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"自动微分:  {ad:.8f}")
print(f"数值差分:  {num:.8f}")
print(f"误差:      {diff:.2e}")
# 误差应 < 1e-5
```

梯度校验在实现新运算时至关重要。如果你的反向传播有 bug，数值校验能抓住它。每个严肃的深度学习实现都会在开发期间运行梯度校验。

**何时使用梯度校验：**

| 场景 | 需要做梯度校验？ |
|------|---------------|
| 给自动微分添加新运算 | 是，永远要做 |
| 调试一个无法收敛的训练循环 | 是，先检查梯度 |
| 生产环境训练 | 否，太慢（每个参数需要 2 次前向传播） |
| 自动微分代码的单元测试 | 是，自动化它 |

### 第 7 步：手动计算验证

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)   # b = 7.0
y = b.relu()         # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

手动验算：`y = relu(x1*x2 + 1)`。由于 `x1*x2 + 1 = 7 > 0`，relu 是恒等函数。
`dy/dx1 = x2 = 3`。`dy/dx2 = x1 = 2`。引擎结果吻合。

## 用现成库

### 与 PyTorch 验证对比

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

梯度一致。你的引擎和 PyTorch 算出同样的结果，因为数学是一样的：通过链式法则的反向模式自动求导。

### 更复杂的表达式

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## 产出

本课产出：
- `outputs/自动微分技能-zh.md` —— 构建与调试自动微分系统的技能文档
- `code/autodiff.py` —— 一个可扩展的极简自动微分引擎

这里构建的 Value 类是第 3 阶段神经网络训练循环的基础。

## 练习题

1. 给 Value 类添加 `__pow__`，使其支持 `x ** n`。验证 `d/dx(x^3)` 在 `x=2` 时等于 `12.0`。

2. 添加 `tanh` 作为激活函数。验证 `tanh'(0) = 1` 和 `tanh'(2) ≈ 0.0707`。

3. 为单个神经元构建计算图：`y = relu(w1*x1 + w2*x2 + b)`。计算全部五个梯度，并与 PyTorch 验证对比。

4. 用对偶数实现前向模式自动求导。创建 `Dual` 类并验证其导数与反向模式引擎一致。

## 关键术语

| 术语 | 大家怎么说的 | 实际上是什么意思 |
|------|-------------|---------------|
| 链式法则 | "导数相乘" | 复合函数的导数等于每个函数的局部导数的乘积，在正确的点求值 |
| 计算图 | "网络图" | 有向无环图，节点是运算，边在正向传播数值、反向传播梯度 |
| 前向模式 | "导数向前推" | 从输入到输出传播导数的自动求导。每个输入变量需要一次遍历。 |
| 反向模式 | "反向传播" | 从输出到输入传播梯度的自动求导。每个输出变量需要一次遍历。 |
| 自动求导 | "自动梯度" | 记录数值上的运算、构建图并通过链式法则计算精确梯度的系统 |
| 对偶数 | "值加导数" | 形式为 a + b*epsilon（epsilon^2 = 0）的数，通过算术携带导数信息 |
| 拓扑排序 | "依赖顺序" | 对图节点排序，使每个节点排在其所有依赖之后。梯度传播需要正确的顺序。 |
| 梯度累加 | "累加，不替换" | 当一个值被多个运算使用时，其梯度是所有传入梯度贡献之和 |
| 动态图 | "运行中定义" | 每次前向传播都重建计算图，允许模型内使用 Python 控制流（PyTorch 风格） |
| 梯度校验 | "数值验证" | 将自动微分梯度与数值有限差分梯度对比以验证正确性。调试必备。 |
| 多层感知器 | "多层感知器" | 有一个或多个隐藏层神经元的神经网络。每个神经元计算加权和加偏置，然后施加激活函数。 |
| 神经元 | "加权求和 + 激活" | 基本单元：输出 = 激活(w1*x1 + w2*x2 + ... + b)。权重和偏置是可学习参数。 |

## 延伸阅读

- [3Blue1Brown：反向传播的微积分](https://www.youtube.com/watch?v=tIeHLnjs5U8) —— 神经网络中链式法则的可视化解释
- [PyTorch Autograd 机制](https://pytorch.org/docs/stable/notes/autograd.html) —— 真实系统的工作原理
- [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) —— 全面参考
