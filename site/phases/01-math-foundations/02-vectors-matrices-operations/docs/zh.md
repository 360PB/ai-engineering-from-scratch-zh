# 向量、矩阵与运算

> 每个神经网络本质上都是矩阵乘法，只是多包了几层。

**类型：** 动手实现
**语言：** Python, Julia
**前置要求：** 第 1 阶段第 1 课（线性代数直觉）
**时长：** ~60 分钟

## 学习目标

- 从零实现 Matrix 类，支持逐元素运算、矩阵乘法、转置、行列式和逆矩阵
- 区分逐元素乘法与矩阵乘法，并说明各自的适用场景
- 仅用手写的 Matrix 类实现一个全连接神经网络层（`relu(W @ x + b)`）
- 解释广播规则，以及神经网络框架中偏置加法的工作原理

## 问题

你想搭一个神经网络。看代码时遇到这一行：

```
output = activation(weights @ input + bias)
```

那个 `@` 就是矩阵乘法。`weights` 是个矩阵，`input` 是个向量。如果你不知道这些运算在做什么，这行代码就是魔法。如果你知道，它就是**一层的完整前向传播，只用了三个运算**。

模型处理的每张图片都是像素值构成的矩阵。每个词嵌入都是向量。神经网络的每一层都是矩阵变换。不懂矩阵运算就搭 AI 系统，就像不懂变量就写代码——行不通。

这节课从零开始建立这种熟练度。

## 概念

### 向量：有序的数字列表

向量是带方向和长度的数字列表。在 AI 中，向量代表数据点、特征或参数。

```
v = [3, 4]        -- 二维向量
w = [1, 0, -2]    -- 三维向量
```

二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度（模）是 5（勾股定理：3-4-5 三角形）。

### 矩阵：数字的网格

矩阵是二维网格。行和列。一个 m × n 矩阵有 m 行 n 列。

```
A = | 1  2  3 |     -- 2×3 矩阵（2 行，3 列）
    | 4  5  6 |
```

在神经网络中，权重矩阵把输入向量变换成输出向量。一个 784 输入、128 输出的层，用的就是 128×784 的权重矩阵。

### 为什么形状很重要

矩阵乘法有严格的规则：`(m × n) @ (n × p) = (m × p)`。中间维度必须匹配。

```
(128 × 784) @ (784 × 1) = (128 × 1)
  权重        输入        输出

中间维度：784 = 784  -- 合法
```

如果你在 PyTorch 里遇到形状不匹配的错误，原因就在这里。

### 运算速查表

| 运算 | 做什么 | 在神经网络中的用途 |
|------|--------|-------------------|
| 加法 | 对应位置相加 | 给输出加偏置 |
| 标量乘法 | 每个元素缩放 | 学习率 × 梯度 |
| 矩阵乘法 | 变换向量 | 层的前向传播 |
| 转置 | 行列互换 | 反向传播 |
| 行列式 | 浓缩成一个数 | 判断是否可逆 |
| 逆矩阵 | 撤销变换 | 解线性方程组 |
| 单位矩阵 | 什么都不做的矩阵 | 初始化、残差连接 |

### 逐元素 vs 矩阵乘法

这个区别让新手栽跟头最频繁。

**逐元素乘法**：对应位置相乘。两个矩阵形状必须完全相同。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

**矩阵乘法**：行和列的点积。中间维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算，不同的结果，不同的规则。

### 广播

把偏置向量加到输出矩阵上时，形状对不上。广播会把小数组"拉伸"去匹配大数组。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

广播把向量沿行方向拉伸：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都自动做这件事。理解它，就不会在形状看起来不对但代码能跑的时候犯迷糊。

## 动手实现

### 步骤 1：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 步骤 2：带核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("矩阵是奇异的，不存在逆矩阵")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 步骤 3：验证一下

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 步骤 4：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是单个全连接层：`output = relu(W @ x + b)`。每个神经网络的每个全连接层，做的都是这个。

## 用现成库

上面所有操作，NumPy 用更少的代码、快几个数量级就能完成。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (逐元素) =\n", A * B)
print("A @ B (矩阵乘法) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\n神经网络层: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 的 `@` 运算符调用的是 `__matmul__`。NumPy 用 C 和 Fortran 写的优化 BLAS 例程来实现它。数学一样，速度快 100 倍。

NumPy 的广播：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动把一维偏置沿两行拉伸。这就是每个神经网络框架中偏置加法的工作原理。

## 产出

本课产出一个用于通过几何直觉教授矩阵运算的提示词。见 `outputs/prompt-matrix-operations-zh.md`。

这里手写的 Matrix 类，是第 3 阶段第 10 课搭建迷你神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()`，确认结果是单位矩阵。用三个不同的 2×2 矩阵试一遍。行列式为零时会发生什么？

2. **实现 3×3 逆矩阵。** 扩展 Matrix 类，用伴随矩阵法计算 3×3 矩阵的逆。和 NumPy 的 `np.linalg.inv` 对比测试。

3. **搭一个两层网络。** 只用你的 Matrix 类（不用 NumPy），创建一个两层神经网络：输入(3) → 隐藏层(4) → 输出(2)。初始化随机权重，跑一遍前向传播，验证所有形状正确。

## 关键术语

| 术语 | 人们怎么说的 | 实际含义 |
|------|-------------|---------|
| 向量 | "一个箭头" | 有序的数字列表。在 AI 中：高维空间中的一个点。 |
| 矩阵 | "一张数表" | 线性变换。把向量从一个空间映射到另一个空间。 |
| 矩阵乘法 | "就把数字乘起来" | 第一个矩阵的每一行与第二个矩阵的每一列做 dot product。顺序很重要。 |
| 转置 | "翻过来" | 行列互换。把 m×n 矩阵变成 n×m。反向传播中很关键。 |
| 行列式 | "矩阵里蹦出来的一个数" | 衡量矩阵把面积(2D)或体积(3D)缩放了多少。为零意味着变换压扁了一个维度。 |
| 逆矩阵 | "撤销矩阵" | 能逆转变换的矩阵。只有行列式不为零时才存在。 |
| 单位矩阵 | "最无聊的矩阵" | 矩阵版的"乘以 1"。用于初始化、残差连接（ResNet）。 |
| 广播 | "自动修形状的黑魔法" | 把小数组沿缺失维度重复拉伸，去匹配大数组。 |
| 逐元素 | "普通乘法" | 对应位置相乘。两个数组形状必须相同（或可广播）。 |

## 延伸阅读

- [3Blue1Brown: 线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) - 本文所有运算的可视化直觉
- [NumPy 广播文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的确切规则
- [Stanford CS229 线性代数复习](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向 ML 的精简参考
