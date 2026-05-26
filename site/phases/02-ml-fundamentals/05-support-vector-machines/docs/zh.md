# 支持向量机

> 在两个类之间找到最宽的街道。这就是全部思想。

**类型：** Build
**语言：** Python
**前置要求：** Phase 1（Lessons 08 优化、14 范数与距离、18 凸优化）
**时长：** 约 90 分钟

## 学习目标

- 从零实现线性 SVM，使用铰链损失和对原始形式的梯度下降
- 解释最大间隔原则，并从训练好的模型中识别支持向量
- 比较线性、多项式和 RBF 核，解释核技巧如何避免显式高维映射
- 评估 C 参数在间隔宽度和分类错误之间的权衡

## 问题背景

你有两类数据点，需要画一条线（或超平面）将它们分开。无限多条线都可以。应该选哪条？

选择间隔最大的那条。间隔是决策边界与每侧最近数据点之间的距离。更宽的间隔意味着分类器更自信，对未见数据泛化更好。

这种直觉引出了支持向量机，ML 中数学上最优雅的算法之一。SVM 在深度学习之前是主导分类方法，对于小数据集、高维数据以及需要原则性、充分理解且有理论保证的模型的问题仍然是最佳选择。

SVM 直接连接到 Phase 1：优化是凸的（Lesson 18），间隔用范数测量（Lesson 14），核技巧利用点积处理非线性边界而无需在高维空间中计算。

## 核心概念

### 最大间隔分类器

给定标签 y_i ∈ {-1, +1} 和特征向量 x_i 的线性可分数据，我们想要一个分离类的超平面 w^T x + b = 0。

点到超平面的距离：

```
distance = |w^T x_i + b| / ||w||
```

对于正确分类的点：y_i * (w^T x_i + b) > 0。间隔是超平面到两侧最近点距离的两倍。

```mermaid
graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["+ 类点"] --> A
    E["- 类点"] --> C
    B --- F["决策边界"]
```

优化问题：

```
maximize    2 / ||w||     （间隔宽度）
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

等价地（最小化 ||w||² 更容易优化）：

```
minimize    (1/2) ||w||^2
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

这是凸二次规划。它有唯一全局解。恰好坐在间隔边界上的数据点（其中 y_i * (w^T x_i + b) = 1）是支持向量。它们是唯一决定决策边界的点。移动或删除任何非支持向量点，边界不变。

### 支持向量：关键少数

大多数训练点无关紧要。只有支持向量重要。这就是为什么 SVM 在预测时内存效率高：你只需要存储支持向量，而不是整个训练集。

支持向量的数量也对泛化误差有界。相对于数据集大小，支持向量越少意味着泛化越好。

### 软间隔：用 C 参数处理噪声

真实数据很少完全可分。有些点可能在边界的错误侧，或在间隔内。软间隔公式通过引入松弛变量允许违规。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i)
subject to  y_i * (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0  for all i
```

松弛变量 xi_i 测量点 i 违反间隔的程度。C 控制权衡：

| C 值 | 行为 |
|------|------|
| 大 C | 严重惩罚违规。窄间隔，更少错误分类。过拟合 |
| 小 C | 允许更多违规。宽间隔，更多错误分类。欠拟合 |

C 是正则化强度的倒数。大 C = 少正则化。小 C = 多正则化。

### 铰链损失：SVM 损失函数

软间隔 SVM 可以重写为无约束优化：

```
minimize    (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))
```

项 max(0, 1 - y_i * f(x_i)) 是铰链损失。当点正确分类且在间隔外时为零。当点在间隔内或错误分类时是线性的。

```
单个点的铰链损失：

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

当 y*f(x) >= 1 时损失为零（正确分类，在间隔外）。
当 y*f(x) < 1 时线性惩罚。
```

与逻辑损失（逻辑回归）比较：

```
铰链:     max(0, 1 - y*f(x))          在间隔处硬截止
逻辑:     log(1 + exp(-y*f(x)))       平滑，永不为零
```

铰链损失产生稀疏解（只有支持向量有非零贡献）。逻辑损失使用所有数据点。这使 SVM 在预测时内存效率更高。

### 用梯度下降训练线性 SVM

你可以用铰链损失加 L2 正则化的梯度下降训练线性 SVM，不需要求解约束 QP：

```
L(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum(max(0, 1 - y_i * (w^T x_i + b)))

对 w 的梯度：
  如果 y_i * (w^T x_i + b) >= 1:  dL/dw = lambda * w
  如果 y_i * (w^T x_i + b) < 1:   dL/dw = lambda * w - y_i * x_i

对 b 的梯度：
  如果 y_i * (w^T x_i + b) >= 1:  dL/db = 0
  如果 y_i * (w^T x_i + b) < 1:   dL/db = -y_i
```

这叫原始形式。每个 epoch 运行在 O(n * d)，其中 n 是样本数，d 是特征数。对于大、高维、稀疏数据（文本分类），这很快。

### 对偶形式和核技巧

SVM 问题的拉格朗日对偶（来自 Phase 1 Lesson 18，KKT 条件）：

```
maximize    sum(alpha_i) - (1/2) * sum_ij(alpha_i * alpha_j * y_i * y_j * (x_i . x_j))
subject to  0 <= alpha_i <= C
            sum(alpha_i * y_i) = 0
```

对偶只涉及数据点之间的点积 x_i . x_j。这是关键洞察。用核函数 K(x_i, x_j) 替换每个点积，SVM 可以在从不显式计算变换的情况下学习非线性边界。

```
线性核:      K(x, z) = x . z
多项式核:  K(x, z) = (x . z + c)^d
RBF（高斯）: K(x, z) = exp(-gamma * ||x - z||^2)
```

RBF 核将数据映射到无限维空间。在输入空间中接近的点有接近 1 的核值。相距甚远的点有接近 0 的核值。它可以学习任何平滑决策边界。

```mermaid
graph LR
    subgraph "输入空间（不可分）"
        A["2D 中的数据点<br>圆形边界"]
    end
    subgraph "特征空间（可分）"
        B["高维中的数据点<br>线性边界"]
    end
    A -->|"核技巧<br>K(x,z) = phi(x).phi(z)"| B
```

核技巧在高维空间中计算点积而无需真的去那里。对于 D 维中 degree d 的多项式核，显式特征空间有 O(D^d) 维。但 K(x, z) 在 O(D) 时间内计算。

### 用于回归的 SVM（SVR）

支持向量回归在数据周围拟合宽度为 epsilon 的管道。管道内的点损失为零。管道外的点被线性惩罚。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i + xi_i*)
subject to  y_i - (w^T x_i + b) <= epsilon + xi_i
            (w^T x_i + b) - y_i <= epsilon + xi_i*
            xi_i, xi_i* >= 0
```

epsilon 参数控制管道宽度。更宽管道 = 更少支持向量 = 更平滑拟合。更窄管道 = 更多支持向量 = 更紧密拟合。

### 为什么 SVM 输给了深度学习（以及何时仍然获胜）

SVM 从 1990 年代末到 2010 年代初主导 ML。深度学习在几个原因上超越了它们：

| 因素 | SVM | 深度学习 |
|------|------|----------|
| 特征工程 | 需要 | 学习特征 |
| 可扩展性 | O(n²) 到 O(n³) 用于核 | O(n) 每 epoch 用 SGD |
| 图像/文本/音频 | 需要手工特征 | 从原始数据学习 |
| 大数据集（>100k） | 慢 | 扩展好 |
| GPU 加速 | 收益有限 | 巨大加速 |

SVM 在以下情况仍然获胜：
- 小数据集（几百到几千）
- 高维稀疏数据（带 TF-IDF 特征的文本）
- 当你需要数学保证（间隔界）时
- 当训练时间必须最少时（线性 SVM 很快）
- 有清晰间隔结构的二分类
- 异常检测（单类 SVM）

## 动手实现

### 步骤 1：铰链损失和梯度

基础。计算一批及其梯度的铰链损失。

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### 步骤 2：通过梯度下降的线性 SVM

通过最小化正则化铰链损失训练。不需要 QP 求解器。

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

### 步骤 3：核函数

实现线性、多项式和 RBF 核。

```python
def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))
```

### 步骤 4：间隔和支持向量识别

训练后，识别哪些点是支持向量并计算间隔宽度。

```python
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors
```

完整实现及所有演示见 `code/svm.py`。

## 用现成库

使用 scikit-learn：

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"准确率: {clf.score(X_test, y_test):.4f}")
print(f"支持向量: {clf['svm'].n_support_}")
```

重要：训练 SVM 前始终缩放特征。SVM 对特征量级敏感，因为间隔依赖于 ||w||，未缩放特征扭曲几何。

对于大数据集，使用 `LinearSVC`（原始形式，O(n) 每 epoch）而不是 `SVC`（对偶形式，O(n²) 到 O(n³)）：

```python
from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])
```

## 练习

1. 生成分离的 2D 线性数据集。训练你的 LinearSVM 并识别支持向量。验证支持向量是离决策边界最近的点。
2. 在噪声数据集上变化 C 从 0.001 到 1000。为每个 C 值绘制决策边界。观察从宽间隔（欠拟合）到窄间隔（过拟合）的过渡。
3. 创建类边界为圆形的数据集（不是线性的）。显示线性 SVM 失败。计算 RBF 核矩阵，显示类在核诱导的特征空间中变得可分。
4. 在同一数据集上比较铰链损失 vs 逻辑损失。训练线性 SVM 和逻辑回归。统计每个模型决策边界贡献了多少训练点（支持向量 vs 所有点）。
5. 实现 SVR（epsilon 不敏感损失）。拟合 y = sin(x) + noise。绘制预测周围的 epsilon 管道并突出支持向量（管道外的点）。

## 关键术语

| 术语 | 真正含义 |
|------|----------|
| 支持向量 | 离决策边界最近的训练点。唯一决定超平面的点 |
| 间隔 | 决策边界与最近支持向量之间的距离。SVM 使其最大化 |
| 铰链损失 | max(0, 1 - y*f(x))。正确分类且在间隔外时为零。否则线性惩罚 |
| C 参数 | 间隔宽度和分类错误之间的权衡。大 C = 窄间隔，小 C = 宽间隔 |
| 软间隔 | 通过松弛变量允许间隔违规的 SVM 公式。处理不可分数据 |
| 核技巧 | 在高维特征空间中计算点积而不显式映射到该空间 |
| 线性核 | K(x, z) = x . z。等价于标准点积。用于线性可分数据 |
| RBF 核 | K(x, z) = exp(-gamma * ||x-z||²)。映射到无限维。学习任何平滑边界 |
| 多项式核 | K(x, z) = (x . z + c)^d。映射到多项式组合的特征空间 |
| 对偶形式 | SVM 问题的重写，只依赖于数据点之间的点积。启用核 |
| SVR | 支持向量回归。在数据周围拟合 epsilon 管道。管道内的点损失为零 |
| 松弛变量 | xi_i：测量点违反间隔的程度。对于正确分类且在间隔外的点为零 |
| 最大间隔 | 选择使每类最近点距离最大化的超平面的原则 |

## 延伸阅读

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1) - SVM 和统计学习的基础文本
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018) - 原始 SVM 论文
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) - 使 SVM 训练实用的 SMO 算法
- [scikit-learn SVM 文档](https://scikit-learn.org/stable/modules/svm.html) - 带实现细节的实践指南
- [LIBSVM: 支持向量机库](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - 大多数 SVM 实现背后的 C++ 库