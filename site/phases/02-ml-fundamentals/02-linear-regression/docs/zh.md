# 从零实现线性回归

> 线性回归通过数据画出一条最佳直线。它是机器学习的"hello world"。

**类型：** Build
**语言：** Python
**前置要求：** Phase 1（线性代数、微积分、优化）、Phase 2 Lesson 1
**时长：** 约 90 分钟

## 学习目标

- 推导均方误差的梯度下降更新规则，从零实现线性回归
- 比较梯度下降和正规方程在计算复杂度上的区别，以及何时使用各自方法
- 构建带特征标准化的多元线性回归模型，并解释学习到的权重
- 解释 Ridge 回归（L2 正则化）如何通过惩罚大权重防止过拟合

## 问题背景

你有数据：房屋面积和销售价格。你想根据面积预测新房屋的价格。你可以凭眼睛在散点图上估计，但你需要一个公式。你需要一条最佳拟合数据的直线，这样你可以输入任意面积得到价格预测。

线性回归给了你这条线。更重要的是，它引入了完整的 ML 训练循环：定义模型，定义成本函数，优化参数。每个 ML 算法都遵循这个模式。在这里用最简单的情况掌握它，你就能在任何地方识别它。

这不仅仅用于简单问题。线性回归用于生产系统中的需求预测、A/B 测试分析、金融建模，以及每个回归任务的基准。

## 核心概念

### 模型

线性回归假设输入 (x) 和输出 (y) 之间存在线性关系：

```
y = wx + b
```

- `w`（权重/斜率）：x 增加 1 时 y 变化多少
- `b`（偏置/截距）：x = 0 时 y 的值

对于多个输入（特征），这扩展为：

```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

或向量形式：`y = w^T * x + b`

目标：找到使预测的 y 在所有训练样本上尽可能接近实际 y 的 w 和 b 值。

### 成本函数（均方误差）

如何衡量"尽可能接近"？你需要单个数字来捕捉预测的错误程度。最常见的选择是均方误差（MSE）：

```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

对每个样本计算误差平方，然后取平均。平方的原因是：
1. 消除负误差（-3 和 +3 都变成 9）
2. 惩罚大误差（4² = 16，比 2² = 4 大得多）

### 梯度下降

为了最小化 MSE，我们迭代调整 w 和 b。每次更新：

```
w = w - learning_rate * gradient
b = b - learning_rate * gradient
```

梯度是 MSE 相对于参数的导数。对于单个特征：

```
∂MSE/∂w = (2/n) * Σ(prediction - actual) * x
∂MSE/∂b = (2/n) * Σ(prediction - actual)
```

### 正规方程

对于小数据集，你可以直接解出最优参数，不需要迭代：

```
θ = (X^T X)^(-1) X^T y
```

优点：无学习率、无迭代、一步得到最优解
缺点：矩阵求逆是 O(n³)，对大特征数不实用

### 特征标准化

线性回归对特征尺度敏感。如果一个特征是房屋面积（1000-5000），另一个是卧室数（1-5），权重会相差很大。

标准化将特征转换为均值为 0、标准差为 1：

```python
def standardize(X):
    return (X - X.mean()) / X.std()
```

### 多元线性回归

现实问题有多个特征。使用所有特征：

```
y = Σ(wi * xi) + b
```

权重 wi 告诉你每个特征对预测的贡献。正权重 = 增加特征增加预测；负权重 = 增加特征减少预测。

### Ridge 回归（L2 正则化）

标准线性回归可能过拟合——权重变得太大，对训练数据拟合得太好而泛化差。

Ridge 添加惩罚项到损失函数：

```
MSE_Ridge = MSE + λ * Σ(wi²)
```

λ（正则化强度）是超参数。λ 越大，权重越小，模型越简单。

为什么有效：
- 惩罚大权重
- 特征权重更均匀分布
- 减少过拟合

## 动手实现

`code/linear_regression.py` 从零实现线性回归和 Ridge 回归。

### 步骤 1：梯度下降实现

```python
def gradient_descent(X, y, lr=0.01, epochs=1000):
    n = len(y)
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(epochs):
        predictions = X @ w + b
        error = predictions - y
        dw = (2/n) * (X.T @ error)
        db = (2/n) * error.sum()
        w -= lr * dw
        b -= lr * db

    return w, b
```

### 步骤 2：正规方程实现

```python
def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y
```

### 步骤 3：Ridge 实现

```python
def ridge_regression(X, y, lambda_):
    n = X.shape[1]
    I = np.eye(n)
    return np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
```

### 步骤 4：在真实数据上测试

使用 scikit-learn 的 diabetes 数据集（回归基准测试）。

## 用现成库

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)
print(f"测试 R²: {model.score(X_test, y_test):.3f}")

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
print(f"Ridge 测试 R²: {ridge.score(X_test, y_test):.3f}")
```

## 产出

本课产出 `outputs/prompt-linear-regression-analyzer.md`——一个分析线性回归结果的 prompt。它解释学习到的权重，检查系数显著性，并识别潜在问题。

## 关键术语

| 术语 | 真正含义 |
|------|----------|
| MSE | 均方误差，衡量预测与真实值差距的平方平均值 |
| 梯度下降 | 迭代向损失函数最小值移动的优化算法 |
| 正规方程 | 一步求解最优参数的闭式解 |
| R² | 决定系数，模型解释的方差比例 |
| L2 正则化 | Ridge 回归，对大权重加惩罚防止过拟合 |

## 练习

1. 在 MSE 中，为什么我们要平方误差而不是取绝对值？
2. 梯度下降的学习率太高会发生什么？太低呢？
3. 何时用正规方程而不是梯度下降？
4. 在有 10000 个特征的数据集上使用 Ridge 回归，解释正则化参数 λ 如何影响权重。

## 延伸阅读

- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) - 直观的线性回归解释
- [Ridge and Lasso Documentation](https://scikit-learn.org/stable/modules/linear_model.html) - sklearn 正则化线性模型