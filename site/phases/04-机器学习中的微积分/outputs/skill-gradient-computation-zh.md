---
name: skill-gradient-computation-zh
description: 计算常见 ML 损失函数、激活函数和层操作的梯度，并选择合适的求导方法
version: 1.0.0
phase: 1
lesson: 4
tags: [calculus, gradients, backpropagation]
---

# ML 中的梯度计算

神经网络中损失函数、激活函数和层操作梯度计算的实用参考。

## 决策清单

1. 函数是否由简单原语（幂、指数、对数、三角函数）组成？使用解析导数和链式法则。
2. 函数是自定义或黑盒操作？使用数值微分：`(f(x+h) - f(x-h)) / (2h)`，h = 1e-7。
3. 函数是否由 PyTorch/JAX 中的张量操作构建？让 autograd 处理。用数值检查验证。
4. 是否需要标量损失对权重矩阵的梯度？通过计算图逐节点应用链式法则。
5. 是否存在不可微操作（argmax、取整、采样）？使用直通估计器或重参数化技巧。

## 何时使用每种方法

| 方法 | 何时使用 | 代价 |
|---|---|---|
| 解析（手推） | 简单函数、验证 autograd 输出 | 运行时免费 |
| 数值（有限差分） | 调试、梯度检查、黑盒函数 | n 个参数需要 2n 次前向传播 |
| 自动微分 | 任何可微计算图（默认） | 一次反向传播 |
| 符号（SymPy, Mathematica） | 推导论文中的闭式梯度 | 仅编译时 |

## 常见导数速查表

| 函数 | f(x) | f'(x) | ML 场景 |
|---|---|---|---|
| MSE 损失 | (1/n) sum(y_hat - y)^2 | (2/n)(y_hat - y) | 回归 |
| 交叉熵（二分类） | -(y log(p) + (1-y) log(1-p)) | p - y（经过 sigmoid 后） | 二分类 |
| 交叉熵（多分类） | -log(p_true_class) | p - one_hot(y)（经过 softmax 后） | 多分类 |
| Sigmoid | 1 / (1 + e^(-x)) | sigma(x) * (1 - sigma(x)) | 输出门、二值输出 |
| Tanh | (e^x - e^(-x)) / (e^x + e^(-x)) | 1 - tanh(x)^2 | 隐藏层激活（遗留） |
| ReLU | max(0, x) | x > 0 时为 1，x < 0 时为 0 | 默认隐藏激活 |
| Leaky ReLU | max(0.01x, x) | x > 0 时为 1，x < 0 时为 0.01 | 避免神经元死亡 |
| GELU | x * Phi(x) | Phi(x) + x * phi(x) | Transformer |
| Softmax_i | e^(x_i) / sum(e^(x_j)) | i=j 时 s_i(1 - s_i)，i≠j 时 -s_i*s_j | 输出层（Jacobian） |
| Log-softmax | x_i - log(sum(e^(x_j))) | 第 i 项为 1 - softmax(x_i) | 数值稳定的 CE |
| 线性层 | y = Wx + b | dL/dW = dL/dy * x^T, dL/db = dL/dy | 每一层 |
| L2 正则化 | lambda * sum(w^2) | 2 * lambda * w | 权重衰减 |
| L1 正则化 | lambda * sum(|w|) | lambda * sign(w) | 稀疏性 |

## 常见错误

- 忘记批平均损失（MSE、交叉熵）中的 1/n 因子。梯度会被批量大小缩放。
- 把 softmax 梯度当作向量计算，而它实际上是 Jacobian 矩阵。对于交叉熵 + softmax 组合，梯度简化为 (p - y)，避免了完整 Jacobian。
- 链式法则顺序搞反。从损失往回推：dL/dW = dL/dy * dy/dW。
- 数值导数用的 h 太大（h = 0.1）或太小（h = 1e-15）。float64 坚持 h = 1e-7。
- 忘记 ReLU 在 x = 0 处导数未定义。实践中设为 0 或 0.5。

## 梯度检查配方

```
对每个参数 w:
  numeric_grad = (loss(w + h) - loss(w - h)) / (2h)
  auto_grad = 反向传播值
  relative_error = |numeric - auto| / max(|numeric|, |auto|, 1e-8)
  assert relative_error < 1e-5
```

相对误差超过 1e-3 说明有问题。在 1e-5 到 1e-3 之间，需要调查。
