# 数值稳定性

> 浮点数是一个有漏洞的抽象。它会在训练期间咬你，而你不会事先察觉。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 1，课程 01-04
**时间：** 约 120 分钟

## 学习目标

- 使用最大值减去技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点运算中的溢出、下溢和灾难性抵消
- 使用中心差分法验证解析梯度与数值梯度
- 解释为什么训练中首选 bfloat16 而非 float16，以及损失缩放如何防止梯度下溢

## 问题背景

你的模型训练了三个小时，然后损失变成了 NaN。你加了一个 print 语句。logits 在第 9000 步没问题。第 9001 步它们是 `inf`。到第 9002 步，每个梯度都是 `nan`，训练死了。

或者：你的模型训练完成但准确率比论文声称的低 2%。你检查了所有东西。架构匹配。超参数匹配。数据匹配。问题是论文用了 float32 而你用了 float16 而没有正确的缩放。三十二位的累积舍入误差悄悄吃掉了你的准确率。

或者：你从零实现了交叉熵损失。它在小 logits 上工作正常。当 logits 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 大于 float32 能表示的范围。每个 ML 框架都用两行代码处理这个问题。你不知道这个技巧存在。

数值稳定性不是理论上的顾虑。它是训练成功运行与静默失败之间的区别。你最终调试的每个严重 ML bug 都归根结底到浮点数。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点值。浮点数有三个部分：符号位、指数和尾数（有效数字）。

```
Float32 布局（32 位总计）：
[1 位符号] [8 位指数] [23 位尾数]

值 = (-1)^符号 * 2^(指数 - 127) * 1.尾数
```

尾数决定精度（多少有效数字）。指数决定范围（一个数能有多大或多小）。

```
格式      位数   指数    尾数     十进制位数  范围（约）
float64    64     11       52        ~15-16          +/- 1.8e308
float32    32     8        23        ~7-8            +/- 3.4e38
float16    16     5        10        ~3-4            +/- 65,504
bfloat16   16     8        7         ~2-3            +/- 3.4e38
```

float32 给你约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但区分不了 1.00000001 和 1.00000002。7 位之后，一切都是舍入噪声。

float16 给你约 3 位。最大可表示数是 65,504。这对于 logits、梯度和激活经常超过此值的 ML 来说小得令人不安。

bfloat16 是 Google 对 float16 范围问题的回答。它与 float32 具有相同的 8 位指数（相同范围，高达 3.4e38），但只有 7 位尾数（精度低于 float16）。对于训练神经网络，范围比精度更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 在二进制浮点数中无法精确表示。在二进制中，它是重复分数：

```
0.1 在二进制 = 0.0001100110011001100110011...（永远重复）
```

Float32 将此截断为 23 位尾数。存储的值大约是 0.100000001490116。类似地，0.2 存储为大约 0.200000002980232。它们的和是 0.300000004470348，不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 的影响：

1. 损失比较如 `if loss < threshold` 可能给出错误答案
2. 累积许多小值（数千步的梯度更新）会偏离真实和
3. 如果用 `==` 比较浮点数，校验和和可重现性测试会失败

修复：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你减去两个几乎相等的浮点数时，有效数字相互抵消，你只剩下被提升到前导位的舍入噪声。

```
a = 1.0000001    （在 float32 中存储为 1.00000011920929）
b = 1.0000000    （在 float32 中存储为 1.00000000000000）

真实差值：  0.0000001
计算值：   0.00000011920929

相对误差： 19.2%
```

单次减法就有 19% 的相对误差。在 ML 中，这发生在：
- 计算大方差数据时：`E[x^2] - E[x]^2` 当 E[x] 很大时
- 减去几乎相等的对数概率
- 用太小的 epsilon 计算有限差分梯度

修复：重新排列公式以避免减去大而几乎相等的数。对于方差，使用 Welford 算法或先中心化数据。对于对数概率，全程在对数空间中工作。

### 溢出与下溢

溢出发生在结果太大而无法表示时。下溢发生在结果太小（接近零）而小于最小可表示正数时。

```
Float32 边界：
  最大值：   3.4028235e+38
  最小正数（正规）： 1.175e-38
  最小正数（次正规）： 1.401e-45
  溢出：   任何 > 3.4e38 的数变成 inf
  下溢：   任何 < 1.4e-45 的数变成 0.0
```

`exp()` 函数是 ML 中溢出的主要来源：

```
exp(88.7)  = 3.40e+38   （勉强放入 float32）
exp(89.0)  = inf         （溢出）
exp(-87.3) = 1.18e-38   （勉强高于下溢）
exp(-104)  = 0.0         （下溢到零）
```

`log()` 函数走向另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （没问题）
log(1e-46) = -inf        （输入下溢到 0，然后 log(0) = -inf）
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。组合 `log(exp(x))` 没有正确的技巧就是雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 溢出。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 下溢到零，而 `log(0)` 是 `-inf`。

技巧：在指数化前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`。不可能溢出。至少有一项是 1，所以和至少是 1，而 `log(1) = 0`。不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （加减 c）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               （提取 exp(c)）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

设 `c = max(x)`，溢出被消除。

此技巧在 ML 中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合
- 变分推断

### 为什么 Softmax 需要最大值减去技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有此技巧，[100, 101, 102] 的 logits 导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些在 float32 中溢出（最大 ~3.4e38）？不，2.69e43 < 3.4e38？实际上：
exp(88.7) 已达到 float32 极限。
exp(100) 在 float32 中为 inf。
```

使用此技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率是相同的。计算是安全的。这不是优化。这是正确性的要求。

### NaN 和 Inf：检测与预防

`nan`（非数）和 `inf`（无穷）在计算中病毒式传播。一个梯度更新中的 `nan` 使权重变为 `nan`，这使每个后续输出都变成 `nan`。训练在一步内死亡。

`inf` 是如何产生的：
- 大正数的 `exp()`
- 除以零：`1.0 / 0.0`
- 累加中的 `float32` 溢出

`nan` 是如何产生的：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 任何涉及已有 `nan` 的算术

检测：

```python
import math

math.isnan(x)       # 如果 x 是 nan 返回 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 返回 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 返回 True
```

预防策略：

1. 将输入限制在 `exp()`：`exp(clamp(x, -80, 80))`
2. 在分母中添加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部添加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪以防止权重爆炸
6. 调试时在每次前向传播后检查 `nan`/`inf`

### 数值梯度检验

解析梯度（来自反向传播）可能有 bug。数值梯度检验通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精确，比前向差分 `(f(x+h) - f(x)) / h`（仅 O(h)）好得多。

选择 h：太大则近似错误，太小则灾难性抵消破坏答案。`h = 1e-5` 到 `1e-7` 是典型值。

检验：计算解析梯度和数值梯度的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验规则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

实现新层或损失函数时始终检验梯度。PyTorch 提供 `torch.autograd.gradcheck()` 来执行此操作。

### 混合精度训练

现代 GPU 有专门的硬件（Tensor Core），计算 float16 矩阵乘法比 float32 快 2-8 倍。混合精度训练利用这一点：

```
1. 维护 float32 主权重副本
2. 前向传播用 float16（快）
3. 用 float32 计算损失（防止溢出）
4. 反向传播用 float16（快）
5. 将梯度转换为 float32
6. 更新 float32 主权重
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。Float16 将低于约 6e-8 的任何值下溢到零。你的模型停止学习，因为所有梯度更新都是零。

修复是损失缩放：

```
1. 将损失乘以一个大比例因子（如 1024）
2. 反向传播计算（loss * 1024）的梯度
3. 所有梯度大 1024 倍（推至 float16 可表示范围上方）
4. 更新权重前将梯度除以 1024
5. 净效果：相同的更新，但没有下溢
```

动态损失缩放自动调整比例因子。从大值（65536）开始。如果梯度溢出到 `inf`，减半。如果 N 步无溢出，翻倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中胜出

```
float16：   [1 位符号] [5 位指数]  [10 位尾数]
bfloat16：  [1 位符号] [8 位指数]  [7 位尾数]
```

float16 精度更高（10 位尾数 vs 7 位）但范围有限（最大 ~65,504）。bfloat16 精度较低但与 float32 范围相同（最大 ~3.4e38）。

对于训练神经网络：

- 激活和 logits 在训练期间经常超过 65,504。float16 溢出；bfloat16 处理。
- 损失缩放在 float16 中必要但在 bfloat16 中通常不需要，因为其范围覆盖梯度量级谱。
- bfloat16 是 float32 的简单截断：丢弃尾数底部 16 位。转换平凡且指数无损。

float16 优于是推理的地方因为值有界且精度更重要。bfloat16 优于是训练的地方因为范围更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）有原生 bfloat16 支持。

### 梯度裁剪

当梯度通过多层指数增长时（常见于 RNN、深度网络和 transformer），会发生梯度爆炸。单个大梯度可以在一步中破坏所有权重。

两种裁剪类型：

**按值裁剪：** 独立地 clamp 每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度方向。这是 `torch.nn.utils.clip_grad_norm_()` 做的事。是标准选择。

典型值：transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，更简单网络用 `max_norm=5.0`。

梯度裁剪不是 hack。它是安全机制。没有它，单个异常批次可以产生大到足以毁掉数周训练的梯度。

### 归一化层作为数值稳定器

批归一化、层归一化和 RMS 归一化通常作为帮助训练收敛的正则器呈现。它们也是数值稳定器。

没有归一化，激活可以通过多层指数增长或收缩：

```
第 1 层：值在 [0, 1]
第 5 层：值在 [0, 100]
第 10 层：值在 [0, 10,000]
第 50 层：值在 [0, inf]
```

归一化在每一层重新中心和重新缩放激活：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常 1e-5）在所有激活相同时防止除以零。学习参数 `gamma` 和 `beta` 让网络恢复它需要的任何缩放。

这在整个网络中保持值处于数值安全范围内，防止前向传播中的溢出和反向传播中的梯度爆炸。

### 常见 ML 数值 Bug

**Bug：几个 epoch 后损失变为 NaN。**
原因：logits 增长太大，softmax 溢出。或者学习率太高，权重发散。
修复：使用稳定 softmax（最大减去），降低学习率，添加梯度裁剪。

**Bug：损失停留在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本不学习。
修复：检查数据标签是否正确，验证损失函数，检查死 ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度没有正确的损失缩放。梯度下溢静默地将小更新清零。
修复：启用动态损失缩放，或切换到 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：死 ReLU 神经元（所有输入为负），或 float16 下溢。
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个 GPU 上工作但在另一个上给出不同结果。**
原因：非确定性浮点累加顺序。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不是结合的。
修复：接受小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：`exp()` 在损失计算中返回 `inf`。**
原因：原始 logits 直接传给 `exp()` 而没有最大减去技巧。
修复：使用 `torch.nn.functional.log_softmax()`，它在内部实现 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度量级或高于 65,504 的激活。
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

## 构建实现

### 步骤 1：演示浮点精度限制

```python
print("=== 浮点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差异: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素 vs 稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"朴素:  {softmax_naive(safe_logits)}")
print(f"稳定: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"稳定: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 步骤 3：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"朴素:  {logsumexp_naive(safe):.6f}")
print(f"稳定: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"稳定: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 步骤 4：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"朴素:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"稳定: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检验

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用方法

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原始范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪范数:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保持: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

参见 `code/numerical.py` 获取包含所有边界情况演示的完整实现。

## 交付成果

本课程产出：
- `code/numerical.py`，包含稳定的 softmax、log-sum-exp、交叉熵、梯度检验和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现在 Phase 3 构建训练循环时以及 Phase 4 实现注意力机制时重新出现。

## 练习

1. **灾难性抵消。** 使用朴素公式 `E[x^2] - E[x]^2` 在 float32 中计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 在线算法计算。相对于真实方差（0.6667）比较误差。

2. **精度搜索。** 找到最小的正 float32 值 `x`，使得 `1.0 + x == 1.0` 在 Python 中。这是机器 epsilon。验证它匹配 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp 边界情况。** 用以下条件测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其余，(c) 所有值非常负（-1000）。验证它在朴素版本失败的地方给出正确结果。

4. **神经网络层的梯度检验。** 实现单个线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证 3×2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建范围在 [1e-9, 1e-3] 内的随机梯度，转换为 float16，测量变为零的比例。然后应用损失缩放（乘以 1024），转换为 float16，缩放回来，再次测量零的比例。

## 核心术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现它。 |
| 机器 epsilon | "精度极限" | 使得 `1.0 + e != 1.0` 在给定浮点格式中的最小值 e。对于 float32，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度损失" | 当减去几乎相等的浮点数时，有效数字抵消而舍入噪声支配结果。 |
| 溢出 | "数太大" | 结果超过最大可表示值并变成 inf。exp(89) 在 float32 中溢出。 |
| 下溢 | "数太小" | 结果接近零而小于最小可表示正数并变成 0.0。exp(-104) 在 float32 中下溢。 |
| Log-sum-exp 技巧 | "先减去最大值" | 通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，防止溢出和下溢。用于 softmax、交叉熵和对数概率运算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在指数化前减去 max(logits)。数值上相同的结果，不可能溢出。 |
| 梯度检验 | "验证你的反向传播" | 将反向传播的解析梯度与有限差分的数值梯度进行比较，以捕获实现 bug。 |
| 混合精度 | "前向 float16，反向 float32" | 在速度关键操作中使用较低精度浮点，在数值敏感操作中使用较高精度浮点。典型加速为 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播前将损失乘以大常数，使梯度保持在 float16 可表示范围内，然后在权重更新前除以相同常数。 |
| bfloat16 | "脑浮点" | Google 的 16 位格式，具有 8 位指数（与 float32 相同范围）和 7 位尾数（精度低于 float16）。优于是训练。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量，使其范数不超过阈值。防止爆炸梯度破坏权重。 |
| NaN | "非数" | 来自未定义操作（0/0、inf-inf、sqrt(-1)）的特殊浮点值。通过所有后续算术传播。 |
| Inf | "无穷" | 来自溢出或除以零的特殊浮点值。可以组合产生 NaN（inf-inf、inf*0）。 |
| 数值梯度 | "暴力求导" | 通过评估 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。慢但可靠，用于验证。 |

## 拓展阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — 权威参考，密集但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) — 引入 float16 训练损失缩放的 NVIDIA 论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) — PyTorch 中混合精度的实用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) — 为什么 Google 选择这种格式用于 TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) — 减少浮点和中舍入误差的算法